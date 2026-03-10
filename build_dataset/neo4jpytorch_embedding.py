"""
双模型图数据导出器 - 导出IOC子图和TTP因果序列

输出文件:
- apt_kg_ioc.pt: IOC异构图（IP/domain/URL/File/CVE/ASN + EVENT）
- apt_kg_ttp.pt: TTP因果序列 + 技术语义嵌入

核心功能:
1. 按MITRE ATT&CK战术阶段生成因果序列 (T1566→T1059→T1547)
2. 技术语义嵌入 (384维 sentence-transformers)
3. IOC节点使用可学习嵌入压缩特征
"""

import os
import json
import warnings
from math import log2
from typing import Tuple
from urllib.parse import urlparse
from collections import defaultdict
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# 尝试导入sentence-transformers（用于Technique语义嵌入）
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMER = True

    # 本地模型路径
    LOCAL_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models',
        'all-MiniLM-L6-v2'
    )

except ImportError:
    HAS_SENTENCE_TRANSFORMER = False
    LOCAL_MODEL_PATH = None
    print("[Warning] sentence-transformers未安装，Technique节点将不包含语义特征")
    print("         安装: pip install sentence-transformers")

# 尝试导入Node2Vec（可选依赖，需要pyg-lib或torch-cluster）
# Node2Vec需要运行时检查，需要实际测试实例化
try:
    from torch_geometric.nn import Node2Vec as _Node2Vec
    # 测试是否可以实例化（需要pyg-lib或torch-cluster）
    import torch
    _test_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    try:
        _Node2Vec(_test_edge_index, walk_length=20, context_size=10, embedding_dim=16, sparse=True)
        Node2Vec = _Node2Vec
        HAS_NODE2VEC = True
        print("[OK] Node2Vec可用，将使用64维图结构特征增强")
    except (ImportError, Exception) as e:
        HAS_NODE2VEC = False
        Node2Vec = None
        print("[Warning] Node2Vec运行时依赖不可用，将跳过节点2vec特征增强")
        print(f"         错误: {e}")
        print("         安装: pip install torch-cluster 或 pip install pyg-lib")
        # 清理测试变量
        del _test_edge_index
except ImportError:
    HAS_NODE2VEC = False
    Node2Vec = None
    print("[Warning] Node2Vec未安装，将跳过节点2vec特征增强")
    print("         安装: pip install torch-cluster 或 pip install pyg-lib")


# ============================================================================
# 嵌入编码器
# ============================================================================

class EdgeSemanticEncoder:
    """
    边语义编码器 - 从边类型提取语义特征

    特征维度: [是否EVENT关联, 是否IOC关联, 是否相似关系, 方向性]
    """
    def __init__(self):
        # EVENT关联的边类型
        self.event_edges = {
            'USES_INFRASTRUCTURE', 'USES_DOMAIN', 'DELIVERS_VIA_URL',
            'EXPLOITS_VULN', 'DROPS_MALWARE', 'USES_TECHNIQUE'  # 新增
        }
        # IOC间关联的边类型
        self.ioc_edges = {
            'BELONGS_TO_NETWORK', 'RESOLVES_TO', 'RESOLVES_FROM',
            'HOSTED_ON_DOMAIN', 'RESOLVES_TO_IP'
        }
        # 相似关系
        self.similarity_edges = {'SIMILAR_TO'}

    def get_edge_semantic_features(self, edge_type: Tuple[str, str, str]) -> np.ndarray:
        """
        获取边的语义特征 [4维]

        Returns:
            [is_event_edge, is_ioc_edge, is_similarity, direction]
        """
        src, rel, dst = edge_type

        feat = np.zeros(4, dtype=np.float32)

        # 1. 是否为EVENT关联边
        if src == 'EVENT':
            feat[0] = 1.0
        elif rel in self.event_edges:
            feat[0] = 1.0

        # 2. 是否为IOC间关联边
        if rel in self.ioc_edges:
            feat[1] = 1.0

        # 3. 是否为相似关系
        if rel in self.similarity_edges:
            feat[2] = 1.0

        # 4. 方向性 (0=无向/双向, 1=单向)
        if rel in {'SIMILAR_TO'}:
            feat[3] = 0.0  # 无向
        else:
            feat[3] = 1.0  # 有向

        return feat


# ============================================
# MITRE ATT&CK 战术阶段逻辑顺序（用于因果序列生成）
# ============================================
TACTIC_PHASE_ORDER = {
    # reconnaissance
    'TA0049': 0,  # Reconnaissance
    # resource development
    'TA0042': 1,  # Resource Development
    # initial access
    'TA0001': 2,  # Initial Access
    # execution
    'TA0002': 3,  # Execution
    # persistence
    'TA0003': 4,  # Persistence
    # privilege escalation
    'TA0004': 5,  # Privilege Escalation
    # defense evasion
    'TA0005': 6,  # Defense Evasion
    # credential access
    'TA0006': 7,  # Credential Access
    # discovery
    'TA0007': 8,  # Discovery
    # lateral movement
    'TA0008': 9,  # Lateral Movement
    # collection
    'TA0009': 10, # Collection
    # command and control
    'TA0011': 11, # Command and Control
    # exfiltration
    'TA0010': 12, # Exfiltration
    # impact
    'TA0040': 13  # Impact
}


def get_technique_phase_order(tech_id, tactic_mapping):
    """
    获取技术所属的战术阶段顺序

    Args:
        tech_id: 技术ID (如 'T1566')
        tactic_mapping: 技术→战术映射字典

    Returns:
        int: 阶段顺序 (0-13), 默认7(Execution)
    """
    if tech_id not in tactic_mapping:
        return 7  # 默认为execution阶段

    tactics = tactic_mapping[tech_id]
    min_phase = 999

    for tactic in tactics:
        tactic_id = tactic['tactic_id']
        if tactic_id in TACTIC_PHASE_ORDER:
            min_phase = min(min_phase, TACTIC_PHASE_ORDER[tactic_id])

    return min_phase if min_phase != 999 else 7


def generate_causal_sequence(tech_ids, tactic_mapping):
    """
    按战术阶段逻辑顺序生成因果序列

    Args:
        tech_ids: List[str] 技术ID列表（无序）
        tactic_mapping: 技术→战术映射字典

    Returns:
        List[str]: 按因果顺序排列的技术ID列表
    """
    # 为每个技术分配阶段顺序
    tech_with_phase = []
    for tech_id in tech_ids:
        phase = get_technique_phase_order(tech_id, tactic_mapping)
        tech_with_phase.append((tech_id, phase))

    # 按阶段排序，同阶段内按技术ID排序（保证稳定性）
    tech_with_phase.sort(key=lambda x: (x[1], x[0]))

    # 返回排序后的技术ID列表
    return [tech_id for tech_id, _ in tech_with_phase]


class EmbeddingEncoder:
    """
    嵌入编码器 - 将离散类别映射到低维连续空间

    特点:
    1. 可学习: 嵌入向量通过训练学习
    2. 低维: 将高维one-hot压缩到8-32维
    3. 语义: 相似的类别在嵌入空间接近
    """

    def __init__(self, categories, embed_dim=16, init_method='xavier'):
        """
        Args:
            categories: 类别列表
            embed_dim: 嵌入维度
            init_method: 初始化方法 ('xavier', 'normal', 'uniform')
        """
        self.categories = list(set(categories))  # 去重
        self.num_categories = len(self.categories)
        self.embed_dim = embed_dim

        # 构建类别到索引的映射
        self.cat_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_cat = {idx: cat for cat, idx in self.cat_to_idx.items()}

        # 初始化嵌入矩阵
        self.embeddings = self._init_embeddings(init_method)

    def _init_embeddings(self, method):
        """初始化嵌入矩阵"""
        if method == 'xavier':
            # Xavier均匀初始化
            bound = np.sqrt(6.0 / (self.num_categories + self.embed_dim))
            embeddings = np.random.uniform(
                -bound, bound,
                (self.num_categories, self.embed_dim)
            ).astype(np.float32)
        elif method == 'normal':
            # 正态分布初始化
            embeddings = np.random.normal(
                0, 0.1,
                (self.num_categories, self.embed_dim)
            ).astype(np.float32)
        else:  # uniform
            embeddings = np.random.uniform(
                -0.1, 0.1,
                (self.num_categories, self.embed_dim)
            ).astype(np.float32)

        return embeddings

    def get_embedding(self, category):
        """获取单个类别的嵌入"""
        idx = self.cat_to_idx.get(category, 0)
        return self.embeddings[idx]

    def encode(self, categories):
        """编码类别列表"""
        indices = [self.cat_to_idx.get(cat, 0) for cat in categories]
        return self.embeddings[indices]

    def save(self, filepath):
        """保存编码器"""
        data = {
            'categories': self.categories,
            'cat_to_idx': self.cat_to_idx,
            'embeddings': self.embeddings.tolist(),
            'embed_dim': self.embed_dim
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath):
        """加载编码器"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        encoder = cls.__new__(cls)
        encoder.categories = data['categories']
        encoder.cat_to_idx = {k: int(v) for k, v in data['cat_to_idx'].items()}
        encoder.embeddings = np.array(data['embeddings'], dtype=np.float32)
        encoder.embed_dim = data['embed_dim']
        encoder.num_categories = len(encoder.categories)

        return encoder


class FrequencyEncoder:
    """频率编码器 - 使用类别频率作为特征"""

    def __init__(self):
        self.freq_map = {}
        self.max_freq = 0

    def fit(self, categories):
        """学习类别频率"""
        from collections import Counter
        freq = Counter(categories)
        self.max_freq = max(freq.values())

        # 归一化频率
        for cat, count in freq.items():
            self.freq_map[cat] = count / self.max_freq

    def encode(self, categories, dim=1):
        """编码类别列表"""
        freqs = [self.freq_map.get(cat, 0.0) for cat in categories]
        return np.array(freqs, dtype=np.float32).reshape(-1, dim)


class HashEncoder:
    """哈希编码器 - 使用哈希函数生成固定长度向量"""

    def __init__(self, dim=8):
        self.dim = dim

    def _hash_to_vector(self, category):
        """将类别哈希到向量"""
        import hashlib
        hash_obj = hashlib.md5(str(category).encode())
        hash_hex = hash_obj.hexdigest()

        # 转换为向量
        vector = np.array([
            int(hash_hex[i:i+2], 16) / 255.0
            for i in range(0, min(len(hash_hex), self.dim * 2), 2)
        ])

        # 填充或截断到指定维度
        if len(vector) < self.dim:
            vector = np.pad(vector, (0, self.dim - len(vector)))
        else:
            vector = vector[:self.dim]

        return vector.astype(np.float32)

    def encode(self, categories):
        """编码类别列表"""
        return np.vstack([self._hash_to_vector(cat) for cat in categories])


# ============================================================================
# 辅助函数
# ============================================================================

def get_country_code_mapper():
    """获取国家代码映射器"""
    fname = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'feature_extraction',
        'helper_files',
        'country_codes.csv'
    )
    if not os.path.exists(fname):
        return {}

    ccs = {}
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            two, three = line.split(',')
            ccs[two] = len(ccs)
            ccs[three] = len(ccs)
    return ccs


def get_tld_headers(top_k=50):
    """获取顶级域名列表"""
    fname = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'feature_extraction',
        'helper_files',
        'ranked_tlds.csv'
    )
    if not os.path.exists(fname):
        return ['COM', 'NET', 'ORG', 'INFO', 'BIZ', 'IO', 'CO', 'XYZ'][:top_k]

    tlds = []
    with open(fname, 'r') as f:
        f.readline()
        for _ in range(top_k):
            line = f.readline()
            if not line:
                break
            tlds.append(line.split(',')[0].strip())
    return tlds


def nlp_features_domain(s):
    """域名NLP特征"""
    if not s or len(s) == 0:
        return {
            'domain_entropy': 0.0,
            'domain_length': 0,
            'num_digits': 0,
            'subdomains': 0
        }

    probs = [s.count(c) / len(s) for c in set(s)]
    entropy = -sum([p * log2(p) for p in probs])

    return {
        'domain_entropy': entropy,
        'domain_length': len(s),
        'num_digits': len([d for d in s if d.isdigit()]),
        'subdomains': s.count('.')
    }


def nlp_features_url(ioc):
    """URL词汇特征"""
    if not ioc or len(ioc) == 0:
        return {
            'url_entropy': 0.0,
            'url_path_entropy': 0.0,
            'url_length': 0,
            'num_periods': 0,
            'num_subdir': 0,
            'num_digits': 0,
            'num_frag': 0,
            'num_params': 0,
            'url_path_length': 0,
            'url_host_length': 0,
            'has_port': 0
        }

    parsed = urlparse(ioc)

    has_port = parsed.netloc.split(':')
    has_port = int(len(has_port) > 1 and has_port[-1].isdigit())

    dirs = parsed.path.strip('/').split('/')
    num_subdirectories = len(dirs) if dirs and dirs[0] != '' else 0

    frags = parsed.fragment
    num_fragments = 0 if not frags or frags == '' else len(frags.split('#'))

    params = parsed.query
    num_params = 0 if not params or params == '' else len(params.split('&'))

    def entropy(s):
        if not s or len(s) == 0:
            return 0.0
        probs = [s.count(c) / len(s) for c in set(s)]
        return -sum([p * log2(p) for p in probs])

    return {
        'url_entropy': entropy(ioc),
        'url_path_entropy': entropy(parsed.path),
        'url_length': len(ioc),
        'num_periods': ioc.count('.'),
        'num_subdir': num_subdirectories,
        'num_digits': len([i for i in ioc if i.isdigit()]),
        'num_frag': num_fragments,
        'num_params': num_params,
        'url_path_length': len(parsed.path),
        'url_host_length': len(parsed.netloc),
        'has_port': has_port
    }


def extract_tld(url, top_tlds):
    """提取TLD"""
    if not url or '.' not in url:
        return 'UNKNOWN'

    tld = url.split('.')[-1].split('/')[0].split(':')[0].upper()
    return tld if tld in top_tlds else 'OTHER'


# ============================================================================
# 改进的图导出器
# ============================================================================

class ImprovedGraphExporter:
    """
    改进的图数据导出器 - 使用嵌入编码替代one-hot

    特征维度对比:
    - IP: 5272维 -> 40维 (压缩99%!)
    - domain: 58维 -> 28维
    - URL: 78维 -> 48维
    - File: 3维 -> 20维
    """

    def __init__(self, uri, user, pwd):
        print("[DEBUG] __init__ 开始", flush=True)
        # 保存认证信息，用于后续创建EnhancedGraphExporter
        self.uri = uri
        self.user = user
        self.pwd = pwd
        print("[DEBUG] 创建Neo4j驱动...", flush=True)
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
        print("[DEBUG] Neo4j驱动创建完成", flush=True)
        self.data = HeteroData()
        self.node_mapping = {}
        self.apt_encoder = LabelEncoder()

        # 编码器字典
        self.encoders = {}

        # 辅助数据
        print("[DEBUG] 加载TLD列表...", flush=True)
        self.top_tlds = get_tld_headers(top_k=50)
        print("[DEBUG] TLD列表加载完成", flush=True)

        # 边语义编码器
        print("[DEBUG] 创建边语义编码器...", flush=True)
        self.edge_semantic_encoder = EdgeSemanticEncoder()
        print("[DEBUG] 边语义编码器创建完成", flush=True)

        # 保存特征配置
        self.feature_config = {}

        # 新增：Technique稀有度缓存 (用于边权重计算)
        self.technique_rarity_map = {}  # {technique_id: rarity_score}

    def close(self):
        self.driver.close()

    def _run_query_df(self, query, params=None, max_retries=3):
        """执行Neo4j查询并返回DataFrame，带重试机制"""
        import time
        from neo4j.exceptions import ServiceUnavailable, TransientError

        for attempt in range(max_retries):
            try:
                with self.driver.session() as session:
                    result = session.run(query, params or {})
                    data = [r.values() for r in result]
                    if not data:
                        return pd.DataFrame()
                    return pd.DataFrame(data, columns=result.keys())
            except (ServiceUnavailable, TransientError) as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 递增等待时间：5, 10, 15秒
                    print(f"    [WARN] Neo4j查询失败，{wait_time}秒后重试 ({attempt + 1}/{max_retries}): {str(e)[:100]}")
                    time.sleep(wait_time)
                else:
                    print(f"    [ERROR] Neo4j查询失败，已达最大重试次数: {str(e)[:100]}")
                    raise
            except Exception as e:
                print(f"    [ERROR] Neo4j查询出错: {str(e)[:200]}")
                raise

    def _register_nodes(self, label, neo4j_ids_series):
        if neo4j_ids_series.empty:
            return 0
        unique_ids = neo4j_ids_series.unique()
        count = len(unique_ids)
        # 调试输出
        if label == 'CVE':
            print(f"       [_register_nodes CVE DEBUG] 输入series长度: {len(neo4j_ids_series)}, 唯一ID数: {len(unique_ids)}, 返回count: {count}")
        mapping = pd.Series(data=np.arange(count), index=unique_ids)
        self.node_mapping[label] = mapping
        return count

    def build_rich_node_features(self):
        """构建改进的节点特征"""
        print("\n[Step 1] 构建节点特征（使用嵌入编码）...")

        # ==================== IP ====================
        print("    -> Processing IP...")
        query = """
        MATCH (n:IP)
        OPTIONAL MATCH (n) -[r:BELONGS_TO_NETWORK]-> (a:ASN)
        RETURN n.value as id,
               n.country_code as cc,
               n.latitude as lat,
               n.longitude as lon,
               n.lat_norm as lat_norm,
               n.lon_norm as lon_norm,
               a.issuer as issuer
        """
        df = self._run_query_df(query)
        if not df.empty:
            num = self._register_nodes('IP', df['id'])

            # 1. 经纬度 (2维)
            lat_norm = df['lat_norm'].fillna(0).values.astype(np.float32).reshape(-1, 1)
            lon_norm = df['lon_norm'].fillna(0).values.astype(np.float32).reshape(-1, 1)

            # 2. 国家代码嵌入 (32维，替代498维one-hot)
            cc_values = df['cc'].fillna('UNKNOWN').values
            cc_encoder = EmbeddingEncoder(cc_values, embed_dim=32)
            cc_embed = cc_encoder.encode(cc_values)
            self.encoders['IP_country_code'] = cc_encoder

            # 3. 发行商嵌入 (64维，替代4772维one-hot)
            iss_values = df['issuer'].fillna('UNKNOWN').values
            iss_encoder = EmbeddingEncoder(iss_values, embed_dim=64)
            iss_embed = iss_encoder.encode(iss_values)
            self.encoders['IP_issuer'] = iss_encoder

            # 4. 发行商频率编码 (1维)
            freq_encoder = FrequencyEncoder()
            freq_encoder.fit(iss_values)
            iss_freq = freq_encoder.encode(iss_values)

            # 拼接所有特征: 2 + 32 + 64 + 1 = 99维 (原来是5272维)
            feats = np.hstack([lat_norm, lon_norm, cc_embed, iss_embed, iss_freq]).astype(np.float32)

            self.data['IP'].x = torch.tensor(feats, dtype=torch.float32)
            self.data['IP'].num_nodes = num

            self.feature_config['IP'] = {
                'dim': feats.shape[1],
                'components': {
                    'lat_lon': 2,
                    'cc_embed': 32,
                    'issuer_embed': 64,
                    'issuer_freq': 1
                }
            }

            print(f"       IP: {num} nodes, dim={feats.shape[1]} (原5272维, 压缩98%)")

        # ==================== domain ====================
        print("    -> Processing Domain...")
        query = """
        MATCH (n:domain)
        RETURN n.value as id,
               n.first_seen as fs,
               n.last_seen as ls,
               n.has_nxdomain as nx,
               n.lifespan_log as lifespan_log
        """
        df = self._run_query_df(query)
        if not df.empty:
            num = self._register_nodes('domain', df['id'])

            # 1. NLP特征 (4维)
            nlp_feats = np.array([
                [
                    nlp_features_domain(val)['domain_entropy'],
                    nlp_features_domain(val)['domain_length'],
                    nlp_features_domain(val)['num_digits'],
                    nlp_features_domain(val)['subdomains']
                ]
                for val in df['id']
            ], dtype=np.float32)

            # 2. TLD嵌入 (16维，替代50维one-hot)
            tld_values = [extract_tld(val, self.top_tlds) for val in df['id']]
            tld_encoder = EmbeddingEncoder(tld_values, embed_dim=16)
            tld_embed = tld_encoder.encode(tld_values)
            self.encoders['domain_TLD'] = tld_encoder

            # 3. 时间特征 (2维)
            def parse_timestamp(ts):
                if not ts or pd.isna(ts) or ts == 0:
                    return 0.0
                if isinstance(ts, (int, float)):
                    return float(ts)
                try:
                    from dateutil.parser import parse
                    return float(parse(ts).timestamp())
                except:
                    return 0.0

            first_seen = np.array([parse_timestamp(ts) for ts in df['fs']], dtype=np.float32).reshape(-1, 1)
            last_seen = np.array([parse_timestamp(ts) for ts in df['ls']], dtype=np.float32).reshape(-1, 1)

            # 4. 其他特征 (2维)
            has_nxdomain = df['nx'].apply(lambda x: 1.0 if x else 0.0).values.astype(np.float32).reshape(-1, 1)
            lifespan_log = df['lifespan_log'].fillna(0).values.astype(np.float32).reshape(-1, 1)

            # 拼接: 4 + 16 + 2 + 2 = 24维 (原来是58维)
            feats = np.hstack([nlp_feats, tld_embed, first_seen, last_seen, has_nxdomain, lifespan_log])

            self.data['domain'].x = torch.tensor(feats, dtype=torch.float32)
            self.data['domain'].num_nodes = num

            self.feature_config['domain'] = {'dim': feats.shape[1]}
            print(f"       domain: {num} nodes, dim={feats.shape[1]} (原58维, 压缩59%)")

        # ==================== URL ====================
        print("    -> Processing URL...")
        query = """
        MATCH (n:URL)
        RETURN n.value as id,
               n.http_code as code,
               n.filetype as ft,
               n.server as srv,
               n.encoding as enc
        """
        df = self._run_query_df(query)
        if not df.empty:
            num = self._register_nodes('URL', df['id'])

            # 1. NLP特征 (11维)
            url_nlp_feats = np.array([
                [
                    nlp_features_url(val)['url_entropy'],
                    nlp_features_url(val)['url_path_entropy'],
                    nlp_features_url(val)['url_length'],
                    nlp_features_url(val)['num_periods'],
                    nlp_features_url(val)['num_subdir'],
                    nlp_features_url(val)['num_digits'],
                    nlp_features_url(val)['num_frag'],
                    nlp_features_url(val)['num_params'],
                    nlp_features_url(val)['url_path_length'],
                    nlp_features_url(val)['url_host_length'],
                    nlp_features_url(val)['has_port']
                ]
                for val in df['id']
            ], dtype=np.float32)

            # 2. TLD嵌入 (16维)
            tld_values = [extract_tld(val, self.top_tlds) for val in df['id']]
            tld_encoder = EmbeddingEncoder(tld_values, embed_dim=16)
            tld_embed = tld_encoder.encode(tld_values)
            self.encoders['URL_TLD'] = tld_encoder

            # 3. HTTP code嵌入 (8维，替代11维one-hot)
            code_values = df['code'].fillna('Unknown').values
            code_encoder = EmbeddingEncoder(code_values, embed_dim=8)
            code_embed = code_encoder.encode(code_values)
            self.encoders['URL_http_code'] = code_encoder

            # 4. Filetype嵌入 (8维，替代6维one-hot)
            ft_values = df['ft'].fillna('Unknown').values
            ft_encoder = EmbeddingEncoder(ft_values, embed_dim=8)
            ft_embed = ft_encoder.encode(ft_values)
            self.encoders['URL_filetype'] = ft_encoder

            # 拼接: 11 + 16 + 8 + 8 = 43维 (原来是78维)
            feats = np.hstack([url_nlp_feats, tld_embed, code_embed, ft_embed])

            self.data['URL'].x = torch.tensor(feats, dtype=torch.float32)
            self.data['URL'].num_nodes = num

            self.feature_config['URL'] = {'dim': feats.shape[1]}
            print(f"       URL: {num} nodes, dim={feats.shape[1]} (原78维, 压缩45%)")

        # ==================== CVE ====================
        print("    -> Processing CVE...")
        query = """
        MATCH (n:CVE)
        RETURN n.id as id, n.year as year, n.year_norm as year_norm
        """
        df = self._run_query_df(query)
        if not df.empty:
            # 调试：检查原始数据
            print(f"       [CVE DEBUG] 查询返回 {len(df)} 行, 唯一ID: {df['id'].nunique()}")

            # 去重：确保每个CVE ID只出现一次（修复CVE节点数量错误的问题）
            df_original = df.copy()
            df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
            print(f"       [CVE DEBUG] 去重后: {len(df)} 行")

            num = self._register_nodes('CVE', df['id'])
            print(f"       [CVE DEBUG] 注册节点数: {num}")

            # CVE特征保持不变 (2维已经很低了)
            year_norm = df['year_norm'].fillna(0).values.reshape(-1, 1)

            cve_nums = []
            for cve_id in df['id']:
                parts = cve_id.split('-')
                if len(parts) >= 3:
                    try:
                        cve_num = int(parts[-1])
                        cve_num_norm = min(cve_num / 99999.0, 1.0)
                        cve_nums.append(cve_num_norm)
                    except:
                        cve_nums.append(0.0)
                else:
                    cve_nums.append(0.0)
            cve_num_norm = np.array(cve_nums).reshape(-1, 1)

            feats = np.hstack([year_norm, cve_num_norm])
            print(f"       [CVE DEBUG] feats.shape: {feats.shape}")

            self.data['CVE'].x = torch.tensor(feats, dtype=torch.float32)
            self.data['CVE'].num_nodes = num
            print(f"       [CVE DEBUG] 设置后 CVE.x.shape: {self.data['CVE'].x.shape}, CVE.num_nodes: {self.data['CVE'].num_nodes}")

            self.feature_config['CVE'] = {'dim': 2}
            print(f"       CVE: {num} nodes, dim=2")
        else:
            print("       [!] CVE查询结果为空")

        # ==================== File ====================
        print("    -> Processing File...")
        query = """
        MATCH (n:File)
        RETURN n.sha256 as id,
               n.imphash as imp,
               n.signature as sig,
               n.ssdeep as ssd,
               n.tlsh as tlsh
        """
        df = self._run_query_df(query)
        if not df.empty:
            num = self._register_nodes('File', df['id'])

            # 1. imphash嵌入 (16维，替代LabelEncoder)
            df['imp_short'] = df['imp'].fillna('00000000000000000000000000000000').str[:8]
            imp_values = df['imp_short'].fillna('unknown').values
            imp_encoder = EmbeddingEncoder(imp_values, embed_dim=16)
            imp_embed = imp_encoder.encode(imp_values)
            self.encoders['File_imphash'] = imp_encoder

            # 2. ssdeep和tlsh保持不变 (2维)
            def extract_ssdeep_bs(ssd):
                if not ssd or pd.isna(ssd):
                    return 0.0
                try:
                    bs = int(str(ssd).split(':')[0])
                    return min(bs / 256.0, 1.0)
                except (ValueError, IndexError, AttributeError):
                    return 0.0

            def extract_tlsh_feat(tlsh):
                if not tlsh or pd.isna(tlsh) or len(str(tlsh)) < 4:
                    return 0.0
                try:
                    val = int(str(tlsh)[:4], 16)
                    return val / 65535.0
                except (ValueError, AttributeError):
                    return 0.0

            ssdeep_bs = np.array([extract_ssdeep_bs(ssd) for ssd in df['ssd']]).reshape(-1, 1).astype(np.float32)
            tlsh_feat = np.array([extract_tlsh_feat(tlsh) for tlsh in df['tlsh']]).reshape(-1, 1).astype(np.float32)

            # 拼接: 16 + 1 + 1 = 18维 (原来是3维)
            feats = np.hstack([imp_embed, ssdeep_bs, tlsh_feat])

            self.data['File'].x = torch.tensor(feats, dtype=torch.float32)
            self.data['File'].num_nodes = num

            self.feature_config['File'] = {'dim': feats.shape[1]}
            print(f"       File: {num} nodes, dim={feats.shape[1]} (原3维, 增加15维)")

        # ==================== ASN ====================
        print("    -> Processing ASN...")
        query = """
        MATCH (n:ASN)
        RETURN n.value as id, n.issuer as issuer
        """
        df = self._run_query_df(query)
        if not df.empty:
            num = self._register_nodes('ASN', df['id'])

            # issuer嵌入 (16维，替代LabelEncoder)
            issuers = df['issuer'].fillna('unknown').astype(str).values
            iss_encoder = EmbeddingEncoder(issuers, embed_dim=16)
            iss_embed = iss_encoder.encode(issuers)
            self.encoders['ASN_issuer'] = iss_encoder

            # 频率编码
            freq_encoder = FrequencyEncoder()
            freq_encoder.fit(issuers)
            iss_freq = freq_encoder.encode(issuers)

            # 拼接: 16 + 1 = 17维 (原来是1维)
            feats = np.hstack([iss_embed, iss_freq])

            self.data['ASN'].x = torch.tensor(feats, dtype=torch.float32)
            self.data['ASN'].num_nodes = num

            self.feature_config['ASN'] = {'dim': feats.shape[1]}
            print(f"       ASN: {num} nodes, dim={feats.shape[1]} (原1维, 增加8维)")

        # ==================== Technique ====================
        print("    -> Processing Technique...")
        query = """
        MATCH (n:Technique)
        RETURN n.id as id, n.name as name, n.description as description
        """
        df = self._run_query_df(query)
        if not df.empty:
            num = self._register_nodes('Technique', df['id'])

            features_list = []

            # 1. 语义嵌入 (384维) - 使用本地sentence-transformers
            if HAS_SENTENCE_TRANSFORMER:
                # 优先使用本地模型
                if os.path.exists(LOCAL_MODEL_PATH):
                    print(f"       -> 使用本地模型: {LOCAL_MODEL_PATH}")
                    model = SentenceTransformer(LOCAL_MODEL_PATH)
                else:
                    print(f"       -> 本地模型不存在，使用默认模型")
                    model = SentenceTransformer('all-MiniLM-L6-v2')

                # 构建技术描述文本
                technique_texts = []
                for _, row in df.iterrows():
                    tech_id = row['id']
                    name = row['name'] if pd.notna(row['name']) else ''
                    desc = row['description'] if pd.notna(row['description']) else ''

                    # 组合: "T1566: Phishing. Spearphishing messages..."
                    text = f"{tech_id}: {name}. {desc}" if name else tech_id
                    technique_texts.append(text)

                semantic_embeds = model.encode(technique_texts, show_progress_bar=False)
                features_list.append(semantic_embeds)
            else:
                # 如果没有sentence-transformers，使用简单的特征
                print("       [警告] 未安装sentence-transformers，使用基础特征")
                # 技术ID的哈希编码作为替代
                technique_hash_embeds = np.array([
                    self._hash_to_vector(tid, dim=64) for tid in df['id']
                ], dtype=np.float32)
                features_list.append(technique_hash_embeds)

            # 2. 稀有度特征 (3维)
            # 计算每个技术被多少个APT组织使用
            rarity_query = """
            MATCH (t:Technique)<-[:USES_TECHNIQUE]-(e:EVENT)
            RETURN t.id as tid, count(DISTINCT e.label) as org_count
            """
            rarity_df = self._run_query_df(rarity_query)

            # 创建技术ID到组织数的映射
            org_count_map = {}
            if not rarity_df.empty:
                for _, row in rarity_df.iterrows():
                    org_count_map[row['tid']] = row['org_count']

            # 查询实际的APT组织总数（而非硬编码20）
            total_orgs_query = """
            MATCH (e:EVENT)
            RETURN count(DISTINCT e.label) as total_orgs
            """
            total_orgs_df = self._run_query_df(total_orgs_query)
            total_orgs = total_orgs_df['total_orgs'].iloc[0] if not total_orgs_df.empty else 20
            total_orgs = max(total_orgs, 1)  # 避免除以0

            rarity_features = []
            for tech_id in df['id']:
                org_count = org_count_map.get(tech_id, 1)
                coverage = org_count / total_orgs  # 覆盖率
                rarity = 1 - coverage  # 稀有度 (越少组织使用，越稀有)
                log_org_count = np.log1p(org_count) / np.log1p(total_orgs)  # 对数归一化

                rarity_features.append([rarity, coverage, log_org_count])

                # Problem 2: 将稀有度保存到缓存，供边权重计算使用
                self.technique_rarity_map[tech_id] = rarity

            rarity_features = np.array(rarity_features, dtype=np.float32)
            features_list.append(rarity_features)

            # 拼接所有特征
            if HAS_SENTENCE_TRANSFORMER:
                # 语义(384) + 稀有度(3) = 387维
                feats = np.hstack(features_list).astype(np.float32)
                print(f"       Technique: {num} nodes, dim=387 (语义384维 + 稀有度3维)")
            else:
                # 哈希(64) + 稀有度(3) = 67维
                feats = np.hstack(features_list).astype(np.float32)
                print(f"       Technique: {num} nodes, dim=67 (哈希64维 + 稀有度3维)")

            self.data['Technique'].x = torch.tensor(feats, dtype=torch.float32)
            self.data['Technique'].num_nodes = num

            self.feature_config['Technique'] = {'dim': feats.shape[1]}

    def _hash_to_vector(self, text, dim=64):
        """将文本哈希到向量（作为语义嵌入的替代）"""
        import hashlib
        hash_obj = hashlib.md5(str(text).encode())
        hash_hex = hash_obj.hexdigest()

        vector = np.array([
            int(hash_hex[i:i+2], 16) / 255.0
            for i in range(0, min(len(hash_hex), dim * 2), 2)
        ])

        if len(vector) < dim:
            vector = np.pad(vector, (0, dim - len(vector)))
        else:
            vector = vector[:dim]

        return vector.astype(np.float32)

    def build_event_nodes(self, ttp_only=False, valid_event_ids=None):
        """EVENT节点聚合技术语义特征

        Args:
            ttp_only: 是否只导出包含TTP的EVENT节点
            valid_event_ids: 可选，指定只导出这些EVENT ID（用于增量更新的TTP过滤）
        """

        print("\n[Step 2] 构建 EVENT 节点（聚合技术语义）...")

        # 查询EVENT及其技术信息
        if valid_event_ids is not None:
            # 优先使用指定的EVENT ID列表（TTP过滤后）
            print(f"    [TTP-FILTERED模式] 只导出指定的 {len(valid_event_ids)} 个EVENT节点")
            valid_ids_list = list(valid_event_ids)
            query = """
            MATCH (e:EVENT)
            WHERE e.id IN $event_ids
            RETURN e.id as id, e.label as label
            """
            df = self._run_query_df(query, params={'event_ids': valid_ids_list})
        elif ttp_only:
            print("    [TTP-ONLY模式] 只导出包含TTP的EVENT节点")
            query = """
            MATCH (e:EVENT)-[:USES_TECHNIQUE]->(t:Technique)
            RETURN DISTINCT e.id as id, e.label as label
            """
            df = self._run_query_df(query)
        else:
            query = """
            MATCH (e:EVENT)
            RETURN e.id as id, e.label as label
            """
            df = self._run_query_df(query)
        if df.empty:
            raise RuntimeError("数据库中没有 EVENT 节点")

        num_nodes = self._register_nodes('EVENT', df['id'])
        event_ids = df['id'].tolist()

        # ==================== 聚合技术语义特征 ====================
        # 注意: 不再从Neo4j查询semantic_embedding（不存在），而是从技术名称生成

        # 检查是否有sentence-transformers模型
        if not HAS_SENTENCE_TRANSFORMER:
            print("    [警告] 无sentence-transformers，使用零特征")
            event_features = np.zeros((num_nodes, 1), dtype=np.float32)
            print(f"  EVENT Ready: {num_nodes} nodes.")
            print(f"  Feature Shape: {self.data['EVENT'].x.shape} (零特征)")
            self.data['EVENT'].x = torch.tensor(event_features, dtype=torch.float32)
            self.data['EVENT'].num_nodes = num_nodes
            return

        # 加载sentence-transformers模型用于技术嵌入
        model = SentenceTransformer(LOCAL_MODEL_PATH)

        # 构建EVENT特征
        event_features_list = []
        for eid in event_ids:
            # 获取该事件的所有技术名称
            tech_names_query = """
            MATCH (e:EVENT {id: $eid})-[:USES_TECHNIQUE]->(t:Technique)
            RETURN t.name as tech_name, t.id as tech_id
            """
            tech_names_df = self._run_query_df(tech_names_query, params={'eid': eid})

            if not tech_names_df.empty:
                # 生成技术嵌入
                tech_names = tech_names_df['tech_name'].tolist()
                tech_embeddings = model.encode(tech_names, show_progress_bar=False)

                # 聚合技术嵌入 (384维)
                tech_mean = np.mean(tech_embeddings, axis=0)  # 平均

                # 技术数量归一化
                num_techs = min(len(tech_names) / 50.0, 1.0)

                # 拼接特征
                event_feat = np.concatenate([tech_mean, [num_techs]])
            else:
                # 没有技术的事件，用零向量
                event_feat = np.concatenate([np.zeros(384), [0.0]])

            event_features_list.append(event_feat)

        event_features = np.array(event_features_list, dtype=np.float32)

        self.data['EVENT'].x = torch.tensor(event_features, dtype=torch.float32)
        self.data['EVENT'].num_nodes = num_nodes

        print(f"  EVENT Ready: {num_nodes} nodes.")
        if ttp_only:
            print(f"  [TTP-ONLY] 仅包含有TTP的EVENT节点")
        print(f"  Feature Shape: {self.data['EVENT'].x.shape} (384维技术聚合 + 1维技术数量 = 385维)")

    def build_edges(self):
        """构建边关系 - 使用边类型嵌入方案"""
        print("\n[Step 3] 提取拓扑结构（边类型嵌入方案）...")

        # 边配置: (源类型, 关系类型, 目标类型, 源属性, 目标属性, 是否有权重)
        edge_configs = [
            ('EVENT', 'USES_INFRASTRUCTURE', 'IP', 'id', 'value', False),
            ('EVENT', 'USES_DOMAIN', 'domain', 'id', 'value', False),
            ('EVENT', 'DELIVERS_VIA_URL', 'URL', 'id', 'value', False),
            ('EVENT', 'EXPLOITS_VULN', 'CVE', 'id', 'id', False),
            ('EVENT', 'DROPS_MALWARE', 'File', 'id', 'sha256', False),
            ('EVENT', 'USES_TECHNIQUE', 'Technique', 'id', 'id', True),  # 技术稀有度权重
            ('IP', 'BELONGS_TO_NETWORK', 'ASN', 'value', 'value', False),
            ('IP', 'RESOLVES_TO', 'domain', 'value', 'value', False),
            ('domain', 'RESOLVES_TO', 'IP', 'value', 'value', False),
            ('URL', 'HOSTED_ON_DOMAIN', 'domain', 'value', 'value', False),
            ('URL', 'RESOLVES_TO_IP', 'IP', 'value', 'value', False),
            ('File', 'SIMILAR_TO', 'File', 'sha256', 'sha256', True)
        ]

        # 创建边类型映射表（用于边类型嵌入）
        edge_type_list = []
        for src, rel, dst, _, _, _ in edge_configs:
            edge_type_list.append((src, rel, dst))
            if src != dst:  # 添加反向边类型
                edge_type_list.append((dst, f'rev_{rel}', src))

        self.edge_type_to_idx = {et: i for i, et in enumerate(edge_type_list)}
        self.num_edge_types = len(edge_type_list)
        print(f"    边类型总数: {self.num_edge_types} (包含反向边)")

        # 统计边信息
        edge_stats = {}
        total_edges = 0

        for src, rel, dst, sp, dp, weighted in edge_configs:
            if src not in self.node_mapping or dst not in self.node_mapping:
                continue

            # 构建查询
            if weighted:
                if src == 'EVENT' and dst == 'Technique':
                    query = f"MATCH (e:{src})-[r:{rel}]->(t:{dst}) RETURN e.{sp} as s, t.{dp} as d"
                else:
                    query = f"MATCH (a:{src})-[r:{rel}]->(b:{dst}) RETURN a.{sp} as s, b.{dp} as d, r.score as w"
            else:
                query = f"MATCH (a:{src})-[r:{rel}]->(b:{dst}) RETURN a.{sp} as s, b.{dp} as d"

            df = self._run_query_df(query)
            if df.empty:
                continue

            # 添加稀有度权重（EVENT-USES_TECHNIQUE）
            if weighted and src == 'EVENT' and dst == 'Technique':
                df['w'] = df['d'].map(lambda tid: self.technique_rarity_map.get(tid, 0.5))

            u = df['s'].map(self.node_mapping[src])
            v = df['d'].map(self.node_mapping[dst])
            valid = u.notna() & v.notna()

            if not valid.any():
                continue

            u = torch.tensor(u[valid].astype(int).values, dtype=torch.long)
            v = torch.tensor(v[valid].astype(int).values, dtype=torch.long)
            edge_index = torch.stack([u, v], dim=0)

            # 保存边索引
            self.data[src, rel, dst].edge_index = edge_index

            # === 边特征：只保留边类型索引和权重 ===
            num_edges = edge_index.shape[1]
            edge_type_idx = self.edge_type_to_idx[(src, rel, dst)]

            # 边权重（归一化到[0,1]）
            edge_score = np.zeros(num_edges, dtype=np.float32)
            if weighted and 'w' in df.columns:
                w = df.loc[valid, 'w'].fillna(0.0).values
                if w.max() > 0:
                    w = w / w.max()
                edge_score = w.astype(np.float32)

            # 边特征：[边类型索引(1维), 权重(1维)]
            edge_type_tensor = torch.full((num_edges, 1), edge_type_idx, dtype=torch.long)
            edge_score_tensor = torch.tensor(edge_score, dtype=torch.float32).unsqueeze(1)
            edge_attr = torch.cat([edge_type_tensor, edge_score_tensor], dim=1)
            self.data[src, rel, dst].edge_attr = edge_attr

            # 统计信息
            edge_stats[(src, rel, dst)] = edge_index.shape[1]
            total_edges += edge_index.shape[1]

            # 双向边
            if src != dst:
                rev_rel = f"rev_{rel}"
                self.data[dst, rev_rel, src].edge_index = torch.stack([v, u], dim=0)

                # 反向边特征
                rev_edge_type_idx = self.edge_type_to_idx[(dst, rev_rel, src)]
                rev_edge_type_tensor = torch.full((num_edges, 1), rev_edge_type_idx, dtype=torch.long)
                rev_edge_attr = torch.cat([rev_edge_type_tensor, edge_score_tensor], dim=1)
                self.data[dst, rev_rel, src].edge_attr = rev_edge_attr

        # 打印边统计
        print(f"    总边数: {total_edges}")
        print(f"    边类型数: {len(edge_stats)}")
        print(f"    边特征: [edge_type_idx, weight] (将转换为嵌入)")
        for (src, rel, dst), count in sorted(edge_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {src}--{rel}-->{dst}: {count} edges")

    def generate_labels(self):
        """生成标签"""
        print("\n[Step 4] 生成标签...")
        df = self._run_query_df("MATCH (e:EVENT) WHERE e.label IS NOT NULL RETURN e.id as eid, e.label as apt")
        if df.empty:
            print("    [!] Error: 没有找到带标签的事件")
            return

        self.apt_encoder.fit(df['apt'])
        df['label'] = self.apt_encoder.transform(df['apt'])

        event_map = self.node_mapping['EVENT']
        df['idx'] = df['eid'].map(event_map)
        df = df.dropna(subset=['idx'])

        indices = df['idx'].astype(int).values
        labels = df['label'].values

        y = torch.full((self.data['EVENT'].num_nodes,), -1, dtype=torch.long)
        y[indices] = torch.tensor(labels, dtype=torch.long)
        self.data['EVENT'].y = y

        self.data._apt_classes = self.apt_encoder.classes_

        print(f" 标签生成完成. Classes: {len(self.apt_encoder.classes_)}")
        print(f"  已标注节点: {len(indices)}")
        print(f"  未标注节点: {self.data['EVENT'].num_nodes - len(indices)}")

    def add_node2vec_features(self, embedding_dim=64, epochs=20, node_types=None):
        """
        使用Node2Vec预训练节点嵌入并拼接特征

        Args:
            embedding_dim: Node2Vec嵌入维度
            epochs: 训练轮数
            node_types: 指定只在哪些节点类型上运行Node2Vec
                       None表示只使用IOC节点（不含TTP节点）
        """
        print(f"\n[Step 4.5] Node2Vec预训练（嵌入维度: {embedding_dim}）...")

        # 检查Node2Vec是否可用
        if not HAS_NODE2VEC:
            print("    [!] 跳过Node2Vec: 未安装pyg-lib或torch-cluster")
            print("    提示: 安装命令 pip install torch-cluster 或 pip install pyg-lib")
            return

        # 1. 构建同构图（将异构图转换为同构图）
        print("    -> 构建同构图...")
        # 确定要包含的节点类型（默认只包含IOC节点，不包括TTP节点）
        if node_types is None:
            ioc_node_types = {'EVENT', 'IP', 'domain', 'URL', 'File', 'ASN', 'CVE'}
            node_types = [nt for nt in self.data.node_types if nt in ioc_node_types]
            print(f"       使用默认IOC节点类型: {node_types}")
        else:
            print(f"       指定节点类型: {node_types}")
        node_type_to_ids = {}  # 记录每种节点类型的全局ID范围
        global_id_offset = 0
        edge_index_list = []

        # 只收集指定的节点类型
        valid_node_types = [nt for nt in node_types if nt in self.data.node_types]
        for node_type in valid_node_types:
            num_nodes = self.data[node_type].num_nodes
            print(f"       DEBUG: {node_type} has {num_nodes} nodes (x.shape={self.data[node_type].x.shape})")
            node_type_to_ids[node_type] = (
                global_id_offset,
                global_id_offset + num_nodes
            )
            print(f"       DEBUG: {node_type} global range: [{global_id_offset}, {global_id_offset + num_nodes})")
            global_id_offset += num_nodes

        total_nodes = global_id_offset
        print(f"       总节点数: {total_nodes}")
        print(f"       包含节点类型: {valid_node_types}")

        # 2. 只收集指定节点类型之间的边
        for edge_type in self.data.edge_types:
            src_type, _, dst_type = edge_type

            # 只收集两端都在指定节点类型中的边
            if src_type not in valid_node_types or dst_type not in valid_node_types:
                continue

            edge_index = self.data[edge_type].edge_index

            # 转换为全局ID
            src_start, _ = node_type_to_ids[src_type]
            dst_start, _ = node_type_to_ids[dst_type]

            global_src = edge_index[0] + src_start
            global_dst = edge_index[1] + dst_start

            edge_index_list.append(torch.stack([global_src, global_dst], dim=0))

        # 合并所有边
        if edge_index_list:
            edge_index = torch.cat(edge_index_list, dim=1)
        else:
            print("    [!] 警告: 没有找到边，跳过Node2Vec")
            return

        print(f"       总边数: {edge_index.shape[1]}")

        # 3. 训练Node2Vec
        print(f"    -> 训练Node2Vec ({epochs} epochs)...")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"       使用设备: {device}")

        node2vec = Node2Vec(
            edge_index,
            embedding_dim=embedding_dim,
            walk_length=20,
            context_size=10,
            walks_per_node=10,
            num_negative_samples=5,
            p=1.0,
            q=1.0,
            sparse=True,
            num_nodes=total_nodes  # 显式指定节点数量，确保孤立节点也有嵌入
        ).to(device)

        optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

        # 训练循环
        node2vec.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            # 使用loader进行小批量训练
            loader = node2vec.loader(batch_size=256, shuffle=True, num_workers=0)

            for pos_rw, neg_rw in loader:
                # 将数据移到正确的设备上
                pos_rw = pos_rw.to(device)
                neg_rw = neg_rw.to(device)

                optimizer.zero_grad()
                loss = node2vec.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0

            if (epoch + 1) % 10 == 0:
                print(f"       Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # 4. 获取嵌入向量
        print("    -> 提取节点嵌入...")
        node2vec.eval()
        with torch.no_grad():
            embeddings = node2vec.embedding.weight.cpu()

        # 5. 将嵌入分割并拼接回各节点类型
        print("    -> 拼接Node2Vec特征...")
        for node_type in node_type_to_ids.keys():
            start, end = node_type_to_ids[node_type]
            print(f"       DEBUG {node_type}: start={start}, end={end}, expected_nodes={end-start}")
            node_type_embeddings = embeddings[start:end]  # [num_nodes, embedding_dim]

            # 获取原始特征
            original_x = self.data[node_type].x
            print(f"       DEBUG {node_type}: original_x.shape={original_x.shape}, node_type_embeddings.shape={node_type_embeddings.shape}")

            # 检查维度匹配
            if original_x.shape[0] != node_type_embeddings.shape[0]:
                print(f"       [!] 跳过 {node_type}: 特征维度不匹配 ({original_x.shape[0]} vs {node_type_embeddings.shape[0]})")
                continue

            # 拼接特征
            new_x = torch.cat([original_x, node_type_embeddings], dim=1)
            self.data[node_type].x = new_x

            # 更新特征配置
            if node_type in self.feature_config:
                self.feature_config[node_type]['dim'] = new_x.shape[1]
                self.feature_config[node_type]['node2vec_dim'] = embedding_dim

            print(f"       {node_type}: {original_x.shape[1]} -> {new_x.shape[1]} (新增{embedding_dim}维)")

        print("    -> Node2Vec特征添加完成!")

    def export_dual_subgraphs(self, output_dir="."):
        """导出IOC和TTP子图用于双模型训练

        通过构建完整图然后过滤的方式来创建两个子图，
        确保都只包含有TTP的EVENT节点

        Args:
            output_dir: 输出目录路径
        """
        print(f"\n[Step 4.6] 导出双模型子图到 {output_dir}...", flush=True)
        print("="*70, flush=True)

        from pathlib import Path
        import pandas as pd

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] 目录创建完成: {output_dir}", flush=True)

        # ==================== 第一步：获取有TTP的EVENT ID列表 ====================
        print("\n[预处理] 查询包含TTP的EVENT节点...", flush=True)
        query = """MATCH (e:EVENT)-[:USES_TECHNIQUE]->(t:Technique)
                    RETURN DISTINCT e.id as event_id"""
        print("[DEBUG] 执行查询...", flush=True)
        df_events = self._run_query_df(query)
        print(f"[DEBUG] 查询完成，返回 {len(df_events)} 行", flush=True)

        if df_events.empty:
            print("[错误] 未找到包含TTP的EVENT节点")
            return

        all_ttp_event_ids = set(df_events['event_id'].tolist())
        print(f"找到 {len(all_ttp_event_ids)} 个包含TTP的EVENT节点", flush=True)
        print(f"占总EVENT比例: ~{len(all_ttp_event_ids) / 3212 * 100:.1f}%", flush=True)

        # ==================== 第二步：构建完整图并Node2Vec（先Node2Vec，不过滤）====================
        print("\n[1/2] 构建完整图并生成Node2Vec特征...", flush=True)

        # 构建完整图（包含所有有TTP的EVENT，不过滤序列）
        print("[DEBUG] 调用build_rich_node_features...", flush=True)
        self.build_rich_node_features()
        print("[DEBUG] build_rich_node_features完成", flush=True)

        print("[DEBUG] 调用build_event_nodes (注册所有有TTP的EVENT)...", flush=True)
        self.build_event_nodes(valid_event_ids=all_ttp_event_ids)  # 注册所有有TTP的EVENT
        print("[DEBUG] build_event_nodes完成", flush=True)

        print("[DEBUG] 调用build_edges...", flush=True)
        self.build_edges()
        print("[DEBUG] build_edges完成", flush=True)

        print("[DEBUG] 调用generate_labels...", flush=True)
        self.generate_labels()
        print("[DEBUG] generate_labels完成", flush=True)

        # 添加Node2Vec特征（为所有有TTP的EVENT生成嵌入）
        print("\n[Node2Vec] 为所有有TTP的EVENT添加Node2Vec特征...", flush=True)
        self.add_node2vec_features(embedding_dim=64, epochs=20)

        # ==================== 第三步：过滤TTP序列（在Node2Vec之后）====================
        print("\n[过滤] 基于Node2Vec图结构，过滤TTP序列...", flush=True)

        # 查询每个事件的技术序列
        print("  -> 查询每个事件的技术序列...")
        query = """
        MATCH (e:EVENT)-[r:USES_TECHNIQUE]->(t:Technique)
        WHERE e.id IN $event_ids
        RETURN e.id as event_id, t.id as tech_id
        ORDER BY e.id, t.id
        """
        df_tech = self._run_query_df(query, params={'event_ids': list(all_ttp_event_ids)})

        if df_tech.empty:
            print("  [错误] 未找到TTP数据")
            return

        # 构建技术ID映射
        all_techniques = df_tech['tech_id'].unique()
        technique_to_idx = {tech_id: idx + 1 for idx, tech_id in enumerate(sorted(all_techniques))}
        num_techniques = len(technique_to_idx)

        print(f"  -> 技术总数: {num_techniques}")

        # 构建事件-技术映射
        event_raw_techniques = {}
        for eid, group in df_tech.groupby('event_id'):
            event_raw_techniques[eid] = group['tech_id'].tolist()

        print("  -> 序列质量筛选（保留完整序列，最低门槛：长度 >= 3）...")

        filtered_sequences = {}
        filtered_event_ids = []
        min_seq_len = 3                      # 只过滤极短序列

        seq_len_all = []                     # 用于统计
        for event_id, tech_ids in event_raw_techniques.items():
            if len(tech_ids) < min_seq_len:
                continue                     # 丢弃长度 < 3 的极短序列
            filtered_sequences[event_id] = tech_ids   # 保留完整序列，不截断
            filtered_event_ids.append(event_id)
            seq_len_all.append(len(tech_ids))

        removed = len(event_raw_techniques) - len(filtered_sequences)
        print(f"     -> 原始事件数:   {len(event_raw_techniques)}")
        print(f"     -> 丢弃（长度<3）: {removed}")
        print(f"     -> 保留事件数:   {len(filtered_sequences)}")
        if seq_len_all:
            print(f"     -> 序列长度统计: "
                f"均值={sum(seq_len_all)/len(seq_len_all):.1f}  "
                f"最小={min(seq_len_all)}  最大={max(seq_len_all)}  "
                f"中位数={sorted(seq_len_all)[len(seq_len_all)//2]}")

        # 更新 valid_event_ids 为筛选后的
        valid_event_ids = set(filtered_event_ids)
        event_raw_techniques = {eid: filtered_sequences[eid] for eid in filtered_event_ids}
        
        # 获取EVENT节点的node_mapping
        event_map = self.node_mapping.get('EVENT')

        if event_map is None:
            print("[错误] EVENT节点未构建")
            return

        # 找到有效的EVENT索引（使用过滤后的event_ids）
        valid_event_indices = []
        for eid in valid_event_ids:
            if eid in event_map.index:
                valid_event_indices.append(event_map[eid])

        valid_event_indices = sorted(valid_event_indices)
        print(f"找到 {len(valid_event_indices)} 个有效的EVENT索引")

        if len(valid_event_indices) == 0:
            print("[错误] 没有找到有效的EVENT索引")
            return

        # 创建旧索引到新索引的映射
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_event_indices)}

        # ==================== 第四步：查询战术阶段映射并生成因果序列 ====================
        print("  -> 查询战术阶段映射...")
        query_tactics = """
        MATCH (t:Technique)-[r:BELONGS_TO]->(tac:Tactic)
        RETURN t.id as tech_id, tac.id as tactic_id, tac.name as tactic_name
        """
        tactic_mapping = defaultdict(list)
        df_tactics = self._run_query_df(query_tactics)

        for _, row in df_tactics.iterrows():
            tech_id = row['tech_id']
            tactic_mapping[tech_id].append({
                'tactic_id': row['tactic_id'],
                'tactic_name': row['tactic_name']
            })

        tactic_mapping = dict(tactic_mapping)
        print(f"     -> 已查询 {len(tactic_mapping)} 个技术的战术映射")

        # 生成因果序列（按战术阶段排序）
        print(f"  -> 生成因果序列（按战术阶段排序）...")
        event_sequences = {}
        for event_id, tech_ids in event_raw_techniques.items():
            if tactic_mapping:
                causal_seq = generate_causal_sequence(tech_ids, tactic_mapping)
                event_sequences[event_id] = [technique_to_idx[t] for t in causal_seq]
            else:
                event_sequences[event_id] = [technique_to_idx[t] for t in sorted(tech_ids)]

        print(f"     -> 生成了 {len(event_sequences)} 个事件的因果序列")

        # ==================== 第四步：构建IOC子图（只包含有TTP的EVENT）====================
        print("\n[构建IOC子图] 过滤完整图，只保留有TTP的EVENT...")
        from torch_geometric.data import HeteroData

        ioc_data = HeteroData()

        # IOC节点类型
        ioc_node_types = {'IP', 'domain', 'URL', 'File', 'CVE', 'ASN', 'EVENT'}

        # 1. 复制非EVENT节点特征
        for node_type in ioc_node_types:
            if node_type != 'EVENT' and node_type in self.data.node_types:
                ioc_data[node_type].x = self.data[node_type].x.clone()
                ioc_data[node_type].num_nodes = self.data[node_type].num_nodes
                print(f"  {node_type}: {self.data[node_type].num_nodes} nodes")

        # 2. 复制EVENT节点特征（只包含有TTP的）
        event_x = self.data['EVENT'].x  # 原始特征（已包含Node2Vec）
        event_y = self.data['EVENT'].y if hasattr(self.data['EVENT'], 'y') else None

        print(f"  EVENT: {len(valid_event_indices)} nodes (只包含TTP)")
        print(f"       特征维度: {event_x.shape[1]}维 (原始特征 + Node2Vec)")

        ioc_data['EVENT'].x = event_x[valid_event_indices].clone()
        ioc_data['EVENT'].num_nodes = len(valid_event_indices)
        if event_y is not None:
            ioc_data['EVENT'].y = event_y[valid_event_indices].clone()

        # 3. 更新EVENT的node_mapping
        ioc_data._node_mapping = {}
        for node_type in ioc_node_types:
            if node_type != 'EVENT' and node_type in self.node_mapping:
                ioc_data._node_mapping[node_type] = self.node_mapping[node_type]
        # EVENT节点重新映射到连续索引
        num_events = len(valid_event_indices)
        ioc_data._node_mapping['EVENT'] = pd.Series(
            data=np.arange(num_events),
            index=pd.RangeIndex(start=0, stop=num_events, step=1)
        )

        # 4. 复制IOC相关边（需要重新映射索引）
        ioc_edge_count = 0
        for edge_type in self.data.edge_types:
            src, rel, dst = edge_type
            # 只保留不涉及Technique的边
            if 'Technique' not in str(edge_type):
                if src in ioc_node_types and dst in ioc_node_types:
                    edge_index = self.data[edge_type].edge_index.clone()

                    # 过滤边：只保留连接到有效EVENT的边
                    mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
                    if src == 'EVENT':
                        mask = torch.isin(edge_index[0], torch.tensor(valid_event_indices))
                        # 重新映射索引
                        edge_index[0] = torch.tensor([old_to_new.get(idx.item(), idx.item()) for idx in edge_index[0]])
                    elif dst == 'EVENT':
                        mask = torch.isin(edge_index[1], torch.tensor(valid_event_indices))
                        # 重新映射索引
                        edge_index[1] = torch.tensor([old_to_new.get(idx.item(), idx.item()) for idx in edge_index[1]])

                    edge_index = edge_index[:, mask]

                    if edge_index.shape[1] == 0:
                        continue

                    ioc_data[edge_type].edge_index = edge_index
                    if hasattr(self.data[edge_type], 'edge_attr'):
                        edge_attr = self.data[edge_type].edge_attr.clone()
                        ioc_data[edge_type].edge_attr = edge_attr[mask]
                    ioc_edge_count += 1

        print(f"  复制了 {ioc_edge_count} 种边类型")

        # 5. 保存IOC子图元数据
        ioc_data._feature_config = {}
        for node_type in ioc_node_types:
            if node_type in ioc_data.node_types:
                if node_type == 'EVENT':
                    ioc_data._feature_config[node_type] = {'dim': ioc_data[node_type].x.shape[1]}
                elif node_type in self.feature_config:
                    ioc_data._feature_config[node_type] = self.feature_config[node_type]

        ioc_data._apt_classes = self.data._apt_classes if hasattr(self.data, '_apt_classes') else self.apt_encoder.classes_

        # 保存IOC子图
        ioc_path = output_dir / "apt_kg_ioc.pt"
        torch.save(ioc_data, ioc_path)
        print(f"\n  [OK] IOC子图已保存: {ioc_path}")
        print(f"     EVENT节点: {ioc_data['EVENT'].num_nodes} (只包含TTP)")
        print(f"     边类型数: {len(ioc_data.edge_types)}")

        # ==================== 第四步：导出TTP变长序列（用于Transformer模型）====================
        print("\n[2/2] 构建TTP变长序列（用于Transformer序列模型）...")

        # 生成技术语义嵌入
        print("  -> 生成技术语义嵌入...")

        # 查询所有技术的描述
        all_techniques = list(set([t for tech_list in event_raw_techniques.values() for t in tech_list]))
        query_tech_desc = """
        MATCH (t:Technique)
        WHERE t.id IN $tech_ids
        RETURN t.id as tech_id, t.name as name, t.description as description
        ORDER BY t.id
        """
        df_tech_desc = self._run_query_df(query_tech_desc, params={'tech_ids': all_techniques})

        # 加载sentence-transformers模型
        import sys
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'all-MiniLM-L6-v2')

        from sentence_transformers import SentenceTransformer
        try:
            model = SentenceTransformer(model_path)
            print(f"     -> 从本地加载模型: {model_path}")
        except Exception as e:
            print(f"     -> 本地加载失败: {e}")
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # 为每个技术生成语义向量
        technique_embeddings = np.zeros((num_techniques + 1, 384), dtype=np.float32)  # +1 for padding

        for _, row in df_tech_desc.iterrows():
            tech_id = row['tech_id']
            idx = technique_to_idx[tech_id]

            # 构建技术描述文本
            text = f"{tech_id}: {row['name']}"
            if pd.notna(row.get('description')) and len(str(row['description']).strip()) > 0:
                text += f". {row['description']}"

            # 生成语义向量
            with torch.no_grad():
                embedding = model.encode([text])[0]

            technique_embeddings[idx] = embedding.astype(np.float32)

        print(f"     -> 语义向量形状: {technique_embeddings.shape}")
        print(f"     -> 向量维度: 384")

        # 获取EVENT标签
        query_labels = """
        MATCH (e:EVENT)
        WHERE e.id IN $event_ids
        RETURN e.id as event_id, e.label as label
        """
        df_labels = self._run_query_df(query_labels, params={'event_ids': list(valid_event_ids)})

        # 构建最终的序列数据（按valid_event_indices的顺序）
        technique_sequences = []
        labels = []

        # 创建event_id到label的映射
        label_map = dict(zip(df_labels['event_id'], df_labels['label']))

        for event_idx in valid_event_indices:
            # 从event_idx找到对应的event_id
            event_id = None
            for eid, idx in event_map.items():
                if idx == event_idx and eid in valid_event_ids:
                    event_id = eid
                    break

            if event_id is not None:
                technique_sequences.append(event_sequences.get(event_id, []))
                labels.append(label_map.get(event_id, 0))

        # 统计序列长度
        seq_lengths = [len(seq) for seq in technique_sequences]
        print(f"  -> 序列统计:")
        print(f"     平均长度: {np.mean(seq_lengths):.1f}")
        print(f"     最大长度: {max(seq_lengths)}")
        print(f"     最小长度: {min(seq_lengths)}")
        print(f"     中位数: {np.median(seq_lengths):.1f}")

        # ==================== 生成阶段感知序列（每个技术对应的战术阶段）====================
        print("\n  -> 生成阶段感知序列...")
        phase_sequences = []
        for seq in technique_sequences:
            phases = []
            for tech_id in seq:
                # 查询技术对应的战术阶段
                tech_id_str = str(tech_id)
                phase = get_technique_phase_order(tech_id_str, tactic_mapping)
                phases.append(phase)
            phase_sequences.append(phases)

        print(f"     -> 阶段序列生成完成")

        # ==================== 生成序列级全局特征 ====================
        print("  -> 生成序列级全局特征...")
        global_features = []

        for i, seq in enumerate(technique_sequences):
            seq_len = len(seq)
            phases = phase_sequences[i]

            if seq_len > 0:
                # 特征1: 序列长度归一化
                norm_len = seq_len / max(seq_lengths) if max(seq_lengths) > 0 else 0

                # 特征2: 阶段跨度
                phase_span = (max(phases) - min(phases)) / 13.0 if len(phases) > 1 else 0

                # 特征3: 阶段覆盖（独特阶段数/14）
                phase_coverage = len(set(phases)) / 14.0

                # 特征4: 技术多样性（独特技术数/序列长度）
                tech_diversity = len(set(seq)) / seq_len

                # 特征5: 攻击深度（最后阶段/13）
                attack_depth = max(phases) / 13.0
            else:
                norm_len = phase_span = phase_coverage = tech_diversity = attack_depth = 0

            global_features.append([norm_len, phase_span, phase_coverage, tech_diversity, attack_depth])

        global_features = torch.tensor(global_features, dtype=torch.float32)
        print(f"     -> 全局特征形状: {global_features.shape}")
        print(f"     -> 特征: [归一化长度, 阶段跨度, 阶段覆盖, 技术多样性, 攻击深度]")

        # 转换为tensor
        labels_tensor = torch.tensor(self.apt_encoder.transform(labels), dtype=torch.long)

        # 保存TTP序列数据（因果序列 + 阶段序列 + 全局特征）
        ttp_data = {
            'causal_sequences': technique_sequences,  # 已按战术阶段排序的因果序列
            'phase_sequences': phase_sequences,  # 每个技术对应的战术阶段ID (0-13)
            'technique_embeddings': torch.tensor(technique_embeddings, dtype=torch.float32),  # [num_techniques+1, 384]
            'global_features': global_features,  # [num_events, 5] 全局特征
            'labels': labels_tensor,
            'num_events': len(technique_sequences),
            'num_techniques': num_techniques,
            'num_classes': len(self.apt_encoder.classes_),
            'apt_classes': self.apt_encoder.classes_,
            'padding_value': 0,
            'semantic_dim': 384,
            'num_phases': 14,  # MITRE ATT&CK 14个战术阶段
            'global_feature_dim': 5,  # 全局特征维度
            'seq_stats': {
                'mean': float(np.mean(seq_lengths)),
                'max': max(seq_lengths),
                'min': min(seq_lengths),
                'median': float(np.median(seq_lengths))
            },
            # 保存战术映射信息（用于验证）
            'tactic_mapping': dict(tactic_mapping) if 'tactic_mapping' in locals() else {},
            'tactic_phase_order': TACTIC_PHASE_ORDER,
            'sequence_type': 'causal_enhanced'  # 标识这是增强的因果序列
        }

        ttp_path = output_dir / "apt_kg_ttp.pt"
        torch.save(ttp_data, ttp_path)
        print(f"\n  [OK] TTP序列数据已保存: {ttp_path}")
        print(f"     数据格式: 增强因果序列（阶段感知 + 全局特征）")
        print(f"     序列数量: {len(technique_sequences)}")
        print(f"     技术数量: {num_techniques}")
        print(f"     语义向量: {technique_embeddings.shape}")
        print(f"     阶段序列: {len(phase_sequences)} 个")
        print(f"     全局特征: {global_features.shape}")
        print(f"     标签: {labels_tensor.shape}")

        print("\n" + "="*70)
        print("双模型子图导出完成！")
        print(f"  IOC子图: 图结构 (IP/domain/URL/File/CVE/ASN + EVENT)")
        print(f"         EVENT数量: {len(valid_event_indices)}")
        print(f"         EVENT特征: 原始特征 + Node2Vec特征")
        print(f"  TTP数据: 变长技术序列 + 预训练语义嵌入 (用于Transformer)")
        print(f"         序列数量: {len(technique_sequences)}")
        print(f"         平均长度: {np.mean(seq_lengths):.1f} 技术数")
        print("="*70)

    def run(self, output_path="apt_kg_embedding.pt"):
        """运行双模型子图导出流程

        Args:
            output_path: 输出目录路径
        """
        print("[DEBUG] run方法开始", flush=True)
        from pathlib import Path
        output_dir = Path(output_path).parent
        print(f"[DEBUG] 输出目录: {output_dir}", flush=True)
        print("[DEBUG] 调用export_dual_subgraphs...", flush=True)
        self.export_dual_subgraphs(output_dir=output_dir)
        print("\n[完成] 双模型子图导出完成！")
        print(f"  - IOC子图: {output_dir / 'apt_kg_ioc.pt'}")
        print(f"  - TTP子图: {output_dir / 'apt_kg_ttp.pt'}")
        print("="*70)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='导出双模型子图（IOC图 + TTP序列）')
    parser.add_argument('--uri', type=str, default='bolt://127.0.0.1:7687', help='Neo4j URI')
    parser.add_argument('--user', type=str, default='neo4j', help='Neo4j用户名')
    parser.add_argument('--password', type=str, default='neo4j123', help='Neo4j密码')
    parser.add_argument('--output', type=str, default='apt_kg_ioc.pt', help='输出文件路径（输出到脚本所在目录）')

    args = parser.parse_args()

    exporter = ImprovedGraphExporter(args.uri, args.user, args.password)
    try:
        exporter.run(output_path=args.output)
    finally:
        exporter.close()
