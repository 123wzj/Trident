import os
import json
import warnings
from math import log2
from typing import Tuple
from urllib.parse import urlparse
from collections import defaultdict
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

try:
    from .config import add_neo4j_args, require_password
except ImportError:
    from config import add_neo4j_args, require_password

warnings.filterwarnings("ignore")


try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMER = True

    LOCAL_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "all-MiniLM-L6-v2"
    )

except ImportError:
    HAS_SENTENCE_TRANSFORMER = False
    LOCAL_MODEL_PATH = None


try:
    from torch_geometric.nn import Node2Vec as _Node2Vec

    import torch

    _test_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    try:
        _Node2Vec(
            _test_edge_index,
            walk_length=20,
            context_size=10,
            embedding_dim=16,
            sparse=True,
        )
        Node2Vec = _Node2Vec
        HAS_NODE2VEC = True
    except (ImportError, Exception) as e:
        HAS_NODE2VEC = False
        Node2Vec = None

        del _test_edge_index
except ImportError:
    HAS_NODE2VEC = False
    Node2Vec = None


class EdgeSemanticEncoder:
    def __init__(self):
        self.event_edges = {
            "USES_INFRASTRUCTURE",
            "USES_DOMAIN",
            "DELIVERS_VIA_URL",
            "EXPLOITS_VULN",
            "DROPS_MALWARE",
            "USES_TECHNIQUE",
        }

        self.ioc_edges = {
            "BELONGS_TO_NETWORK",
            "RESOLVES_TO",
            "RESOLVES_FROM",
            "HOSTED_ON_DOMAIN",
            "RESOLVES_TO_IP",
        }

        self.similarity_edges = {"SIMILAR_TO"}

    def get_edge_semantic_features(self, edge_type: Tuple[str, str, str]) -> np.ndarray:
        src, rel, dst = edge_type

        feat = np.zeros(4, dtype=np.float32)

        if src == "EVENT":
            feat[0] = 1.0
        elif rel in self.event_edges:
            feat[0] = 1.0

        if rel in self.ioc_edges:
            feat[1] = 1.0

        if rel in self.similarity_edges:
            feat[2] = 1.0

        if rel in {"SIMILAR_TO"}:
            feat[3] = 0.0
        else:
            feat[3] = 1.0

        return feat


TACTIC_PHASE_ORDER = {
    "TA0049": 0,
    "TA0042": 1,
    "TA0001": 2,
    "TA0002": 3,
    "TA0003": 4,
    "TA0004": 5,
    "TA0005": 6,
    "TA0006": 7,
    "TA0007": 8,
    "TA0008": 9,
    "TA0009": 10,
    "TA0011": 11,
    "TA0010": 12,
    "TA0040": 13,
}


def get_technique_phase_order(tech_id, tactic_mapping):
    if tech_id not in tactic_mapping:
        return 7

    tactics = tactic_mapping[tech_id]
    min_phase = 999

    for tactic in tactics:
        tactic_id = tactic["tactic_id"]
        if tactic_id in TACTIC_PHASE_ORDER:
            min_phase = min(min_phase, TACTIC_PHASE_ORDER[tactic_id])

    return min_phase if min_phase != 999 else 7


def generate_causal_sequence(tech_ids, tactic_mapping):
    tech_with_phase = []
    for tech_id in tech_ids:
        phase = get_technique_phase_order(tech_id, tactic_mapping)
        tech_with_phase.append((tech_id, phase))

    tech_with_phase.sort(key=lambda x: (x[1], x[0]))

    return [tech_id for tech_id, _ in tech_with_phase]


class EmbeddingEncoder:
    def __init__(self, categories, embed_dim=16, init_method="xavier"):
        self.categories = list(set(categories))
        self.num_categories = len(self.categories)
        self.embed_dim = embed_dim

        self.cat_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_cat = {idx: cat for cat, idx in self.cat_to_idx.items()}

        self.embeddings = self._init_embeddings(init_method)

    def _init_embeddings(self, method):
        if method == "xavier":
            bound = np.sqrt(6.0 / (self.num_categories + self.embed_dim))
            embeddings = np.random.uniform(
                -bound, bound, (self.num_categories, self.embed_dim)
            ).astype(np.float32)
        elif method == "normal":
            embeddings = np.random.normal(
                0, 0.1, (self.num_categories, self.embed_dim)
            ).astype(np.float32)
        else:
            embeddings = np.random.uniform(
                -0.1, 0.1, (self.num_categories, self.embed_dim)
            ).astype(np.float32)

        return embeddings

    def get_embedding(self, category):
        idx = self.cat_to_idx.get(category, 0)
        return self.embeddings[idx]

    def encode(self, categories):
        indices = [self.cat_to_idx.get(cat, 0) for cat in categories]
        return self.embeddings[indices]

    def save(self, filepath):
        data = {
            "categories": self.categories,
            "cat_to_idx": self.cat_to_idx,
            "embeddings": self.embeddings.tolist(),
            "embed_dim": self.embed_dim,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)

        encoder = cls.__new__(cls)
        encoder.categories = data["categories"]
        encoder.cat_to_idx = {k: int(v) for k, v in data["cat_to_idx"].items()}
        encoder.embeddings = np.array(data["embeddings"], dtype=np.float32)
        encoder.embed_dim = data["embed_dim"]
        encoder.num_categories = len(encoder.categories)

        return encoder


class FrequencyEncoder:
    def __init__(self):
        self.freq_map = {}
        self.max_freq = 0

    def fit(self, categories):
        from collections import Counter

        freq = Counter(categories)
        self.max_freq = max(freq.values())

        for cat, count in freq.items():
            self.freq_map[cat] = count / self.max_freq

    def encode(self, categories, dim=1):
        freqs = [self.freq_map.get(cat, 0.0) for cat in categories]
        return np.array(freqs, dtype=np.float32).reshape(-1, dim)


class HashEncoder:
    def __init__(self, dim=8):
        self.dim = dim

    def _hash_to_vector(self, category):
        import hashlib

        hash_obj = hashlib.md5(str(category).encode())
        hash_hex = hash_obj.hexdigest()

        vector = np.array(
            [
                int(hash_hex[i : i + 2], 16) / 255.0
                for i in range(0, min(len(hash_hex), self.dim * 2), 2)
            ]
        )

        if len(vector) < self.dim:
            vector = np.pad(vector, (0, self.dim - len(vector)))
        else:
            vector = vector[: self.dim]

        return vector.astype(np.float32)

    def encode(self, categories):
        return np.vstack([self._hash_to_vector(cat) for cat in categories])


def get_country_code_mapper():
    fname = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "feature_extraction",
        "helper_files",
        "country_codes.csv",
    )
    if not os.path.exists(fname):
        return {}

    ccs = {}
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            two, three = line.split(",")
            ccs[two] = len(ccs)
            ccs[three] = len(ccs)
    return ccs


def get_tld_headers(top_k=50):
    fname = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "feature_extraction",
        "helper_files",
        "ranked_tlds.csv",
    )
    if not os.path.exists(fname):
        return ["COM", "NET", "ORG", "INFO", "BIZ", "IO", "CO", "XYZ"][:top_k]

    tlds = []
    with open(fname, "r") as f:
        f.readline()
        for _ in range(top_k):
            line = f.readline()
            if not line:
                break
            tlds.append(line.split(",")[0].strip())
    return tlds


def nlp_features_domain(s):
    if not s or len(s) == 0:
        return {
            "domain_entropy": 0.0,
            "domain_length": 0,
            "num_digits": 0,
            "subdomains": 0,
        }

    probs = [s.count(c) / len(s) for c in set(s)]
    entropy = -sum([p * log2(p) for p in probs])

    return {
        "domain_entropy": entropy,
        "domain_length": len(s),
        "num_digits": len([d for d in s if d.isdigit()]),
        "subdomains": s.count("."),
    }


def nlp_features_url(ioc):
    if not ioc or len(ioc) == 0:
        return {
            "url_entropy": 0.0,
            "url_path_entropy": 0.0,
            "url_length": 0,
            "num_periods": 0,
            "num_subdir": 0,
            "num_digits": 0,
            "num_frag": 0,
            "num_params": 0,
            "url_path_length": 0,
            "url_host_length": 0,
            "has_port": 0,
        }

    parsed = urlparse(ioc)

    has_port = parsed.netloc.split(":")
    has_port = int(len(has_port) > 1 and has_port[-1].isdigit())

    dirs = parsed.path.strip("/").split("/")
    num_subdirectories = len(dirs) if dirs and dirs[0] != "" else 0

    frags = parsed.fragment
    num_fragments = 0 if not frags or frags == "" else len(frags.split("#"))

    params = parsed.query
    num_params = 0 if not params or params == "" else len(params.split("&"))

    def entropy(s):
        if not s or len(s) == 0:
            return 0.0
        probs = [s.count(c) / len(s) for c in set(s)]
        return -sum([p * log2(p) for p in probs])

    return {
        "url_entropy": entropy(ioc),
        "url_path_entropy": entropy(parsed.path),
        "url_length": len(ioc),
        "num_periods": ioc.count("."),
        "num_subdir": num_subdirectories,
        "num_digits": len([i for i in ioc if i.isdigit()]),
        "num_frag": num_fragments,
        "num_params": num_params,
        "url_path_length": len(parsed.path),
        "url_host_length": len(parsed.netloc),
        "has_port": has_port,
    }


def extract_tld(url, top_tlds):
    if not url or "." not in url:
        return "UNKNOWN"

    tld = url.split(".")[-1].split("/")[0].split(":")[0].upper()
    return tld if tld in top_tlds else "OTHER"


class ImprovedGraphExporter:
    def __init__(self, uri, user, pwd):
        if GraphDatabase is None:
            raise ModuleNotFoundError(
                "Missing dependency 'neo4j'. Install it in this Python environment: pip install neo4j"
            )

        self.uri = uri
        self.user = user
        self.pwd = pwd
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
        self.data = HeteroData()
        self.node_mapping = {}
        self.apt_encoder = LabelEncoder()
        self.encoders = {}
        self.top_tlds = get_tld_headers(top_k=50)
        self.edge_semantic_encoder = EdgeSemanticEncoder()
        self.feature_config = {}
        self.technique_rarity_map = {}

    def close(self):
        self.driver.close()

    def _run_query_df(self, query, params=None, max_retries=3):
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
                    wait_time = (attempt + 1) * 5
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                raise

    def _register_nodes(self, label, neo4j_ids_series):
        if neo4j_ids_series.empty:
            return 0
        unique_ids = neo4j_ids_series.unique()
        count = len(unique_ids)

        mapping = pd.Series(data=np.arange(count), index=unique_ids)
        self.node_mapping[label] = mapping
        return count

    def build_rich_node_features(self):
        print("[INFO] Building node features")
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
            num = self._register_nodes("IP", df["id"])

            lat_norm = df["lat_norm"].fillna(0).values.astype(np.float32).reshape(-1, 1)
            lon_norm = df["lon_norm"].fillna(0).values.astype(np.float32).reshape(-1, 1)

            cc_values = df["cc"].fillna("UNKNOWN").values
            cc_encoder = EmbeddingEncoder(cc_values, embed_dim=32)
            cc_embed = cc_encoder.encode(cc_values)
            self.encoders["IP_country_code"] = cc_encoder

            iss_values = df["issuer"].fillna("UNKNOWN").values
            iss_encoder = EmbeddingEncoder(iss_values, embed_dim=64)
            iss_embed = iss_encoder.encode(iss_values)
            self.encoders["IP_issuer"] = iss_encoder

            freq_encoder = FrequencyEncoder()
            freq_encoder.fit(iss_values)
            iss_freq = freq_encoder.encode(iss_values)

            feats = np.hstack(
                [lat_norm, lon_norm, cc_embed, iss_embed, iss_freq]
            ).astype(np.float32)

            self.data["IP"].x = torch.tensor(feats, dtype=torch.float32)
            self.data["IP"].num_nodes = num

            self.feature_config["IP"] = {
                "dim": feats.shape[1],
                "components": {
                    "lat_lon": 2,
                    "cc_embed": 32,
                    "issuer_embed": 64,
                    "issuer_freq": 1,
                },
            }

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
            num = self._register_nodes("domain", df["id"])

            nlp_feats = np.array(
                [
                    [
                        nlp_features_domain(val)["domain_entropy"],
                        nlp_features_domain(val)["domain_length"],
                        nlp_features_domain(val)["num_digits"],
                        nlp_features_domain(val)["subdomains"],
                    ]
                    for val in df["id"]
                ],
                dtype=np.float32,
            )

            tld_values = [extract_tld(val, self.top_tlds) for val in df["id"]]
            tld_encoder = EmbeddingEncoder(tld_values, embed_dim=16)
            tld_embed = tld_encoder.encode(tld_values)
            self.encoders["domain_TLD"] = tld_encoder

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

            first_seen = np.array(
                [parse_timestamp(ts) for ts in df["fs"]], dtype=np.float32
            ).reshape(-1, 1)
            last_seen = np.array(
                [parse_timestamp(ts) for ts in df["ls"]], dtype=np.float32
            ).reshape(-1, 1)

            has_nxdomain = (
                df["nx"]
                .apply(lambda x: 1.0 if x else 0.0)
                .values.astype(np.float32)
                .reshape(-1, 1)
            )
            lifespan_log = (
                df["lifespan_log"].fillna(0).values.astype(np.float32).reshape(-1, 1)
            )

            feats = np.hstack(
                [
                    nlp_feats,
                    tld_embed,
                    first_seen,
                    last_seen,
                    has_nxdomain,
                    lifespan_log,
                ]
            )

            self.data["domain"].x = torch.tensor(feats, dtype=torch.float32)
            self.data["domain"].num_nodes = num

            self.feature_config["domain"] = {"dim": feats.shape[1]}
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
            num = self._register_nodes("URL", df["id"])

            url_nlp_feats = np.array(
                [
                    [
                        nlp_features_url(val)["url_entropy"],
                        nlp_features_url(val)["url_path_entropy"],
                        nlp_features_url(val)["url_length"],
                        nlp_features_url(val)["num_periods"],
                        nlp_features_url(val)["num_subdir"],
                        nlp_features_url(val)["num_digits"],
                        nlp_features_url(val)["num_frag"],
                        nlp_features_url(val)["num_params"],
                        nlp_features_url(val)["url_path_length"],
                        nlp_features_url(val)["url_host_length"],
                        nlp_features_url(val)["has_port"],
                    ]
                    for val in df["id"]
                ],
                dtype=np.float32,
            )

            tld_values = [extract_tld(val, self.top_tlds) for val in df["id"]]
            tld_encoder = EmbeddingEncoder(tld_values, embed_dim=16)
            tld_embed = tld_encoder.encode(tld_values)
            self.encoders["URL_TLD"] = tld_encoder

            code_values = df["code"].fillna("Unknown").values
            code_encoder = EmbeddingEncoder(code_values, embed_dim=8)
            code_embed = code_encoder.encode(code_values)
            self.encoders["URL_http_code"] = code_encoder

            ft_values = df["ft"].fillna("Unknown").values
            ft_encoder = EmbeddingEncoder(ft_values, embed_dim=8)
            ft_embed = ft_encoder.encode(ft_values)
            self.encoders["URL_filetype"] = ft_encoder

            feats = np.hstack([url_nlp_feats, tld_embed, code_embed, ft_embed])

            self.data["URL"].x = torch.tensor(feats, dtype=torch.float32)
            self.data["URL"].num_nodes = num

            self.feature_config["URL"] = {"dim": feats.shape[1]}
        query = """
        MATCH (n:CVE)
        RETURN n.id as id, n.year as year, n.year_norm as year_norm
        """
        df = self._run_query_df(query)
        if not df.empty:
            df_original = df.copy()
            df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

            num = self._register_nodes("CVE", df["id"])

            year_norm = df["year_norm"].fillna(0).values.reshape(-1, 1)

            cve_nums = []
            for cve_id in df["id"]:
                parts = cve_id.split("-")
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

            self.data["CVE"].x = torch.tensor(feats, dtype=torch.float32)
            self.data["CVE"].num_nodes = num

            self.feature_config["CVE"] = {"dim": 2}

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
            num = self._register_nodes("File", df["id"])

            df["imp_short"] = (
                df["imp"].fillna("00000000000000000000000000000000").str[:8]
            )
            imp_values = df["imp_short"].fillna("unknown").values
            imp_encoder = EmbeddingEncoder(imp_values, embed_dim=16)
            imp_embed = imp_encoder.encode(imp_values)
            self.encoders["File_imphash"] = imp_encoder

            def extract_ssdeep_bs(ssd):
                if not ssd or pd.isna(ssd):
                    return 0.0
                try:
                    bs = int(str(ssd).split(":")[0])
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

            ssdeep_bs = (
                np.array([extract_ssdeep_bs(ssd) for ssd in df["ssd"]])
                .reshape(-1, 1)
                .astype(np.float32)
            )
            tlsh_feat = (
                np.array([extract_tlsh_feat(tlsh) for tlsh in df["tlsh"]])
                .reshape(-1, 1)
                .astype(np.float32)
            )

            feats = np.hstack([imp_embed, ssdeep_bs, tlsh_feat])

            self.data["File"].x = torch.tensor(feats, dtype=torch.float32)
            self.data["File"].num_nodes = num

            self.feature_config["File"] = {"dim": feats.shape[1]}
        query = """
        MATCH (n:ASN)
        RETURN n.value as id, n.issuer as issuer
        """
        df = self._run_query_df(query)
        if not df.empty:
            num = self._register_nodes("ASN", df["id"])

            issuers = df["issuer"].fillna("unknown").astype(str).values
            iss_encoder = EmbeddingEncoder(issuers, embed_dim=16)
            iss_embed = iss_encoder.encode(issuers)
            self.encoders["ASN_issuer"] = iss_encoder

            freq_encoder = FrequencyEncoder()
            freq_encoder.fit(issuers)
            iss_freq = freq_encoder.encode(issuers)

            feats = np.hstack([iss_embed, iss_freq])

            self.data["ASN"].x = torch.tensor(feats, dtype=torch.float32)
            self.data["ASN"].num_nodes = num

            self.feature_config["ASN"] = {"dim": feats.shape[1]}
        query = """
        MATCH (n:Technique)
        RETURN n.id as id, n.name as name, n.description as description
        """
        df = self._run_query_df(query)
        if not df.empty:
            num = self._register_nodes("Technique", df["id"])

            features_list = []

            if HAS_SENTENCE_TRANSFORMER:
                if os.path.exists(LOCAL_MODEL_PATH):
                    model = SentenceTransformer(LOCAL_MODEL_PATH)
                else:
                    model = SentenceTransformer("all-MiniLM-L6-v2")

                technique_texts = []
                for _, row in df.iterrows():
                    tech_id = row["id"]
                    name = row["name"] if pd.notna(row["name"]) else ""
                    desc = row["description"] if pd.notna(row["description"]) else ""

                    text = f"{tech_id}: {name}. {desc}" if name else tech_id
                    technique_texts.append(text)

                semantic_embeds = model.encode(technique_texts, show_progress_bar=False)
                features_list.append(semantic_embeds)
            else:
                technique_hash_embeds = np.array(
                    [self._hash_to_vector(tid, dim=64) for tid in df["id"]],
                    dtype=np.float32,
                )
                features_list.append(technique_hash_embeds)

            rarity_query = """
            MATCH (t:Technique)<-[:USES_TECHNIQUE]-(e:EVENT)
            RETURN t.id as tid, count(DISTINCT e.label) as org_count
            """
            rarity_df = self._run_query_df(rarity_query)

            org_count_map = {}
            if not rarity_df.empty:
                for _, row in rarity_df.iterrows():
                    org_count_map[row["tid"]] = row["org_count"]

            total_orgs_query = """
            MATCH (e:EVENT)
            RETURN count(DISTINCT e.label) as total_orgs
            """
            total_orgs_df = self._run_query_df(total_orgs_query)
            total_orgs = (
                total_orgs_df["total_orgs"].iloc[0] if not total_orgs_df.empty else 20
            )
            total_orgs = max(total_orgs, 1)

            rarity_features = []
            for tech_id in df["id"]:
                org_count = org_count_map.get(tech_id, 1)
                coverage = org_count / total_orgs
                rarity = 1 - coverage
                log_org_count = np.log1p(org_count) / np.log1p(total_orgs)

                rarity_features.append([rarity, coverage, log_org_count])

                self.technique_rarity_map[tech_id] = rarity

            rarity_features = np.array(rarity_features, dtype=np.float32)
            features_list.append(rarity_features)

            if HAS_SENTENCE_TRANSFORMER:
                feats = np.hstack(features_list).astype(np.float32)

            else:
                feats = np.hstack(features_list).astype(np.float32)

            self.data["Technique"].x = torch.tensor(feats, dtype=torch.float32)
            self.data["Technique"].num_nodes = num

            self.feature_config["Technique"] = {"dim": feats.shape[1]}

    def _hash_to_vector(self, text, dim=64):
        import hashlib

        hash_obj = hashlib.md5(str(text).encode())
        hash_hex = hash_obj.hexdigest()

        vector = np.array(
            [
                int(hash_hex[i : i + 2], 16) / 255.0
                for i in range(0, min(len(hash_hex), dim * 2), 2)
            ]
        )

        if len(vector) < dim:
            vector = np.pad(vector, (0, dim - len(vector)))
        else:
            vector = vector[:dim]

        return vector.astype(np.float32)

    def build_event_nodes(self, ttp_only=False, valid_event_ids=None):
        if valid_event_ids is not None:
            valid_ids_list = list(valid_event_ids)
            query = """
            MATCH (e:EVENT)
            WHERE e.id IN $event_ids
            RETURN e.id as id, e.label as label
            """
            df = self._run_query_df(query, params={"event_ids": valid_ids_list})
        elif ttp_only:
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
            raise RuntimeError("No EVENT nodes found for export.")

        num_nodes = self._register_nodes("EVENT", df["id"])
        event_ids = df["id"].tolist()

        if not HAS_SENTENCE_TRANSFORMER:
            event_features = np.zeros((num_nodes, 1), dtype=np.float32)
            self.data["EVENT"].x = torch.tensor(event_features, dtype=torch.float32)
            self.data["EVENT"].num_nodes = num_nodes
            print(f"[INFO] Built {num_nodes} EVENT nodes")
            return

        model = SentenceTransformer(LOCAL_MODEL_PATH)

        event_features_list = []
        for eid in event_ids:
            tech_names_query = """
            MATCH (e:EVENT {id: $eid})-[:USES_TECHNIQUE]->(t:Technique)
            RETURN t.name as tech_name, t.id as tech_id
            """
            tech_names_df = self._run_query_df(tech_names_query, params={"eid": eid})

            if not tech_names_df.empty:
                tech_names = tech_names_df["tech_name"].tolist()
                tech_embeddings = model.encode(tech_names, show_progress_bar=False)

                tech_mean = np.mean(tech_embeddings, axis=0)

                num_techs = min(len(tech_names) / 50.0, 1.0)

                event_feat = np.concatenate([tech_mean, [num_techs]])
            else:
                event_feat = np.concatenate([np.zeros(384), [0.0]])

            event_features_list.append(event_feat)

        event_features = np.array(event_features_list, dtype=np.float32)

        self.data["EVENT"].x = torch.tensor(event_features, dtype=torch.float32)
        self.data["EVENT"].num_nodes = num_nodes

        print(f"[INFO] Built {num_nodes} EVENT nodes")

    def build_edges(self):
        edge_configs = [
            ("EVENT", "USES_INFRASTRUCTURE", "IP", "id", "value", False),
            ("EVENT", "USES_DOMAIN", "domain", "id", "value", False),
            ("EVENT", "DELIVERS_VIA_URL", "URL", "id", "value", False),
            ("EVENT", "EXPLOITS_VULN", "CVE", "id", "id", False),
            ("EVENT", "DROPS_MALWARE", "File", "id", "sha256", False),
            ("EVENT", "USES_TECHNIQUE", "Technique", "id", "id", True),
            ("IP", "BELONGS_TO_NETWORK", "ASN", "value", "value", False),
            ("IP", "RESOLVES_TO", "domain", "value", "value", False),
            ("domain", "RESOLVES_TO", "IP", "value", "value", False),
            ("URL", "HOSTED_ON_DOMAIN", "domain", "value", "value", False),
            ("URL", "RESOLVES_TO_IP", "IP", "value", "value", False),
            ("File", "SIMILAR_TO", "File", "sha256", "sha256", True),
        ]

        edge_type_list = []
        for src, rel, dst, _, _, _ in edge_configs:
            edge_type_list.append((src, rel, dst))
            if src != dst:
                edge_type_list.append((dst, f"rev_{rel}", src))

        self.edge_type_to_idx = {et: i for i, et in enumerate(edge_type_list)}
        self.num_edge_types = len(edge_type_list)
        edge_stats = {}
        total_edges = 0

        for src, rel, dst, sp, dp, weighted in edge_configs:
            if src not in self.node_mapping or dst not in self.node_mapping:
                continue

            if weighted:
                if src == "EVENT" and dst == "Technique":
                    query = f"MATCH (e:{src})-[r:{rel}]->(t:{dst}) RETURN e.{sp} as s, t.{dp} as d"
                else:
                    query = f"MATCH (a:{src})-[r:{rel}]->(b:{dst}) RETURN a.{sp} as s, b.{dp} as d, r.score as w"
            else:
                query = f"MATCH (a:{src})-[r:{rel}]->(b:{dst}) RETURN a.{sp} as s, b.{dp} as d"

            df = self._run_query_df(query)
            if df.empty:
                continue

            if weighted and src == "EVENT" and dst == "Technique":
                df["w"] = df["d"].map(
                    lambda tid: self.technique_rarity_map.get(tid, 0.5)
                )

            u = df["s"].map(self.node_mapping[src])
            v = df["d"].map(self.node_mapping[dst])
            valid = u.notna() & v.notna()

            if not valid.any():
                continue

            u = torch.tensor(u[valid].astype(int).values, dtype=torch.long)
            v = torch.tensor(v[valid].astype(int).values, dtype=torch.long)
            edge_index = torch.stack([u, v], dim=0)

            self.data[src, rel, dst].edge_index = edge_index

            num_edges = edge_index.shape[1]
            edge_type_idx = self.edge_type_to_idx[(src, rel, dst)]

            edge_score = np.zeros(num_edges, dtype=np.float32)
            if weighted and "w" in df.columns:
                w = df.loc[valid, "w"].fillna(0.0).values
                if w.max() > 0:
                    w = w / w.max()
                edge_score = w.astype(np.float32)

            edge_type_tensor = torch.full(
                (num_edges, 1), edge_type_idx, dtype=torch.long
            )
            edge_score_tensor = torch.tensor(edge_score, dtype=torch.float32).unsqueeze(
                1
            )
            edge_attr = torch.cat([edge_type_tensor, edge_score_tensor], dim=1)
            self.data[src, rel, dst].edge_attr = edge_attr

            edge_stats[(src, rel, dst)] = edge_index.shape[1]
            total_edges += edge_index.shape[1]

            if src != dst:
                rev_rel = f"rev_{rel}"
                self.data[dst, rev_rel, src].edge_index = torch.stack([v, u], dim=0)

                rev_edge_type_idx = self.edge_type_to_idx[(dst, rev_rel, src)]
                rev_edge_type_tensor = torch.full(
                    (num_edges, 1), rev_edge_type_idx, dtype=torch.long
                )
                rev_edge_attr = torch.cat(
                    [rev_edge_type_tensor, edge_score_tensor], dim=1
                )
                self.data[dst, rev_rel, src].edge_attr = rev_edge_attr

        print(
            f"[INFO] Built {total_edges} graph edges across {len(edge_stats)} relation types"
        )

    def generate_labels(self):
        df = self._run_query_df(
            "MATCH (e:EVENT) WHERE e.label IS NOT NULL RETURN e.id as eid, e.label as apt"
        )
        if df.empty:
            return

        self.apt_encoder.fit(df["apt"])
        df["label"] = self.apt_encoder.transform(df["apt"])

        event_map = self.node_mapping["EVENT"]
        df["idx"] = df["eid"].map(event_map)
        df = df.dropna(subset=["idx"])

        indices = df["idx"].astype(int).values
        labels = df["label"].values

        y = torch.full((self.data["EVENT"].num_nodes,), -1, dtype=torch.long)
        y[indices] = torch.tensor(labels, dtype=torch.long)
        self.data["EVENT"].y = y

        self.data._apt_classes = self.apt_encoder.classes_
        print(f"[INFO] Encoded {len(self.apt_encoder.classes_)} APT labels")

    def add_node2vec_features(self, embedding_dim=64, epochs=20, node_types=None):
        if not HAS_NODE2VEC:
            return

        if node_types is None:
            ioc_node_types = {"EVENT", "IP", "domain", "URL", "File", "ASN", "CVE"}
            node_types = [nt for nt in self.data.node_types if nt in ioc_node_types]

        node_type_to_ids = {}
        global_id_offset = 0
        edge_index_list = []

        valid_node_types = [nt for nt in node_types if nt in self.data.node_types]
        for node_type in valid_node_types:
            num_nodes = self.data[node_type].num_nodes
            node_type_to_ids[node_type] = (
                global_id_offset,
                global_id_offset + num_nodes,
            )
            global_id_offset += num_nodes

        total_nodes = global_id_offset

        for edge_type in self.data.edge_types:
            src_type, _, dst_type = edge_type

            if src_type not in valid_node_types or dst_type not in valid_node_types:
                continue

            edge_index = self.data[edge_type].edge_index

            src_start, _ = node_type_to_ids[src_type]
            dst_start, _ = node_type_to_ids[dst_type]

            global_src = edge_index[0] + src_start
            global_dst = edge_index[1] + dst_start

            edge_index_list.append(torch.stack([global_src, global_dst], dim=0))

        if edge_index_list:
            edge_index = torch.cat(edge_index_list, dim=1)
        else:
            return

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Training Node2Vec on {total_nodes} nodes")

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
            num_nodes=total_nodes,
        ).to(device)

        optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

        node2vec.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            loader = node2vec.loader(batch_size=256, shuffle=True, num_workers=0)

            for pos_rw, neg_rw in loader:
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

        node2vec.eval()
        with torch.no_grad():
            embeddings = node2vec.embedding.weight.cpu()

        for node_type in node_type_to_ids.keys():
            start, end = node_type_to_ids[node_type]
            node_type_embeddings = embeddings[start:end]

            original_x = self.data[node_type].x

            if original_x.shape[0] != node_type_embeddings.shape[0]:
                continue

            new_x = torch.cat([original_x, node_type_embeddings], dim=1)
            self.data[node_type].x = new_x

            if node_type in self.feature_config:
                self.feature_config[node_type]["dim"] = new_x.shape[1]
                self.feature_config[node_type]["node2vec_dim"] = embedding_dim
        print("[INFO] Node2Vec features added")

    def export_dual_subgraphs(self, output_dir="."):
        print("=" * 70, flush=True)

        from pathlib import Path
        import pandas as pd

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        query = """MATCH (e:EVENT)-[:USES_TECHNIQUE]->(t:Technique)
                    RETURN DISTINCT e.id as event_id"""

        df_events = self._run_query_df(query)

        if df_events.empty:
            return

        all_ttp_event_ids = set(df_events["event_id"].tolist())

        self.build_rich_node_features()

        self.build_event_nodes(valid_event_ids=all_ttp_event_ids)

        self.build_edges()

        self.generate_labels()

        self.add_node2vec_features(embedding_dim=64, epochs=20)

        query = """
        MATCH (e:EVENT)-[r:USES_TECHNIQUE]->(t:Technique)
        WHERE e.id IN $event_ids
        RETURN e.id as event_id, t.id as tech_id
        ORDER BY e.id, t.id
        """
        df_tech = self._run_query_df(
            query, params={"event_ids": list(all_ttp_event_ids)}
        )

        if df_tech.empty:
            return

        all_techniques = df_tech["tech_id"].unique()
        technique_to_idx = {
            tech_id: idx + 1 for idx, tech_id in enumerate(sorted(all_techniques))
        }
        num_techniques = len(technique_to_idx)

        event_raw_techniques = {}
        for eid, group in df_tech.groupby("event_id"):
            event_raw_techniques[eid] = group["tech_id"].tolist()

        filtered_sequences = {}
        filtered_event_ids = []
        min_seq_len = 3

        seq_len_all = []
        for event_id, tech_ids in event_raw_techniques.items():
            if len(tech_ids) < min_seq_len:
                continue
            filtered_sequences[event_id] = tech_ids
            filtered_event_ids.append(event_id)
            seq_len_all.append(len(tech_ids))

        valid_event_ids = set(filtered_event_ids)
        event_raw_techniques = {
            eid: filtered_sequences[eid] for eid in filtered_event_ids
        }

        event_map = self.node_mapping.get("EVENT")

        if event_map is None:
            return

        valid_event_indices = []
        for eid in valid_event_ids:
            if eid in event_map.index:
                valid_event_indices.append(event_map[eid])

        valid_event_indices = sorted(valid_event_indices)

        if len(valid_event_indices) == 0:
            return

        old_to_new = {
            old_idx: new_idx for new_idx, old_idx in enumerate(valid_event_indices)
        }

        query_tactics = """
        MATCH (t:Technique)-[r:BELONGS_TO]->(tac:Tactic)
        RETURN t.id as tech_id, tac.id as tactic_id, tac.name as tactic_name
        """
        tactic_mapping = defaultdict(list)
        df_tactics = self._run_query_df(query_tactics)

        for _, row in df_tactics.iterrows():
            tech_id = row["tech_id"]
            tactic_mapping[tech_id].append(
                {"tactic_id": row["tactic_id"], "tactic_name": row["tactic_name"]}
            )

        tactic_mapping = dict(tactic_mapping)

        event_sequences = {}
        for event_id, tech_ids in event_raw_techniques.items():
            if tactic_mapping:
                causal_seq = generate_causal_sequence(tech_ids, tactic_mapping)
                event_sequences[event_id] = [technique_to_idx[t] for t in causal_seq]
            else:
                event_sequences[event_id] = [
                    technique_to_idx[t] for t in sorted(tech_ids)
                ]

        from torch_geometric.data import HeteroData

        ioc_data = HeteroData()

        ioc_node_types = {"IP", "domain", "URL", "File", "CVE", "ASN", "EVENT"}

        for node_type in ioc_node_types:
            if node_type != "EVENT" and node_type in self.data.node_types:
                ioc_data[node_type].x = self.data[node_type].x.clone()
                ioc_data[node_type].num_nodes = self.data[node_type].num_nodes

        event_x = self.data["EVENT"].x
        event_y = self.data["EVENT"].y if hasattr(self.data["EVENT"], "y") else None

        ioc_data["EVENT"].x = event_x[valid_event_indices].clone()
        ioc_data["EVENT"].num_nodes = len(valid_event_indices)
        if event_y is not None:
            ioc_data["EVENT"].y = event_y[valid_event_indices].clone()

        ioc_data._node_mapping = {}
        for node_type in ioc_node_types:
            if node_type != "EVENT" and node_type in self.node_mapping:
                ioc_data._node_mapping[node_type] = self.node_mapping[node_type]

        num_events = len(valid_event_indices)
        ioc_data._node_mapping["EVENT"] = pd.Series(
            data=np.arange(num_events),
            index=pd.RangeIndex(start=0, stop=num_events, step=1),
        )

        ioc_edge_count = 0
        for edge_type in self.data.edge_types:
            src, rel, dst = edge_type

            if "Technique" not in str(edge_type):
                if src in ioc_node_types and dst in ioc_node_types:
                    edge_index = self.data[edge_type].edge_index.clone()

                    mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
                    if src == "EVENT":
                        mask = torch.isin(
                            edge_index[0], torch.tensor(valid_event_indices)
                        )

                        edge_index[0] = torch.tensor(
                            [
                                old_to_new.get(idx.item(), idx.item())
                                for idx in edge_index[0]
                            ]
                        )
                    elif dst == "EVENT":
                        mask = torch.isin(
                            edge_index[1], torch.tensor(valid_event_indices)
                        )

                        edge_index[1] = torch.tensor(
                            [
                                old_to_new.get(idx.item(), idx.item())
                                for idx in edge_index[1]
                            ]
                        )

                    edge_index = edge_index[:, mask]

                    if edge_index.shape[1] == 0:
                        continue

                    ioc_data[edge_type].edge_index = edge_index
                    if hasattr(self.data[edge_type], "edge_attr"):
                        edge_attr = self.data[edge_type].edge_attr.clone()
                        ioc_data[edge_type].edge_attr = edge_attr[mask]
                    ioc_edge_count += 1

        ioc_data._feature_config = {}
        for node_type in ioc_node_types:
            if node_type in ioc_data.node_types:
                if node_type == "EVENT":
                    ioc_data._feature_config[node_type] = {
                        "dim": ioc_data[node_type].x.shape[1]
                    }
                elif node_type in self.feature_config:
                    ioc_data._feature_config[node_type] = self.feature_config[node_type]

        ioc_data._apt_classes = (
            self.data._apt_classes
            if hasattr(self.data, "_apt_classes")
            else self.apt_encoder.classes_
        )

        ioc_path = output_dir / "apt_kg_ioc.pt"
        torch.save(ioc_data, ioc_path)
        print(f"[INFO] Saved IOC graph to {ioc_path}")

        all_techniques = list(
            set([t for tech_list in event_raw_techniques.values() for t in tech_list])
        )
        query_tech_desc = """
        MATCH (t:Technique)
        WHERE t.id IN $tech_ids
        RETURN t.id as tech_id, t.name as name, t.description as description
        ORDER BY t.id
        """
        df_tech_desc = self._run_query_df(
            query_tech_desc, params={"tech_ids": all_techniques}
        )

        import sys

        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "all-MiniLM-L6-v2")

        from sentence_transformers import SentenceTransformer

        try:
            model = SentenceTransformer(model_path)

        except Exception as e:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        technique_embeddings = np.zeros((num_techniques + 1, 384), dtype=np.float32)

        for _, row in df_tech_desc.iterrows():
            tech_id = row["tech_id"]
            idx = technique_to_idx[tech_id]

            text = f"{tech_id}: {row['name']}"
            if (
                pd.notna(row.get("description"))
                and len(str(row["description"]).strip()) > 0
            ):
                text += f". {row['description']}"

            with torch.no_grad():
                embedding = model.encode([text])[0]

            technique_embeddings[idx] = embedding.astype(np.float32)

        query_labels = """
        MATCH (e:EVENT)
        WHERE e.id IN $event_ids
        RETURN e.id as event_id, e.label as label
        """
        df_labels = self._run_query_df(
            query_labels, params={"event_ids": list(valid_event_ids)}
        )

        technique_sequences = []
        labels = []

        label_map = dict(zip(df_labels["event_id"], df_labels["label"]))

        for event_idx in valid_event_indices:
            event_id = None
            for eid, idx in event_map.items():
                if idx == event_idx and eid in valid_event_ids:
                    event_id = eid
                    break

            if event_id is not None:
                technique_sequences.append(event_sequences.get(event_id, []))
                labels.append(label_map.get(event_id, 0))

        seq_lengths = [len(seq) for seq in technique_sequences]

        phase_sequences = []
        for seq in technique_sequences:
            phases = []
            for tech_id in seq:
                tech_id_str = str(tech_id)
                phase = get_technique_phase_order(tech_id_str, tactic_mapping)
                phases.append(phase)
            phase_sequences.append(phases)

        global_features = []

        for i, seq in enumerate(technique_sequences):
            seq_len = len(seq)
            phases = phase_sequences[i]

            if seq_len > 0:
                norm_len = seq_len / max(seq_lengths) if max(seq_lengths) > 0 else 0

                phase_span = (
                    (max(phases) - min(phases)) / 13.0 if len(phases) > 1 else 0
                )

                phase_coverage = len(set(phases)) / 14.0

                tech_diversity = len(set(seq)) / seq_len

                attack_depth = max(phases) / 13.0
            else:
                norm_len = (
                    phase_span
                ) = phase_coverage = tech_diversity = attack_depth = 0

            global_features.append(
                [norm_len, phase_span, phase_coverage, tech_diversity, attack_depth]
            )

        global_features = torch.tensor(global_features, dtype=torch.float32)

        labels_tensor = torch.tensor(
            self.apt_encoder.transform(labels), dtype=torch.long
        )

        ttp_data = {
            "causal_sequences": technique_sequences,
            "phase_sequences": phase_sequences,
            "technique_embeddings": torch.tensor(
                technique_embeddings, dtype=torch.float32
            ),
            "global_features": global_features,
            "labels": labels_tensor,
            "num_events": len(technique_sequences),
            "num_techniques": num_techniques,
            "num_classes": len(self.apt_encoder.classes_),
            "apt_classes": self.apt_encoder.classes_,
            "padding_value": 0,
            "semantic_dim": 384,
            "num_phases": 14,
            "global_feature_dim": 5,
            "seq_stats": {
                "mean": float(np.mean(seq_lengths)),
                "max": max(seq_lengths),
                "min": min(seq_lengths),
                "median": float(np.median(seq_lengths)),
            },
            "tactic_mapping": dict(tactic_mapping)
            if "tactic_mapping" in locals()
            else {},
            "tactic_phase_order": TACTIC_PHASE_ORDER,
            "sequence_type": "causal_enhanced",
        }

        ttp_path = output_dir / "apt_kg_ttp.pt"
        torch.save(ttp_data, ttp_path)
        print(f"[INFO] Saved TTP sequence data to {ttp_path}")

    def run(self, output_path="apt_kg_embedding.pt"):
        from pathlib import Path

        output_dir = Path(output_path).parent

        self.export_dual_subgraphs(output_dir=output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export PyTorch Geometric graph data.")
    add_neo4j_args(parser)
    parser.add_argument("--output", type=str, default="apt_kg_ioc.pt")

    args = parser.parse_args()

    exporter = ImprovedGraphExporter(
        args.uri, args.user, require_password(args.password)
    )
    try:
        exporter.run(output_path=args.output)
    finally:
        exporter.close()
