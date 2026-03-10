import ipaddress
import os
import re
import json
from threading import Lock
from typing import List, Optional, Dict, Literal, Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.globals import set_debug
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, field_validator, ConfigDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from agent.my_llm import deepseek_model, deepseek_reasoner, gpt, qwen3_max, qwen3_plus
from cti_extrac.prompts import TTP_EXTRACTION_SYSTEM, TTP_EXTRACTION_USER, IOC_EXTRACTION_SYSTEM, IOC_EXTRACTION_USER, \
    ADVERSARIAL_CRITIC_SYSTEM, ADVERSARIAL_CRITIC_USER, SUMMARY_SYSTEM, SUMMARY_USER

# 开启调试模式，这会打印出 LLM 的完整输入 prompt 和原始输出
# set_debug(True)

# ==========================================
# 1. 配置层 (Configuration)
# ==========================================
llm = deepseek_model

llm_critic = deepseek_reasoner
# 自洽性推理配置
SELF_CONSISTENCY_ROUNDS = 3
CONFIDENCE_THRESHOLD = 0.66

# 增量式处理配置
CHUNK_SIZE = 8000  # 每块文本大小（字符数）
CHUNK_OVERLAP = 1000  # 重叠部分，防止边界实体丢失

# Batch 处理配置（防止 API 限流）
ADVERSARIAL_CRITIC_BATCH_SIZE = 10  # 对抗性审核每批处理数量
MAX_BATCH_REQUESTS = 20  # 单次 batch 最大请求数（防止超时或限流）

# ==========================================
# 全局 Prompt 模板（性能优化：避免重复创建）
# ==========================================
_TTP_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TTP_EXTRACTION_SYSTEM),
    ("user", TTP_EXTRACTION_USER)
])

_IOC_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", IOC_EXTRACTION_SYSTEM),
    ("user", IOC_EXTRACTION_USER)
])

_ADVERSARIAL_CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ADVERSARIAL_CRITIC_SYSTEM),
    ("user", ADVERSARIAL_CRITIC_USER)
])

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SUMMARY_SYSTEM),
    ("user", SUMMARY_USER)
])


# ==========================================
# 2. 数据模型层 (Schema Definition)
# ==========================================

class TTP(BaseModel):
    """描述一个从文中明确提取的战术、技术或过程"""
    technique_id: str = Field(..., description="MITRE ATT&CK ID, e.g., T1566.001. If not explicit, use 'Unknown'.")
    technique_name: str = Field(..., description="The name of the technique, e.g., 'Phishing'.")
    description: str = Field(..., description="Specific details from the text describing how this technique was used.")
    confidence_score: float = Field(default=1.0, ge=0, le=1, description="自洽性置信度")
    chunk_id: Optional[int] = Field(None, description="来源文本块ID")

    @field_validator('technique_id')
    def validate_id(cls, v):
        return v.upper().strip()


class BaseIOC(BaseModel):
    role: str = Field(
        ...,
        description="The tactical role of this indicator in the attack (e.g., C2, Payload_Delivery, Victim, Benign, Malware, etc.)"
    )
    context: Optional[str] = Field(None, description="Short snippet or reason why this is considered an IOC.")
    confidence_score: float = Field(default=1.0, ge=0, le=1, description="自洽性置信度")
    chunk_id: Optional[int] = Field(None, description="来源文本块ID")
    cross_chunk_reference: bool = Field(default=False, description="是否跨块引用")


class IpIOC(BaseIOC):
    value: str = Field(..., description="The IP address value (e.g., 192.168.1.1).")
    version: Literal["IPv4", "IPv6"] = Field("IPv4", description="IP Version")

    @field_validator('value')
    def clean_ip(cls, v):
        return v.replace("[.]", ".").strip()


class DomainIOC(BaseIOC):
    value: str = Field(..., description="The domain name or hostname.")
    record_type: Optional[str] = Field(None, description="DNS record type")

    @field_validator('value')
    def clean_domain(cls, v):
        return v.replace("[.]", ".").lower().strip()


class UrlIOC(BaseIOC):
    value: str = Field(..., description="The full URL.")
    protocol: Literal["http", "https", "ftp", "ws", "wss", "other"] = Field("http")

    @field_validator('value')
    def clean_url(cls, v):
        return v.replace("hxxp", "http").replace("[.]", ".").strip()


class FileIOC(BaseIOC):
    filename: Optional[str] = Field(None)
    md5: Optional[str] = Field(None)
    sha1: Optional[str] = Field(None)
    sha256: Optional[str] = Field(None)
    file_path: Optional[str] = Field(None)

    @field_validator('sha256')
    def validate_hash(cls, v):
        if not v:
            return None
        v = v.strip()
        if len(v) != 64:
            return None
        return v


class CveIOC(BaseModel):
    cve_id: str = Field(...)
    description: Optional[str] = Field(None)
    confidence_score: float = Field(default=1.0, ge=0, le=1)
    chunk_id: Optional[int] = Field(None)

    @field_validator('cve_id')
    def format_cve(cls, v):
        match = re.search(r'(CVE-\d{4}-\d{4,})', v.upper())
        return match.group(1) if match else v.upper()


class ExtractionResult(BaseModel):
    """单次提取结果"""
    summary: str = Field(default="")
    ttps: List[TTP] = Field(default_factory=list)
    ips: List[IpIOC] = Field(default_factory=list)
    domains: List[DomainIOC] = Field(default_factory=list)
    urls: List[UrlIOC] = Field(default_factory=list)
    files: List[FileIOC] = Field(default_factory=list)
    cves: List[CveIOC] = Field(default_factory=list)


# ==========================================
# 实体寄存器 (Entity Register)
# ==========================================
class EntityRegister(BaseModel):
    """跨窗口实体记忆 """
    # 存储已见过的实体，键为实体值，值为首次出现的上下文
    known_ips: Dict[str, str] = Field(default_factory=dict)
    known_domains: Dict[str, str] = Field(default_factory=dict)
    known_urls: Dict[str, str] = Field(default_factory=dict)
    known_files: Dict[str, str] = Field(default_factory=dict)
    known_cves: Dict[str, str] = Field(default_factory=dict)
    known_ttps: Dict[str, str] = Field(default_factory=dict)

    # 滚动摘要：每个chunk的关键信息
    rolling_summary: str = Field(default="")

    def register_entity(self, entity_type: str, value: str, context: str):
        """注册实体到寄存器"""
        register_map = {
            "ip": self.known_ips,
            "domain": self.known_domains,
            "url": self.known_urls,
            "file": self.known_files,
            "cve": self.known_cves,
            "ttp": self.known_ttps
        }
        if entity_type in register_map:
            target_dict = register_map[entity_type]

            if value not in target_dict:
                # 首次发现
                target_dict[value] = context[:200]
            else:
                # 防止无限增长：只保留最新的补充信息，或者去重
                current_ctx = target_dict[value]
                if context not in current_ctx and len(current_ctx) < 400:  # 限制总长度
                    target_dict[value] += f" | {context}"

    def get_context_for_entity(self, entity_type: str, value: str) -> Optional[str]:
        """查询实体的历史上下文"""
        register_map = {
            "ip": self.known_ips,
            "domain": self.known_domains,
            "url": self.known_urls,
            "file": self.known_files,
            "cve": self.known_cves,
            "ttp": self.known_ttps
        }
        return register_map.get(entity_type, {}).get(value)

    def update_rolling_summary(self, new_summary: str):
        """智能更新滚动摘要"""
        full_text = self.rolling_summary + "\n\n" + new_summary

        if len(full_text) > 2000:
            # 压缩旧摘要
            candidate = full_text[-1500:]
            # 找到第一个换行符的位置，丢弃前面的残句
            first_newline = candidate.find('\n')
            if first_newline != -1:
                self.rolling_summary = candidate[first_newline + 1:]
            else:
                self.rolling_summary = candidate  # 实在找不到换行符就硬切
        else:
            self.rolling_summary = full_text

    def get_register_summary(self) -> str:
        """生成带上下文的高质量摘要"""
        summary_parts = []

        def format_items(data: Dict[str, str], label: str, limit: int = 5):
            if not data:
                return None
            # 输出格式包含 Value 和 Context
            items = []
            # 取最新的 limit 个实体
            for k, v in list(data.items())[-limit:]:
                # 截取前 50 个字符的上下文，防止 Prompt 过长
                snippet = v[:50] + "..." if len(v) > 50 else v
                items.append(f"{k} ({snippet})")
            return f"{label}: {', '.join(items)}"

        if self.known_ips:
            summary_parts.append(format_items(self.known_ips, "Known IPs", limit=8))
        if self.known_domains:
            summary_parts.append(format_items(self.known_domains, "Known Domains", limit=8))
        if self.known_urls:
            summary_parts.append(format_items(self.known_urls, "Known URLs", limit=5))
        if self.known_files:
            summary_parts.append(format_items(self.known_files, "Known Files", limit=5))
        if self.known_cves:
            summary_parts.append(format_items(self.known_cves, "Known CVEs", limit=5))
        if self.known_ttps:
            summary_parts.append(format_items(self.known_ttps, "Known TTPs", limit=5))

        return " | ".join(filter(None, summary_parts)) if summary_parts else "No prior context."


# ==========================================
# 3. 状态定义 (State Definition)
# ==========================================
class AgentState(BaseModel):
    """LangGraph 工作流的状态容器"""
    # 1. 原始输入
    raw_text: str = Field(..., description="输入的 Markdown 文本")

    # 2. 文本分块
    text_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="文本块列表")
    current_chunk_idx: int = Field(default=0, description="当前处理的块索引")

    # 3. 实体寄存器
    entity_register: EntityRegister = Field(default_factory=EntityRegister)

    # 4. 累积的提取结果
    extracted_ttps: List[TTP] = Field(default_factory=list)
    extracted_files: List[FileIOC] = Field(default_factory=list)
    extracted_ips: List[IpIOC] = Field(default_factory=list)
    extracted_domains: List[DomainIOC] = Field(default_factory=list)
    extracted_urls: List[UrlIOC] = Field(default_factory=list)
    extracted_cves: List[CveIOC] = Field(default_factory=list)

    # 5. 文本摘要
    summary: str = Field(default="", description="威胁情报摘要")

    # 6. 最终输出
    final_output: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ==========================================
# 4. 工具函数
# ==========================================
# 基础 IPv4
IPV4_PATTERN = re.compile(r'\b(?:\d{1,3}(?:\[\.\]|\(\.\)|\.|\[dot\])){3}\d{1,3}\b', re.IGNORECASE)

# IPv6
IPV6_PATTERN = re.compile(
    r'\b([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}\b|\b([0-9a-fA-F]{1,4}:){1,7}:|\b([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|\b([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|\b([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|\b([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|\b([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|\b[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|\b:((:[0-9a-fA-F]{1,4}){1,7}|:)\b')

# 域名
DOMAIN_PATTERN = re.compile(
    r'\b((?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\[\.\]|\.|\[dot\]))+[a-zA-Z]{2,63})\b',
    re.IGNORECASE
)

# URL
URL_PATTERN = re.compile(r'(?:hxxp|h\*\*p|http|https|ftp|ws|wss):\/\/[^\s"\']+', re.IGNORECASE)

# Hashes
MD5_PATTERN = re.compile(r'\b[a-fA-F0-9]{32}\b')
SHA1_PATTERN = re.compile(r'\b[a-fA-F0-9]{40}\b')
SHA256_PATTERN = re.compile(r'\b[a-fA-F0-9]{64}\b')

# CVE
CVE_PATTERN = re.compile(r'\bCVE-\d{4}-\d{4,}\b', re.IGNORECASE)


def regex_extract_iocs(text: str) -> Dict[str, List[str]]:
    """
    全量正则提取与清洗 IOC
    返回字典结构，方便后续分类处理
    """
    IGNORE_DOMAINS = {
        "cdn-mineru.openxlab.org.cn",  # MinerU 的图片托管域
        "w3.org",  # 常见的 XML/HTML 命名空间干扰
        "schemas.microsoft.com",  # Office 文档常见的干扰
        "linkedin.com",  # 社交媒体（通常是参考资料，可视情况保留或过滤）
        "twitter.com",
        "cyble.com",  # 报告发布者自己的域名（防止误报为 C2）
        "virustotal.com",  # 常见的分析平台
        "mitre.org"  # 框架引用
    }

    # --- 1. 深度去防 (Defanging) ---
    # 将常见的 CTI 混淆还原，以便正则和后续处理能识别
    def defang(s: str) -> str:
        s = s.replace("[.]", ".").replace("(.)", ".").replace("[dot]", ".")
        s = s.replace("hxxp", "http").replace("h**p", "http")
        s = s.replace("[at]", "@").replace("[@]", "@")
        return s

    # 这里只用于提取 IOC
    clean_text = defang(text)

    iocs = {
        "ips": set(),
        "domains": set(),
        "urls": set(),
        "emails": set(),
        "hashes": set(),
        "cves": set()
    }

    # --- 2. 提取与验证 IPv4 ---
    raw_ips_clean = IPV4_PATTERN.findall(clean_text)

    for ip in raw_ips_clean:
        try:
            # 核心优化：使用 ipaddress 库校验
            # 这会自动过滤版本号 (如 2023.12.01) 和无效 IP (如 999.999.999.999)
            ip_obj = ipaddress.ip_address(ip)
            # 排除 0.0.0.0, 广播地址等，保留内网IP(因为攻击者常在内网横向移动)
            if not ip_obj.is_unspecified:
                iocs["ips"].add(str(ip_obj))
        except ValueError:
            # 忽略无效 IP：版本号 (1.2.3), 日期 (2023.12.01), 超范围 IP (999.999.999.999) 等
            continue

    # --- 3. 提取 IPv6 ---
    ipv6_candidates = IPV6_PATTERN.findall(clean_text)
    for ipv6 in ipv6_candidates:
        try:
            # 使用 ipaddress 库验证 IPv6 格式
            ipv6_obj = ipaddress.ip_address(ipv6)
            if not ipv6_obj.is_unspecified:
                iocs["ips"].add(str(ipv6_obj))
        except ValueError:
            # 忽略无效的 IPv6 格式
            continue

    # --- 4. 提取 URL ---
    raw_urls = URL_PATTERN.findall(clean_text)
    iocs["urls"].update(raw_urls)

    # --- 5. 提取 Domain  ---
    raw_domains = DOMAIN_PATTERN.findall(clean_text)
    for domain in raw_domains:
        domain_lower = domain.lower()

        # 1. 白名单过滤
        if any(ignore in domain_lower for ignore in IGNORE_DOMAINS):
            continue

        # 2. 排除已被 URL 包含的域名 (避免重复)
        if any(domain in url for url in iocs["urls"]):
            continue

        # 3. 排除常用误报词
        if domain_lower in ["inc.", "ltd.", "corp.", "fig.", "ver.", "vol."]:
            continue

        try:
            ipaddress.ip_address(domain)  # 排除 IP
        except ValueError:
            iocs["domains"].add(domain)

    # --- 7. 提取 Hashes ---
    iocs["hashes"].update(MD5_PATTERN.findall(clean_text))
    iocs["hashes"].update(SHA1_PATTERN.findall(clean_text))
    iocs["hashes"].update(SHA256_PATTERN.findall(clean_text))

    # --- 8. 提取 CVE ---
    iocs["cves"].update([c.upper() for c in CVE_PATTERN.findall(clean_text)])

    # 转为列表返回
    return {k: list(v) for k, v in iocs.items()}


class TokenCostHandler(BaseCallbackHandler):
    """
    自定义 Token 计费回调处理器
    支持多种 LLM 提供商的计费统计

    使用方法:
    1. 自动模式：通过模型名称自动匹配价格
    2. 手动模式：为模型实例设置价格

    示例:
        handler = TokenCostHandler()
        # 添加自定义模型价格
        handler.add_model_price("my-custom-model", input_price=0.5, output_price=1.0)
        # 为 LLM 实例设置价格（当模型名称无法自动识别时）
        handler.set_price_for_llm(my_llm_instance, input_price=0.3, output_price=0.6)
    """

    def __init__(self):
        self.lock = Lock()
        self.usage_stats = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
            "model_breakdown": {}
        }

        # LLM 实例到价格的映射（用于模型名称无法识别的情况）
        self.llm_price_map = {}

        # === 定价配置 (单位: USD per 1M tokens) ===
        # 数据来源: 各模型官方定价（2025年1月）
        self.pricing = {
            # === DeepSeek ===
            "deepseek-chat": {"input": 0.29, "output": 0.42, "note": "DeepSeek V3.2"},
            "deepseek-reasoner": {"input": 0.29, "output": 0.42, "note": "DeepSeek V3.2"},

            # === zhipu ===
            "glm-4.7":{"input": 0.43, "output": 2.00, "note": "glm-4.7"},


            # === 阿里云 Qwen ===
            "qwen-max": {"input": 0.46, "output": 1.83, "note": "Qwen Max"},
            "qwen-plus": {"input": 0.12, "output": 0.29, "note": "Qwen Plus"},
            "qwen-turbo": {"input": 0.04, "output": 0.08, "note": "Qwen Turbo"},

            # === Google Gemini ===
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30, "note": "Gemini 1.5 Flash"},
            "gemini-1.5-pro": {"input": 3.50, "output": 10.50, "note": "Gemini 1.5 Pro"},
            "gemini-2.0-flash": {"input": 0.10, "output": 0.40, "note": "Gemini 2.0 Flash"},
            "gemini-2.5-flash": {"input": 0.075, "output": 0.30, "note": "Gemini 2.5 Flash (估算)"},
            "gemini-2.5-pro": {"input": 1.25, "output": 5.00, "note": "Gemini 2.5 Pro (估算)"},

            # === OpenAI ===
            "gpt-4o": {"input": 2.50, "output": 10.00, "note": "GPT-4o"},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60, "note": "GPT-4o Mini"},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00, "note": "GPT-4 Turbo"},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50, "note": "GPT-3.5 Turbo"},
            "o1": {"input": 15.00, "output": 60.00, "note": "OpenAI o1"},
            "o1-mini": {"input": 1.10, "output": 4.40, "note": "OpenAI o1-mini"},

            # === Anthropic Claude ===
            "claude-3-5-sonnet": {"input": 3.00, "output": 15.00, "note": "Claude 3.5 Sonnet"},
            "claude-3-5-haiku": {"input": 0.80, "output": 4.00, "note": "Claude 3.5 Haiku"},
            "claude-3-opus": {"input": 15.00, "output": 75.00, "note": "Claude 3 Opus"},

            # === 其他流行模型 ===
            "llama-3-70b": {"input": 0.70, "output": 0.70, "note": "Llama 3 70B (估算)"},
            "mistral-large": {"input": 2.00, "output": 6.00, "note": "Mistral Large"},
            "command-r": {"input": 0.50, "output": 1.50, "note": "Command R"},

            # === 默认 fallback ===
            "default": {"input": 1.0, "output": 1.0, "note": "Default pricing"}
        }

    def add_model_price(self, model_name: str, input_price: float, output_price: float, note: str = ""):
        """
        添加或更新模型价格

        Args:
            model_name: 模型名称（支持部分匹配）
            input_price: 输入价格（美元/百万 tokens）
            output_price: 输出价格（美元/百万 tokens）
            note: 备注说明
        """
        self.pricing[model_name] = {
            "input": input_price,
            "output": output_price,
            "note": note
        }

    def set_price_for_llm(self, llm_instance, input_price: float, output_price: float):
        """
        为特定 LLM 实例设置价格
        当模型名称无法自动识别时使用

        Args:
            llm_instance: LLM 对象实例
            input_price: 输入价格（美元/百万 tokens）
            output_price: 输出价格（美元/百万 tokens）
        """
        self.llm_price_map[id(llm_instance)] = {
            "input": input_price,
            "output": output_price
        }

    def _match_price(self, model_name: str, llm_instance=None) -> dict:
        """
        匹配模型价格配置

        优先级:
        1. LLM 实例映射（手动设置）
        2. 精确匹配模型名称
        3. 模糊匹配（包含关系）
        4. 默认价格
        """
        # 1. 检查 LLM 实例映射
        if llm_instance and id(llm_instance) in self.llm_price_map:
            return self.llm_price_map[id(llm_instance)]

        # 2. 精确匹配
        if model_name in self.pricing:
            return self.pricing[model_name]

        # 3. 模糊匹配（如 "gpt-4o-2024-08-06" -> "gpt-4o"）
        for key, config in self.pricing.items():
            if key != "default" and key in model_name:
                return config

        # 4. 默认价格
        return self.pricing["default"]

    def on_llm_end(self, response, **kwargs):
        with self.lock:
            if not response.llm_output:
                return

            # 获取 Token 使用量
            usage = response.llm_output.get("token_usage", {})
            if not usage:
                return

            p_tokens = usage.get("prompt_tokens", 0)
            c_tokens = usage.get("completion_tokens", 0)
            t_tokens = usage.get("total_tokens", 0)

            # 获取模型名称
            model_name = response.llm_output.get("model_name", "unknown")

            # 尝试获取 LLM 实例
            llm_instance = kwargs.get("llm")

            # 匹配定价
            price_config = self._match_price(model_name, llm_instance)

            # 计算本次成本
            cost = (p_tokens / 1_000_000 * price_config["input"]) + \
                   (c_tokens / 1_000_000 * price_config["output"])

            # 更新总计
            self.usage_stats["total_tokens"] += t_tokens
            self.usage_stats["prompt_tokens"] += p_tokens
            self.usage_stats["completion_tokens"] += c_tokens
            self.usage_stats["total_cost"] += cost

            # 更新分模型统计
            if model_name not in self.usage_stats["model_breakdown"]:
                self.usage_stats["model_breakdown"][model_name] = {
                    "calls": 0, "tokens": 0, "cost": 0.0
                }
            self.usage_stats["model_breakdown"][model_name]["calls"] += 1
            self.usage_stats["model_breakdown"][model_name]["tokens"] += t_tokens
            self.usage_stats["model_breakdown"][model_name]["cost"] += cost

    def print_report(self):
        """打印计费报告"""
        print("\n" + "=" * 50)
        print("LLM COST & USAGE REPORT")
        print("=" * 50)
        print(f"{'Model':<25} | {'Calls':<6} | {'Tokens':<10} | {'Cost ($)':<10}")
        print("-" * 64)

        for model, stats in self.usage_stats["model_breakdown"].items():
            # 获取价格备注
            price_info = self._match_price(model)
            note = price_info.get("note", "")
            display_name = f"{model} ({note})" if note and len(note) < 15 else model
            print(f"{display_name:<25} | {stats['calls']:<6} | {stats['tokens']:<10} | ${stats['cost']:.5f}")

        print("-" * 64)
        total_cost = self.usage_stats["total_cost"]
        print(
            f"{'TOTAL':<25} | {'-':<6} | {self.usage_stats['total_tokens']:<10} | ${total_cost:.5f}")
        print("=" * 50 + "\n")

    def get_total_cost(self) -> float:
        """获取总成本（美元）"""
        return self.usage_stats["total_cost"]

    def get_total_tokens(self) -> int:
        """获取总 token 数"""
        return self.usage_stats["total_tokens"]

    def get_report(self) -> Dict[str, Any]:
        """返回成本报告的字典格式（用于保存到 JSON 文件）"""
        model_breakdown = []
        for model, stats in self.usage_stats["model_breakdown"].items():
            # 获取价格备注
            price_info = self._match_price(model)
            note = price_info.get("note", "")
            model_breakdown.append({
                "model": model,
                "note": note,
                "calls": stats["calls"],
                "tokens": stats["tokens"],
                "cost": round(stats["cost"], 5)
            })

        return {
            "total_tokens": self.usage_stats["total_tokens"],
            "prompt_tokens": self.usage_stats["prompt_tokens"],
            "completion_tokens": self.usage_stats["completion_tokens"],
            "total_cost": round(self.usage_stats["total_cost"], 5),
            "model_breakdown": model_breakdown
        }


def split_text_into_chunks(text: str) -> List[Dict[str, Any]]:
    """将文本分块，保留重叠部分"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)

    return [
        {
            "id": i,
            "text": chunk,
            "start_pos": i * (CHUNK_SIZE - CHUNK_OVERLAP),
            "end_pos": i * (CHUNK_SIZE - CHUNK_OVERLAP) + len(chunk)
        }
        for i, chunk in enumerate(chunks)
    ]


# ==========================================
# 5. 核心创新节点实现
# ==========================================

# ============================================
# 文本分块预处理
# ============================================
def text_chunking_node(state: AgentState):
    """将长文档分块"""
    print("--- [Node] Text Chunking ---")

    text_length = len(state.raw_text)
    print(f"  Total text length: {text_length} characters")

    if text_length <= CHUNK_SIZE:
        # 文本足够短，不需要分块
        chunks = [{
            "id": 0,
            "text": state.raw_text,
            "start_pos": 0,
            "end_pos": text_length
        }]
        print(f"  No chunking needed (single chunk)")
    else:
        chunks = split_text_into_chunks(state.raw_text)
        print(f"  Split into {len(chunks)} chunks")

    return {"text_chunks": chunks}


# ============================================
# 创新点2：增量式跨窗口TTP提取
# ============================================
def incremental_ttp_extractor(state: AgentState):
    """增量式TTP提取，带实体寄存器"""
    print(
        f"--- [Innovation 2] Incremental TTP Extraction (Chunk {state.current_chunk_idx + 1}/{len(state.text_chunks)}) ---")

    if state.current_chunk_idx >= len(state.text_chunks):
        print("  All chunks processed.")
        return {}

    current_chunk = state.text_chunks[state.current_chunk_idx]
    chunk_text = current_chunk["text"]

    # 获取实体寄存器的上下文
    register_context = state.entity_register.get_register_summary()
    rolling_summary = state.entity_register.rolling_summary

    # 使用全局 prompt 模板（性能优化）
    structured_llm = llm.with_structured_output(ExtractionResult)
    chain = _TTP_EXTRACTION_PROMPT | structured_llm

    # 自洽性推理 (并行化优化)
    # 构造多次调用的输入列表
    batch_inputs = [{
        "chunk_id": current_chunk["id"],  # 0-based，与存储一致
        "total_chunks": len(state.text_chunks),
        "chunk_text": chunk_text,
        "register_context": register_context,
        "rolling_summary": rolling_summary[:500] if rolling_summary else "None"
    } for _ in range(SELF_CONSISTENCY_ROUNDS)]

    try:
        # 使用 batch 并行执行
        batch_results = chain.batch(
            batch_inputs,
            config={"max_concurrency": 5}  # 限制同时只有 5 个请求在飞，防止 API 崩溃
        )
        all_rounds_results = [res.ttps for res in batch_results]
    except Exception as e:
        print(f"Error in batch execution: {e}")
        # 降级为串行或返回空
        all_rounds_results = [[] for _ in range(SELF_CONSISTENCY_ROUNDS)]

    # 聚合结果
    ttp_votes = {}
    for round_ttps in all_rounds_results:
        for ttp in round_ttps:
            key = ttp.technique_id
            if key not in ttp_votes:
                ttp_votes[key] = []
            ttp_votes[key].append(ttp)

    # 过滤并标记chunk_id
    chunk_ttps = []
    for technique_id, ttp_list in ttp_votes.items():
        vote_count = len(ttp_list)
        confidence = vote_count / SELF_CONSISTENCY_ROUNDS

        if confidence >= CONFIDENCE_THRESHOLD:
            best_ttp = max(ttp_list, key=lambda t: len(t.description))
            best_ttp.confidence_score = confidence
            best_ttp.chunk_id = current_chunk["id"]
            chunk_ttps.append(best_ttp)

            # 注册到实体寄存器
            state.entity_register.register_entity(
                "ttp",
                technique_id,
                best_ttp.description
            )
            print(f" Chunk {current_chunk['id']}: {technique_id} (conf={confidence:.2f})")

    # 累积结果
    updated_ttps = state.extracted_ttps + chunk_ttps

    # 更新滚动摘要
    if chunk_ttps:
        chunk_summary = f"Chunk {current_chunk['id']}: Found {len(chunk_ttps)} TTPs - " + \
                        ", ".join([t.technique_id for t in chunk_ttps])
        state.entity_register.update_rolling_summary(chunk_summary)

    return {
        "extracted_ttps": updated_ttps,
        "entity_register": state.entity_register
    }


# ============================================
# 增量式跨窗口IOC提取
# ============================================
def incremental_ioc_extractor(state: AgentState):
    """增量式IOC提取，带实体寄存器 """
    print(
        f"--- [Innovation 2] Incremental IOC Extraction (Chunk {state.current_chunk_idx + 1}/{len(state.text_chunks)}) ---")

    # 边界检查
    if state.current_chunk_idx >= len(state.text_chunks):
        return {}

    current_chunk = state.text_chunks[state.current_chunk_idx]
    chunk_text = current_chunk["text"]

    # 1. 正则预提取 (使用优化后的 regex_extract_iocs 函数)
    # 返回格式: {'ips': [...], 'domains': [...], ...}
    regex_results = regex_extract_iocs(chunk_text)

    # 2. 构造正则候选字符串 (作为 LLM 的参考提示)
    candidates_str = (
        f"Potential IPs: {regex_results.get('ips', [])}\n"
        f"Potential Domains: {regex_results.get('domains', [])}\n"
        f"Potential URLs: {regex_results.get('urls', [])}\n"
        f"Potential Files/Hashes: {regex_results.get('hashes', [])}\n"
        f"Potential CVEs: {regex_results.get('cves', [])}"
    )

    # 3. 获取历史记忆
    register_context = state.entity_register.get_register_summary()
    rolling_summary = state.entity_register.rolling_summary

    # 4. 准备 Chain（使用全局 prompt 模板）
    structured_llm = llm.with_structured_output(ExtractionResult)
    chain = _IOC_EXTRACTION_PROMPT | structured_llm

    # 5. 自洽性推理循环 (并行化优化)
    batch_inputs = [{
        "chunk_id": current_chunk["id"],  # 0-based，与存储一致
        "total_chunks": len(state.text_chunks),
        "chunk_text": chunk_text,
        "register_context": register_context,  # 历史记忆
        "rolling_summary": rolling_summary[:500] if rolling_summary else "None",
        "regex_candidates": candidates_str  # 当前正则提示
    } for _ in range(SELF_CONSISTENCY_ROUNDS)]

    try:
        all_rounds_results = chain.batch(
            batch_inputs,
            config={"max_concurrency": 5}  # 限制同时只有 5 个请求在飞，防止 API 崩溃
        )
    except Exception as e:
        print(f"    Error in batch execution: {e}")
        all_rounds_results = [ExtractionResult() for _ in range(SELF_CONSISTENCY_ROUNDS)]

    # 6. 定义内部聚合函数
    def aggregate_chunk_iocs(ioc_type: str, all_results: List[ExtractionResult]):
        """
        通用IOC聚合函数
        ioc_type: "ips", "domains", "urls", "files", "cves"
        """
        votes = {}
        # 映射 ExtractionResult 字段名到 EntityRegister 类型名
        entity_type_map = {
            "ips": "ip",
            "domains": "domain",
            "urls": "url",
            "files": "file",
            "cves": "cve"
        }

        # 收集投票
        for result in all_results:
            # getattr 获取 result.ips, result.domains 等列表
            iocs = getattr(result, ioc_type, [])
            for ioc in iocs:
                # 获取唯一标识符 (Key)
                if ioc_type == "cves":
                    key = ioc.cve_id
                elif ioc_type == "files":
                    # 文件优先使用哈希，其次文件名
                    key = ioc.sha256 or ioc.md5 or ioc.sha1 or ioc.filename or "unknown"
                else:
                    key = ioc.value

                if key not in votes:
                    votes[key] = []
                votes[key].append(ioc)

        chunk_iocs = []
        # 统计投票并过滤
        for key, ioc_list in votes.items():
            confidence = len(ioc_list) / SELF_CONSISTENCY_ROUNDS

            if confidence >= CONFIDENCE_THRESHOLD:
                # 选出信息最全的一个对象 (比如有 context 的优先)
                best_ioc = max(ioc_list, key=lambda x: len(getattr(x, 'context', '') or '') + len(
                    getattr(x, 'description', '') or ''))

                best_ioc.confidence_score = confidence
                best_ioc.chunk_id = current_chunk["id"]
                chunk_iocs.append(best_ioc)

                # 【关键步骤】注册实体到寄存器，供下一个 Chunk 使用
                entity_type_key = entity_type_map.get(ioc_type)
                if entity_type_key:
                    context_str = getattr(best_ioc, 'context', None) or getattr(best_ioc, 'description', '')
                    # 调用 EntityRegister 的注册方法
                    state.entity_register.register_entity(
                        entity_type_key,
                        key,
                        context_str
                    )

        return chunk_iocs

    # 7. 执行聚合
    chunk_ips = aggregate_chunk_iocs("ips", all_rounds_results)
    chunk_domains = aggregate_chunk_iocs("domains", all_rounds_results)
    chunk_urls = aggregate_chunk_iocs("urls", all_rounds_results)
    chunk_files = aggregate_chunk_iocs("files", all_rounds_results)
    chunk_cves = aggregate_chunk_iocs("cves", all_rounds_results)

    # 打印日志
    print(f"  Chunk {current_chunk['id']}: Found {len(chunk_ips)} IPs, {len(chunk_domains)} Domains, "
          f"{len(chunk_urls)} URLs, {len(chunk_files)} Files, {len(chunk_cves)} CVEs")

    # 8. 更新滚动摘要 (简短统计，用于传递给下一个 Prompt)
    chunk_summary_text = (f"Chunk {current_chunk['id']} findings: "
                          f"{len(chunk_ips)} IPs, {len(chunk_domains)} domains, "
                          f"{len(chunk_files)} files.")
    state.entity_register.update_rolling_summary(chunk_summary_text)

    # 9. 返回增量更新 (追加到现有列表)
    return {
        "extracted_ips": state.extracted_ips + chunk_ips,
        "extracted_domains": state.extracted_domains + chunk_domains,
        "extracted_urls": state.extracted_urls + chunk_urls,
        "extracted_files": state.extracted_files + chunk_files,
        "extracted_cves": state.extracted_cves + chunk_cves,
        "entity_register": state.entity_register  # 必须返回更新后的寄存器
    }


# ============================================
# 块迭代控制节点
# ============================================
def chunk_iterator_node(state: AgentState):
    """控制分块迭代"""
    next_idx = state.current_chunk_idx + 1

    if next_idx < len(state.text_chunks):
        print(f"\n--- Moving to Chunk {next_idx + 1}/{len(state.text_chunks)} ---\n")
        return {"current_chunk_idx": next_idx}
    else:
        print("\n--- All chunks processed, moving to final stages ---\n")
        return {"current_chunk_idx": next_idx}  # 超出范围，后续节点会跳过


def should_continue_chunking(state: AgentState) -> str:
    """判断是否继续处理下一个chunk"""
    if state.current_chunk_idx < len(state.text_chunks):
        return "continue"
    else:
        return "finish"


# ============================================
# 创新点3：对抗性去偏（魔鬼代言人）
# ============================================
def adversarial_critic_node(state: AgentState):
    """
    全量对抗性校验节点 (DeepSeek-R1)
    覆盖范围：TTPs, IPs, Domains, URLs, Files, CVEs
    """
    print("--- [Innovation 3] Adversarial Debiasing - All Entities ---")

    # 1. 初始化 Chain（使用全局 prompt 模板）
    chain = _ADVERSARIAL_CRITIC_PROMPT | llm_critic

    # 2. 定义通用的审核辅助函数 (并行化优化)
    def audit_list(items: list, item_type_label: str, value_extractor) -> list:
        """
        items: 待审核的对象列表
        item_type_label: 传给 Prompt 的类型 (如 'IP Address', 'File')
        value_extractor: 一个函数，用于从对象中提取可读的 value 字符串
        """
        if not items:
            return []

        print(f"  > Auditing {len(items)} {item_type_label}s...")

        # 准备 batch 输入
        batch_inputs = []
        for item in items:
            item_value = value_extractor(item)
            item_context = getattr(item, 'context', None) or getattr(item, 'description', 'No context provided')
            batch_inputs.append({
                "item_type": item_type_label,
                "value": item_value,
                "description": item_context,
                "confidence": getattr(item, 'confidence_score', 1.0),
                "chunk_id": getattr(item, 'chunk_id', 0)
            })

        # 执行 batch 调用
        # 注意：如果 items 数量非常大，可能需要分批次 batch (例如每批 10-20 个)
        BATCH_SIZE = ADVERSARIAL_CRITIC_BATCH_SIZE
        filtered_items = []

        for i in range(0, len(batch_inputs), BATCH_SIZE):
            current_batch = batch_inputs[i:i + BATCH_SIZE]
            current_items = items[i:i + BATCH_SIZE]

            try:
                responses = chain.batch(
                    batch_inputs,
                    config={"max_concurrency": 5}  # 限制同时只有 5 个请求在飞，防止 API 崩溃
                )

                for item, response in zip(current_items, responses):
                    content = response.content
                    item_value = value_extractor(item)

                    # 鲁棒的 JSON 解析 (兼容 R1 的 <think> 标签)
                    json_match = re.search(r"\{.*\}", content, re.DOTALL)
                    is_accepted = False
                    reason = "Unknown"

                    if json_match:
                        try:
                            result_json = json.loads(json_match.group(0))
                            verdict = result_json.get("verdict", "REJECT").upper()
                            reason = result_json.get("reason", "No reason provided")
                            if "ACCEPT" in verdict:
                                is_accepted = True
                        except json.JSONDecodeError:
                            if "ACCEPT" in content.upper():
                                is_accepted = True
                                reason = "JSON Parse Error (Fallback Accept)"
                    else:
                        if "ACCEPT" in content.upper():
                            is_accepted = True
                            reason = "No JSON Found (Fallback Accept)"

                    if is_accepted:
                        filtered_items.append(item)
                    else:
                        print(f"REJECTED: {item_value} | Reason: {reason}")

            except Exception as e:
                print(f"    ! Error auditing batch {i}: {e}")
                # 出错时保留该批次所有项目，避免误删
                filtered_items.extend(current_items)

        return filtered_items

    # 3. 执行 TTP 审核
    # TTP 的 value 是 ID + Name
    state.extracted_ttps = audit_list(
        state.extracted_ttps,
        "TTP (Tactic/Technique)",
        lambda x: f"{x.technique_id} ({x.technique_name})"
    )

    # 4. 执行 IOC 审核
    # IP
    state.extracted_ips = audit_list(
        state.extracted_ips,
        "IP Address",
        lambda x: x.value
    )

    # Domain
    state.extracted_domains = audit_list(
        state.extracted_domains,
        "Domain Name",
        lambda x: x.value
    )

    # URL
    state.extracted_urls = audit_list(
        state.extracted_urls,
        "URL",
        lambda x: x.value
    )

    # File (文件比较特殊，可能是 Hash 或 Filename)
    def get_file_label(f):
        if f.sha256: return f"SHA256: {f.sha256}"
        if f.md5: return f"MD5: {f.md5}"
        if f.filename: return f"File: {f.filename}"
        return "Unknown File"

    state.extracted_files = audit_list(
        state.extracted_files,
        "File Artifact",
        get_file_label
    )

    # CVE
    state.extracted_cves = audit_list(
        state.extracted_cves,
        "CVE Vulnerability",
        lambda x: x.cve_id
    )

    # 5. 返回更新后的状态
    return {
        "extracted_ttps": state.extracted_ttps,
        "extracted_ips": state.extracted_ips,
        "extracted_domains": state.extracted_domains,
        "extracted_urls": state.extracted_urls,
        "extracted_files": state.extracted_files,
        "extracted_cves": state.extracted_cves
    }


# ============================================
# 数据聚合
# ============================================
def aggregator_node(state: AgentState):
    print("--- [Node] Aggregating Results ---")

    # 生成最终摘要（使用全局 prompt 模板）
    try:
        summary_chain = _SUMMARY_PROMPT | llm
        summary_response = summary_chain.invoke({
            "rolling_summary": state.entity_register.rolling_summary,
            "ttps": "\n".join([f"{t.technique_id}: {t.technique_name}" for t in state.extracted_ttps[:10]])
        })
        final_summary = summary_response.content
    except:
        final_summary = state.entity_register.rolling_summary[:500]

    final_data = {
        "summary": final_summary,
        "document_stats": {
            "total_length": len(state.raw_text),
            "num_chunks": len(state.text_chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        },
        "innovation_metrics": {
            "self_consistency_rounds": SELF_CONSISTENCY_ROUNDS,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "cross_chunk_references": sum(1 for ip in state.extracted_ips if ip.cross_chunk_reference) +
                                      sum(1 for d in state.extracted_domains if d.cross_chunk_reference) +
                                      sum(1 for u in state.extracted_urls if u.cross_chunk_reference),
            "entity_register_size": (
                    len(state.entity_register.known_ips) +
                    len(state.entity_register.known_domains) +
                    len(state.entity_register.known_urls) +
                    len(state.entity_register.known_files) +
                    len(state.entity_register.known_cves) +
                    len(state.entity_register.known_ttps)
            )
        },
        "total_indicators": (
                len(state.extracted_files) +
                len(state.extracted_ips) +
                len(state.extracted_domains) +
                len(state.extracted_urls)
        ),
        "ttps": [t.model_dump() for t in state.extracted_ttps],
        "iocs": {
            "files": [i.model_dump() for i in state.extracted_files],
            "ips": [i.model_dump() for i in state.extracted_ips],
            "domains": [i.model_dump() for i in state.extracted_domains],
            "urls": [i.model_dump() for i in state.extracted_urls],
            "cves": [i.model_dump() for i in state.extracted_cves]
        },
        "entity_register_summary": {
            "known_ips": list(state.entity_register.known_ips.keys()),
            "known_domains": list(state.entity_register.known_domains.keys()),
            "known_urls": list(state.entity_register.known_urls.keys()),
            "known_files": list(state.entity_register.known_files.keys()),
            "known_cves": list(state.entity_register.known_cves.keys()),
            "known_ttps": list(state.entity_register.known_ttps.keys())
        }
    }

    return {"final_output": final_data, "summary": final_summary}


# ==========================================
# 6. 构建图 (Graph Construction with Loops)
# ==========================================
def build_cti_graph():
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("chunk_text", text_chunking_node)
    workflow.add_node("extract_ttp_incremental", incremental_ttp_extractor)
    workflow.add_node("extract_ioc_incremental", incremental_ioc_extractor)
    workflow.add_node("next_chunk", chunk_iterator_node)
    workflow.add_node("adversarial_critic", adversarial_critic_node)
    workflow.add_node("aggregate", aggregator_node)

    # 定义工作流（带循环）
    workflow.set_entry_point("chunk_text")
    workflow.add_edge("chunk_text", "extract_ttp_incremental")
    workflow.add_edge("extract_ttp_incremental", "extract_ioc_incremental")
    workflow.add_edge("extract_ioc_incremental", "next_chunk")

    # 条件边：判断是否继续处理下一个chunk
    workflow.add_conditional_edges(
        "next_chunk",
        should_continue_chunking,
        {
            "continue": "extract_ttp_incremental",  # 循环回去处理下一个chunk
            "finish": "adversarial_critic"  # 全部处理完，进入批评阶段
        }
    )

    workflow.add_edge("adversarial_critic", "aggregate")
    workflow.add_edge("aggregate", END)

    return workflow.compile()


# ==========================================
# 7. 执行入口
# ==========================================
if __name__ == "__main__":

    cost_handler = TokenCostHandler()
    llm.callbacks = [cost_handler]
    llm_critic.callbacks = [cost_handler]

    # input_file_path = "./cti/warp-panda-cloud-threats.md"
    # input_file_path = "./cti/ESET_MQsTTang-MustangPandas-backdoor-Qt-MQTT(03-02-2023).md"
    # input_file_path = "./cti/ESET_WinorDLL64-Lazarus-arsenal(02-23-2023).md"
    # input_file_path = "./cti/MoonWalk_A_deep_dive_into_the_updated_arsenal_of_APT41.md"
    # input_file_path = "./cti/Hide Your RDP-Password Spray Leads to RansomHub Deployment.md"
    # input_file_path = "./cti/Cyble_OperationShadowCat-Targeting-Indian-Political-Observers(07-24-2024).md"
    input_file_path = "./cti/DodgeBox_A_deep_dive_into_the_updated_arsenal_of_APT41.md"
    # input_file_path = "E:/PythonProject/langchain/src/cti_extrac/data/apt41-arisen-from-dust.txt"
    # input_file_path = "E:/PythonProject/langchain/src/cti_extrac/data/arid-viper-poisons-android-apps-with-aridspy.txt"
    with open(input_file_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    print("\nBuilding Agent Graph...")
    app = build_cti_graph()

    print("Running Incremental Extraction...\n")
    inputs = AgentState(raw_text=markdown_content)
    config = {"recursion_limit": 1000}

    result = app.invoke(inputs, config=config)

    cost_handler.print_report()

    # 将成本报告添加到最终输出
    result['final_output']['cost_report'] = cost_handler.get_report()


    # 保存结果
    # 1. 获取文件名 (不带路径) -> "report.md"
    filename = os.path.basename(input_file_path)
    # 2. 去除扩展名 -> "report"
    base_name = os.path.splitext(filename)[0]
    # 3. 拼接输出路径 -> "output/report.json"
    output_dir = "output/deepseek"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 如果 output 文件夹不存在则自动创建

    output_file_path = os.path.join(output_dir, f"{base_name}.json")

    # 保存结果
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(result['final_output'], f, indent=2, ensure_ascii=False)

    print(f"\n[Success] Results saved to: {output_file_path}")