import ipaddress
import json
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cti_extrac.prompts import (
    ADVERSARIAL_CRITIC_SYSTEM,
    IOC_EXTRACTION_SYSTEM,
    IOC_EXTRACTION_USER,
    SUMMARY_SYSTEM,
    SUMMARY_USER,
    TTP_EXTRACTION_SYSTEM,
    TTP_EXTRACTION_USER,
)
from my_llm import gpt5_5


llm = gpt5_5
llm_critic = gpt5_5

SELF_CONSISTENCY_ROUNDS = 1
CONFIDENCE_THRESHOLD = 0.66
ENABLE_ADVERSARIAL_CRITIC = True

CHUNK_SIZE = 8000
CHUNK_OVERLAP = 1000
ADVERSARIAL_CRITIC_BATCH_SIZE = 20
LLM_RETRY_ATTEMPTS = 4
LLM_RETRY_BACKOFF_SECONDS = 8

SUPPORTED_INPUT_SUFFIXES = {".md", ".txt"}
CTI_INPUT_ROOT = Path("./cti")
DEFAULT_OUTPUT_ROOT = Path("output/gpt-5.5")
TARGET_FOLDERS = ["Sapphire Mushroom"]
TARGET_FILE = ""


_TTP_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TTP_EXTRACTION_SYSTEM),
    ("user", TTP_EXTRACTION_USER),
])

_IOC_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", IOC_EXTRACTION_SYSTEM),
    ("user", IOC_EXTRACTION_USER),
])

_ADVERSARIAL_CRITIC_BATCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ADVERSARIAL_CRITIC_SYSTEM),
    ("user", """Audit the following extraction items as a batch.

Type: {item_type}
Items JSON:
{items_json}

Return one JSON object only:
{{
  "accepted_values": ["value-1", "value-2"],
  "rejected": [{{"value": "value-3", "reason": "brief reason"}}]
}}

Only include an item in accepted_values if it is a valid malicious indicator or behavior used by the adversary."""),
])

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SUMMARY_SYSTEM),
    ("user", SUMMARY_USER),
])


class TTP(BaseModel):
    """Observed MITRE ATT&CK technique."""

    technique_id: str = Field(..., description="MITRE ATT&CK ID, e.g., T1566.001. Use 'Unknown' if absent.")
    technique_name: str = Field(..., description="Technique name, e.g., 'Phishing'.")
    description: str = Field(..., description="Evidence-backed usage details from the source text.")
    confidence_score: float = Field(default=1.0, ge=0, le=1, description="Self-consistency confidence.")
    chunk_id: Optional[int] = Field(None, description="Source chunk ID.")

    @field_validator("technique_id")
    def validate_id(cls, value):
        return value.upper().strip()


class BaseIOC(BaseModel):
    role: str = Field(
        ...,
        description="Indicator role in the attack, e.g., C2, Payload_Delivery, Victim, Benign, Malware.",
    )
    context: Optional[str] = Field(None, description="Brief evidence or reason for this IOC.")
    confidence_score: float = Field(default=1.0, ge=0, le=1, description="Self-consistency confidence.")
    chunk_id: Optional[int] = Field(None, description="Source chunk ID.")
    cross_chunk_reference: bool = Field(default=False, description="Whether this refers to prior chunks.")


class IpIOC(BaseIOC):
    value: str = Field(..., description="IP address value.")
    version: Literal["IPv4", "IPv6"] = Field("IPv4", description="IP version.")

    @field_validator("value")
    def clean_ip(cls, value):
        return value.replace("[.]", ".").strip()


class DomainIOC(BaseIOC):
    value: str = Field(..., description="Domain name or hostname.")
    record_type: Optional[str] = Field(None, description="DNS record type.")

    @field_validator("value")
    def clean_domain(cls, value):
        return value.replace("[.]", ".").lower().strip()


class UrlIOC(BaseIOC):
    value: str = Field(..., description="Full URL.")
    protocol: Literal["http", "https", "ftp", "ws", "wss", "other"] = Field("http")

    @field_validator("value")
    def clean_url(cls, value):
        return value.replace("hxxp", "http").replace("[.]", ".").strip()


class FileIOC(BaseIOC):
    filename: Optional[str] = Field(None)
    md5: Optional[str] = Field(None)
    sha1: Optional[str] = Field(None)
    sha256: Optional[str] = Field(None)
    file_path: Optional[str] = Field(None)

    @field_validator("sha256")
    def validate_hash(cls, value):
        value = value.strip() if value else None
        return value if value and len(value) == 64 else None


class CveIOC(BaseModel):
    cve_id: str = Field(...)
    description: Optional[str] = Field(None)
    confidence_score: float = Field(default=1.0, ge=0, le=1)
    chunk_id: Optional[int] = Field(None, description="Source chunk ID.")

    @field_validator("cve_id")
    def format_cve(cls, value):
        match = re.search(r"(CVE-\d{4}-\d{4,})", value.upper())
        return match.group(1) if match else value.upper()


class ExtractionResult(BaseModel):
    """Single structured extraction result."""

    summary: str = Field(default="", description="Threat intelligence summary.")
    ttps: List[TTP] = Field(default_factory=list)
    ips: List[IpIOC] = Field(default_factory=list)
    domains: List[DomainIOC] = Field(default_factory=list)
    urls: List[UrlIOC] = Field(default_factory=list)
    files: List[FileIOC] = Field(default_factory=list)
    cves: List[CveIOC] = Field(default_factory=list)


class EntityRegister(BaseModel):
    """Short memory for cross-chunk entity context."""

    known_ips: Dict[str, str] = Field(default_factory=dict)
    known_domains: Dict[str, str] = Field(default_factory=dict)
    known_urls: Dict[str, str] = Field(default_factory=dict)
    known_files: Dict[str, str] = Field(default_factory=dict)
    known_cves: Dict[str, str] = Field(default_factory=dict)
    known_ttps: Dict[str, str] = Field(default_factory=dict)
    rolling_summary: str = Field(default="")

    def _registers(self) -> Dict[str, Dict[str, str]]:
        return {
            "ip": self.known_ips,
            "domain": self.known_domains,
            "url": self.known_urls,
            "file": self.known_files,
            "cve": self.known_cves,
            "ttp": self.known_ttps,
        }

    def register_entity(self, entity_type: str, value: str, context: str):
        target = self._registers().get(entity_type)
        if target is None:
            return
        if value not in target:
            target[value] = context[:200]
            return
        if context not in target[value] and len(target[value]) < 400:
            target[value] += f" | {context}"

    def get_context_for_entity(self, entity_type: str, value: str) -> Optional[str]:
        return self._registers().get(entity_type, {}).get(value)

    def update_rolling_summary(self, new_summary: str):
        text = f"{self.rolling_summary}\n\n{new_summary}"
        if len(text) <= 2000:
            self.rolling_summary = text
            return
        candidate = text[-1500:]
        first_newline = candidate.find("\n")
        self.rolling_summary = candidate[first_newline + 1:] if first_newline != -1 else candidate

    def get_register_summary(self) -> str:
        def format_items(data: Dict[str, str], label: str, limit: int) -> Optional[str]:
            if not data:
                return None
            items = []
            for key, value in list(data.items())[-limit:]:
                snippet = value[:50] + "..." if len(value) > 50 else value
                items.append(f"{key} ({snippet})")
            return f"{label}: {', '.join(items)}"

        parts = [
            format_items(self.known_ips, "Known IPs", 8),
            format_items(self.known_domains, "Known Domains", 8),
            format_items(self.known_urls, "Known URLs", 5),
            format_items(self.known_files, "Known Files", 5),
            format_items(self.known_cves, "Known CVEs", 5),
            format_items(self.known_ttps, "Known TTPs", 5),
        ]
        return " | ".join(filter(None, parts)) if any(parts) else "No prior context."


class AgentState(BaseModel):
    """LangGraph state container."""

    raw_text: str = Field(..., description="Input Markdown text.")
    text_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Text chunks.")
    current_chunk_idx: int = Field(default=0, description="Current chunk index.")
    entity_register: EntityRegister = Field(default_factory=EntityRegister)
    extracted_ttps: List[TTP] = Field(default_factory=list)
    extracted_files: List[FileIOC] = Field(default_factory=list)
    extracted_ips: List[IpIOC] = Field(default_factory=list)
    extracted_domains: List[DomainIOC] = Field(default_factory=list)
    extracted_urls: List[UrlIOC] = Field(default_factory=list)
    extracted_cves: List[CveIOC] = Field(default_factory=list)
    summary: str = Field(default="", description="Threat intelligence summary.")
    final_output: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}(?:\[\.\]|\(\.\)|\.|\[dot\])){3}\d{1,3}\b", re.IGNORECASE)
IPV6_PATTERN = re.compile(
    r"\b([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}\b|\b([0-9a-fA-F]{1,4}:){1,7}:|"
    r"\b([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|\b([0-9a-fA-F]{1,4}:){1,5}"
    r"(:[0-9a-fA-F]{1,4}){1,2}|\b([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|"
    r"\b([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|\b([0-9a-fA-F]{1,4}:){1,2}"
    r"(:[0-9a-fA-F]{1,4}){1,5}|\b[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|"
    r"\b:((:[0-9a-fA-F]{1,4}){1,7}|:)\b"
)
DOMAIN_PATTERN = re.compile(
    r"\b((?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\[\.\]|\.|\[dot\]))+[a-zA-Z]{2,63})\b",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"(?:hxxp|h\*\*p|http|https|ftp|ws|wss):\/\/[^\s\"']+", re.IGNORECASE)
MD5_PATTERN = re.compile(r"\b[a-fA-F0-9]{32}\b")
SHA1_PATTERN = re.compile(r"\b[a-fA-F0-9]{40}\b")
SHA256_PATTERN = re.compile(r"\b[a-fA-F0-9]{64}\b")
CVE_PATTERN = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)


def regex_extract_iocs(text: str) -> Dict[str, List[str]]:
    """Pre-extract IOC candidates for later LLM verification."""

    ignore_domains = {
        "cdn-mineru.openxlab.org.cn",
        "w3.org",
        "schemas.microsoft.com",
        "linkedin.com",
        "twitter.com",
        "cyble.com",
        "virustotal.com",
        "mitre.org",
    }

    def defang(value: str) -> str:
        return (
            value.replace("[.]", ".")
            .replace("(.)", ".")
            .replace("[dot]", ".")
            .replace("hxxp", "http")
            .replace("h**p", "http")
            .replace("[at]", "@")
            .replace("[@]", "@")
        )

    clean_text = defang(text)
    iocs = {"ips": set(), "domains": set(), "urls": set(), "emails": set(), "hashes": set(), "cves": set()}

    for candidate in IPV4_PATTERN.findall(clean_text):
        try:
            ip_obj = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if not ip_obj.is_unspecified:
            iocs["ips"].add(str(ip_obj))

    for match in IPV6_PATTERN.finditer(clean_text):
        try:
            ip_obj = ipaddress.ip_address(match.group(0))
        except ValueError:
            continue
        if not ip_obj.is_unspecified:
            iocs["ips"].add(str(ip_obj))

    iocs["urls"].update(URL_PATTERN.findall(clean_text))

    for domain in DOMAIN_PATTERN.findall(clean_text):
        domain_lower = domain.lower()
        if any(item in domain_lower for item in ignore_domains):
            continue
        if any(domain in url for url in iocs["urls"]):
            continue
        if domain_lower in {"inc.", "ltd.", "corp.", "fig.", "ver.", "vol."}:
            continue
        try:
            ipaddress.ip_address(domain)
        except ValueError:
            iocs["domains"].add(domain)

    iocs["hashes"].update(MD5_PATTERN.findall(clean_text))
    iocs["hashes"].update(SHA1_PATTERN.findall(clean_text))
    iocs["hashes"].update(SHA256_PATTERN.findall(clean_text))
    iocs["cves"].update(cve.upper() for cve in CVE_PATTERN.findall(clean_text))
    return {key: list(value) for key, value in iocs.items()}


class TokenCostHandler(BaseCallbackHandler):
    """Track token usage and estimate cost from local price settings."""

    def __init__(self):
        self.lock = Lock()
        self.usage_stats = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
            "model_breakdown": {},
        }
        self.llm_price_map = {}
        self.pricing = {
            "gpt-5.2": {"input": 1.75, "output": 14.00, "note": "GPT-5.2"},
            "gpt-5.4": {"input": 2.50, "output": 15.00, "note": "GPT-5.4"},
            "gpt-5.5": {"input": 5.00, "output": 30.00, "note": "GPT-5.5"},
            "default": {"input": 5.00, "output": 30.00, "note": "GPT-5.5"},
        }

    def add_model_price(self, model_name: str, input_price: float, output_price: float, note: str = ""):
        self.pricing[model_name] = {"input": input_price, "output": output_price, "note": note}

    def set_price_for_llm(self, llm_instance, input_price: float, output_price: float):
        self.llm_price_map[id(llm_instance)] = {"input": input_price, "output": output_price}

    def _match_price(self, model_name: str, llm_instance=None) -> dict:
        if llm_instance and id(llm_instance) in self.llm_price_map:
            return self.llm_price_map[id(llm_instance)]
        if model_name in self.pricing:
            return self.pricing[model_name]
        for key, config in self.pricing.items():
            if key != "default" and key in model_name:
                return config
        return self.pricing["default"]

    def on_llm_end(self, response, **kwargs):
        if not response.llm_output:
            return
        usage = response.llm_output.get("token_usage", {})
        if usage:
            self.record_usage(
                response.llm_output.get("model_name", "unknown"),
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage.get("total_tokens", 0),
                kwargs.get("llm"),
            )

    def record_usage(self, model_name: str, prompt_tokens: int, completion_tokens: int, total_tokens: int, llm_instance=None):
        with self.lock:
            price = self._match_price(model_name, llm_instance)
            cost = prompt_tokens / 1_000_000 * price["input"] + completion_tokens / 1_000_000 * price["output"]
            self.usage_stats["total_tokens"] += total_tokens
            self.usage_stats["prompt_tokens"] += prompt_tokens
            self.usage_stats["completion_tokens"] += completion_tokens
            self.usage_stats["total_cost"] += cost

            breakdown = self.usage_stats["model_breakdown"].setdefault(model_name, {"calls": 0, "tokens": 0, "cost": 0.0})
            breakdown["calls"] += 1
            breakdown["tokens"] += total_tokens
            breakdown["cost"] += cost

    def print_report(self):
        print("\n" + "=" * 50)
        print("LLM COST & USAGE REPORT")
        print("=" * 50)
        print(f"{'Model':<25} | {'Calls':<6} | {'Tokens':<10} | {'Cost ($)':<10}")
        print("-" * 64)
        for model, stats in self.usage_stats["model_breakdown"].items():
            note = self._match_price(model).get("note", "")
            display_name = f"{model} ({note})" if note and len(note) < 15 else model
            print(f"{display_name:<25} | {stats['calls']:<6} | {stats['tokens']:<10} | ${stats['cost']:.5f}")
        print("-" * 64)
        print(f"{'TOTAL':<25} | {'-':<6} | {self.usage_stats['total_tokens']:<10} | ${self.usage_stats['total_cost']:.5f}")
        print("=" * 50 + "\n")

    def get_total_cost(self) -> float:
        return self.usage_stats["total_cost"]

    def get_total_tokens(self) -> int:
        return self.usage_stats["total_tokens"]

    def get_report(self) -> Dict[str, Any]:
        model_breakdown = []
        for model, stats in self.usage_stats["model_breakdown"].items():
            model_breakdown.append({
                "model": model,
                "note": self._match_price(model).get("note", ""),
                "calls": stats["calls"],
                "tokens": stats["tokens"],
                "cost": round(stats["cost"], 5),
            })
        return {
            "total_tokens": self.usage_stats["total_tokens"],
            "prompt_tokens": self.usage_stats["prompt_tokens"],
            "completion_tokens": self.usage_stats["completion_tokens"],
            "total_cost": round(self.usage_stats["total_cost"], 5),
            "model_breakdown": model_breakdown,
        }


def split_text_into_chunks(text: str) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [
        {
            "id": index,
            "text": chunk,
            "start_pos": index * (CHUNK_SIZE - CHUNK_OVERLAP),
            "end_pos": index * (CHUNK_SIZE - CHUNK_OVERLAP) + len(chunk),
        }
        for index, chunk in enumerate(splitter.split_text(text))
    ]


def text_chunking_node(state: AgentState):
    print("--- [Node] Text Chunking ---")
    text_length = len(state.raw_text)
    print(f"  Total text length: {text_length} characters")
    if text_length <= CHUNK_SIZE:
        chunks = [{"id": 0, "text": state.raw_text, "start_pos": 0, "end_pos": text_length}]
        print("  No chunking needed (single chunk)")
    else:
        chunks = split_text_into_chunks(state.raw_text)
        print(f"  Split into {len(chunks)} chunks")
    return {"text_chunks": chunks}


def invoke_llm_with_retries(chain, payload: dict, label: str):
    last_error = None
    for attempt in range(1, LLM_RETRY_ATTEMPTS + 1):
        try:
            return chain.invoke(payload)
        except Exception as exc:
            last_error = exc
            print(f"    ! {label} failed ({attempt}/{LLM_RETRY_ATTEMPTS}): {exc}")
            if attempt < LLM_RETRY_ATTEMPTS:
                time.sleep(LLM_RETRY_BACKOFF_SECONDS * attempt)
    raise last_error


def incremental_ttp_extractor(state: AgentState):
    print(f"--- [Node] Incremental TTP Extraction ({state.current_chunk_idx + 1}/{len(state.text_chunks)}) ---")
    if state.current_chunk_idx >= len(state.text_chunks):
        print("  All chunks processed.")
        return {}

    current_chunk = state.text_chunks[state.current_chunk_idx]
    chain = _TTP_EXTRACTION_PROMPT | llm.with_structured_output(ExtractionResult)
    payload = {
        "chunk_id": current_chunk["id"],
        "total_chunks": len(state.text_chunks),
        "chunk_text": current_chunk["text"],
        "register_context": state.entity_register.get_register_summary(),
        "rolling_summary": state.entity_register.rolling_summary[:500] or "None",
    }

    try:
        results = []
        for round_idx in range(1, SELF_CONSISTENCY_ROUNDS + 1):
            print(f"  TTP round {round_idx}/{SELF_CONSISTENCY_ROUNDS}...")
            results.append(invoke_llm_with_retries(chain, payload, f"TTP chunk {current_chunk['id']} round {round_idx}").ttps)
    except Exception as exc:
        print(f"Error in TTP extraction: {exc}")
        results = [[] for _ in range(SELF_CONSISTENCY_ROUNDS)]

    votes: Dict[str, List[TTP]] = {}
    for round_ttps in results:
        for ttp in round_ttps:
            votes.setdefault(ttp.technique_id, []).append(ttp)

    chunk_ttps = []
    for technique_id, ttp_list in votes.items():
        confidence = len(ttp_list) / SELF_CONSISTENCY_ROUNDS
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        best_ttp = max(ttp_list, key=lambda item: len(item.description))
        best_ttp.confidence_score = confidence
        best_ttp.chunk_id = current_chunk["id"]
        chunk_ttps.append(best_ttp)
        state.entity_register.register_entity("ttp", technique_id, best_ttp.description)
        print(f"  Chunk {current_chunk['id']}: {technique_id} (conf={confidence:.2f})")

    if chunk_ttps:
        ids = ", ".join(ttp.technique_id for ttp in chunk_ttps)
        state.entity_register.update_rolling_summary(f"Chunk {current_chunk['id']}: Found {len(chunk_ttps)} TTPs - {ids}")

    return {"extracted_ttps": state.extracted_ttps + chunk_ttps, "entity_register": state.entity_register}


def incremental_ioc_extractor(state: AgentState):
    print(f"--- [Node] Incremental IOC Extraction ({state.current_chunk_idx + 1}/{len(state.text_chunks)}) ---")
    if state.current_chunk_idx >= len(state.text_chunks):
        return {}

    current_chunk = state.text_chunks[state.current_chunk_idx]
    regex_results = regex_extract_iocs(current_chunk["text"])
    candidates = (
        f"Potential IPs: {regex_results.get('ips', [])}\n"
        f"Potential Domains: {regex_results.get('domains', [])}\n"
        f"Potential URLs: {regex_results.get('urls', [])}\n"
        f"Potential Files/Hashes: {regex_results.get('hashes', [])}\n"
        f"Potential CVEs: {regex_results.get('cves', [])}"
    )
    payload = {
        "chunk_id": current_chunk["id"],
        "total_chunks": len(state.text_chunks),
        "chunk_text": current_chunk["text"],
        "register_context": state.entity_register.get_register_summary(),
        "rolling_summary": state.entity_register.rolling_summary[:500] or "None",
        "regex_candidates": candidates,
    }
    chain = _IOC_EXTRACTION_PROMPT | llm.with_structured_output(ExtractionResult)

    try:
        results = []
        for round_idx in range(1, SELF_CONSISTENCY_ROUNDS + 1):
            print(f"  IOC round {round_idx}/{SELF_CONSISTENCY_ROUNDS}...")
            results.append(invoke_llm_with_retries(chain, payload, f"IOC chunk {current_chunk['id']} round {round_idx}"))
    except Exception as exc:
        print(f"    Error in IOC extraction: {exc}")
        results = [ExtractionResult() for _ in range(SELF_CONSISTENCY_ROUNDS)]

    def aggregate(ioc_type: str):
        entity_type_map = {"ips": "ip", "domains": "domain", "urls": "url", "files": "file", "cves": "cve"}
        votes = {}
        for result in results:
            for ioc in getattr(result, ioc_type, []):
                if ioc_type == "cves":
                    key = ioc.cve_id
                elif ioc_type == "files":
                    key = ioc.sha256 or ioc.md5 or ioc.sha1 or ioc.filename or "unknown"
                else:
                    key = ioc.value
                votes.setdefault(key, []).append(ioc)

        chunk_iocs = []
        for key, ioc_list in votes.items():
            confidence = len(ioc_list) / SELF_CONSISTENCY_ROUNDS
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            best_ioc = max(
                ioc_list,
                key=lambda item: len(getattr(item, "context", "") or "") + len(getattr(item, "description", "") or ""),
            )
            best_ioc.confidence_score = confidence
            best_ioc.chunk_id = current_chunk["id"]
            chunk_iocs.append(best_ioc)
            context = getattr(best_ioc, "context", None) or getattr(best_ioc, "description", "")
            state.entity_register.register_entity(entity_type_map[ioc_type], key, context)
        return chunk_iocs

    chunk_ips = aggregate("ips")
    chunk_domains = aggregate("domains")
    chunk_urls = aggregate("urls")
    chunk_files = aggregate("files")
    chunk_cves = aggregate("cves")

    print(
        f"  Chunk {current_chunk['id']}: Found {len(chunk_ips)} IPs, {len(chunk_domains)} Domains, "
        f"{len(chunk_urls)} URLs, {len(chunk_files)} Files, {len(chunk_cves)} CVEs"
    )
    state.entity_register.update_rolling_summary(
        f"Chunk {current_chunk['id']} findings: {len(chunk_ips)} IPs, {len(chunk_domains)} domains, {len(chunk_files)} files."
    )
    return {
        "extracted_ips": state.extracted_ips + chunk_ips,
        "extracted_domains": state.extracted_domains + chunk_domains,
        "extracted_urls": state.extracted_urls + chunk_urls,
        "extracted_files": state.extracted_files + chunk_files,
        "extracted_cves": state.extracted_cves + chunk_cves,
        "entity_register": state.entity_register,
    }


def chunk_iterator_node(state: AgentState):
    next_idx = state.current_chunk_idx + 1
    if next_idx < len(state.text_chunks):
        print(f"\n--- Moving to Chunk {next_idx + 1}/{len(state.text_chunks)} ---\n")
    else:
        print("\n--- All chunks processed, moving to final stages ---\n")
    return {"current_chunk_idx": next_idx}


def should_continue_chunking(state: AgentState) -> str:
    return "continue" if state.current_chunk_idx < len(state.text_chunks) else "finish"


def adversarial_critic_node(state: AgentState):
    if not ENABLE_ADVERSARIAL_CRITIC:
        print("--- [Node] Adversarial Critic skipped ---")
        return _extracted_state(state)

    print("--- [Node] Adversarial Debiasing ---")
    chain = _ADVERSARIAL_CRITIC_BATCH_PROMPT | llm_critic

    def audit_list(items: list, item_type_label: str, value_extractor) -> list:
        if not items:
            return []
        print(f"  > Auditing {len(items)} {item_type_label}s...")
        batch_inputs = [
            {
                "item_type": item_type_label,
                "value": value_extractor(item),
                "description": getattr(item, "context", None) or getattr(item, "description", "No context provided"),
                "confidence": getattr(item, "confidence_score", 1.0),
                "chunk_id": getattr(item, "chunk_id", 0),
            }
            for item in items
        ]

        filtered_items = []
        for index in range(0, len(batch_inputs), ADVERSARIAL_CRITIC_BATCH_SIZE):
            current_batch = batch_inputs[index:index + ADVERSARIAL_CRITIC_BATCH_SIZE]
            current_items = items[index:index + ADVERSARIAL_CRITIC_BATCH_SIZE]
            try:
                print(f"    Audit {item_type_label} batch {index + 1}-{index + len(current_batch)}/{len(batch_inputs)}...")
                response = invoke_llm_with_retries(
                    chain,
                    {"item_type": item_type_label, "items_json": json.dumps(current_batch, ensure_ascii=False, indent=2)},
                    f"{item_type_label} audit batch {index // ADVERSARIAL_CRITIC_BATCH_SIZE + 1}",
                )
                json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
                if not json_match:
                    print("    ! Batch audit returned no JSON; keeping this batch.")
                    filtered_items.extend(current_items)
                    continue
                try:
                    result_json = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    print("    ! Batch audit JSON parse failed; keeping this batch.")
                    filtered_items.extend(current_items)
                    continue

                accepted = {str(value) for value in result_json.get("accepted_values", [])}
                rejected_reasons = {
                    str(item.get("value")): item.get("reason", "Rejected by batch audit")
                    for item in result_json.get("rejected", [])
                    if isinstance(item, dict)
                }
                for item in current_items:
                    value = value_extractor(item)
                    if value in accepted:
                        filtered_items.append(item)
                    else:
                        print(f"REJECTED: {value} | Reason: {rejected_reasons.get(value, 'Not accepted by batch audit')}")
            except Exception as exc:
                print(f"    ! Error auditing batch {index}: {exc}")
                filtered_items.extend(current_items)
        return filtered_items

    def file_label(file_ioc):
        if file_ioc.sha256:
            return f"SHA256: {file_ioc.sha256}"
        if file_ioc.md5:
            return f"MD5: {file_ioc.md5}"
        if file_ioc.filename:
            return f"File: {file_ioc.filename}"
        return "Unknown File"

    state.extracted_ttps = audit_list(state.extracted_ttps, "TTP (Tactic/Technique)", lambda item: f"{item.technique_id} ({item.technique_name})")
    state.extracted_ips = audit_list(state.extracted_ips, "IP Address", lambda item: item.value)
    state.extracted_domains = audit_list(state.extracted_domains, "Domain Name", lambda item: item.value)
    state.extracted_urls = audit_list(state.extracted_urls, "URL", lambda item: item.value)
    state.extracted_files = audit_list(state.extracted_files, "File Artifact", file_label)
    state.extracted_cves = audit_list(state.extracted_cves, "CVE Vulnerability", lambda item: item.cve_id)
    return _extracted_state(state)


def _extracted_state(state: AgentState) -> Dict[str, Any]:
    return {
        "extracted_ttps": state.extracted_ttps,
        "extracted_ips": state.extracted_ips,
        "extracted_domains": state.extracted_domains,
        "extracted_urls": state.extracted_urls,
        "extracted_files": state.extracted_files,
        "extracted_cves": state.extracted_cves,
    }


def _entity_quality_score(item) -> tuple:
    context = getattr(item, "context", None) or getattr(item, "description", "") or ""
    return getattr(item, "confidence_score", 0), len(context)


def deduplicate_entities(items: list, key_func) -> list:
    deduped = {}
    for item in items:
        key = key_func(item) or json.dumps(item.model_dump(), sort_keys=True, ensure_ascii=False)
        if key not in deduped or _entity_quality_score(item) > _entity_quality_score(deduped[key]):
            deduped[key] = item
    return list(deduped.values())


def aggregator_node(state: AgentState):
    print("--- [Node] Aggregating Results ---")
    deduped_ttps = deduplicate_entities(state.extracted_ttps, lambda item: item.technique_id)
    deduped_ips = deduplicate_entities(state.extracted_ips, lambda item: item.value)
    deduped_domains = deduplicate_entities(state.extracted_domains, lambda item: item.value)
    deduped_urls = deduplicate_entities(state.extracted_urls, lambda item: item.value)
    deduped_files = deduplicate_entities(state.extracted_files, lambda item: item.sha256 or item.sha1 or item.md5 or item.filename or item.file_path)
    deduped_cves = deduplicate_entities(state.extracted_cves, lambda item: item.cve_id)

    try:
        summary_response = (_SUMMARY_PROMPT | llm).invoke({
            "rolling_summary": state.entity_register.rolling_summary,
            "ttps": "\n".join(f"{ttp.technique_id}: {ttp.technique_name}" for ttp in deduped_ttps[:10]),
        })
        final_summary = summary_response.content
    except Exception:
        final_summary = state.entity_register.rolling_summary[:500]

    final_data = {
        "summary": final_summary,
        "document_stats": {
            "total_length": len(state.raw_text),
            "num_chunks": len(state.text_chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        "innovation_metrics": {
            "self_consistency_rounds": SELF_CONSISTENCY_ROUNDS,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "cross_chunk_references": sum(1 for item in deduped_ips + deduped_domains + deduped_urls if item.cross_chunk_reference),
            "entity_register_size": sum(len(store) for store in state.entity_register._registers().values()),
        },
        "total_indicators": len(deduped_files) + len(deduped_ips) + len(deduped_domains) + len(deduped_urls),
        "ttps": [item.model_dump() for item in deduped_ttps],
        "iocs": {
            "files": [item.model_dump() for item in deduped_files],
            "ips": [item.model_dump() for item in deduped_ips],
            "domains": [item.model_dump() for item in deduped_domains],
            "urls": [item.model_dump() for item in deduped_urls],
            "cves": [item.model_dump() for item in deduped_cves],
        },
        "entity_register_summary": {
            "known_ips": list(state.entity_register.known_ips.keys()),
            "known_domains": list(state.entity_register.known_domains.keys()),
            "known_urls": list(state.entity_register.known_urls.keys()),
            "known_files": list(state.entity_register.known_files.keys()),
            "known_cves": list(state.entity_register.known_cves.keys()),
            "known_ttps": list(state.entity_register.known_ttps.keys()),
        },
    }
    return {"final_output": final_data, "summary": final_summary}


def build_cti_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("chunk_text", text_chunking_node)
    workflow.add_node("extract_ttp_incremental", incremental_ttp_extractor)
    workflow.add_node("extract_ioc_incremental", incremental_ioc_extractor)
    workflow.add_node("next_chunk", chunk_iterator_node)
    workflow.add_node("adversarial_critic", adversarial_critic_node)
    workflow.add_node("aggregate", aggregator_node)
    workflow.set_entry_point("chunk_text")
    workflow.add_edge("chunk_text", "extract_ttp_incremental")
    workflow.add_edge("extract_ttp_incremental", "extract_ioc_incremental")
    workflow.add_edge("extract_ioc_incremental", "next_chunk")
    workflow.add_conditional_edges(
        "next_chunk",
        should_continue_chunking,
        {"continue": "extract_ttp_incremental", "finish": "adversarial_critic"},
    )
    workflow.add_edge("adversarial_critic", "aggregate")
    workflow.add_edge("aggregate", END)
    return workflow.compile()


def discover_cti_folder_batches(cti_root: Path, selected_folder_names: Optional[List[str]] = None) -> list:
    if not cti_root.exists():
        raise FileNotFoundError(f"CTI input directory not found: {cti_root}")

    selected = {name.lower() for name in selected_folder_names} if selected_folder_names else None
    folder_batches = []
    for folder_path in sorted((path for path in cti_root.iterdir() if path.is_dir()), key=lambda path: path.name.lower()):
        if selected and folder_path.name.lower() not in selected:
            continue
        report_files = sorted(
            (path for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES),
            key=lambda path: path.name.lower(),
        )
        if report_files:
            folder_batches.append({"folder_name": folder_path.name, "folder_path": folder_path, "files": report_files})
    return folder_batches


def resolve_cti_file(cti_root: Path, folder_name: str, file_name: str) -> Path:
    folder_path = cti_root / folder_name
    if not folder_path.is_dir():
        available = ", ".join(sorted(path.name for path in cti_root.iterdir() if path.is_dir()))
        raise FileNotFoundError(f"CTI folder not found: {folder_name}. Available folders: {available}")

    direct_path = folder_path / file_name
    if direct_path.is_file():
        return direct_path

    wanted_name = file_name.lower()
    wanted_stem = Path(file_name).stem.lower()
    for candidate in sorted(folder_path.iterdir(), key=lambda path: path.name.lower()):
        if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            continue
        if candidate.name.lower() == wanted_name or candidate.stem.lower() == wanted_stem:
            return candidate

    available_files = ", ".join(
        sorted(path.name for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES)
    )
    raise FileNotFoundError(f"CTI file not found under {folder_path}: {file_name}. Available files: {available_files}")


def read_report_text(file_path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(encoding="utf-8", errors="replace")


def run_extraction_for_file(app, input_file_path: Path, output_root: Path, folder_name: str) -> Path:
    cost_handler = TokenCostHandler()
    llm.callbacks = [cost_handler]
    llm_critic.callbacks = [cost_handler]

    print(f"\n[Input] Folder: {folder_name} | File: {input_file_path.name}")
    print("Running incremental extraction...\n")
    result = app.invoke(AgentState(raw_text=read_report_text(input_file_path)), config={"recursion_limit": 1000})
    if not result.get("final_output"):
        raise RuntimeError(f"No final output generated for {input_file_path}")

    cost_handler.print_report()
    result["final_output"]["cost_report"] = cost_handler.get_report()
    result["final_output"]["source"] = {
        "cti_folder": folder_name,
        "file_name": input_file_path.name,
        "file_path": str(input_file_path),
    }

    output_dir = output_root / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / f"{input_file_path.stem}.json"
    with output_file_path.open("w", encoding="utf-8") as file:
        json.dump(result["final_output"], file, indent=2, ensure_ascii=False)

    print(f"\n[Success] Results saved to: {output_file_path}")
    return output_file_path


def normalize_target_folders(value) -> List[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [folder.strip() for folder in value if folder and folder.strip()]


def main():
    print("\nBuilding Agent Graph...")
    app = build_cti_graph()
    target_folders = normalize_target_folders(TARGET_FOLDERS)
    target_file = TARGET_FILE.strip() if TARGET_FILE else ""

    if target_file and len(target_folders) != 1:
        raise ValueError("TARGET_FILE must be used with exactly one folder in TARGET_FOLDERS")

    if target_file:
        target_folder = target_folders[0]
        input_file_path = resolve_cti_file(CTI_INPUT_ROOT, target_folder, target_file)
        run_extraction_for_file(app, input_file_path, DEFAULT_OUTPUT_ROOT, target_folder)
        return

    folder_batches = discover_cti_folder_batches(CTI_INPUT_ROOT, target_folders or None)
    total_files = sum(len(batch["files"]) for batch in folder_batches)
    print(f"Discovered {len(folder_batches)} CTI folders and {total_files} report files under {CTI_INPUT_ROOT}")

    saved_files = []
    failures = []
    for batch in folder_batches:
        print(f"\n=== Processing CTI folder: {batch['folder_name']} ({len(batch['files'])} files) ===")
        for input_file_path in batch["files"]:
            try:
                saved_files.append(run_extraction_for_file(app, input_file_path, DEFAULT_OUTPUT_ROOT, batch["folder_name"]))
            except Exception as exc:
                failures.append((input_file_path, str(exc)))
                print(f"\n[Failed] {input_file_path}: {exc}")

    print(f"\n[Done] Saved {len(saved_files)} result files to {DEFAULT_OUTPUT_ROOT}")
    if failures:
        print(f"[Warning] {len(failures)} files failed:")
        for input_file_path, reason in failures:
            print(f"  - {input_file_path}: {reason}")


if __name__ == "__main__":
    main()
