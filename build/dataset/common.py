import json
import os
import time
from collections import defaultdict
from pathlib import Path
from random import shuffle
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


WORKERS_PER_KEY = 5
OTX_RATE_SECONDS = 0.3704
DEFAULT_TOPK = 25

IOC_TYPES = {"IPv4", "IPv6", "domain", "hostname", "URL"}
CVE_TYPE = "CVE"
FILE_HASH_TYPE = "FileHash-SHA256"

TARGET_APTS = {
    "APT35",
    "APT38",
    "APT41",
    "BLACKENERGY",
    "COBALT GROUP",
    "FIN11",
    "KIMSUKY",
    "MAGECART",
    "MUDDYWATER",
    "MUSTANG PANDA",
    "PAT BEAR",
    "SAFE",
    "SAPPHIRE MUSHROOM",
    "TA551",
    "TURLA",
}


def dataset_root() -> Path:
    return Path(__file__).resolve().parent


def read_json(path: os.PathLike):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: os.PathLike, data, indent: int = 2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)


def get_env_list(*names: str) -> List[str]:
    values = []
    for name in names:
        raw = os.getenv(name, "")
        values.extend(item.strip() for item in raw.replace(";", ",").split(",") if item.strip())
    return values


def get_otx_api_keys() -> List[str]:
    keys = get_env_list("OTX_API_KEYS", "OTX_KEY", "OTX_API_KEY")
    if not keys:
        raise KeyError("Set OTX_API_KEYS or OTX_KEY before running this script.")
    return keys


def get_malwarebazaar_api_key() -> str:
    key = os.getenv("MALWAREBAZAAR_API_KEY") or os.getenv("MALWARE_BAZAAR_API_KEY")
    if not key:
        raise KeyError("Set MALWAREBAZAAR_API_KEY before enriching file hashes.")
    return key


def top_apts(apt_to_ids: Dict[str, Sequence[str]], topk: int = DEFAULT_TOPK) -> List[Tuple[str, int]]:
    ranked = sorted(((apt, len(ids)) for apt, ids in apt_to_ids.items()), key=lambda item: item[1], reverse=True)
    return ranked[:topk]


def overlapping_pulses(apt_to_ids: Dict[str, Sequence[str]]) -> Set[str]:
    pulse_to_apts = defaultdict(set)
    for apt, pulse_ids in apt_to_ids.items():
        for pulse_id in pulse_ids:
            pulse_to_apts[pulse_id].add(apt)
    return {pulse_id for pulse_id, apts in pulse_to_apts.items() if len(apts) > 1}


def load_pulse_ids(
    pulse_ids_path: os.PathLike,
    topk: int = DEFAULT_TOPK,
    target_apts: Optional[Iterable[str]] = None,
    drop_overlaps: bool = True,
) -> Dict[str, List[str]]:
    apt_to_ids = read_json(pulse_ids_path)
    ignored = overlapping_pulses(apt_to_ids) if drop_overlaps else set()
    filtered = {apt: [pulse_id for pulse_id in ids if pulse_id not in ignored] for apt, ids in apt_to_ids.items()}

    selected = {apt for apt, _ in top_apts(filtered, topk)}
    if target_apts:
        selected &= {apt.upper() for apt in target_apts}

    return {apt: filtered[apt] for apt in selected if filtered.get(apt)}


def build_jobs(pulse_ids: Dict[str, Sequence[str]], output_dir: os.PathLike) -> List[Tuple[str, str, Path]]:
    jobs = []
    for apt, events in pulse_ids.items():
        apt_dir = Path(output_dir) / apt
        apt_dir.mkdir(parents=True, exist_ok=True)
        for event in events:
            if not (apt_dir / f"{event}.json").exists():
                jobs.append((event, apt, apt_dir))
    shuffle(jobs)
    return jobs


def rate_limit(start_time: float, workers_per_key: int = WORKERS_PER_KEY):
    wait_time = max((OTX_RATE_SECONDS * workers_per_key) - (time.time() - start_time), 0)
    if wait_time:
        time.sleep(wait_time)
