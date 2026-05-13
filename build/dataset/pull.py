import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

import OTXv2 as otx_sdk
from IndicatorTypes import DOMAIN, IPv4, URL
from OTXv2 import NotFound, OTXv2, RetryError
from joblib import Parallel, delayed

from common import (
    CVE_TYPE,
    FILE_HASH_TYPE,
    IOC_TYPES,
    TARGET_APTS,
    WORKERS_PER_KEY,
    build_jobs,
    dataset_root,
    get_otx_api_keys,
    load_pulse_ids,
    rate_limit,
    write_json,
)

BadRequest = getattr(otx_sdk, "BadRequest", ValueError)


def sanitize(value: str) -> str:
    return value.replace("{", "{{").replace("}", "}}")


def get_indicator_details(otx, indicator_type, value, section: Optional[str] = None):
    try:
        if section:
            return otx.get_indicator_details_by_section(indicator_type, value, section=section)
        return otx.get_indicator_details_full(indicator_type, value)
    except (NotFound, BadRequest, RetryError):
        return {}


def enrich_url(otx, value: str) -> Dict:
    result = {"ioc": value, "type": "URL", "hostname": urlparse(value).netloc}
    details = get_indicator_details(otx, URL, sanitize(value))
    if not details:
        return result

    result.update({
        key: value
        for key, value in details.get("general", {}).items()
        if key in {"net_loc", "city", "region", "latitude", "longitude", "country_code"}
    })

    url_list = details.get("url_list", {}).get("url_list") or []
    server_details = url_list[0].get("result") if url_list else None
    if not server_details:
        return result

    urlworker = server_details.get("urlworker", {})
    for key in ("ip", "filetype", "fileclass", "http_code"):
        result[key] = urlworker.get(key)

    headers = {key.upper(): value for key, value in (urlworker.get("http_response") or {}).items()}
    result.update({
        "server": headers.get("SERVER"),
        "expires": headers.get("EXPIRES"),
        "cache-control": headers.get("CACHE-CONTROL"),
        "encoding": headers.get("CONTENT-ENCODING"),
        "content-type": headers.get("CONTENT-TYPE"),
    })

    extractor = server_details.get("extractor")
    if extractor:
        result.update({f"extracted-{key}": value for key, value in extractor.items()})
    return result


def enrich_ip(otx, value: str) -> Dict:
    result = {"ioc": value, "type": "IP", "resolves_to": []}
    details = get_indicator_details(otx, IPv4, value, section="general")
    result.update({
        key: val
        for key, val in details.items()
        if key in {"net_loc", "city", "region", "latitude", "longitude", "country_code", "asn"}
    })

    passive_dns = get_indicator_details(otx, IPv4, value, section="passive_dns").get("passive_dns") or []
    if passive_dns:
        result.setdefault("asn", passive_dns[0].get("asn"))
    for record in passive_dns:
        result["resolves_to"].append({
            "host": record.get("hostname"),
            "record_type": record.get("record_type"),
            "first_seen": record.get("first"),
            "last_seen": record.get("last"),
        })
    return result


def enrich_domain(otx, value: str, kind: str = "domain") -> Dict:
    result = {"ioc": value, "type": kind, "dns_records": []}
    passive_dns = get_indicator_details(otx, DOMAIN, value, section="passive_dns").get("passive_dns") or []
    for record in passive_dns:
        result["dns_records"].append({
            key: record.get(key)
            for key in ("address", "first", "last", "record_type", "asn")
        })
    return result


def enrich_ioc(otx, value: str, ioc_type: str) -> Dict:
    if ioc_type.lower() == "domain":
        return enrich_domain(otx, value)
    if ioc_type.lower() == "hostname":
        return enrich_domain(otx, value, kind="hostname")
    if ioc_type in {"IPv4", "IPv6", "IP"}:
        return enrich_ip(otx, value)
    if ioc_type == "URL":
        return enrich_url(otx, value)
    raise TypeError(f"Unsupported IOC type: {ioc_type}")


def get_pulse_indicators(otx, event_id: str) -> List[Dict]:
    try:
        return otx.get_pulse_indicators(event_id, include_inactive=True)
    except NotFound:
        return []
    except RetryError:
        time.sleep(5)
        try:
            return otx.get_pulse_indicators(event_id, include_inactive=True)
        except Exception:
            return []
    except Exception:
        return []


def get_pulse_details(otx, event_id: str) -> Dict:
    try:
        details = otx.get_pulse_details(event_id)
    except RetryError:
        time.sleep(5)
        try:
            details = otx.get_pulse_details(event_id)
        except RetryError:
            details = {}
    return {key: details.get(key, "") for key in ("name", "description", "tags")}


def save_cve_event(otx, event_id: str, apt: str, out_dir: Path, message: str):
    out_file = out_dir / f"{event_id}.json"
    if out_file.exists():
        return

    start = time.time()
    cves = [
        item.get("indicator")
        for item in get_pulse_indicators(otx, event_id)
        if item.get("type") == CVE_TYPE and item.get("indicator")
    ]
    rate_limit(start)
    if cves:
        write_json(out_file, {"event_id": event_id, "apt": apt, "count": len(cves), "indicators": cves}, indent=4)
    print(f"{message} CVE {apt}/{event_id}: {len(cves)}")


def save_ioc_event(otx, event_id: str, apt: str, out_dir: Path, message: str, max_iocs: int):
    out_file = out_dir / f"{event_id}.json"
    if out_file.exists():
        return

    start = time.time()
    raw_iocs = [
        (item.get("indicator"), item.get("type"))
        for item in get_pulse_indicators(otx, event_id)
        if item.get("type") in IOC_TYPES and item.get("indicator")
    ]
    details = get_pulse_details(otx, event_id)
    rate_limit(start)

    if len(raw_iocs) > max_iocs:
        print(f"{message} IOC {apt}/{event_id}: skipped {len(raw_iocs)} indicators")
        return

    enriched = [enrich_ioc(otx, value, ioc_type) for value, ioc_type in raw_iocs]
    write_json(out_file, {"event_id": event_id, "label": apt, "details": details, "iocs": enriched})
    print(f"{message} IOC {apt}/{event_id}: {len(enriched)}")


def fetch_file_hash_event(otx, event_id: str, apt: str, out_dir: Path, message: str) -> Optional[Dict]:
    start = time.time()
    hashes = {
        item.get("indicator")
        for item in get_pulse_indicators(otx, event_id)
        if item.get("type") == FILE_HASH_TYPE and item.get("indicator")
    }
    rate_limit(start)
    print(f"{message} FileHash {apt}/{event_id}: {len(hashes)}")
    if not hashes:
        return None
    return {"event_id": event_id, "apt": apt, "out_dir": str(out_dir), "indicators": sorted(hashes)}


def save_unique_file_hashes(events: Sequence[Dict]):
    counts = defaultdict(int)
    for event in events:
        for file_hash in event["indicators"]:
            counts[file_hash] += 1

    saved = 0
    discarded = 0
    for event in events:
        indicators = [file_hash for file_hash in event["indicators"] if counts[file_hash] == 1]
        discarded += len(event["indicators"]) - len(indicators)
        if not indicators:
            continue
        out_file = Path(event["out_dir"]) / f"{event['event_id']}.json"
        write_json(
            out_file,
            {
                "event_id": event["event_id"],
                "apt": event["apt"],
                "count": len(indicators),
                "indicators": indicators,
            },
            indent=4,
        )
        saved += 1

    print(f"Saved {saved} file-hash events; discarded {discarded} duplicate hash instances.")


def build_dataset(
    mode: str,
    output_dir: Path,
    pulse_ids_path: Path,
    topk: int,
    target_apts: Optional[Iterable[str]],
    max_iocs: int,
):
    pulse_ids = load_pulse_ids(pulse_ids_path, topk=topk, target_apts=target_apts)
    jobs = build_jobs(pulse_ids, output_dir)
    otxs = [OTXv2(key) for key in get_otx_api_keys()]
    print(f"Prepared {len(jobs)} jobs with {len(otxs)} OTX keys.")

    if mode == "file":
        events = Parallel(n_jobs=len(otxs) * WORKERS_PER_KEY, prefer="threads")(
            delayed(fetch_file_hash_event)(otxs[index % len(otxs)], event, apt, out_dir, f"({index + 1}/{len(jobs)})")
            for index, (event, apt, out_dir) in enumerate(jobs)
        )
        save_unique_file_hashes([event for event in events if event])
        return

    if mode == "ioc":
        Parallel(n_jobs=len(otxs) * WORKERS_PER_KEY, prefer="threads")(
            delayed(save_ioc_event)(
                otxs[index % len(otxs)],
                event,
                apt,
                out_dir,
                f"({index + 1}/{len(jobs)})",
                max_iocs,
            )
            for index, (event, apt, out_dir) in enumerate(jobs)
        )
        return

    Parallel(n_jobs=len(otxs) * WORKERS_PER_KEY, prefer="threads")(
        delayed(save_cve_event)(otxs[index % len(otxs)], event, apt, out_dir, f"({index + 1}/{len(jobs)})")
        for index, (event, apt, out_dir) in enumerate(jobs)
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build OTX-backed CTI datasets.")
    parser.add_argument("mode", choices=("ioc", "cve", "file"), help="Dataset type to build.")
    parser.add_argument("--pulse-ids", default=str(dataset_root() / "pulse_ids.json"))
    parser.add_argument("--output", default=str(dataset_root() / "dataset"))
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--all-apts", action="store_true", help="Use top-k APTs without TARGET_APTS filtering.")
    parser.add_argument("--max-iocs", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()
    target_apts = None if args.all_apts else TARGET_APTS
    build_dataset(args.mode, Path(args.output), Path(args.pulse_ids), args.topk, target_apts, args.max_iocs)


if __name__ == "__main__":
    main()
