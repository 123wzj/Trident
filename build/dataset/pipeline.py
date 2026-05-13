import argparse
import asyncio
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import aiohttp
from tqdm import tqdm

from common import TARGET_APTS, get_malwarebazaar_api_key, read_json, write_json


MALWAREBAZAAR_URL = "https://mb-api.abuse.ch/api/v1/"
CSV_HEADERS = ["index", "sha256", "apt", "signature", "imphash", "ssdeep", "tlsh"]


def aggregate_hash_dataset(dataset_dir: os.PathLike, output_file: os.PathLike):
    apt_hashes: Dict[str, Set[str]] = {}
    total_files = 0

    for file_path in Path(dataset_dir).rglob("*.json"):
        total_files += 1
        try:
            data = read_json(file_path)
        except Exception as exc:
            print(f"[ERR] Cannot read {file_path}: {exc}")
            continue

        apt = data.get("apt", "Unknown_APT")
        apt_hashes.setdefault(apt, set()).update(data.get("indicators", []))
        if total_files % 500 == 0:
            print(f"Scanned {total_files} files...")

    output = {apt: sorted(hashes) for apt, hashes in apt_hashes.items()}
    write_json(output_file, output, indent=4)
    print(f"Aggregated {sum(len(items) for items in output.values())} hashes from {len(output)} APTs.")


async def fetch_file_info(session, api_key: str, apt: str, file_hash: str, semaphore, max_retries: int):
    payload = {"query": "get_info", "hash": file_hash}
    headers = {"Auth-Key": api_key, "User-Agent": "Python-Async"}

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(MALWAREBAZAAR_URL, data=payload, headers=headers, timeout=30) as response:
                    if response.status == 429 or response.status >= 500:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    if response.status != 200:
                        return None

                    body = await response.json()
                    if body.get("query_status") != "ok" or not body.get("data"):
                        return None

                    data = body["data"][0]
                    return {
                        "sha256": file_hash,
                        "apt": apt,
                        "signature": data.get("signature", ""),
                        "imphash": data.get("imphash", ""),
                        "ssdeep": data.get("ssdeep", ""),
                        "tlsh": data.get("tlsh", ""),
                    }
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
            except Exception:
                return None
    return None


async def enrich_hashes_with_malwarebazaar(
    input_json: os.PathLike,
    output_csv: os.PathLike,
    api_key: Optional[str] = None,
    concurrency: int = 50,
    max_retries: int = 3,
):
    api_key = api_key or get_malwarebazaar_api_key()
    apt_data = read_json(input_json)
    tasks_data = [(apt, file_hash) for apt, hashes in apt_data.items() for file_hash in set(hashes)]

    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    semaphore = asyncio.Semaphore(concurrency)
    output_csv = Path(output_csv)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            fetch_file_info(session, api_key, apt, file_hash, semaphore, max_retries)
            for apt, file_hash in tasks_data
        ]
        with output_csv.open("w", encoding="utf-8-sig", newline="", buffering=8192) as file:
            writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
            writer.writeheader()
            success_count = 0
            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), unit="hash"):
                result = await future
                if not result:
                    continue
                success_count += 1
                result["index"] = success_count
                writer.writerow(result)
    print(f"Enriched {success_count} hashes to {output_csv}.")


def build_hash_to_event_map(dataset_dir: os.PathLike) -> Dict[str, str]:
    hash_to_event = {}
    for file_path in Path(dataset_dir).rglob("*.json"):
        try:
            data = read_json(file_path)
        except Exception:
            continue
        for file_hash in data.get("indicators", []):
            hash_to_event[file_hash] = file_path.stem
    print(f"Indexed {len(hash_to_event)} hashes from {dataset_dir}.")
    return hash_to_event


def filter_enriched_csv(
    input_csv: os.PathLike,
    output_csv: os.PathLike,
    dataset_dir: os.PathLike,
    target_apts: Iterable[str] = TARGET_APTS,
):
    target_apts = set(target_apts)
    hash_to_event = build_hash_to_event_map(dataset_dir)
    written = 0
    missing = 0

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as source, open(
        output_csv, "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.reader(source)
        writer = csv.writer(target)
        header = next(reader)
        header.insert(1, "event_id")
        writer.writerow(header)

        for row in reader:
            if not row:
                continue
            file_hash = row[1].strip()
            apt = row[2].strip()
            if apt not in target_apts:
                continue
            event_id = hash_to_event.get(file_hash, "Unknown")
            missing += int(event_id == "Unknown")
            row.insert(1, event_id)
            writer.writerow(row)
            written += 1

    print(f"Wrote {written} rows to {output_csv}; {missing} rows have unknown event_id.")


def count_known_signatures(csv_path: os.PathLike, aliases_json: os.PathLike) -> Tuple[int, int]:
    aliases = read_json(aliases_json)
    names = set()
    for name, alias_list in aliases.items():
        names.add(name)
        names.update(alias_list or [])

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        signatures = {row.get("signature", "") for row in csv.DictReader(file)}
    match_count = sum(1 for signature in signatures if signature in names)
    print(f"Unique signatures: {len(signatures)}")
    print(f"Known malware aliases matched: {match_count}")
    return len(signatures), match_count


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate and enrich file-hash datasets.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    aggregate = subcommands.add_parser("aggregate")
    aggregate.add_argument("--dataset", default="./dataset")
    aggregate.add_argument("--output", default="all_apt_hashes.json")

    enrich = subcommands.add_parser("enrich")
    enrich.add_argument("--input", default="all_apt_hashes.json")
    enrich.add_argument("--output", default="filehash_info.csv")
    enrich.add_argument("--concurrency", type=int, default=50)

    filter_csv = subcommands.add_parser("filter")
    filter_csv.add_argument("--input", default="filehash_info.csv")
    filter_csv.add_argument("--output", default="apt_filtered.csv")
    filter_csv.add_argument("--dataset", default="./dataset")

    count = subcommands.add_parser("count-signatures")
    count.add_argument("--csv", default="apt_filtered.csv")
    count.add_argument("--aliases", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "aggregate":
        aggregate_hash_dataset(args.dataset, args.output)
    elif args.command == "enrich":
        asyncio.run(enrich_hashes_with_malwarebazaar(args.input, args.output, concurrency=args.concurrency))
    elif args.command == "filter":
        filter_enriched_csv(args.input, args.output, args.dataset)
    elif args.command == "count-signatures":
        count_known_signatures(args.csv, args.aliases)


if __name__ == "__main__":
    main()
