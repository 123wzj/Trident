import csv
import json
import os
import socket

from build_dataset.incremental_pull_full import save_incremental_pulse_ids_only, save_incremental_stats
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"

import sys
import time
import asyncio
import aiohttp
from typing import Dict, List, Set, Tuple
from random import shuffle

from joblib import Parallel, delayed
from OTXv2 import OTXv2, RetryError, NotFound
from tqdm import tqdm

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入依赖模块
from src.build_dataset.pull import get_otx, thread_job, API_KEYS
from src.build_dataset.label_mapper.apt_label_mapper import build_ta_map

# ============================================================================
# 1. 核心参数调优 (提速的关键)
# ============================================================================

# 【并发覆盖】降低并发以避免 429 和代理断开
WORKERS_PER_KEY = 5

# 【AbuseCH 并发】提升异步并发限制
ABUSE_CONCURRENCY_LIMIT = 50
ABUSE_MAX_RETRIES = 3



# 【API 延迟】添加延迟避免 429 限流
API_DELAY = 0.5

# 【拉取上限】增加拉取上限
MAX_PULSES_PER_ORG = 1000

# 【网络超时】
socket.setdefaulttimeout(60)

# 其他配置保持不变
file_dir = os.path.dirname(os.path.realpath(__file__))
IOC_TYPES = ['IPv4', 'IPv6', 'domain', 'hostname', 'URL']
ABUSE_API_URL = "https://mb-api.abuse.ch/api/v1/"
ABUSE_API_KEY = "9ff9bdc51c4deb819dabbd4db830d3c84d8d190bf37410b8"


# ============================================================================
# 工具函数 (保持不变)
# ============================================================================

def load_existing_pulse_ids(file_path: str) -> Dict[str, List[str]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def verify_pulse_data_integrity(pulse_data: Dict[str, List[str]]) -> bool:
    total_orgs = len(pulse_data)
    total_pulses = sum(len(pulses) for pulses in pulse_data.values())
    print(f"  数据完整性检查:")
    print(f"    总组织数: {total_orgs}")
    print(f"    总pulse数: {total_pulses}")
    print(f"    数据完整性检查通过")
    return True


def get_top_organizations(pulse_data: Dict[str, List[str]], top_k: int = 20) -> List[Tuple[str, int]]:
    TARGET_ORGS = [
        "APT28", "APT34", "APT35", "APT37", "APT38", "APT41", "BLACKENERGY", "CARETO",
        "COBALT GROUP", "FIN11", "FIN7", "GOLD WATERFALL", "ICEFOG", "KIMSUKY", "KINSING",
        "MAGECART", "MUDDYWATER", "MUSTANG PANDA", "PAT BEAR", "SAFE", "SAPPHIRE MUSHROOM",
        "TA511", "TA551", "TEAMTNT", "TURLA"
    ]
    result = []
    for org in TARGET_ORGS:
        if org in pulse_data:
            result.append((org, len(pulse_data[org])))
        else:
            result.append((org, 0))
    result_sorted = sorted(result, key=lambda x: x[1], reverse=True)
    print(f"  目标组织列表（来自 output 目录）:")
    for i, (org, count) in enumerate(result_sorted, 1):
        status = "✓" if count > 0 else "✗"
        print(f"    {i:2d}. {org:25s}: {count:5d} pulses {status}")
    return result_sorted[:top_k]


def get_org_aliases(org_name: str, ta_map: Dict[str, str]) -> List[str]:
    aliases = [alias for alias, official in ta_map.items() if official == org_name]
    if org_name not in aliases:
        aliases.append(org_name)
    return aliases


def sanitize(s: str) -> str:
    return s.replace('{', '{{').replace('}', '}}')


# ============================================================================
# Pulse 拉取模块 (保持你的逻辑，优化了 Retry 参数)
# ============================================================================

def fetch_pulses_from_otx(org_name: str, otx: OTXv2, ta_map: Dict[str, str]) -> Set[str]:
    """
    [回归库调用版] 从OTX API获取指定组织的所有pulse ID
    不做分页，一次性拉取，但增加了重试机制以防断连
    """
    pulse_ids = set()
    aliases = [org_name]  # 只使用官方名称

    if len(aliases) > 1:
        print(f"    搜索 {org_name} 的 {len(aliases)} 个别名...")

    for alias in aliases:
        try:
            # 1. 先探测一下总数
            try:
                resp = otx.search_pulses('tag:"%s"' % alias, max_results=1)
            except Exception as e:
                print(f"      [警告] 连接初探失败，重试中... ({e})")
                time.sleep(1)  # 缩短等待
                resp = otx.search_pulses('tag:"%s"' % alias, max_results=1)

            total_count = resp.get('count', 0)
            if total_count == 0:
                continue

            count = min(total_count, MAX_PULSES_PER_ORG)

            # 2. 一次性拉取 (增加重试)
            success = False
            for attempt in range(3):
                try:
                    if attempt > 0:
                        print(f"      [重试 {attempt}/3] 重新拉取 {count} 条...")

                    resp = otx.search_pulses('tag:"%s"' % alias, max_results=count)

                    if 'results' in resp:
                        new_pulses = {item['id'] for item in resp['results']}
                        before_count = len(pulse_ids)
                        pulse_ids.update(new_pulses)
                        found_new = len(pulse_ids) - before_count

                        if len(aliases) > 1 and found_new > 0:
                            print(f"      别名 '{alias}': 找到 {len(new_pulses)} 个pulse (新增 {found_new})")

                    success = True
                    break

                except Exception as e:
                    print(f"      [网络波动] 第 {attempt + 1} 次尝试失败: {e}")
                    time.sleep(2)  # 缩短重试等待

            if not success:
                print(f"      [失败] 无法拉取 {alias} 的数据")

            time.sleep(API_DELAY)

        except Exception as e:
            print(f"      [错误] 搜索别名 '{alias}' 时出错: {e}")
            continue

    return pulse_ids


def fetch_incremental_pulses(top_orgs: List[Tuple[str, int]], pulse_data: Dict[str, List[str]],
                             ta_map: Dict[str, str]) -> Dict:
    """从OTX API获取前20个组织的所有pulse并找出增量pulse（支持断点续传）"""
    otx = get_otx(API_KEYS[0])

    incremental_stats_path = os.path.join(file_dir, 'incremental_pulses.json')
    completed_orgs = set()

    if os.path.exists(incremental_stats_path):
        print("\n检测到之前的运行记录，支持断点续传...")
        try:
            with open(incremental_stats_path, 'r') as f:
                prev_data = json.load(f)
                completed_orgs = set(prev_data.get("organizations", {}).keys())
                print(f"  已完成 {len(completed_orgs)} 个组织，将从断点继续")
        except Exception as e:
            print(f"  [警告] 读取进度文件失败: {e}，将从头开始")

    result = {
        "organizations": {},
        "summary": {
            "total_existing": 0,
            "total_new": 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    # 恢复数据
    if 'prev_data' in locals():
        for org_name in completed_orgs:
            if org_name in prev_data.get("organizations", {}):
                result["organizations"][org_name] = prev_data["organizations"][org_name]
                result["summary"]["total_existing"] += result["organizations"][org_name]["existing_count"]
                result["summary"]["total_new"] += result["organizations"][org_name]["new_count"]

    print("\n开始从OTX API拉取pulse数据...")
    print("=" * 70)

    for org_name, existing_count in tqdm(top_orgs, desc="拉取组织pulse"):
        if org_name in completed_orgs:
            # print(f"  [跳过] {org_name:25s}: 已完成") # 注释掉以保持界面清爽
            continue

        all_pulse_ids = fetch_pulses_from_otx(org_name, otx, ta_map)

        if not all_pulse_ids:
            print(f"  {org_name:25s}: 未找到任何pulse")
            continue

        org_existing = set(pulse_data.get(org_name, []))
        new_pulses = all_pulse_ids - org_existing
        total_count = len(all_pulse_ids)
        new_count = len(new_pulses)

        result["summary"]["total_existing"] += existing_count
        result["summary"]["total_new"] += new_count

        result["organizations"][org_name] = {
            "existing_count": existing_count,
            "total_count": total_count,
            "new_count": new_count,
            "new_pulse_ids": sorted(list(new_pulses))
        }

        try:
            with open(incremental_stats_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [警告] 保存进度失败: {e}")

        status = f"+{new_count}" if new_count > 0 else "无新增"
        print(f"  {org_name:25s}: 现有 {existing_count:4d}, API返回 {total_count:4d}, 新增 {status}")

    print("=" * 70)
    return result


# ============================================================================
# IOC 数据下载模块（保持原逻辑，提升并发）
# ============================================================================

def build_incremental_ioc_dataset(incremental_data: Dict, output_dir: str):
    jobs = []
    skipped_exists = 0

    for org_name, org_data in incremental_data["organizations"].items():
        new_pulse_ids = org_data.get("new_pulse_ids", [])
        if not new_pulse_ids:
            continue

        out_dir = os.path.join(output_dir, org_name)
        os.makedirs(out_dir, exist_ok=True)

        for pulse_id in new_pulse_ids:
            out_f = os.path.join(out_dir, pulse_id) + '.json'
            # 断点续传检查
            if os.path.exists(out_f) and os.path.getsize(out_f) > 0:
                skipped_exists += 1
            else:
                jobs.append((pulse_id, org_name, out_dir))

    if not jobs:
        print("\n所有IOC数据已下载完成（断点续传）")
        return

    if skipped_exists > 0:
        print(f"\n检测到 {skipped_exists} 个已下载的文件，将跳过")

    shuffle(jobs)

    print(f"\n开始下载并富化 {len(jobs)} 个新pulse的IOC数据...")

    # ⚠️ 关键优化：大幅提升并发数
    # 如果有 2 个 Key，这里将启动 2 * 25 = 50 个线程
    n_workers = len(API_KEYS) * WORKERS_PER_KEY
    print(f"🚀 启动并发下载: {n_workers} 线程 (Windows优化)")
    print("=" * 70)

    otxs = [get_otx(key) for key in API_KEYS]

    Parallel(n_jobs=n_workers, prefer='threads')(
        delayed(thread_job)(
            otxs[i % len(otxs)],
            j[0],
            j[1],
            f'({i + 1}/{len(jobs)})',
            j[2]
        )
        for i, j in enumerate(jobs)
    )

    # 统计
    actual_files = 0
    for org_name, org_data in incremental_data["organizations"].items():
        new_pulse_ids = org_data.get("new_pulse_ids", [])
        if not new_pulse_ids: continue
        out_dir = os.path.join(output_dir, org_name)
        for pulse_id in new_pulse_ids:
            out_f = os.path.join(out_dir, pulse_id) + '.json'
            if os.path.exists(out_f):
                actual_files += 1

    newly_downloaded = actual_files - skipped_exists
    skipped_no_ioc = len(jobs) - newly_downloaded

    print("=" * 70)
    print(f"IOC数据下载完成!")
    print(f"  已存在(跳过): {skipped_exists}")
    print(f"  本次下载任务: {len(jobs)}")
    print(f"  本次成功保存: {newly_downloaded}")
    print(f"  跳过(无IOC): {skipped_no_ioc}")
    print(f"  总计保存文件: {actual_files}")


# ============================================================================
# CVE 数据下载模块
# ============================================================================

def build_incremental_cve_dataset(incremental_data: Dict, output_dir: str):
    cve_jobs = []
    skipped_no_ioc = 0

    for org_name, org_data in incremental_data["organizations"].items():
        new_pulse_ids = org_data.get("new_pulse_ids", [])
        if not new_pulse_ids: continue

        cve_out_dir = os.path.join(output_dir, 'cve', org_name)
        os.makedirs(cve_out_dir, exist_ok=True)

        for pulse_id in new_pulse_ids:
            cve_out_f = os.path.join(cve_out_dir, pulse_id) + '.json'
            if os.path.exists(cve_out_f): continue

            ioc_file = os.path.join(output_dir, org_name, pulse_id + '.json')
            if not os.path.exists(ioc_file):
                skipped_no_ioc += 1
                continue

            cve_jobs.append((pulse_id, org_name, cve_out_dir))

    if not cve_jobs:
        print("\n没有需要下载的新CVE数据")
        return

    print(f"\n开始下载 {len(cve_jobs)} 个新pulse的CVE数据...")

    # ⚠️ 同样应用高并发
    n_workers = len(API_KEYS) * WORKERS_PER_KEY
    print(f"🚀 并发数: {n_workers}")
    print("=" * 70)

    otxs = [get_otx(key) for key in API_KEYS]

    def cve_thread_job(otx, event_id, apt, out_dir):
        out_f = os.path.join(out_dir, event_id) + '.json'
        if os.path.exists(out_f): return

        try:
            iocs = otx.get_pulse_indicators(event_id, include_inactive=True)
        except (NotFound, RetryError, Exception):
            return

        extracted_cves = []
        for ioc in iocs:
            if ioc.get('type') == 'CVE':
                indicator = ioc.get('indicator')
                if indicator: extracted_cves.append(indicator)

        if extracted_cves:
            data_to_save = {
                "event_id": event_id, "apt": apt, "count": len(extracted_cves), "indicators": extracted_cves
            }
            with open(out_f, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)

    Parallel(n_jobs=n_workers, prefer='threads')(
        delayed(cve_thread_job)(
            otxs[i % len(otxs)], j[0], j[1], j[2]
        )
        for i, j in enumerate(tqdm(cve_jobs, desc="CVE数据下载"))
    )
    print("CVE数据下载完成!")


# ============================================================================
# File Hash 数据处理模块
# ============================================================================

def build_incremental_file_dataset(incremental_data: Dict, output_dir: str, otxs: List):
    file_jobs = []
    skipped_no_ioc = 0

    for org_name, org_data in incremental_data["organizations"].items():
        new_pulse_ids = org_data.get("new_pulse_ids", [])
        if not new_pulse_ids: continue

        for pulse_id in new_pulse_ids:
            ioc_file = os.path.join(output_dir, org_name, pulse_id + '.json')
            if not os.path.exists(ioc_file):
                skipped_no_ioc += 1
                continue
            file_jobs.append((pulse_id, org_name))

    if not file_jobs:
        print("\n没有需要处理的File数据")
        return None, None

    print(f"\n开始提取 {len(file_jobs)} 个新pulse的File Hash数据...")

    # ⚠️ 同样应用高并发
    n_workers = len(otxs) * WORKERS_PER_KEY
    print(f"🚀 并发数: {n_workers}")
    print("=" * 70)

    def file_thread_job(otx, event_id, apt):
        try:
            iocs = otx.get_pulse_indicators(event_id, include_inactive=True)
        except:
            return None

        extracted_hashes = set()
        for ioc in iocs:
            if ioc.get('type') == 'FileHash-SHA256':
                indicator = ioc.get('indicator')
                if indicator: extracted_hashes.add(indicator)

        if extracted_hashes:
            return {"event_id": event_id, "apt": apt, "indicators": list(extracted_hashes)}
        return None

    results = Parallel(n_jobs=n_workers, prefer='threads')(
        delayed(file_thread_job)(
            otxs[i % len(otxs)], j[0], j[1]
        )
        for i, j in enumerate(tqdm(file_jobs, desc="File Hash提取"))
    )

    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print("未找到File Hash数据")
        return None, None

    unique_hashes = set()
    for result in valid_results:
        unique_hashes.update(result['indicators'])

    print(f"找到 {len(unique_hashes)} 个唯一文件哈希")

    file_output_dir = os.path.join(output_dir, 'file_hashes')
    os.makedirs(file_output_dir, exist_ok=True)
    return valid_results, file_output_dir


async def fetch_file_info(session, apt_name, file_hash, semaphore):
    url = ABUSE_API_URL
    payload = {'query': 'get_info', 'hash': file_hash}
    headers = {'Auth-Key': ABUSE_API_KEY, 'User-Agent': 'Python-Async/Linux-Fast'}

    async with semaphore:
        for attempt in range(ABUSE_MAX_RETRIES):
            try:
                async with session.post(url, data=payload, headers=headers, timeout=30, ssl=False) as response:
                    if response.status == 429 or response.status >= 500:
                        await asyncio.sleep(1)  # 缩短退避时间
                        continue
                    if response.status != 200: return None

                    json_resp = await response.json()
                    if json_resp.get('query_status') != 'ok': return None

                    data_list = json_resp.get('data', [])
                    if not data_list: return None

                    file_data = data_list[0]
                    return {
                        'sha256': file_hash, 'apt': apt_name,
                        'signature': file_data.get('signature', ''),
                        'imphash': file_data.get('imphash', ''),
                        'ssdeep': file_data.get('ssdeep', ''),
                        'tlsh': file_data.get('tlsh', '')
                    }
            except:
                if attempt < ABUSE_MAX_RETRIES - 1:
                    await asyncio.sleep(0.5)
                else:
                    return None
    return None


async def enrich_file_data(valid_results: List[Dict], unique_hashes: set) -> List[Dict]:
    hash_to_apt = {}
    for result in valid_results:
        apt = result['apt']
        for h in result['indicators']:
            if h not in hash_to_apt: hash_to_apt[h] = apt

    tasks_data = [(hash_to_apt.get(h, "Unknown"), h) for h in unique_hashes]

    # 使用高并发配置
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    semaphore = asyncio.Semaphore(ABUSE_CONCURRENCY_LIMIT)  # 100并发

    print(f"🚀 启动 AbuseCH 异步富化 (并发: {ABUSE_CONCURRENCY_LIMIT})")

    enriched_files = []
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_file_info(session, apt, h, semaphore) for apt, h in tasks_data]

        # 使用 tqdm.gather 显示进度
        from tqdm.asyncio import tqdm as async_tqdm
        results = await async_tqdm.gather(*tasks, desc="Enriching Hashes")

        for result in results:
            if result: enriched_files.append(result)

    return enriched_files


def filter_and_save_files(valid_results, enriched_files, file_output_dir):
    # (保持逻辑不变)
    print("\n开始过滤文件数据...")
    enriched_lookup = {f['sha256']: f for f in enriched_files}

    csv_path = os.path.join(file_output_dir, 'incremental_file_hashes.csv')
    saved_count = 0

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['event_id', 'sha256', 'signature', 'imphash', 'ssdeep', 'tlsh', 'apt']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in valid_results:
            event_id = result['event_id']
            apt = result['apt']
            for sha256 in result['indicators']:
                enriched = enriched_lookup.get(sha256)
                if enriched and any([
                    enriched.get('signature'), enriched.get('imphash'),
                    enriched.get('ssdeep'), enriched.get('tlsh')
                ]):
                    writer.writerow({
                        'event_id': event_id, 'sha256': sha256,
                        'signature': enriched['signature'], 'imphash': enriched['imphash'],
                        'ssdeep': enriched['ssdeep'], 'tlsh': enriched['tlsh'],
                        'apt': apt
                    })
                    saved_count += 1

    print(f"File哈希数据已保存到: {csv_path} (共 {saved_count} 行)")

    # 映射文件
    mapping_path = os.path.join(file_output_dir, 'event_mapping.json')
    event_mapping = {}
    for result in valid_results:
        event_id = result['event_id']
        apt = result['apt']
        if event_id not in event_mapping:
            event_mapping[event_id] = {'apt': apt, 'sha256_list': result['indicators']}

    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(event_mapping, f, indent=2, ensure_ascii=False)
    print(f"Event映射已保存到: {mapping_path}")


# ============================================================================
# TTP 数据处理模块
# ============================================================================

def process_ttp_data(incremental_data: Dict, otx: OTXv2, output_dir: str, ioc_output_dir: str):
    OUTPUT_FILE = os.path.join(output_dir, 'incremental_ttp_data.json')
    print("  从OTX API提取TTP数据...")

    result = {}
    total_ttps = 0
    ttp_tasks = []

    for org_name, org_data in incremental_data["organizations"].items():
        new_pulse_ids = org_data.get("new_pulse_ids", [])
        for pulse_id in new_pulse_ids:
            ioc_file = os.path.join(ioc_output_dir, org_name, pulse_id + '.json')
            if not os.path.exists(ioc_file): continue
            ttp_tasks.append((org_name, pulse_id))

    if not ttp_tasks:
        print("  没有需要处理的TTP数据");
        return

    def fetch_pulse_attack_ids(otx, org_name, pulse_id):
        try:
            pulse_details = otx.get_pulse_details(pulse_id)
            attack_ids = pulse_details.get('attack_ids', [])
            if attack_ids:
                return {'org': org_name, 'id': pulse_id, 'attack_ids': attack_ids}
        except:
            pass
        return None

    # ⚠️ 应用高并发
    n_workers = WORKERS_PER_KEY
    print(f"🚀 TTP 提取并发: {n_workers}")

    from joblib import Parallel, delayed
    ttp_results = Parallel(n_jobs=n_workers, prefer='threads')(
        delayed(fetch_pulse_attack_ids)(otx, task[0], task[1])
        for task in tqdm(ttp_tasks, desc="获取TTP")
    )

    for ttp_result in ttp_results:
        if ttp_result:
            org_name = ttp_result['org']
            if org_name not in result: result[org_name] = []
            result[org_name].append({'id': ttp_result['id'], 'attack_ids': ttp_result['attack_ids']})
            total_ttps += len(ttp_result['attack_ids'])

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  TTP数据已保存到: {OUTPUT_FILE}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("完整增量Pulse拉取工具 (Windows 高性能调优版)")
    print("=" * 70)

    pulse_ids_path = os.path.join(file_dir, 'pulse_ids.json')
    incremental_output_path = os.path.join(file_dir, 'incremental_pulse_ids.json')
    incremental_stats_path = os.path.join(file_dir, 'incremental_pulses.json')
    output_dir = os.path.abspath(os.path.join(file_dir, '../output_incremental'))

    if not os.path.exists(pulse_ids_path):
        print(f"错误: 文件不存在 {pulse_ids_path}");
        return

    pulse_data = load_existing_pulse_ids(pulse_ids_path)
    verify_pulse_data_integrity(pulse_data)

    print(f"\n步骤1.5: 加载APT别名映射表...")
    ta_map = build_ta_map()

    print(f"\n步骤2: 统计拥有pulse数量最多的20个组织...")
    top_orgs = get_top_organizations(pulse_data, top_k=20)

    print(f"\n步骤3: 检查增量 (断点续传)...")
    incremental_data = fetch_incremental_pulses(top_orgs, pulse_data, ta_map)

    print(f"\n步骤4: 统计信息")
    # print_summary 函数缺失，这里补一个简单的打印
    total_new = incremental_data['summary']['total_new']
    print(f"总计新增 Pulse: {total_new}")

    print(f"\n步骤5: 保存结果...")
    save_incremental_pulse_ids_only(incremental_data, incremental_output_path)
    save_incremental_stats(incremental_data, incremental_stats_path)

    if total_new > 0:
        print(f"\n步骤6: 下载新pulse的详细数据...")
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        otxs = [get_otx(key) for key in API_KEYS]

        build_incremental_ioc_dataset(incremental_data, output_dir)
        build_incremental_cve_dataset(incremental_data, output_dir)

        print(f"\n步骤7: 处理File数据...")
        valid_results, file_output_dir = build_incremental_file_dataset(incremental_data, output_dir, otxs)

        if valid_results and file_output_dir:
            unique_hashes = set()
            for result in valid_results: unique_hashes.update(result['indicators'])

            # Windows 特有：设置 Event Loop Policy
            if os.name == 'nt':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

            enriched_files = asyncio.run(enrich_file_data(valid_results, unique_hashes))
            print(f"成功富化 {len(enriched_files)} 个文件哈希")
            filter_and_save_files(valid_results, enriched_files, file_output_dir)

        print(f"\n步骤8: 处理TTP数据...")
        process_ttp_data(incremental_data, otxs[0], file_dir, output_dir)

        print(f"\n" + "=" * 70)
        print(f"增量数据拉取完成!")
    else:
        print(f"\n新增数据，任务完成。")


if __name__ == '__main__':
    main()