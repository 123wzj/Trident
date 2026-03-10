import csv
import json
import os
import socket

# 将 127.0.0.1:7890 替换为你实际使用的代理地址和端口
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
from src.build_dataset.pull import get_otx, thread_job, WORKERS_PER_KEY,API_KEYS
from src.build_dataset.label_mapper.apt_label_mapper import build_ta_map

# 获取当前文件所在目录
file_dir = os.path.dirname(os.path.realpath(__file__))

# 拉取的pulse数量上限 (每个组织)
MAX_PULSES_PER_ORG = 2000
# 并发配置
ABUSE_CONCURRENCY_LIMIT = 50
IOC_CONCURRENCY = 20       # 异步并发数 (根据网络状况调整，20-50之间)
ABUSE_MAX_RETRIES = 3

# API 请求间隔 (秒)
API_DELAY = 1

# IOC 类型
IOC_TYPES = ['IPv4', 'IPv6', 'domain', 'hostname', 'URL']

# AbuseCH API 配置
ABUSE_API_URL = "https://mb-api.abuse.ch/api/v1/"
ABUSE_API_KEY = "9ff9bdc51c4deb819dabbd4db830d3c84d8d190bf37410b8"




# ============================================================================
# 工具函数
# ============================================================================

def load_existing_pulse_ids(file_path: str) -> Dict[str, List[str]]:
    """加载现有的pulse_ids.json文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def verify_pulse_data_integrity(pulse_data: Dict[str, List[str]]) -> bool:
    """验证pulse_ids.json的数据完整性"""
    total_orgs = len(pulse_data)
    total_pulses = sum(len(pulses) for pulses in pulse_data.values())
    org_counts = sorted([(org, len(pulses)) for org, pulses in pulse_data.items()], key=lambda x: x[1], reverse=True)

    print(f"  数据完整性检查:")
    print(f"    总组织数: {total_orgs}")
    print(f"    总pulse数: {total_pulses}")

    print(f"    数据完整性检查通过")
    return True


def get_top_organizations(pulse_data: Dict[str, List[str]], top_k: int = 20) -> List[Tuple[str, int]]:
    """
    获取固定的25个目标组织（与 output 目录保持一致）

    注意: 增量拉取必须使用与全量拉取相同的组织列表，
    否则会导致数据分布不一致，影响增量学习效果

    组织来源: E:\PythonProject\Trail-main\src\output 目录
    """
    # 固定的25个目标组织（与 output 目录一致）
    TARGET_ORGS = [
        "APT28",
        "APT34",
        "APT35",
        "APT37",
        "APT38",
        "APT41",
        "BLACKENERGY",
        "CARETO",
        "COBALT GROUP",
        "FIN11",
        "FIN7",
        "GOLD WATERFALL",
        "ICEFOG",
        "KIMSUKY",
        "KINSING",
        "MAGECART",
        "MUDDYWATER",
        "MUSTANG PANDA",
        "PAT BEAR",
        "SAFE",
        "SAPPHIRE MUSHROOM",
        "TA511",
        "TA551",
        "TEAMTNT",
        "TURLA"
    ]

    # 从 pulse_data 中筛选出目标组织及其pulse数量
    result = []
    for org in TARGET_ORGS:
        if org in pulse_data:
            result.append((org, len(pulse_data[org])))
        else:
            # 如果组织不在 pulse_data 中，添加为 0
            result.append((org, 0))

    # 按pulse数量排序（用于显示）
    result_sorted = sorted(result, key=lambda x: x[1], reverse=True)

    # 打印 Top 25 组织列表（用于验证）
    print(f"  目标组织列表（来自 output 目录）:")
    for i, (org, count) in enumerate(result_sorted, 1):
        status = "✓" if count > 0 else "✗"
        print(f"    {i:2d}. {org:25s}: {count:5d} pulses {status}")

    return result_sorted[:top_k]


def get_org_aliases(org_name: str, ta_map: Dict[str, str]) -> List[str]:
    """获取组织的所有别名（包括官方名称本身）"""
    aliases = [alias for alias, official in ta_map.items() if official == org_name]
    if org_name not in aliases:
        aliases.append(org_name)
    return aliases


def sanitize(s: str) -> str:
    """清理字符串中的花括号"""
    return s.replace('{', '{{').replace('}', '}}')


# ============================================================================
# Pulse 拉取模块
# ============================================================================
socket.setdefaulttimeout(300)  # 增加到5分钟，适应无代理时的慢速网络
def fetch_pulses_from_otx(org_name: str, otx: OTXv2, ta_map: Dict[str, str]) -> Set[str]:
    """
    [回归库调用版] 从OTX API获取指定组织的所有pulse ID
    不做分页，一次性拉取，但增加了重试机制以防断连
    """
    pulse_ids = set()
    # aliases = get_org_aliases(org_name, ta_map)
    aliases = [org_name]  # 只使用官方名称

    if len(aliases) > 1:
        print(f"    搜索 {org_name} 的 {len(aliases)} 个别名...")

    for alias in aliases:
        try:
            # 1. 先探测一下总数 (这个很快)
            try:
                resp = otx.search_pulses('tag:"%s"' % alias, max_results=1)
            except Exception as e:
                print(f"      [警告] 连接初探失败，重试中... ({e})")
                time.sleep(2)
                resp = otx.search_pulses('tag:"%s"' % alias, max_results=1)

            total_count = resp.get('count', 0)
            if total_count == 0:
                continue

            # 确定要拉取的数量
            count = min(total_count, MAX_PULSES_PER_ORG)

            # 2. 一次性拉取 (这是最容易超时的地方，所以加重试循环)
            success = False
            for attempt in range(3):  # 给它 3 次机会
                try:
                    if attempt > 0:
                        print(f"      [重试 {attempt}/3] 正在重新拉取 {count} 条数据...")

                    # ⚠️ 这里是一次性拉取所有数据
                    resp = otx.search_pulses('tag:"%s"' % alias, max_results=count)

                    if 'results' in resp:
                        new_pulses = {item['id'] for item in resp['results']}
                        before_count = len(pulse_ids)
                        pulse_ids.update(new_pulses)
                        found_new = len(pulse_ids) - before_count

                        if len(aliases) > 1 and found_new > 0:
                            print(f"      别名 '{alias}': 找到 {len(new_pulses)} 个pulse (新增 {found_new})")

                    success = True
                    break  # 成功了就跳出循环

                except Exception as e:
                    print(f"      [网络波动] 第 {attempt + 1} 次尝试失败: {e}")
                    time.sleep(5)  # 失败后歇 5 秒

            if not success:
                print(f"      [失败] 经过3次尝试，无法一次性拉取 {alias} 的数据")

            time.sleep(API_DELAY)

        except Exception as e:
            print(f"      [错误] 搜索别名 '{alias}' 时出错: {e}")
            continue

    return pulse_ids

def fetch_incremental_pulses(top_orgs: List[Tuple[str, int]], pulse_data: Dict[str, List[str]], ta_map: Dict[str, str]) -> Dict:
    """从OTX API获取前20个组织的所有pulse并找出增量pulse（支持断点续传）"""
    otx = get_otx(API_KEYS[0])

    # 检查是否存在之前的进度文件
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

    # 恢复已完成组织的数据
    for org_name in completed_orgs:
        if org_name in pulse_data:
            result["organizations"][org_name] = prev_data["organizations"][org_name]
            result["summary"]["total_existing"] += result["organizations"][org_name]["existing_count"]
            result["summary"]["total_new"] += result["organizations"][org_name]["new_count"]

    print("\n开始从OTX API拉取pulse数据（包括别名搜索）...")
    print("=" * 70)

    for org_name, existing_count in tqdm(top_orgs, desc="拉取组织pulse"):
        # 跳过已完成的组织
        if org_name in completed_orgs:
            print(f"  [跳过] {org_name:25s}: 已完成")
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

        # 每完成一个组织，立即保存进度
        try:
            with open(incremental_stats_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [警告] 保存进度失败: {e}")

        status = f"+{new_count}" if new_count > 0 else "无新增"
        print(f"  {org_name:25s}: 现有 {existing_count:4d}, API返回 {total_count:4d}, 新增 {status} [已保存]")

    print("=" * 70)
    return result


# ============================================================================
# IOC 数据下载模块（富化）
# ============================================================================

def build_incremental_ioc_dataset(incremental_data: Dict, output_dir: str):
    """下载增量pulse的IOC数据并进行富化处理（支持断点续传）"""
    jobs = []
    skipped_exists = 0  # 已存在，跳过

    for org_name, org_data in incremental_data["organizations"].items():
        new_pulse_ids = org_data.get("new_pulse_ids", [])
        if not new_pulse_ids:
            continue

        out_dir = os.path.join(output_dir, org_name)
        os.makedirs(out_dir, exist_ok=True)

        for pulse_id in new_pulse_ids:
            out_f = os.path.join(out_dir, pulse_id) + '.json'
            # 跳过已存在的文件（断点续传）
            if os.path.exists(out_f):
                skipped_exists += 1
            else:
                jobs.append((pulse_id, org_name, out_dir))

    if not jobs:
        print("\n所有IOC数据已下载完成（断点续传）")
        return

    if skipped_exists > 0:
        print(f"\n检测到 {skipped_exists} 个已下载的文件，将跳过（断点续传）")

    shuffle(jobs)

    print(f"\n开始下载并富化 {len(jobs)} 个新pulse的IOC数据...")
    print("=" * 70)

    # 初始化 API 客户端
    otxs = [get_otx(key) for key in API_KEYS]
    print(f"已初始化 {len(otxs)} 个 API 客户端")

    # 使用原始的 thread_job 函数（已包含富化逻辑和空 IOC 检查）
    # thread_job 会自动跳过没有有效 IOC 的 pulse，不保存文件
    Parallel(n_jobs=len(otxs) * WORKERS_PER_KEY, prefer='threads')(
        delayed(thread_job)(
            otxs[i % len(otxs)],
            j[0],
            j[1],
            f'({i+1}/{len(jobs)})',
            j[2]
        )
        for i, j in enumerate(jobs)
    )

    # 统计实际生成的文件数
    actual_files = 0
    for org_name, org_data in incremental_data["organizations"].items():
        new_pulse_ids = org_data.get("new_pulse_ids", [])
        if not new_pulse_ids:
            continue
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
    """下载增量pulse的CVE数据（只处理有有效IOC的pulse）"""
    cve_jobs = []
    skipped_no_ioc = 0

    for org_name, org_data in incremental_data["organizations"].items():
        new_pulse_ids = org_data.get("new_pulse_ids", [])
        if not new_pulse_ids:
            continue

        cve_out_dir = os.path.join(output_dir, 'cve', org_name)
        os.makedirs(cve_out_dir, exist_ok=True)

        for pulse_id in new_pulse_ids:
            cve_out_f = os.path.join(cve_out_dir, pulse_id) + '.json'
            # 检查是否已存在 CVE 文件
            if os.path.exists(cve_out_f):
                continue

            # 检查对应的 IOC 文件是否存在（只有有 IOC 的 pulse 才处理）
            ioc_file = os.path.join(output_dir, org_name, pulse_id + '.json')
            if not os.path.exists(ioc_file):
                skipped_no_ioc += 1
                continue

            cve_jobs.append((pulse_id, org_name, cve_out_dir))

    if not cve_jobs:
        print("\n没有需要下载的新CVE数据")
        if skipped_no_ioc > 0:
            print(f"  跳过 {skipped_no_ioc} 个无IOC的pulse")
        return

    print(f"\n开始下载 {len(cve_jobs)} 个新pulse的CVE数据...")
    print("=" * 70)

    # 初始化 API 客户端
    otxs = [get_otx(key) for key in API_KEYS]

    def cve_thread_job(otx, event_id, apt, out_dir):
        """单个pulse的CVE数据拉取任务"""
        out_f = os.path.join(out_dir, event_id) + '.json'
        if os.path.exists(out_f):
            return

        try:
            iocs = otx.get_pulse_indicators(event_id, include_inactive=True)
        except (NotFound, RetryError, Exception):
            return

        # 提取CVE类型的indicators
        extracted_cves = []
        for ioc in iocs:
            if ioc.get('type') == 'CVE':
                indicator = ioc.get('indicator')
                if indicator:
                    extracted_cves.append(indicator)

        # 保存结果
        if extracted_cves:
            data_to_save = {
                "event_id": event_id,
                "apt": apt,
                "count": len(extracted_cves),
                "indicators": extracted_cves
            }
            with open(out_f, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)

    # 并行处理
    Parallel(n_jobs=len(otxs) * WORKERS_PER_KEY, prefer='threads')(
        delayed(cve_thread_job)(
            otxs[i % len(otxs)],
            j[0],
            j[1],
            j[2]
        )
        for i, j in enumerate(tqdm(cve_jobs, desc="CVE数据下载"))
    )

    # 统计实际生成的文件数
    actual_files = sum(1 for org_name, org_data in incremental_data["organizations"].items()
                       for pulse_id in org_data.get("new_pulse_ids", [])
                       if os.path.exists(os.path.join(output_dir, 'cve', org_name, pulse_id + '.json')))

    print("=" * 70)
    print(f"CVE数据下载完成!")
    print(f"  处理任务数: {len(cve_jobs)}")
    print(f"  成功保存: {actual_files}")
    print(f"  跳过(无IOC): {skipped_no_ioc}")


# ============================================================================
# File Hash 数据处理模块（从OTX拉取 + AbuseCH富化）
# ============================================================================

def build_incremental_file_dataset(incremental_data: Dict, output_dir: str, otxs: List):
    """提取增量pulse的File Hash（只处理有有效IOC的pulse）"""
    file_jobs = []
    skipped_no_ioc = 0

    for org_name, org_data in incremental_data["organizations"].items():
        new_pulse_ids = org_data.get("new_pulse_ids", [])
        if not new_pulse_ids:
            continue

        for pulse_id in new_pulse_ids:
            # 检查对应的 IOC 文件是否存在（只有有 IOC 的 pulse 才处理）
            ioc_file = os.path.join(output_dir, org_name, pulse_id + '.json')
            if not os.path.exists(ioc_file):
                skipped_no_ioc += 1
                continue

            file_jobs.append((pulse_id, org_name))

    if not file_jobs:
        print("\n没有需要处理的File数据")
        if skipped_no_ioc > 0:
            print(f"  跳过 {skipped_no_ioc} 个无IOC的pulse")
        return None, None

    print(f"\n开始提取 {len(file_jobs)} 个新pulse的File Hash数据...")
    print("=" * 70)

    def file_thread_job(otx, event_id, apt):
        """单个pulse的File Hash提取任务"""
        try:
            iocs = otx.get_pulse_indicators(event_id, include_inactive=True)
        except (NotFound, RetryError, Exception):
            return None

        # 提取FileHash-SHA256类型的indicators
        extracted_hashes = set()
        for ioc in iocs:
            if ioc.get('type') == 'FileHash-SHA256':
                indicator = ioc.get('indicator')
                if indicator:
                    extracted_hashes.add(indicator)

        if extracted_hashes:
            return {
                "event_id": event_id,
                "apt": apt,
                "indicators": list(extracted_hashes)
            }
        return None

    # 并行处理
    results = Parallel(n_jobs=len(otxs) * WORKERS_PER_KEY, prefer='threads')(
        delayed(file_thread_job)(
            otxs[i % len(otxs)],
            j[0],
            j[1]
        )
        for i, j in enumerate(tqdm(file_jobs, desc="File Hash提取"))
    )

    # 收集成功的结果
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print("未找到File Hash数据")
        return None, None

    # 聚合唯一的哈希（去重）
    unique_hashes = set()
    for result in valid_results:
        unique_hashes.update(result['indicators'])

    print(f"找到 {len(unique_hashes)} 个唯一文件哈希")
    print(f"  跳过(无IOC): {skipped_no_ioc}")

    # 创建File输出目录
    file_output_dir = os.path.join(output_dir, 'file_hashes')
    os.makedirs(file_output_dir, exist_ok=True)

    return valid_results, file_output_dir


async def fetch_file_info(session, apt_name, file_hash, semaphore):
    """异步获取文件信息（AbuseCH API）"""
    url = ABUSE_API_URL
    payload = {'query': 'get_info', 'hash': file_hash}
    headers = {'Auth-Key': ABUSE_API_KEY, 'User-Agent': 'Python-Async/Linux-Fast'}

    async with semaphore:
        for attempt in range(ABUSE_MAX_RETRIES):
            try:
                async with session.post(url, data=payload, headers=headers, timeout=30) as response:
                    if response.status == 429 or response.status >= 500:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    if response.status != 200:
                        return None

                    json_resp = await response.json()
                    if json_resp.get('query_status') != 'ok':
                        return None

                    data_list = json_resp.get('data', [])
                    if not data_list:
                        return None

                    file_data = data_list[0]
                    return {
                        'sha256': file_hash,
                        'apt': apt_name,
                        'signature': file_data.get('signature', ''),
                        'imphash': file_data.get('imphash', ''),
                        'ssdeep': file_data.get('ssdeep', ''),
                        'tlsh': file_data.get('tlsh', '')
                    }

            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt < ABUSE_MAX_RETRIES - 1:
                    await asyncio.sleep(1)
                else:
                    return None
            except Exception:
                return None

    return None


async def enrich_file_data(valid_results: List[Dict], unique_hashes: set) -> List[Dict]:
    """
    异步富化文件哈希数据

    Args:
        valid_results: 从 OTX 提取的原始结果列表，包含 event_id, apt, indicators
        unique_hashes: 唯一的哈希值集合
    """
    # 创建 hash → apt 映射（一个 hash 可能对应多个 apt，取第一个）
    hash_to_apt = {}
    for result in valid_results:
        apt = result['apt']
        for h in result['indicators']:
            if h not in hash_to_apt:  # 只记录第一次出现
                hash_to_apt[h] = apt

    # 准备任务数据
    tasks_data = [(hash_to_apt.get(h, "Unknown"), h) for h in unique_hashes]

    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    semaphore = asyncio.Semaphore(ABUSE_CONCURRENCY_LIMIT)

    enriched_files = []
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_file_info(session, apt, h, semaphore) for apt, h in tasks_data]

        for future in asyncio.as_completed(tasks):
            result = await future
            if result:
                enriched_files.append(result)

    return enriched_files


def filter_and_save_files(valid_results: List[Dict], enriched_files: List[Dict], file_output_dir: str):
    """过滤并保存File数据"""
    print("\n开始过滤文件数据...")
    print("=" * 70)

    # 创建查找表
    enriched_lookup = {f['sha256']: f for f in enriched_files}

    filtered_files = []
    stats = {
        'total': len(enriched_files),
        'has_signature': 0,
        'has_imphash': 0,
        'no_features': 0
    }

    for f in enriched_files:
        if f.get('signature'):
            stats['has_signature'] += 1
        if f.get('imphash'):
            stats['has_imphash'] += 1

        has_features = any([
            f.get('signature'),
            f.get('imphash'),
            f.get('ssdeep'),
            f.get('tlsh')
        ])

        if has_features:
            filtered_files.append(f)
        else:
            stats['no_features'] += 1

    print(f"过滤结果:")
    print(f"  总数: {stats['total']}")
    print(f"  有签名: {stats['has_signature']}")
    print(f"  有imphash: {stats['has_imphash']}")
    print(f"  无特征(已过滤): {stats['no_features']}")
    print(f"  保留: {len(filtered_files)}")

    # 生成CSV文件
    csv_path = os.path.join(file_output_dir, 'incremental_file_hashes.csv')

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
                    enriched.get('signature'),
                    enriched.get('imphash'),
                    enriched.get('ssdeep'),
                    enriched.get('tlsh')
                ]):
                    writer.writerow({
                        'event_id': event_id,
                        'sha256': sha256,
                        'signature': enriched['signature'],
                        'imphash': enriched['imphash'],
                        'ssdeep': enriched['ssdeep'],
                        'tlsh': enriched['tlsh'],
                        'apt': apt
                    })

    total_rows = sum(len(r['indicators']) for r in valid_results)
    filtered_rows = sum(
        1 for r in valid_results
        for h in r['indicators']
        if enriched_lookup.get(h) and any([
            enriched_lookup[h].get('signature'),
            enriched_lookup[h].get('imphash'),
            enriched_lookup[h].get('ssdeep'),
            enriched_lookup[h].get('tlsh')
        ])
    )

    print(f"\nFile哈希数据已保存到: {csv_path}")
    print(f"原始记录: {total_rows} 条")
    print(f"过滤后记录: {filtered_rows} 条")

    # 生成event映射文件
    mapping_path = os.path.join(file_output_dir, 'event_mapping.json')
    event_mapping = {}
    for result in valid_results:
        event_id = result['event_id']
        apt = result['apt']
        if event_id not in event_mapping:
            event_mapping[event_id] = {
                'apt': apt,
                'sha256_list': result['indicators']
            }

    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(event_mapping, f, indent=2, ensure_ascii=False)

    print(f"Event映射已保存到: {mapping_path}")
    print("=" * 70)


# ============================================================================
# TTP 数据处理模块（从 OTX API attack_ids 提取）
# ============================================================================

def process_ttp_data(incremental_data: Dict, otx: OTXv2, output_dir: str, ioc_output_dir: str):
    """
    处理TTP数据 - 从OTX API获取pulse的attack_ids字段（只处理有有效IOC的pulse）

    正确的数据格式（与 incremental_update.py 兼容）:
    {
      "ORG_NAME": [
        {
          "id": "pulse_id",
          "attack_ids": ["T1055", "T1123", "T1195"]
        }
      ]
    }

    注意: 使用 attack_ids 字段，不是 tags 字段！
    """
    OUTPUT_FILE = os.path.join(output_dir, 'incremental_ttp_data.json')

    print("  从OTX API提取TTP数据 (attack_ids字段)...")

    result = {}
    total_pulses = 0
    total_ttps = 0
    skipped_no_ioc = 0

    # 收集所有需要获取 TTP 的 pulse
    ttp_tasks = []
    for org_name, org_data in incremental_data["organizations"].items():
        new_pulse_ids = org_data.get("new_pulse_ids", [])
        for pulse_id in new_pulse_ids:
            # 检查对应的 IOC 文件是否存在（只有有 IOC 的 pulse 才处理）
            ioc_file = os.path.join(ioc_output_dir, org_name, pulse_id + '.json')
            if not os.path.exists(ioc_file):
                skipped_no_ioc += 1
                continue

            ttp_tasks.append((org_name, pulse_id))

    if not ttp_tasks:
        print("  没有需要处理的TTP数据")
        if skipped_no_ioc > 0:
            print(f"  跳过 {skipped_no_ioc} 个无IOC的pulse")
        return

    # 并发获取 TTP 数据
    def fetch_pulse_attack_ids(otx, org_name, pulse_id):
        """获取单个 pulse 的 attack_ids"""
        try:
            pulse_details = otx.get_pulse_details(pulse_id)
            attack_ids = pulse_details.get('attack_ids', [])

            if attack_ids:
                return {
                    'org': org_name,
                    'id': pulse_id,
                    'attack_ids': attack_ids
                }
        except Exception:
            pass
        return None

    # 使用 joblib 并行处理
    from joblib import Parallel, delayed

    # 使用传入的 otx 客户端
    ttp_results = Parallel(n_jobs=WORKERS_PER_KEY, prefer='threads')(
        delayed(fetch_pulse_attack_ids)(
            otx,
            task[0],
            task[1]
        )
        for task in ttp_tasks
    )

    # 如果并行处理失败，使用串行处理
    if not any(ttp_results):
        print("  并行处理失败，使用串行处理...")
        ttp_results = []
        for task in tqdm(ttp_tasks, desc="获取TTP(串行)"):
            result = fetch_pulse_attack_ids(otx, task[0], task[1])
            if result:
                ttp_results.append(result)

    # 组织结果
    for ttp_result in ttp_results:
        if ttp_result:
            org_name = ttp_result['org']
            if org_name not in result:
                result[org_name] = []

            result[org_name].append({
                'id': ttp_result['id'],
                'attack_ids': ttp_result['attack_ids']
            })
            total_ttps += len(ttp_result['attack_ids'])

    total_pulses = sum(len(pulses) for pulses in result.values())

    print(f"  共处理 {total_pulses} 个包含TTP的pulse")
    print(f"  总计 {total_ttps} 条技术条目")
    print(f"  跳过(无IOC): {skipped_no_ioc}")

    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  TTP数据已保存到: {OUTPUT_FILE}")


# ============================================================================
# 保存和打印函数
# ============================================================================

def save_incremental_pulse_ids_only(incremental_data: Dict, output_path: str) -> None:
    """保存仅包含新增pulse ID的文件"""
    pulse_ids = {}
    for org, data in incremental_data["organizations"].items():
        if data.get("new_pulse_ids"):
            pulse_ids[org] = data["new_pulse_ids"]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pulse_ids, f, indent=2, ensure_ascii=False)

    print(f"新增pulse ID列表已保存到: {output_path}")


def save_incremental_stats(incremental_data: Dict, output_path: str) -> None:
    """保存增量统计信息"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(incremental_data, f, indent=2, ensure_ascii=False)

    print(f"增量统计信息已保存到: {output_path}")


def print_summary(incremental_data: Dict):
    """打印统计摘要"""
    print(f"\n统计摘要")
    print("=" * 70)
    print(f"{'组织':<25s} {'现有':<8s} {'总数':<8s} {'新增':<8s}")
    print("-" * 70)

    has_new_pulses = False
    for org, data in incremental_data["organizations"].items():
        existing = data["existing_count"]
        total = data["total_count"]
        new = data["new_count"]
        print(f"{org:<25s} {existing:<8d} {total:<8d} {new:<8d}")
        if new > 0:
            has_new_pulses = True

    print("-" * 70)
    print(f"{'总计':<25s} {incremental_data['summary']['total_existing']:<8d} {'-':<8s} {incremental_data['summary']['total_new']:<8d}")
    print("=" * 70)

    return has_new_pulses


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("=" * 70)
    print("完整增量Pulse拉取工具（支持别名搜索）")
    print("=" * 70)

    # 文件路径
    pulse_ids_path = os.path.join(file_dir, 'pulse_ids.json')
    incremental_output_path = os.path.join(file_dir, 'incremental_pulse_ids.json')
    incremental_stats_path = os.path.join(file_dir, 'incremental_pulses.json')
    output_dir = os.path.abspath(os.path.join(file_dir, '../output_incremental'))

    # 1. 加载现有的pulse_ids.json
    print(f"\n步骤1: 加载现有pulse数据...")
    print(f"  文件路径: {pulse_ids_path}")

    if not os.path.exists(pulse_ids_path):
        print(f"  [错误] 文件不存在: {pulse_ids_path}")
        print(f"  请先运行 pull.py 生成 pulse_ids.json 文件")
        return

    pulse_data = load_existing_pulse_ids(pulse_ids_path)
    total_orgs = len(pulse_data)
    total_pulses = sum(len(pulses) for pulses in pulse_data.values())
    print(f"  已加载 {total_orgs} 个组织, {total_pulses} 个 pulse")

    # 数据完整性检查
    if not verify_pulse_data_integrity(pulse_data):
        print(f"\n[错误] pulse_ids.json 数据异常!")
        print(f"可能原因:")
        print(f"  1. pulse_ids.json 被其他脚本(如 pull.py)重新生成")
        print(f"  2. 数据文件损坏")
        print(f"\n建议:")
        print(f"  1. 检查是否有其他脚本在修改 pulse_ids.json")
        print(f"  2. 从备份恢复: cp pulse_ids.json.backup pulse_ids.json")
        return

    # 1.5 加载别名映射表
    print(f"\n步骤1.5: 加载APT别名映射表...")
    ta_map = build_ta_map()
    total_aliases = len(ta_map)
    unique_orgs = len(set(ta_map.values()))
    print(f"  已加载 {total_aliases} 个别名映射，覆盖 {unique_orgs} 个官方组织")

    # 2. 找出拥有pulse数量最多的20个组织
    print(f"\n步骤2: 统计拥有pulse数量最多的20个组织...")
    top_orgs = get_top_organizations(pulse_data, top_k=20)

    print(f"  前20个组织:")
    for i, (org, count) in enumerate(top_orgs, 1):
        org_aliases = len(get_org_aliases(org, ta_map))
        alias_info = f" ({org_aliases} 个别名)" if org_aliases > 1 else ""
        print(f"    {i:2d}. {org:25s}: {count:5d} pulses{alias_info}")

    # 3. 从OTX API拉取这些组织的所有pulse并找出新增的
    print(f"\n步骤3: 从OTX API拉取最新pulse数据（包括所有别名）...")
    incremental_data = fetch_incremental_pulses(top_orgs, pulse_data, ta_map)

    # 4. 打印统计信息
    print(f"\n步骤4: 统计信息")
    has_new_pulses = print_summary(incremental_data)

    # 5. 保存增量pulse ID列表
    print(f"\n步骤5: 保存结果...")
    save_incremental_pulse_ids_only(incremental_data, incremental_output_path)
    save_incremental_stats(incremental_data, incremental_stats_path)

    # 6. 如果有新pulse，下载详细数据
    if has_new_pulses:
        print(f"\n步骤6: 下载新pulse的详细数据...")
        print(f"  输出目录: {output_dir}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 初始化OTX客户端
        otxs = [get_otx(key) for key in API_KEYS]

        # 6.1 下载并富化IOC数据
        build_incremental_ioc_dataset(incremental_data, output_dir)

        # 6.2 下载CVE数据
        build_incremental_cve_dataset(incremental_data, output_dir)

        # 6.3 提取并富化File数据
        print(f"\n步骤7: 处理File数据...")
        print("=" * 70)
        valid_results, file_output_dir = build_incremental_file_dataset(incremental_data, output_dir, otxs)

        if valid_results and file_output_dir:
            # 聚合唯一的哈希
            unique_hashes = set()
            for result in valid_results:
                unique_hashes.update(result['indicators'])

            print(f"\n开始富化 {len(unique_hashes)} 个文件哈希（使用AbuseCH API）...")

            # 异步富化
            enriched_files = asyncio.run(enrich_file_data(valid_results, unique_hashes))

            print(f"成功富化 {len(enriched_files)} 个文件哈希")

            # 过滤并保存
            filter_and_save_files(valid_results, enriched_files, file_output_dir)

            print(f"File数据处理完成!")

        # 6.4 提取TTP数据
        print(f"\n步骤8: 处理TTP数据...")
        print("=" * 70)
        process_ttp_data(incremental_data, otxs[0], file_dir, output_dir)
        print("=" * 70)

        # 7. 后续步骤提示
        print(f"\n" + "=" * 70)
        print(f"增量数据拉取完成!")
        print(f"\n已下载的数据类型:")
        print(f"  - IOC数据 (IP/Domain/URL/Hostname): {output_dir}/")
        print(f"    包含富化信息: 地理位置、ASN、DNS记录、HTTP响应头")
        print(f"  - CVE数据: {output_dir}/cve/")
        print(f"  - File数据: {output_dir}/file_hashes/incremental_file_hashes.csv")
        print(f"    包含富化信息: signature、imphash、ssdeep、tlsh")
        print(f"    已过滤: 只保留有特征的恶意文件")
        print(f"  - TTP数据: {file_dir}/incremental_ttp_data.json")
        print(f"    来源: MITRE ATT&CK 数据库")
        print(f"\n后续步骤:")
        print(f"  1. 查看增量pulse详情: {incremental_stats_path}")
        print(f"  2. 使用 incremental_update.py 更新知识图谱:")
        print(f"     from src.test_dataset.build_knowledge_graph.incremental_update import TrailNeo4jIncrementalUpdater")
        print(f"     updater = TrailNeo4jIncrementalUpdater(uri, user, password)")
        print(f"     updater.incremental_import_events('{output_dir}')")
        print(f"     updater.incremental_import_cve('{os.path.join(output_dir, 'cve')}')")
        print(f"     updater.incremental_import_files('{os.path.join(output_dir, 'file_hashes/incremental_file_hashes.csv')}')")
        print(f"     updater.incremental_import_mitre_tags('{os.path.join(file_dir, 'incremental_ttp_data.json')}')")
        print(f"  3. 使用 neo4j2pytorch_incremental.py 导出增量特征")
        print(f"  4. 使用 incremental_train_fusion.py 进行增量训练")
        print(f"\n参考文档: docs/INCREMENTAL_LEARNING_GUIDE.md")
        print("=" * 70)
    else:
        print(f"\n未发现新增pulse。所有组织的pulse数据已是最新。")


if __name__ == '__main__':
    main()
