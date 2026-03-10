from collections import defaultdict
import json
import os
import time
from random import shuffle

from joblib import Parallel, delayed
from OTXv2 import OTXv2, RetryError, NotFound
from tqdm import tqdm

# 假设这些是你本地的模块，保持引用不变
from build_dataset.enrich import enrich
from build_dataset.label_mapper.apt_label_mapper import build_ta_map

# 每个 API 密钥支持的并发工作线程数
WORKERS_PER_KEY = 10

# 单个线程最多处理的 IOC 数量上限
MAX_JOB_SIZE = 1000

# 我们关注的 IOC 类型(忽略文件哈希)
iocs_we_want = ['IPv4', 'IPv6', 'domain', 'hostname', 'URL']

# 获取当前文件所在目录
file_dir = os.path.dirname(os.path.realpath(__file__))

# ==========================================
# 修改 1: 配置多个 API Key
# ==========================================
API_KEYS = [
    "0534b7f0120a3d5d7605393c33e975773e9c12eb91666816e8870f7697176a62",
    "b2a8ad4dc6394583bd2bb854066ade53ad5e485823eac8c11426758be18f10fd",
    "249cb3d557f711254d16e59ee1e8fa60c3173cb179c32a977ec8ddfc85dbc037"
]

def get_otx(api_key):
    """根据传入的 key 返回一个 OTX 客户端实例"""
    return OTXv2(api_key)

def get_otx_pulse_ids():
    """
    获取各个 APT 组织关联的 Pulse IDs 并保存到文件。
    只需要一个客户端即可完成此任务，默认使用列表中的第一个 Key。
    """
    print("正在通过 API 获取 Pulse IDs 列表...")
    # 使用第一个 Key 进行搜索即可
    otx = get_otx(API_KEYS[0])
    
    ta_map = build_ta_map()
    saved = defaultdict(set)
    apts = ta_map.keys()
    
    # 增加进度条显示
    for apt in tqdm(apts, desc="搜索 APT Pulse"):
        try:
            resp = otx.search_pulses('tag:"%s"' % apt, max_results=1)
            hits = min(resp['count'], 1000)
            if hits:
                resp = otx.search_pulses('tag:"%s"' % apt, max_results=hits)
                nresults = len(resp['results'])
                saved[ta_map[apt]].update(
                    [resp['results'][i]['id'] for i in range(nresults)]
                )
        except Exception as e:
            print(f"搜索 {apt} 时出错: {e}")
            continue

    # 保存文件
    save_path = os.path.join(file_dir, 'pulse_ids.json')
    with open(save_path, 'w') as f:
        f.write(json.dumps({k:list(v) for k,v in saved.items()}, indent=1))
    
    print(f"Pulse IDs 已保存至: {save_path}")
    return saved


def get_top_apts(db, topk):
    lens = [(k,len(v)) for k,v in db.items()]
    lens.sort(key=lambda x : x[1], reverse=True)
    return lens[:topk]


def get_overlapping_pulses(apt_to_ids):
    apt_to_ids = {k:set(v) for k,v in apt_to_ids.items()}
    overlaps = defaultdict(set)

    for k1,v1 in apt_to_ids.items():
        for k2,v2 in apt_to_ids.items():
            if k1 == k2:
                continue
            if (overlap := v1.intersection(v2)):
                [overlaps[uuid].update([k1,k2]) for uuid in overlap]
    print("已过滤重叠 pulses")
    return list(overlaps.keys())


def build_list_of_pulse_ids(topk=25):
    """
    读取并过滤 Pulse IDs。
    修改 2: 如果文件不存在，先调用 get_otx_pulse_ids 生成文件。
    """
    pulse_file_path = os.path.join(file_dir, 'pulse_ids.json')
    
    # 检查文件是否存在
    if not os.path.exists(pulse_file_path):
        print(f"警告: {pulse_file_path} 不存在，正在尝试重新获取...")
        get_otx_pulse_ids()

    with open(pulse_file_path, 'r') as f:
        apt_to_ids = json.load(f)

    ignore = get_overlapping_pulses(apt_to_ids)
    filtered = dict()
    for k, v in apt_to_ids.items():
        filtered[k] = [v_ for v_ in v if v_ not in ignore]

    topk_names = [x[0] for x in get_top_apts(filtered, topk)]
    pulse_dict = {k: filtered[k] for k in topk_names}

    output_file = os.path.join(file_dir, 'filtered_pulse_ids.json')
    with open(output_file, 'w') as f:
        json.dump(pulse_dict, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Top {topk} APT 组织及其 Pulse 数量统计:")
    print("=" * 60)
    total_pulses = 0
    for apt_name, pulse_ids in pulse_dict.items():
        pulse_count = len(pulse_ids)
        total_pulses += pulse_count
        print(f"{apt_name:30s}: {pulse_count:4d} pulses")
    print("=" * 60)
    print(f"总计: {len(pulse_dict)} 个组织, {total_pulses} 个 pulses")
    print(f"结果已保存到: {output_file}")
    print("=" * 60)

    return pulse_dict


def get_iocs(otx, iocs, pulse_id, apt, message=''):
    """获取并丰富化一组 IOC 的详细信息（多线程并行）"""
    from joblib import Parallel, delayed

    if not iocs:
        return dict(event_id=pulse_id, label=apt, iocs=[])

    # 显示处理信息
    print(f"\r{message} 富化 {len(iocs)} 个 IOCs...", end='', flush=True)

    # 多线程并行富化 IOC
    def enrich_single(ioc_tuple):
        try:
            return enrich(otx, *ioc_tuple)
        except Exception as e:
            # 遇到限流时等待
            error_msg = str(e)
            if '429' in error_msg or 'rate limit' in error_msg.lower():
                time.sleep(2)
            return None

    # 使用多线程并行处理（当前进程内）
    enriched_iocs = Parallel(n_jobs=10, prefer='threads', require='sharedmem')(
        delayed(enrich_single)(ioc) for ioc in iocs
    )

    # 过滤掉失败的结果
    enriched_iocs = [ioc for ioc in enriched_iocs if ioc is not None]

    # 完成后显示结果
    print(f"\r{message} 完成: {len(enriched_iocs)}/{len(iocs)} 个 IOCs", flush=True)

    return dict(event_id=pulse_id, label=apt, iocs=enriched_iocs)


def get_ioc_job(otx, ioc, message, idx, tot_iocs):
    """单个 IOC 的处理任务"""
    try:
        enriched = enrich(otx, *ioc)
        print(f'\r{message}: {idx}/{tot_iocs}', end='')
        return enriched if enriched else None
    except Exception as e:
        # 遇到限流时等待
        if '429' in str(e) or 'rate limit' in str(e).lower():
            time.sleep(2)
        return None


def thread_job(otx, event, apt, message, out_dir):
    """
    单个威胁情报的完整处理流程
    """
    out_f = os.path.join(out_dir, event) + '.json'

    if os.path.exists(out_f):
        return

    st = time.time()
    try:
        iocs = otx.get_pulse_indicators(event, include_inactive=True)
    except NotFound:
        print(f"\n{message} 威胁情报 {event} 未找到，跳过")
        return
    except RetryError:
        time.sleep(5)
        try:
            iocs = otx.get_pulse_indicators(event, include_inactive=True)
        except RetryError:
            print(f"\n{message} 威胁情报 {event} 获取失败，跳过")
            return
    except Exception as e:
        print(f"\n{message} 未知错误: {e}")
        return

    elapsed = time.time()-st
    # 去掉延迟，加快处理速度
    # time.sleep(max(0.5-elapsed, 0))

    # 获取元数据
    st = time.time()
    try:
        deets = otx.get_pulse_details(event)
    except Exception:
        # 如果获取详情失败，给一个空字典，不阻断流程
        deets = dict()

    deets = {
        k:deets.get(k, '')
        for k in ['name', 'description', 'tags']
    }

    # 过滤 IOC 类型
    to_fetch = []
    if iocs:
        for ioc in iocs:
            if (ioc_type:=ioc.get('type')) in iocs_we_want:
                to_fetch.append((ioc.get('indicator'), ioc_type))

    if not to_fetch:
        # print(f"\n{message} 威胁情报 {event} 没有有效的 IOC，跳过保存")
        return

    if len(to_fetch) > MAX_JOB_SIZE:
        print(f"\n{message} 威胁情报 {event} IOC 数量 ({len(to_fetch)}) 超过 {MAX_JOB_SIZE}，跳过")
        return

    elapsed = time.time()-st
    # 去掉延迟，加快处理速度
    # time.sleep(max(0.5-elapsed, 0))

    # 丰富化 IOC
    blob = get_iocs(otx, to_fetch, event, apt, message=message)
    blob['details'] = deets

    if blob['iocs']:
        with open(out_f, 'w', encoding='utf-8') as f:
            json.dump(blob, f, indent=2, ensure_ascii=False)
        # print(f"\n{message} 保存成功: {len(blob['iocs'])} 个 IOCs")
    else:
        print(f"\n{message} 威胁情报 {event} 所有 IOC 丰富化失败，跳过保存")


def fmt_time(elapsed):
    min = int(elapsed / 60)
    sec = int(elapsed % 60)
    return f'{min}m.{sec}s'


def inter_thread_job(otxs, jobs):
    """
    交叉线程处理模式 (Inter-thread processing)
    """
    last_used = 0
    n_keys = len(otxs)
    tot_events = len(jobs)

    for i, (event, apt, out_dir) in enumerate(jobs):
        st = time.time()
        out_f = os.path.join(out_dir, event) + '.json'

        if os.path.exists(out_f):
            continue
            
        # 轮询获取 IOC 列表
        try:
            current_otx = otxs[last_used % n_keys]
            iocs = current_otx.get_pulse_indicators(event, include_inactive=True)
            last_used += 1
        except (RetryError, NotFound):
            time.sleep(5)
            try:
                # 换一个 Key 再试一次
                last_used += 1
                current_otx = otxs[last_used % n_keys]
                iocs = current_otx.get_pulse_indicators(event, include_inactive=True)
            except Exception:
                print(f"\n({i}/{tot_events}) 威胁情报 {event} 获取失败，跳过")
                continue

        # 获取详情
        st = time.time()
        try:
            current_otx = otxs[last_used % n_keys]
            deets = current_otx.get_pulse_details(event)
            last_used += 1
        except Exception:
            deets = dict()

        deets = {
            k:deets.get(k, '')
            for k in ['name', 'description', 'tags']
        }
        
        to_fetch = []
        if iocs:
            for ioc in iocs:
                if (ioc_type:=ioc.get('type')) in iocs_we_want:
                    to_fetch.append((ioc.get('indicator'), ioc_type))
        
        if not to_fetch:
            continue
            
        blob = dict(event_id=event, label=apt, details=deets)
        
        # 并行处理 IOC，利用所有 Key 的 worker
        # n_jobs = API Key 数量 * 每个 Key 的 Worker 数
        iocs_result = Parallel(
            prefer='threads',
            n_jobs=len(otxs) * WORKERS_PER_KEY,
            batch_size=len(otxs)
        )(
            delayed(get_ioc_job)(
                otxs[(last_used + j) % n_keys], # 轮询分配每个 IOC 给不同的 Key
                ioc,
                f'({i}/{tot_events})',
                j + 1,
                len(to_fetch)
            ) for j, ioc in enumerate(to_fetch)
        )

        last_used += len(to_fetch)
        
        valid_iocs = [ioc for ioc in iocs_result if ioc is not None]
        
        if valid_iocs:
            blob['iocs'] = valid_iocs
            with open(out_f, 'w', encoding='utf-8') as f:
                json.dump(blob, f, indent=2, ensure_ascii=False)
            print(f' 保存成功: {len(valid_iocs)} 个 IOCs, 耗时: {fmt_time(time.time()-st)}')
        else:
            print(f' 所有 IOC 丰富化失败，跳过保存')


def build_dataset(location, inter_thread=False):
    """构建完整的威胁情报数据集"""
    # 确保调用的是修复后的 build_list_of_pulse_ids，它会处理文件不存在的情况
    ids = build_list_of_pulse_ids()

    # ==========================================
    # 修改 3: 初始化所有 Key 的客户端
    # ==========================================
    otxs = [get_otx(key) for key in API_KEYS]
    print(f"已初始化 {len(otxs)} 个 API 客户端。")
    
    jobs = []
    for apt, events in ids.items():
        out_dir = os.path.join(location, apt)
        os.makedirs(out_dir, exist_ok=True)

        [
            jobs.append((event, apt, out_dir))
            for event in events
            # 只有不存在时才加入任务队列
            if not os.path.exists(os.path.join(out_dir, event) + '.json')
        ]

    shuffle(jobs)
    
    print(f"\n总共需要处理 {len(jobs)} 个威胁情报任务")

    if not inter_thread:
        # 普通模式：任务级并行
        # 使用 len(otxs) * WORKERS_PER_KEY 作为总线程数
        # otxs[i % len(otxs)] 实现客户端的轮询复用
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
    else:
        # 交叉模式：IOC级并行
        inter_thread_job(otxs, jobs)


if __name__ == '__main__':
    # 修正输出目录路径，避免相对路径问题
    output_dir = os.path.abspath(os.path.join(file_dir, '../output'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"开始构建数据集到: {output_dir}")
    build_dataset(output_dir)
    print("数据集构建完成！")