"""
APT 标签映射器模块

功能说明:
1. 构建 APT 组织名称的别名映射表
2. 将各种 APT 别名统一映射到官方名称
3. 从 MISP 事件中提取和标准化 APT 标签
4. 支持多个数据源(apt_names.csv + galaxy-labels.json)

使用场景:
- APT28 / Fancy Bear / Sofacy / Sednit → 统一标记为 APT28
- 确保数据集中 APT 标签的一致性
- 从威胁情报事件中自动提取 APT 归属
"""

import json
import os
from pathlib import Path

# 获取当前文件所在目录(用于定位数据文件)
file_dir = Path(os.path.dirname(os.path.realpath(__file__)))


# ============================================================================
# APT 标签映射器类
# ============================================================================

class AptLabelMapper():
    """
    APT 标签映射器

    功能:
    - 维护 APT 别名到官方名称的映射关系
    - 从 MISP 事件中提取 APT 标签
    - 标准化 APT 名称

    使用示例:
        mapper = AptLabelMapper()
        label = mapper.get_label(misp_event)  # 从事件提取 APT
        official = mapper.get_label_from_str('Fancy Bear')  # APT28
        all_labels = mapper.get_all_labels()  # 所有官方 APT 名称
    """

    def __init__(self):
        """
        初始化映射器

        在初始化时构建完整的 APT 别名映射表
        映射表格式: {别名: 官方名称}
        例如: {'FANCY BEAR': 'APT28', 'SOFACY': 'APT28'}
        """
        self.ta_map = build_ta_map()

    def get_label(self, misp_event):
        """
        从 MISP 事件中提取 APT 标签

        提取策略(按优先级):
        1. 优先从事件的 Tag 中查找 threat-actor 标签
        2. 如果没有标签,尝试从事件标题中匹配 APT 名称
        3. 如果都找不到,返回 None

        参数:
            misp_event: MISP 事件 JSON 对象

        返回:
            str: 标准化的 APT 官方名称,或 None

        示例:
            event = {
                'Event': {
                    'Tag': [
                        {'name': 'misp-galaxy:threat-actor="Fancy Bear"'}
                    ],
                    'info': 'APT28 Campaign'
                }
            }
            label = mapper.get_label(event)  # 返回 'APT28'
        """
        # ========== 策略 1: 从标签中提取 ==========
        # MISP 使用 galaxy 标签标记威胁行为者
        for tag in misp_event['Event'].get('Tag', []):
            name = tag['name']

            # 检查是否是威胁行为者标签
            if 'threat-actor' in name:
                has_apt = True

                # 清理标签格式
                # 'misp-galaxy:threat-actor="Fancy Bear"' → 'FANCY BEAR'
                k = name.replace(
                    'misp-galaxy:threat-actor=', ''  # 移除前缀
                ).replace('"', '').upper()  # 移除引号并转大写

                # 在映射表中查找
                if k in self.ta_map:
                    return self.ta_map[k]

        # ========== 策略 2: 从事件标题中匹配 ==========
        # 有些事件没有标签,但标题中包含 APT 名称
        # 例如: "APT28 targets government networks"
        title = misp_event['Event']['info']

        # 遍历所有已知的 APT 别名
        for apt in self.ta_map.keys():
            if apt in title.upper():
                # 找到匹配的别名,返回对应的官方名称
                return self.ta_map[apt]

        # ========== 无法识别 APT ==========
        # 如果无法从标签或标题中推断 APT,返回 None
        return None

    def get_label_from_str(self, label):
        """
        将 APT 别名转换为官方名称

        参数:
            label: APT 别名字符串

        返回:
            str: 官方标准名称,如果不在映射表中则返回原值

        示例:
            mapper.get_label_from_str('Fancy Bear')  # 返回 'APT28'
            mapper.get_label_from_str('Unknown')     # 返回 'Unknown'
        """
        return self.ta_map.get(label, label)

    def get_all_labels(self):
        """
        获取所有唯一的官方 APT 标签

        返回:
            list: 去重后的官方 APT 名称列表

        示例:
            ['APT28', 'APT29', 'APT1', ...]
        """
        # 从映射表的值中提取唯一的官方名称
        # 因为多个别名可能映射到同一个官方名称
        return list(set(list(self.ta_map.values())))


# ============================================================================
# 核心映射表构建函数
# ============================================================================

def build_ta_map():
    """
    构建 APT 威胁行为者的别名映射表

    数据源(按优先级合并):
    1. apt_names.csv - 手工整理的 APT 别名表
    2. galaxy-labels.json - MISP galaxy 威胁行为者数据库

    构建策略:
    - 建立 {别名: 官方名称} 的映射关系
    - 两个数据源的名称如果冲突,以第一个出现的为准
    - 所有名称转为大写以实现不区分大小写匹配

    缓存机制:
    - 首次运行时构建映射表并保存到 label_map.json
    - 后续运行直接加载缓存文件,提升性能

    返回:
        dict: {别名: 官方名称} 映射字典

    示例输出:
        {
            'APT28': 'APT28',
            'FANCY BEAR': 'APT28',
            'SOFACY': 'APT28',
            'SEDNIT': 'APT28',
            ...
        }
    """
    # 缓存文件路径
    outf = file_dir / 'label_map.json'
    ta_map = dict()

    # ========== 检查缓存 ==========
    # 如果映射表已经构建过,直接加载
    if os.path.exists(outf):
        with open(outf, 'r') as f:
            lmap = json.load(f)
        return lmap

    # ========== 数据源 1: apt_names.csv ==========
    # CSV 格式: Official Name, Confidence, Type, Country, synonym1, synonym2, ...
    # 示例: APT28,high,state-sponsored,Russia,Fancy Bear,Sofacy,Sednit
    with open(file_dir / 'apt_names.csv', 'r') as f:
        ta_names = f.read()

    # 跳过第一行标题,处理每一行数据
    for line in ta_names.split('\n')[1:]:
        if not line.strip():  # 跳过空行
            continue

        tokens = line.split(',')

        # 第一列是官方名称,作为映射的目标值
        value = tokens[0].upper()

        # 官方名称映射到自己
        ta_map[value] = value

        # 处理同义词(从第 5 列开始)
        # tokens[0-3] 是固定字段,tokens[4:] 是同义词列表
        for synonym in tokens[4:]:
            if synonym:  # 过滤空字符串
                # 所有同义词都映射到官方名称
                ta_map[synonym.upper()] = value

    # ========== 数据源 2: galaxy-labels.json ==========
    # MISP Galaxy 威胁行为者数据库
    # 包含更详细的 APT 信息和别名
    with open(file_dir / 'galaxy-labels.json', 'r', encoding='utf-8') as f:
        tas = json.load(f)

    # 处理每个威胁行为者
    for ta in tas['values']:
        # 主要名称
        k = ta['value'].upper()

        # 获取所有同义词
        aliases = ta.get('meta', dict()).get('synonyms', [])

        # 确定官方名称(需要考虑数据源优先级)
        official = k

        # ===== 策略 1: 检查主名称是否已在映射表中 =====
        if k in ta_map:
            # 如果主名称已存在,使用现有的官方名称
            # 这确保了 apt_names.csv 的优先级
            official = ta_map[k]
        else:
            # ===== 策略 2: 检查同义词是否已在映射表中 =====
            # 尝试找到两个数据源都认可的官方名称
            for alias in aliases:
                alias = alias.upper()
                if alias in ta_map:
                    # 找到匹配,使用该官方名称
                    official = ta_map[alias]
                    break

        # 将主名称和所有同义词都映射到确定的官方名称
        for ta_name in aliases + [k]:
            ta_map[ta_name.upper()] = official

    # ========== 保存映射表到缓存 ==========
    # 避免下次运行时重复构建
    with open(outf, 'w') as f:
        json.dump(ta_map, f, indent=2)

    return ta_map


# ============================================================================
# 独立函数(向后兼容)
# ============================================================================

def get_label(misp_event):
    """
    从 MISP 事件中提取 APT 标签(独立函数版本)

    这是一个便捷函数,内部使用 build_ta_map() 构建映射表
    如果需要多次调用,建议使用 AptLabelMapper 类以避免重复构建映射表

    参数:
        misp_event: MISP 事件 JSON 对象

    返回:
        str: APT 官方名称,或 None

    注意:
        每次调用都会加载映射表,频繁调用建议使用类版本
    """
    # 构建或加载映射表
    ta_map = build_ta_map()

    # ========== 策略 1: 从标签提取 ==========
    for tag in misp_event['Event'].get('Tag', []):
        name = tag['name']

        if 'threat-actor' in name:
            has_apt = True

            # 清理标签格式
            k = name.replace(
                'misp-galaxy:threat-actor=', ''
            ).replace('"', '').upper()

            # 查找映射
            if k in ta_map:
                return ta_map[k]

    # ========== 策略 2: 从标题匹配 ==========
    title = misp_event['Event']['info']
    for apt in ta_map.keys():
        if apt in title.upper():
            return ta_map[apt]

    # ========== 无法识别 ==========
    return None


# ============================================================================
# 使用示例和测试
# ============================================================================
"""
# 示例 1: 使用类版本(推荐)
mapper = AptLabelMapper()

# 从 MISP 事件提取 APT
misp_event = {
    'Event': {
        'Tag': [
            {'name': 'misp-galaxy:threat-actor="Fancy Bear"'}
        ],
        'info': 'Russian APT campaign'
    }
}
label = mapper.get_label(misp_event)
print(f"提取的 APT: {label}")  # APT28

# 标准化 APT 名称
official = mapper.get_label_from_str('Sofacy')
print(f"官方名称: {official}")  # APT28

# 获取所有 APT 列表
all_apts = mapper.get_all_labels()
print(f"共 {len(all_apts)} 个 APT 组织")

# 示例 2: 使用独立函数(简单场景)
label = get_label(misp_event)

# 示例 3: 直接构建映射表
ta_map = build_ta_map()
print(f"Fancy Bear 的官方名称: {ta_map.get('FANCY BEAR')}")
print(f"Sofacy 的官方名称: {ta_map.get('SOFACY')}")
print(f"共 {len(ta_map)} 个别名映射")
"""


# ============================================================================
# 映射表统计分析
# ============================================================================

def analyze_mapping():
    """
    分析映射表的统计信息(调试和验证用)
    """
    ta_map = build_ta_map()

    print("\n" + "=" * 70)
    print("APT 映射表统计分析")
    print("=" * 70)

    # 统计官方名称数量
    official_names = set(ta_map.values())
    print(f"\n📊 基本统计:")
    print(f"  - 总别名数: {len(ta_map)}")
    print(f"  - 官方 APT 数: {len(official_names)}")
    print(f"  - 平均每个 APT 的别名数: {len(ta_map) / len(official_names):.1f}")

    # 找出别名最多的 APT
    from collections import Counter
    apt_counts = Counter(ta_map.values())

    print(f"\n🏆 别名最多的 TOP 10 APT:")
    for apt, count in apt_counts.most_common(10):
        print(f"  {apt}: {count} 个别名")

    # 显示一些示例映射
    print(f"\n📝 映射示例:")
    examples = list(ta_map.items())[:10]
    for alias, official in examples:
        if alias != official:
            print(f"  {alias} → {official}")


# 如果直接运行此文件,执行分析
if __name__ == "__main__":
    analyze_mapping()