"""
将 incremental_ttp_data.json 转换为与 ttp_features_by_organization.json 相同的格式

输入:
- incremental_ttp_data.json: 简化格式 {org: [{id, attack_ids}]}
- enterprise-attack.json: MITRE ATT&CK 完整数据

输出:
- incremental_ttp_data_converted.json: 完整格式 {org: [{pulse_id, attack_ids, tactics}]}
"""
import json
from pathlib import Path
from collections import defaultdict

# 数据路径
SCRIPT_DIR = Path(__file__).parent
INCREMENTAL_TTP_FILE = SCRIPT_DIR / "incremental_ttp_data.json"
ENTERPRISE_ATTACK_FILE = Path(r"/\src\test_dataset\cti\attck_data\enterprise-attack.json")
OUTPUT_FILE = SCRIPT_DIR / "incremental_ttp_data_converted.json"


def load_mitre_data(enterprise_attack_file):
    """加载 MITRE ATT&CK 数据并构建映射"""
    print(f"[1/3] 加载 MITRE ATT&CK 数据: {enterprise_attack_file}")

    with open(enterprise_attack_file, 'r', encoding='utf-8') as f:
        mitre_data = json.load(f)

    # 构建 tactic_shortname -> tactic_id 映射
    # (kill_chain_phases 使用小写名称，需要映射到 TA000x 格式)
    tactic_shortname_to_id = {}
    tactic_details = {}  # {tactic_id: {name, description}}

    # 构建 technique -> tactics 映射 (使用正确的 tactic_id)
    technique_to_tactics = defaultdict(list)
    technique_details = {}  # {technique_id: {name, description}}

    for obj in mitre_data['objects']:
        obj_type = obj.get('type')

        # 处理战术 (x-mitre-tactic) - 先处理，因为技术需要引用
        if obj_type == 'x-mitre-tactic':
            external_refs = obj.get('external_references', [])
            tactic_id = None
            for ref in external_refs:
                if ref.get('source_name') == 'mitre-attack':
                    tactic_id = ref.get('external_id')
                    break

            if tactic_id:
                # 获取小写名称（x_mitre_shortname 或从 name 生成）
                tactic_shortname = obj.get('x_mitre_shortname')
                if not tactic_shortname:
                    tactic_shortname = obj.get('name', '').lower().replace(' ', '-').replace('(', '').replace(')', '')

                tactic_shortname_to_id[tactic_shortname] = tactic_id

                tactic_details[tactic_id] = {
                    'id': tactic_id,
                    'name': obj.get('name', ''),
                    'description': obj.get('description', '')
                }

    # 处理技术 (attack-pattern)
    for obj in mitre_data['objects']:
        obj_type = obj.get('type')

        if obj_type == 'attack-pattern':
            # 获取 technique ID (如 T1055)
            external_refs = obj.get('external_references', [])
            technique_id = None
            for ref in external_refs:
                if ref.get('source_name') == 'mitre-attack':
                    technique_id = ref.get('external_id')
                    break

            if technique_id:
                # 获取关联的战术，并转换为 TA000x 格式
                kill_chains = obj.get('kill_chain_phases', [])
                tactic_ids = []
                for kc in kill_chains:
                    if kc.get('kill_chain_name') == 'mitre-attack':
                        phase_name = kc.get('phase_name')  # 小写名称
                        # 映射到正确的战术ID
                        tactic_id = tactic_shortname_to_id.get(phase_name)
                        if tactic_id:
                            tactic_ids.append(tactic_id)
                        else:
                            # 如果找不到映射，直接使用 phase_name
                            tactic_ids.append(phase_name)

                technique_to_tactics[technique_id] = tactic_ids

                # 保存技术详情
                technique_details[technique_id] = {
                    'id': technique_id,
                    'name': obj.get('name', ''),
                    'description': obj.get('description', '')
                }

    print(f"  - 加载了 {len(technique_details)} 个技术")
    print(f"  - 加载了 {len(tactic_details)} 个战术")
    print(f"  - 构建了 {len(technique_to_tactics)} 个技术->战术映射")
    print(f"  - 战术ID映射示例: {list(tactic_shortname_to_id.items())[:5]}")

    return technique_to_tactics, technique_details, tactic_details


def convert_incremental_data(incremental_file, technique_to_tactics, technique_details, tactic_details):
    """转换增量数据格式"""
    print(f"\n[2/3] 转换增量数据: {incremental_file}")

    with open(incremental_file, 'r', encoding='utf-8') as f:
        incremental_data = json.load(f)

    converted_data = {}
    total_events = 0
    total_techniques = 0
    missing_techniques = set()

    for org_name, events in incremental_data.items():
        converted_events = []

        for event in events:
            event_id = event.get('id')
            attack_ids = event.get('attack_ids', [])

            if not event_id or not attack_ids:
                continue

            # 按战术分组技术
            tactics_dict = defaultdict(lambda: {'techniques': []})

            for tech_id in attack_ids:
                # 获取该技术关联的战术
                tactic_ids = technique_to_tactics.get(tech_id, [])

                if not tactic_ids:
                    missing_techniques.add(tech_id)
                    # 如果没有找到战术映射，跳过该技术
                    continue

                # 获取技术详情
                tech_detail = technique_details.get(tech_id, {
                    'id': tech_id,
                    'name': '',
                    'description': ''
                })

                # 添加到每个关联的战术
                for tactic_id in tactic_ids:
                    tactics_dict[tactic_id]['techniques'].append(tech_detail)

            # 转换为目标格式
            tactics_output = {}
            for tactic_id, tech_data in tactics_dict.items():
                # 获取战术详情
                tactic_detail = tactic_details.get(tactic_id, {
                    'id': tactic_id,
                    'name': tactic_id.replace('-', ' ').title(),
                    'description': ''
                })

                tactics_output[tactic_id] = {
                    'name': tactic_detail['name'],
                    'description': tactic_detail['description'],
                    'techniques': tech_data['techniques']
                }

            # 只保留有TTP的事件
            if tactics_output:
                converted_event = {
                    'pulse_id': event_id,
                    'attack_ids': attack_ids,
                    'tactics': tactics_output
                }

                converted_events.append(converted_event)
                total_events += 1
                total_techniques += len(attack_ids)

        if converted_events:
            converted_data[org_name] = converted_events
            print(f"  - {org_name}: {len(converted_events)} 个事件")

    print(f"\n  转换统计:")
    print(f"  - 总事件数: {total_events}")
    print(f"  - 总技术数: {total_techniques}")
    if missing_techniques:
        print(f"  - 未映射的技术数: {len(missing_techniques)}")
        print(f"  - 未映射技术示例: {list(missing_techniques)[:10]}")

    return converted_data


def save_converted_data(data, output_file):
    """保存转换后的数据"""
    print(f"\n[3/3] 保存转换后的数据: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  - 保存成功!")
    print(f"  - 文件大小: {output_file.stat().st_size / 1024:.2f} KB")


def main():
    print("=" * 80)
    print("增量 TTP 数据格式转换".center(80))
    print("=" * 80)

    # Step 1: 加载 MITRE 数据
    technique_to_tactics, technique_details, tactic_details = load_mitre_data(ENTERPRISE_ATTACK_FILE)

    # Step 2: 转换增量数据
    converted_data = convert_incremental_data(
        INCREMENTAL_TTP_FILE,
        technique_to_tactics,
        technique_details,
        tactic_details
    )

    # Step 3: 保存结果
    save_converted_data(converted_data, OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("转换完成!".center(80))
    print("=" * 80)

    # 验证格式
    print("\n[验证] 检查输出格式:")
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        verify_data = json.load(f)

    sample_org = list(verify_data.keys())[0]
    sample_event = verify_data[sample_org][0] if verify_data[sample_org] else {}

    print(f"  示例组织: {sample_org}")
    print(f"  示例事件 ID: {sample_event.get('pulse_id')}")
    print(f"  attack_ids 数量: {len(sample_event.get('attack_ids', []))}")
    print(f"  tactics 数量: {len(sample_event.get('tactics', {}))}")

    for tactic_id, tactic_data in list(sample_event.get('tactics', {}).items())[:3]:
        print(f"    - {tactic_id}: {tactic_data.get('name')} ({len(tactic_data.get('techniques', []))} 个技术)")


if __name__ == "__main__":
    main()
