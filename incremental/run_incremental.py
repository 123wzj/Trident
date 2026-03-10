"""
增量学习完整测试流程

按照 neo4j2pytorch_incremental.py 的流程：
1. Step 1: 导出旧数据 (apt_kg_old.pt)
2. Step 2: 增量更新Neo4j数据库
3. Step 3: 导出更新后的数据 (apt_kg_updated.pt)
"""

import sys
import torch
from pathlib import Path
from datetime import datetime
from neo4j import GraphDatabase

# 配置
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j123"

# 数据路径
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "incremental_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 支持跨平台路径
PROJECT_ROOT = SCRIPT_DIR.parent.parent
INCREMENTAL_DATA_DIR = PROJECT_ROOT / "src" / "output_incremental"

print("=" * 80)
print("增量学习完整测试流程".center(80))
print("=" * 80)
print("=" * 80)


# ============================================================================
# Step 1: 导出旧数据（IOC + TTP）
# ============================================================================

def step1_export_old_data():
    """Step 1: 导出旧数据（IOC图数据 + TTP序列数据）"""
    print("\n" + "=" * 80)
    print("Step 1: 导出旧数据 (apt_kg_ioc_old.pt + apt_kg_ttp_old.pt)".center(80))
    print("=" * 80)

    try:
        # 导入双模型导出器
        from build_dataset.neo4jpytorch_embedding import ImprovedGraphExporter

        print(f"\n  [导出] 连接Neo4j并导出旧数据...")
        exporter = ImprovedGraphExporter(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            pwd=NEO4J_PASSWORD
        )

        # 导出双模型子图
        exporter.export_dual_subgraphs(output_dir=str(OUTPUT_DIR))
        exporter.close()

        # 重命名导出的文件为 _old 后缀
        ioc_old_path = OUTPUT_DIR / "apt_kg_ioc_old.pt"
        ttp_old_path = OUTPUT_DIR / "apt_kg_ttp_old.pt"

        ioc_src = OUTPUT_DIR / "apt_kg_ioc.pt"
        ttp_src = OUTPUT_DIR / "apt_kg_ttp.pt"

        if ioc_src.exists():
            import shutil
            shutil.copy(ioc_src, ioc_old_path)
            print(f"\n  [IOC] 旧数据已保存: {ioc_old_path}")

        if ttp_src.exists():
            import shutil
            shutil.copy(ttp_src, ttp_old_path)
            print(f"  [TTP] 旧数据已保存: {ttp_old_path}")

        # 验证导出
        if ioc_old_path.exists() and ttp_old_path.exists():
            print(f"\n  [验证] 导出成功!")

            # IOC数据
            ioc_data = torch.load(ioc_old_path, weights_only=False)
            print(f"    [IOC] 文件大小: {ioc_old_path.stat().st_size / 1024 / 1024:.2f} MB")
            print(f"    [IOC] 节点类型: {list(ioc_data.node_types)}")
            for node_type in ioc_data.node_types:
                num_nodes = ioc_data[node_type].num_nodes
                print(f"      - {node_type}: {num_nodes:,} 个节点")

            # TTP数据
            ttp_data = torch.load(ttp_old_path, weights_only=False)
            print(f"    [TTP] 文件大小: {ttp_old_path.stat().st_size / 1024 / 1024:.2f} MB")
            print(f"    [TTP] 序列数量: {len(ttp_data['causal_sequences'])}")
            print(f"    [TTP] 标签数量: {ttp_data['labels'].shape[0]}")
            print(f"    [TTP] 类别数: {ttp_data['num_classes']}")

            print(f"\n  [OK] Step 1 完成")
            return True
        else:
            print(f"\n  [ERROR] 导出文件未生成")
            return False

    except Exception as e:
        print(f"\n  [ERROR] Step 1 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Step 2: 增量更新Neo4j数据库
# ============================================================================

def step2_incremental_update():
    """Step 2: 增量更新Neo4j数据库（先导入TTP获取有效事件ID，只导入有TTP的事件）"""
    print("\n" + "=" * 80)
    print("Step 2: 增量更新Neo4j数据库".center(80))
    print("=" * 80)

    try:
        from incremental.incremental_update import TrailNeo4jIncrementalUpdater

        updater = TrailNeo4jIncrementalUpdater(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        # 检查数据库状态
        print(f"\n  [检查] 更新前数据库状态:")
        updater.check_database_health()

        # ============================================================================
        # Step 2.1: 先导入TTP数据，获取有TTP的事件ID列表
        # ============================================================================
        ttp_file = PROJECT_ROOT / "src" / "build_dataset" / "incremental_ttp_data_converted.json"
        valid_event_ids = None

        if ttp_file.exists():
            print(f"\n  [导入] 增量TTP数据: {ttp_file}")
            valid_event_ids = updater.incremental_import_ttp(str(ttp_file))
            print(f"    - 有效事件数: {len(valid_event_ids)}")
        else:
            print(f"  [!] TTP数据文件不存在: {ttp_file}")
            print(f"  [警告] 将导入所有事件（无TTP过滤）")

        # ============================================================================
        # Step 2.2: 增量导入IOC数据（只导入有TTP的事件）
        # ============================================================================
        ioc_dir = INCREMENTAL_DATA_DIR / "ioc"
        if ioc_dir.exists():
            print(f"\n  [导入] 增量IOC数据: {ioc_dir}")
            stats_ioc = updater.incremental_import_events(str(ioc_dir), valid_event_ids=valid_event_ids)
            print(f"    - 成功: {stats_ioc['success']}, 跳过: {stats_ioc['skipped']}, 失败: {stats_ioc['failed']}", end='')
            if stats_ioc.get('filtered', 0) > 0:
                print(f", 过滤(无TTP): {stats_ioc['filtered']}")
            else:
                print()
        else:
            print(f"  [!] IOC数据目录不存在: {ioc_dir}")

        # ============================================================================
        # Step 2.3: 增量导入CVE数据
        # ============================================================================
        cve_dir = INCREMENTAL_DATA_DIR / "cve"
        if cve_dir.exists():
            print(f"\n  [导入] 增量CVE数据: {cve_dir}")
            stats_cve = updater.incremental_import_cve(str(cve_dir), valid_event_ids=valid_event_ids)
        else:
            print(f"  [!] CVE数据目录不存在: {cve_dir}")

        # ============================================================================
        # Step 2.4: 增量导入文件数据
        # ============================================================================
        file_csv = INCREMENTAL_DATA_DIR / "file_hashes" / "incremental_file_hashes.csv"
        if file_csv.exists():
            print(f"\n  [导入] 增量文件数据: {file_csv}")
            stats_file = updater.incremental_import_files(str(file_csv), valid_event_ids=valid_event_ids)
        else:
            print(f"  [!] 文件数据不存在: {file_csv}")

        # 检查更新后状态
        print(f"\n  [检查] 更新后数据库状态:")
        updater.check_database_health()

        # 清理孤立的EVENT节点
        updater.cleanup_isolated_events()

        updater.close()

        print(f"\n  [OK] Step 2 完成: 数据库已更新（只导入有TTP的事件）")
        return True

    except Exception as e:
        print(f"\n  [ERROR] Step 2 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Step 3: 导出更新后的数据
# ============================================================================

def step3_export_updated_data():
    """Step 3: 导出更新后的数据（IOC图数据 + TTP序列数据）"""
    print("\n" + "=" * 80)
    print("Step 3: 导出更新后的数据 (apt_kg_ioc_updated.pt + apt_kg_ttp_updated.pt)".center(80))
    print("=" * 80)

    try:
        # ============================================================================
        # 检查数据库当前状态
        # ============================================================================
        print(f"\n  [检查] 数据库当前状态:")
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            # 总EVENT数
            total_events = session.run("MATCH (e:EVENT) RETURN count(e) as c").single()['c']
            # 有TTP的EVENT数
            ttp_events = session.run("MATCH (e:EVENT)-[:USES_TECHNIQUE]->(:Technique) RETURN count(DISTINCT e) as c").single()['c']
            # 各节点类型统计
            node_stats = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
            """).data()

            print(f"    - EVENT总数: {total_events:,}")
            print(f"    - 有TTP的EVENT: {ttp_events:,} ({ttp_events/total_events*100:.1f}%)")
            print(f"    - 各节点类型:")
            for stat in node_stats[:8]:
                print(f"        {stat['label']}: {stat['count']:,}")
        driver.close()

        from build_dataset.neo4jpytorch_embedding import ImprovedGraphExporter

        old_ioc_path = OUTPUT_DIR / "apt_kg_ioc_old.pt"
        old_ttp_path = OUTPUT_DIR / "apt_kg_ttp_old.pt"

        if not old_ioc_path.exists() or not old_ttp_path.exists():
            print(f"\n  [ERROR] 旧数据文件不存在")
            return False

        print(f"\n  [导出] 连接Neo4j并导出更新后的数据...")
        exporter = ImprovedGraphExporter(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            pwd=NEO4J_PASSWORD
        )

        # 导出双模型子图
        exporter.export_dual_subgraphs(output_dir=str(OUTPUT_DIR))
        exporter.close()

        # 重命名导出的文件为 _updated 后缀
        ioc_updated_path = OUTPUT_DIR / "apt_kg_ioc_updated.pt"
        ttp_updated_path = OUTPUT_DIR / "apt_kg_ttp_updated.pt"

        ioc_src = OUTPUT_DIR / "apt_kg_ioc.pt"
        ttp_src = OUTPUT_DIR / "apt_kg_ttp.pt"

        if ioc_src.exists():
            import shutil
            shutil.copy(ioc_src, ioc_updated_path)
            print(f"\n  [IOC] 更新后数据已保存: {ioc_updated_path}")

        if ttp_src.exists():
            import shutil
            shutil.copy(ttp_src, ttp_updated_path)
            print(f"  [TTP] 更新后数据已保存: {ttp_updated_path}")

        # 验证导出
        if ioc_updated_path.exists() and ttp_updated_path.exists():
            print(f"\n  [验证] 导出成功!")

            # IOC数据对比
            old_ioc_data = torch.load(old_ioc_path, weights_only=False)
            new_ioc_data = torch.load(ioc_updated_path, weights_only=False)

            print(f"\n  [对比] IOC数据变化:")
            for node_type in new_ioc_data.node_types:
                old_num = old_ioc_data[node_type].num_nodes if node_type in old_ioc_data.node_types else 0
                new_num = new_ioc_data[node_type].num_nodes
                delta = new_num - old_num
                if delta > 0:
                    print(f"    - {node_type}: {old_num:,} -> {new_num:,} (+{delta:,})")
                else:
                    print(f"    - {node_type}: {old_num:,} -> {new_num:,}")

            # TTP数据对比
            old_ttp_data = torch.load(old_ttp_path, weights_only=False)
            new_ttp_data = torch.load(ttp_updated_path, weights_only=False)

            print(f"\n  [对比] TTP数据变化:")
            print(f"    - 序列数量: {len(old_ttp_data['causal_sequences']):,} -> {len(new_ttp_data['causal_sequences']):,}")
            print(f"    - 标签数量: {old_ttp_data['labels'].shape[0]:,} -> {new_ttp_data['labels'].shape[0]:,}")

            print(f"\n  [OK] Step 3 完成")
            return True
        else:
            print(f"\n  [ERROR] 导出文件未生成")
            return False

    except Exception as e:
        print(f"\n  [ERROR] Step 3 失败: {e}")
        import traceback
        traceback.print_exc()
        return False



def filter_to_same_classes(data, target_classes):
    """过滤数据，只保留指定的类别集合（确保新旧数据使用相同的类别）

    Args:
        data: HeteroData IOC图数据
        target_classes: 目标类别名称列表（来自旧数据的Top-K类别）

    Returns:
        过滤后的数据
    """
    import torch
    import numpy as np

    print(f"\n  [类别过滤] 过滤到指定的 {len(target_classes)} 个类别...")

    # 获取当前数据的类别映射
    current_classes = data._apt_classes if hasattr(data, '_apt_classes') else []
    class_to_label = {name: idx for idx, name in enumerate(current_classes)}

    # 找出目标类别在当前数据中的标签索引
    target_labels = []
    found_classes = []
    missing_classes = []

    for class_name in target_classes:
        if class_name in class_to_label:
            target_labels.append(class_to_label[class_name])
            found_classes.append(class_name)
        else:
            missing_classes.append(class_name)

    if missing_classes:
        print(f"    [警告] 以下类别在新数据中不存在: {missing_classes}")
        print(f"    [警告] 将只保留找到的 {len(found_classes)} 个类别")

    if not found_classes:
        print(f"    [ERROR] 没有找到任何匹配的类别！")
        return data

    target_label_set = set(target_labels)

    print(f"    目标类别数: {len(target_classes)}")
    print(f"    实际找到: {len(found_classes)} 个类别")
    print(f"    保留的类别:")
    for label in target_labels:
        class_name = current_classes[label]
        # 统计该类别的样本数
        valid_idx = data['EVENT'].y != -1
        valid_y = data['EVENT'].y[valid_idx]
        count = torch.sum(valid_y == label).item()
        print(f"      {class_name:<30}: {count:>5} 个样本")

    # 过滤数据：只保留目标类别的样本
    filtered_mask = torch.zeros(len(data['EVENT'].y), dtype=torch.bool)
    for i, label in enumerate(data['EVENT'].y):
        if label != -1 and label.item() in target_label_set:
            filtered_mask[i] = True

    # 创建新的标签映射 (0..len(target_labels)-1)
    # 保持与旧数据相同的标签顺序
    old_to_new_label = {}
    for new_label, old_label in enumerate(target_labels):
        old_to_new_label[old_label] = new_label

    # 更新标签
    new_y = torch.full((len(data['EVENT'].y),), -1, dtype=torch.long)
    kept_samples = 0
    for i in range(len(data['EVENT'].y)):
        if filtered_mask[i]:
            old_label = data['EVENT'].y[i].item()
            new_y[i] = old_to_new_label[old_label]
            kept_samples += 1

    data['EVENT'].y = new_y

    # 更新类别名称（使用目标类别列表，只保留找到的）
    new_apt_classes = [cls for cls in target_classes if cls in found_classes]
    data._apt_classes = np.array(new_apt_classes)

    print(f"    过滤后样本数: {kept_samples}")
    print(f"    过滤后类别数: {len(new_apt_classes)}")
    if len(new_apt_classes) > 0:
        print(f"    平均每类: {kept_samples // len(new_apt_classes)} 个样本")

    return data


def filter_ttp_labels(ttp_data, ioc_data):
    """过滤TTP数据，使其与IOC数据的过滤完全一致
    参照 train_dual_or_fusion.py 的过滤逻辑
    """
    import torch

    # IOC数据的过滤掩码：y != -1 表示保留的样本
    ioc_valid_mask = ioc_data['EVENT'].y != -1

    # 获取IOC保留的样本索引
    valid_idx = torch.where(ioc_valid_mask)[0].numpy()

    print(f"    TTP原始长度: {len(ttp_data['labels'])}")
    print(f"    IOC原始长度: {len(ioc_data['EVENT'].y)}")
    print(f"    valid_idx范围: {valid_idx.min()}-{valid_idx.max()}, 大小: {len(valid_idx)}")

    # 获取新类别列表
    new_classes = ioc_data._apt_classes if hasattr(ioc_data, '_apt_classes') else []

    # 参照 train_dual_or_fusion.py：直接使用 valid_idx 过滤 TTP 数据
    sequences = ttp_data.get('causal_sequences', ttp_data.get('technique_sequences'))
    ttp_sequences = [sequences[i] for i in valid_idx]
    ttp_labels = ioc_data['EVENT'].y[valid_idx].clone()  # 直接使用IOC的标签

    # 创建过滤后的数据（与 train_dual_or_fusion.py 一致）
    filtered_ttp = {
        'causal_sequences' if 'causal_sequences' in ttp_data else 'technique_sequences': ttp_sequences,
        'labels': ttp_labels,
        'num_techniques': ttp_data.get('num_techniques', 369),
        'phase_sequences': [ttp_data['phase_sequences'][i] for i in valid_idx] if 'phase_sequences' in ttp_data else [],
        'global_features': ttp_data['global_features'][valid_idx] if 'global_features' in ttp_data else None,
        'technique_embeddings': ttp_data.get('technique_embeddings'),
        'num_events': len(ttp_labels),
        'num_classes': len(new_classes),
        'apt_classes': new_classes,
        'padding_value': ttp_data.get('padding_value'),
        'semantic_dim': ttp_data.get('semantic_dim'),
        'num_phases': ttp_data.get('num_phases'),
        'global_feature_dim': ttp_data.get('global_feature_dim'),
        'seq_stats': ttp_data.get('seq_stats', {}),
        'tactic_mapping': ttp_data.get('tactic_mapping', {}),
        'tactic_phase_order': ttp_data.get('tactic_phase_order', {}),
        'sequence_type': ttp_data.get('sequence_type', 'causal_enhanced'),
        'valid_idx': valid_idx  # 保存valid_idx，用于后续索引
    }

    # 保留因果序列键名
    if 'causal_sequences' in ttp_data:
        filtered_ttp['causal_sequences'] = ttp_sequences
    else:
        filtered_ttp['technique_sequences'] = ttp_sequences

    print(f"    TTP过滤: {len(ttp_data['labels'])} -> {len(ttp_labels)}")

    return filtered_ttp

    return filtered_ttp


# ============================================================================
# 主流程
# ============================================================================

def main():
    """主流程"""
    import argparse

    parser = argparse.ArgumentParser(description='增量学习测试')
    parser.add_argument('--step', type=str, default='all',
                       choices=['all', '1', '2', '3'],
                       help='运行指定步骤 (默认: all)')

    args = parser.parse_args()

    start_time = datetime.now()
    print(f"\n开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    if args.step in ['all', '1']:
        results['step1'] = step1_export_old_data()
    else:
        print("\n[跳过] Step 1")

    if args.step in ['all', '2']:
        results['step2'] = step2_incremental_update()
    else:
        print("\n[跳过] Step 2")

    if args.step in ['all', '3']:
        results['step3'] = step3_export_updated_data()
    else:
        print("\n[跳过] Step 3")

    # 输出结果
    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 80)
    print("测试结果汇总".center(80))
    print("=" * 80)

    for step, passed in results.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {step}")

    print(f"\n总耗时: {elapsed / 60:.1f} 分钟")

    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("所有步骤完成!".center(80))
    else:
        print("部分步骤失败，请查看详细信息".center(80))
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
