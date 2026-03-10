"""
增量更新知识图谱模块
支持新增事件、IOCs、CVE、File等数据到现有Neo4j图谱中
"""
import json
import glob
import os
from tqdm import tqdm
from neo4j import GraphDatabase
import ppdeep
import tlsh


class TrailNeo4jIncrementalUpdater:
    """增量更新Neo4j知识图谱"""

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"[*] 已连接到 Neo4j: {uri}")

    def cleanup_isolated_events(self):
        """清理孤立的EVENT节点（没有连接到任何IOC或TTP的节点）

        Returns:
            删除的孤立EVENT节点数量
        """
        print("\n[清理] 检查并删除孤立的EVENT节点...")

        with self.driver.session() as session:
            # 查找孤立的EVENT节点：没有连接到任何IOC或Technique
            query = """
                MATCH (e:EVENT)
                WHERE NOT (e)-[:USES_INFRASTRUCTURE|USES_DOMAIN|DELIVERS_VIA_URL|EXPLOITS_VULN|DROPS_MALWARE|USES_TECHNIQUE]->()
                RETURN count(e) as isolated_count
            """
            result = session.run(query).single()
            isolated_count = result['isolated_count']

            if isolated_count == 0:
                print(f"  [OK] 没有发现孤立的EVENT节点")
                return 0

            print(f"  发现 {isolated_count} 个孤立的EVENT节点，正在删除...")

            # 删除孤立的EVENT节点
            delete_query = """
                MATCH (e:EVENT)
                WHERE NOT (e)-[:USES_INFRASTRUCTURE|USES_DOMAIN|DELIVERS_VIA_URL|EXPLOITS_VULN|DROPS_MALWARE|USES_TECHNIQUE]->()
                DETACH DELETE e
                RETURN count(e) as deleted_count
            """
            result = session.run(delete_query).single()
            deleted_count = result['deleted_count']

            print(f"  [OK] 已删除 {deleted_count} 个孤立的EVENT节点")
            return deleted_count

    def close(self):
        self.driver.close()

    def check_database_health(self):
        """检查数据库状态"""
        with self.driver.session() as session:
            # 统计现有节点数
            stats = session.run("""
                MATCH (n)
                RETURN labels(n) as label, count(n) as count
                ORDER BY count DESC
            """).data()

            print("\n[数据库状态] 现有节点统计:")
            for stat in stats:
                print(f"  {stat['label'][0] if stat['label'] else 'Unknown'}: {stat['count']:,}")

            return stats

    def incremental_import_events(self, new_data_dir: str, batch_size: int = 100, valid_event_ids=None):
        """
        增量导入新的事件和IOCs
        Args:
            new_data_dir: 新数据目录路径
            batch_size: 批量处理大小
            valid_event_ids: 有效事件ID列表，只导入这些事件（如不提供则导入所有）
        """
        print(f"\n[增量更新] 导入新事件数据: {new_data_dir}")

        if valid_event_ids is not None:
            print(f"  [过滤] 只导入有TTP的 {len(valid_event_ids)} 个事件")

        if not os.path.exists(new_data_dir):
            print(f"[!] 目录不存在: {new_data_dir}")
            return {'success': 0, 'failed': 0, 'skipped': 0}

        apt_dirs = [d for d in glob.glob(os.path.join(new_data_dir, '*')) if os.path.isdir(d)]

        stats = {'success': 0, 'failed': 0, 'skipped': 0, 'filtered': 0}
        valid_event_set = set(valid_event_ids) if valid_event_ids is not None else None

        with self.driver.session() as session:
            for apt_dir in tqdm(apt_dirs, desc="增量Event导入"):
                apt_name = os.path.basename(apt_dir)
                json_files = glob.glob(os.path.join(apt_dir, '*.json'))

                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            event_data = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"\n[!] JSON格式错误 {json_file}: {e}")
                        stats['failed'] += 1
                        continue
                    except Exception as e:
                        print(f"\n[!] 文件读取错误 {json_file}: {e}")
                        stats['failed'] += 1
                        continue

                    try:

                        event_id = event_data['event_id']

                        # 检查是否在有效事件列表中（如果提供了过滤）
                        if valid_event_set is not None and event_id not in valid_event_set:
                            stats['filtered'] += 1
                            continue

                        # 获取IOC列表
                        iocs = event_data.get('iocs', [])
                        if not iocs:
                            # 没有IOC的事件跳过
                            stats['skipped'] += 1
                            continue

                        # 检查事件是否已存在（应该通过TTP导入创建）
                        existing = session.run(
                            "MATCH (e:EVENT {id: $eid}) RETURN count(e) as count",
                            eid=event_id
                        ).single()

                        event_exists = existing['count'] > 0

                        if event_exists:
                            # EVENT已存在（通过TTP导入创建的）
                            # 只处理IOC关联，不重新创建EVENT节点
                            # 处理IOCs
                            connected_count = 0
                            for ioc_data in iocs:
                                ioc_val = ioc_data.get('ioc')
                                if not ioc_val:
                                    continue

                                itype = ioc_data.get('type', '').upper()
                                try:
                                    if itype in ['IP', 'IPV4', 'IPV6']:
                                        self._process_ip(session, event_id, ioc_data)
                                        connected_count += 1
                                    elif itype in ['DOMAIN', 'HOSTNAME']:
                                        self._process_domain(session, event_id, ioc_data)
                                        connected_count += 1
                                    elif itype == 'URL':
                                        self._process_url(session, event_id, ioc_data)
                                        connected_count += 1
                                except Exception as e:
                                    pass

                            if connected_count > 0:
                                stats['success'] += 1
                            else:
                                stats['skipped'] += 1
                        else:
                            # EVENT不存在：这说明TTP导入失败了
                            # 采用严格模式：跳过并记录警告，不创建没有TTP的EVENT
                            if 'missing_event' not in stats:
                                stats['missing_event'] = 0
                            stats['missing_event'] += 1

                            # 只在第一次出现时打印警告
                            if stats['missing_event'] == 1:
                                print(f"\n  [警告] 发现EVENT在TTP列表中但不存在于数据库，已跳过")
                                print(f"         （这可能是因为TTP导入时创建失败）")

                    except Exception as e:
                        error_msg = str(e)
                        # 检查是否是索引限制错误
                        if 'Property value is too large to index' in error_msg or 'property size' in error_msg:
                            print(f"\n[!] tags属性过大 {json_file}: 尝试使用空tags重新导入")
                            try:
                                # 重新创建Event，但不设置tags
                                session.run("""
                                    MERGE (e:EVENT {id: $eid})
                                    SET e.label = $apt,
                                        e.source = $src,
                                        e.name = $name,
                                        e.description = $desc,
                                        e.created_at = datetime()
                                """, eid=event_id, apt=apt_name,
                                   src=event_data.get('source', 'OTX'),
                                   name=event_data.get('details', {}).get('name', ''),
                                   desc=event_data.get('details', {}).get('description', ''))
                                stats['success'] += 1
                            except:
                                print(f"\n[!] 即使不设置tags也无法导入 {json_file}")
                                stats['failed'] += 1
                        else:
                            print(f"\n[!] 文件处理错误 {json_file}: {e}")
                            stats['failed'] += 1

        msg = f"\n[增量更新完成] 成功: {stats['success']}, 跳过: {stats['skipped']}, 失败: {stats['failed']}"
        if stats.get('filtered', 0) > 0:
            msg += f", 过滤(无TTP): {stats['filtered']}"
        if stats.get('missing_event', 0) > 0:
            msg += f", 缺失EVENT(TTP导入失败): {stats['missing_event']}"
        print(msg)
        return stats

    def incremental_import_ttp(self, ttp_file: str, create_missing_nodes=True):
        """
        增量导入TTP数据（Technique + Tactic节点及关系）

        参考 build_knowledge_graph.py 的 import_ttp_nodes 方法
        图结构: EVENT -[:USES_TECHNIQUE]-> Technique -[:BELONGS_TO]-> Tactic

        Args:
            ttp_file: incremental_ttp_data_converted.json 文件路径
            create_missing_nodes: 是否创建缺失的Technique节点（默认True）

        Returns:
            valid_event_ids: 有TTP数据的事件ID列表（用于过滤IOC导入）
        """
        print(f"\n[增量更新] 导入TTP数据: {ttp_file}")

        if not os.path.exists(ttp_file):
            print(f"[!] 文件不存在: {ttp_file}")
            return []

        with open(ttp_file, 'r', encoding='utf-8') as f:
            ttp_data = json.load(f)

        # 收集所有事件-技术关系和技术-战术关系
        event_techniques = []  # [(event_id, technique_id)]
        technique_tactics = []  # [(technique_id, tactic_id)]
        valid_event_ids = set()  # 有TTP的事件ID

        # 收集Technique和Tactic的详细信息
        technique_details = {}  # {tech_id: {name, description}}
        tactic_details = {}  # {tactic_id: {name, description}}

        # 收集EVENT的详细信息（用于创建EVENT节点）
        event_details = {}  # {event_id: {label (org_name), source}}

        # 统计信息
        all_techniques = set()
        all_tactics = set()

        for org_name, pulse_list in ttp_data.items():
            for pulse in pulse_list:
                event_id = pulse.get('pulse_id')
                tactics = pulse.get('tactics', {})

                if not event_id:
                    continue

                # 记录有效事件ID
                valid_event_ids.add(event_id)

                # 保存事件详细信息（用于创建EVENT节点）
                if event_id not in event_details:
                    event_details[event_id] = {
                        'id': event_id,
                        'label': org_name,  # 组织名称作为label
                        'source': pulse.get('source', 'TTP')
                    }

                # 遍历战术
                for tactic_id, tactic_data in tactics.items():
                    all_tactics.add(tactic_id)

                    # 保存Tactic详细信息
                    if tactic_id not in tactic_details:
                        tactic_details[tactic_id] = {
                            'id': tactic_id,
                            'name': tactic_data.get('name', ''),
                            'description': tactic_data.get('description', '')
                        }

                    # 遍历该战术下的技术
                    techniques = tactic_data.get('techniques', [])
                    for tech in techniques:
                        tech_id = tech.get('id')
                        if not tech_id:
                            continue

                        all_techniques.add(tech_id)

                        # 保存Technique详细信息
                        if tech_id not in technique_details:
                            technique_details[tech_id] = {
                                'id': tech_id,
                                'name': tech.get('name', ''),
                                'description': tech.get('description', '')
                            }

                        # EVENT-Technique关系
                        event_techniques.append((event_id, tech_id))

                        # Technique-Tactic关系
                        technique_tactics.append((tech_id, tactic_id))

        print(f"  发现 {len(valid_event_ids)} 个有TTP的事件")
        print(f"  发现 {len(all_tactics)} 个唯一Tactic")
        print(f"  发现 {len(all_techniques)} 个唯一Technique")
        print(f"  发现 {len(event_techniques)} 个EVENT-Technique关系")

        if not event_techniques:
            print("  [!] 没有找到有效的TTP数据")
            return []

        batch_size = 1000
        stats = {
            'event_technique_rels': 0,
            'technique_tactic_rels': 0,
            'created_technique_nodes': 0,
            'created_tactic_nodes': 0,
            'created_event_nodes': 0,
            'missing_technique_nodes': 0,
            'missing_tactic_nodes': 0
        }

        with self.driver.session() as session:
            # ============================================================================
            # Step 0: 先创建缺失的EVENT节点（只创建基本结构，不包含IOC）
            # ============================================================================
            print(f"  -> 检查并创建缺失的EVENT节点...")
            existing_events = session.run("""
                MATCH (e:EVENT)
                WHERE e.id IN $event_ids
                RETURN e.id as id
            """, event_ids=list(valid_event_ids))

            existing_event_ids = set()
            for record in existing_events:
                existing_event_ids.add(record['id'])

            missing_events = valid_event_ids - existing_event_ids
            if missing_events:
                print(f"     创建 {len(missing_events)} 个缺失的EVENT节点...")
                create_events_query = """
                    UNWIND $batch AS row
                    MERGE (e:EVENT {id: row.id})
                    SET e.label = row.label,
                        e.source = row.source,
                        e.created_at = datetime()
                    RETURN count(e) as count
                """
                event_list = list(missing_events)
                for i in range(0, len(event_list), batch_size):
                    batch = [event_details[eid] for eid in event_list[i:i + batch_size]]
                    result = session.run(create_events_query, batch=batch).single()
                    stats['created_event_nodes'] += result['count']
                print(f"     EVENT节点: {len(existing_event_ids)} + {stats['created_event_nodes']} = {len(valid_event_ids)} 个")
            else:
                print(f"     所有EVENT节点已存在 ({len(valid_event_ids)} 个)")

            # Step 1: 确保Tactic节点存在
            print(f"  -> 检查Tactic节点...")
            tactic_list = list(all_tactics)
            create_tactics_query = """
                UNWIND $batch AS row
                MERGE (t:Tactic {id: row.id})
                SET t.name = row.name,
                    t.description = row.description
                RETURN count(t) as count
            """
            for i in range(0, len(tactic_list), batch_size):
                batch = [tactic_details[tid] for tid in tactic_list[i:i + batch_size]]
                result = session.run(create_tactics_query, batch=batch).single()
                stats['created_tactic_nodes'] += result['count']
            print(f"     Tactic节点: {stats['created_tactic_nodes']} 个")

            # Step 2: 确保Technique节点存在（创建缺失的）
            print(f"  -> 检查Technique节点...")
            # 先查询已存在的
            existing_techniques = session.run("""
                MATCH (t:Technique)
                WHERE t.id IN $tech_ids
                RETURN t.id as id
            """, tech_ids=list(all_techniques))

            existing_tech_ids = set()
            for record in existing_techniques:
                existing_tech_ids.add(record['id'])

            missing_techniques = all_techniques - existing_tech_ids
            if missing_techniques and create_missing_nodes:
                print(f"     创建 {len(missing_techniques)} 个缺失的Technique节点...")
                create_techniques_query = """
                    UNWIND $batch AS row
                    MERGE (t:Technique {id: row.id})
                    SET t.name = row.name,
                        t.description = row.description
                    RETURN count(t) as count
                """
                tech_list = list(missing_techniques)
                for i in range(0, len(tech_list), batch_size):
                    batch = [technique_details[tid] for tid in tech_list[i:i + batch_size]]
                    result = session.run(create_techniques_query, batch=batch).single()
                    stats['created_technique_nodes'] += result['count']
                print(f"     Technique节点: {len(existing_tech_ids)} + {stats['created_technique_nodes']} = {len(all_techniques)} 个")
            elif missing_techniques:
                print(f"     [警告] {len(missing_techniques)} 个技术节点不存在于数据库")
                stats['missing_technique_nodes'] = len(missing_techniques)

            # Step 3: 创建EVENT-Technique关系（所有EVENT节点现在应该都存在了）
            print(f"  -> 创建EVENT-USES_TECHNIQUE关系（{len(event_techniques)} 条）...")
            event_techniques = list(set(event_techniques))  # 去重

            create_event_tech_query = """
                UNWIND $batch AS row
                MATCH (e:EVENT {id: row.event_id})
                MATCH (t:Technique {id: row.tech_id})
                MERGE (e)-[:USES_TECHNIQUE]->(t)
                RETURN count(*) as count
            """

            for i in range(0, len(event_techniques), batch_size):
                batch = [{'event_id': e, 'tech_id': t} for e, t in event_techniques[i:i + batch_size]]
                result = session.run(create_event_tech_query, batch=batch).single()
                stats['event_technique_rels'] += result['count']

            print(f"     已创建EVENT-Technique关系: {stats['event_technique_rels']} 条")

            # Step 4: 创建Technique-Tactic关系
            print(f"  -> 创建Technique-BELONGS_TO-Tactic关系...")
            technique_tactics = list(set(technique_tactics))  # 去重
            create_tech_tactic_query = """
                UNWIND $batch AS row
                MATCH (t:Technique {id: row.tech_id})
                MATCH (tac:Tactic {id: row.tactic_id})
                MERGE (t)-[:BELONGS_TO]->(tac)
                RETURN count(*) as count
            """

            for i in range(0, len(technique_tactics), batch_size):
                batch = [{'tech_id': t, 'tactic_id': tac} for t, tac in technique_tactics[i:i + batch_size]]
                result = session.run(create_tech_tactic_query, batch=batch).single()
                stats['technique_tactic_rels'] += result['count']

        print(f"\n  [TTP导入完成]")
        print(f"     创建EVENT节点: {stats['created_event_nodes']}")
        print(f"     EVENT-Technique关系: {stats['event_technique_rels']}")
        print(f"     创建Technique节点: {stats['created_technique_nodes']}")
        print(f"     创建Tactic节点: {stats['created_tactic_nodes']}")
        print(f"     Technique-Tactic关系: {stats['technique_tactic_rels']}")
        print(f"     有效事件数: {len(valid_event_ids)}")

        return list(valid_event_ids)

    def incremental_import_mitre_tags(self, new_tags_file: str):
        """增量导入MITRE ATT&CK技术标签（已弃用，请使用 incremental_import_ttp）"""
        print(f"\n[增量更新] 导入MITRE技术标签: {new_tags_file}")
        print(f"  [警告] 此方法已弃用，请使用 incremental_import_ttp()")

        if not os.path.exists(new_tags_file):
            print(f"[!] 文件不存在: {new_tags_file}")
            return

        with open(new_tags_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        query = """
        MATCH (e:EVENT {id: $pulse_id})
        SET e.tags = COALESCE(e.tags, []) + [t IN $new_tags WHERE NOT t IN e.tags]
        """

        stats = {'updated': 0, 'not_found': 0}

        with self.driver.session() as session:
            for org_name, event_list in tqdm(data.items(), desc="TTPs增量追加"):
                if not isinstance(event_list, list):
                    continue

                for event_item in event_list:
                    pulse_id = event_item.get("id")
                    if not pulse_id:
                        continue

                    # 检查事件是否存在
                    existing = session.run(
                        "MATCH (e:EVENT {id: $eid}) RETURN count(e) as count",
                        eid=pulse_id
                    ).single()

                    if existing['count'] == 0:
                        stats['not_found'] += 1
                        continue

                    raw_attack_ids = event_item.get("attack_ids", [])
                    if isinstance(raw_attack_ids, list) and raw_attack_ids:
                        tech_ids_to_add = list(set([str(tid) for tid in raw_attack_ids if tid]))
                        if tech_ids_to_add:
                            session.run(query, pulse_id=pulse_id, new_tags=tech_ids_to_add)
                            stats['updated'] += 1

        print(f"[MITRE标签更新] 更新成功: {stats['updated']}, 事件未找到: {stats['not_found']}")
        return stats

    def incremental_import_cve(self, new_cve_dir: str, valid_event_ids=None):
        """增量导入CVE数据 - 只导入已存在的Event的CVE关联，不创建孤立Event节点

        Args:
            new_cve_dir: CVE数据目录路径
            valid_event_ids: 有效事件ID列表，只导入这些事件（如不提供则导入所有）
        """
        print(f"\n[增量更新] 导入CVE数据: {new_cve_dir}")

        if not os.path.exists(new_cve_dir):
            return

        valid_event_set = set(valid_event_ids) if valid_event_ids is not None else None
        if valid_event_set is not None:
            print(f"  [过滤] 只为有TTP的 {len(valid_event_set)} 个事件导入CVE")

        json_files = glob.glob(os.path.join(new_cve_dir, '**', '*.json'), recursive=True)
        query = """
            MATCH (e:EVENT {id: $eid})
            UNWIND $cves as cve_id
            MERGE (c:CVE {id: cve_id})
            MERGE (e)-[:EXPLOITS]->(c)
        """

        stats = {'success': 0, 'failed': 0, 'skipped_no_event': 0, 'skipped_no_cve': 0, 'filtered_no_ttp': 0}

        with self.driver.session() as session:
            for json_file in tqdm(json_files, desc="CVE增量导入"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    eid = data.get('event_id')
                    cves = data.get('indicators', [])

                    # 跳过无效数据
                    if not eid:
                        stats['skipped_no_event'] += 1
                        continue

                    if not cves:
                        stats['skipped_no_cve'] += 1
                        continue

                    # 检查是否在有效事件列表中（如果提供了过滤）
                    if valid_event_set is not None and eid not in valid_event_set:
                        stats['filtered_no_ttp'] += 1
                        continue

                    # 检查事件是否存在（不自动创建）
                    existing = session.run(
                        "MATCH (e:EVENT {id: $eid}) RETURN count(e) as count",
                        eid=eid
                    ).single()

                    if existing['count'] > 0:
                        # Event存在，才导入CVE关联
                        session.run(query, eid=eid, cves=cves)
                        stats['success'] += 1
                    else:
                        # Event不存在，跳过（不创建孤立Event）
                        stats['skipped_no_event'] += 1

                except Exception as e:
                    stats['failed'] += 1

        msg = f"[CVE更新] 成功: {stats['success']}, 失败: {stats['failed']}, "
        msg += f"跳过(Event不存在): {stats['skipped_no_event']}, 跳过(无CVE): {stats['skipped_no_cve']}"
        if stats.get('filtered_no_ttp', 0) > 0:
            msg += f", 过滤(无TTP): {stats['filtered_no_ttp']}"
        print(msg)
        return stats

    def incremental_import_files(self, new_csv_path: str, valid_event_ids=None):
        """增量导入恶意文件Hash - 只导入已存在的Event的File关联，不创建孤立Event节点

        Args:
            new_csv_path: File CSV文件路径
            valid_event_ids: 有效事件ID列表，只导入这些事件（如不提供则导入所有）
        """
        print(f"\n[增量更新] 导入恶意文件Hash: {new_csv_path}")

        if not os.path.exists(new_csv_path):
            return

        valid_event_set = set(valid_event_ids) if valid_event_ids is not None else None
        if valid_event_set is not None:
            print(f"  [过滤] 只为有TTP的 {len(valid_event_set)} 个事件导入File")

        # 使用MATCH而不是MERGE，只处理已存在的Event
        query = """
            MATCH (e:EVENT {id: row.event_id})
            MERGE (f:File {sha256: row.sha256})
            SET f.signature = row.signature,
                f.imphash = row.imphash,
                f.ssdeep = row.ssdeep,
                f.tlsh = row.tlsh,
                f.apt_tag = row.apt,
                f.type = 'File'
            MERGE (e)-[:CONTAINS]->(f)
        """

        stats = {'success': 0, 'failed': 0, 'skipped_no_event': 0, 'skipped_invalid': 0, 'filtered_no_ttp': 0}

        import csv
        with open(new_csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            batch = []

            with self.driver.session() as session:
                for row in tqdm(reader, desc="File增量导入"):
                    eid = row.get('event_id')

                    # 跳过无效数据
                    if not eid or eid == "Unknown":
                        stats['skipped_invalid'] += 1
                        continue

                    # 检查是否在有效事件列表中（如果提供了过滤）
                    if valid_event_set is not None and eid not in valid_event_set:
                        stats['filtered_no_ttp'] += 1
                        continue

                    # 检查事件是否存在（不自动创建）
                    existing = session.run(
                        "MATCH (e:EVENT {id: $eid}) RETURN count(e) as count",
                        eid=eid
                    ).single()

                    if existing['count'] > 0:
                        # Event存在，才导入File关联
                        batch.append(row)
                        if len(batch) >= 2000:
                            session.run(f"UNWIND $batch as row {query}", batch=batch)
                            stats['success'] += len(batch)
                            batch = []
                    else:
                        # Event不存在，跳过（不创建孤立Event）
                        stats['skipped_no_event'] += 1

                if batch:
                    session.run(f"UNWIND $batch as row {query}", batch=batch)
                    stats['success'] += len(batch)

        msg = f"[File更新] 成功: {stats['success']}, 失败: {stats['failed']}, "
        msg += f"跳过(Event不存在): {stats['skipped_no_event']}, 跳过(无效数据): {stats['skipped_invalid']}"
        if stats.get('filtered_no_ttp', 0) > 0:
            msg += f", 过滤(无TTP): {stats['filtered_no_ttp']}"
        print(msg)
        return stats

    def incremental_update_similarity(self):
        """增量更新文件相似度边（仅对新文件）"""
        print("\n[增量更新] 计算新增文件相似度...")

        # 查找没有相似边的新文件
        fetch_query = """
            MATCH (f:File)
            WHERE NOT (f)-[:SIMILAR]-()
            RETURN f.sha256 as sha256, f.apt_tag as apt,
                   f.imphash as imphash, f.ssdeep as ssdeep, f.tlsh as tlsh
        """

        apt_groups = {}
        with self.driver.session() as session:
            results = session.run(fetch_query)
            for r in results:
                apt = r['apt']
                if apt not in apt_groups:
                    apt_groups[apt] = []
                apt_groups[apt].append(dict(r))

        if not apt_groups:
            print("  没有需要计算的新文件。")
            return

        similarity_edges = []
        TH_SSDEEP = 80
        TH_TLSH = 50

        for apt, files in tqdm(apt_groups.items(), desc="APT组内计算"):
            n = len(files)
            if n < 2:
                continue
            processed_pairs = set()

            # Imphash匹配
            imphash_map = {}
            for f in files:
                imp = f['imphash']
                if imp:
                    imphash_map.setdefault(imp, []).append(f)

            for imp, group in imphash_map.items():
                if len(group) > 1:
                    for i in range(len(group)):
                        for j in range(i + 1, len(group)):
                            f1, f2 = group[i], group[j]
                            pair_key = tuple(sorted((f1['sha256'], f2['sha256'])))
                            processed_pairs.add(pair_key)
                            similarity_edges.append({
                                'sha1': f1['sha256'], 'sha2': f2['sha256'],
                                'score': 100, 'reason': "Imphash_Match"
                            })

            # SSDeep匹配
            ssdeep_buckets = {}
            for f in files:
                ssd = f['ssdeep']
                if ssd:
                    try:
                        bs = int(ssd.split(':')[0])
                        ssdeep_buckets.setdefault(bs, []).append(f)
                    except:
                        pass

            block_sizes = sorted(ssdeep_buckets.keys())
            for i, bs in enumerate(block_sizes):
                current_bucket = ssdeep_buckets[bs]
                self._compare_bucket(current_bucket, current_bucket, processed_pairs,
                                    similarity_edges, TH_SSDEEP)
                if i + 1 < len(block_sizes):
                    next_bs = block_sizes[i + 1]
                    if next_bs == bs * 2:
                        next_bucket = ssdeep_buckets[next_bs]
                        self._compare_bucket(current_bucket, next_bucket, processed_pairs,
                                            similarity_edges, TH_SSDEEP)

            # TLSH匹配（已注释，环境未安装tlsh）
            if n < 5000:
                self._run_tlsh_fallback(files, processed_pairs, similarity_edges, TH_TLSH)

        if not similarity_edges:
            print("  未发现相似文件。")
            return

        print(f"  计算完成，发现 {len(similarity_edges)} 条相似边，正在写入...")
        write_query = """
            MATCH (a:File {sha256: row.sha1})
            MATCH (b:File {sha256: row.sha2})
            MERGE (a)-[r:SIMILAR]-(b)
            SET r.score = row.score, r.reason = row.reason
        """

        with self.driver.session() as session:
            batch_size = 2000
            for i in tqdm(range(0, len(similarity_edges), batch_size), desc="写入Neo4j"):
                batch = similarity_edges[i:i + batch_size]
                session.run(f"UNWIND $batch as row {write_query}", batch=batch)

        print(f"  相似度更新完成，共 {len(similarity_edges)} 条边")

    # 复用原辅助方法
    def _process_ip(self, session, event_id, ioc_data):
        ip_value = ioc_data['ioc']
        session.run("""
            MERGE (i:IP {value: $val})
            SET i.country_code = $cc, i.city = $city, i.region = $region,
                i.latitude = $lat, i.longitude = $lon
            WITH i
            MATCH (e:EVENT {id: $eid})
            MERGE (e)-[:InReport]->(i)
        """, val=ip_value, eid=event_id, cc=ioc_data.get('country_code'),
           city=ioc_data.get('city'), region=ioc_data.get('region'),
           lat=ioc_data.get('latitude'), lon=ioc_data.get('longitude'))

        asn_str = ioc_data.get('asn')
        if asn_str:
            parts = asn_str.split(' ', 1)
            asn_val = parts[0]
            issuer = parts[1] if len(parts) > 1 else ''
            session.run("""
                MATCH (i:IP {value: $val})
                MERGE (a:ASN {value: $asn}) SET a.issuer = $iss
                MERGE (i)-[:InGroup]->(a)
            """, val=ip_value, asn=asn_val, iss=issuer)

        for res in ioc_data.get('resolves_to', []):
            host = res.get('host')
            if host:
                session.run("""
                    MATCH (i:IP {value: $val})
                    MERGE (d:domain {value: $host})
                    MERGE (i)-[:RESOLVES_TO]->(d)
                """, val=ip_value, host=host)

    def _process_domain(self, session, event_id, ioc_data):
        domain_value = ioc_data['ioc']
        dns_records = ioc_data.get('dns_records', [])
        first_seen, last_seen = None, None
        timestamps = []
        has_nxdomain = False

        for record in dns_records:
            if record.get('address') == 'NXDOMAIN':
                has_nxdomain = True
            if record.get('first'):
                timestamps.append(record['first'])
            if record.get('last'):
                timestamps.append(record['last'])

        if timestamps:
            try:
                sorted_ts = sorted(timestamps)
                first_seen = sorted_ts[0]
                last_seen = sorted_ts[-1]
            except:
                pass

        session.run("""
            MERGE (d:domain {value: $val})
            SET d.first_seen = $fs, d.last_seen = $ls, d.has_nxdomain = $nx, d.type = 'Domain'
            WITH d
            MATCH (e:EVENT {id: $eid})
            MERGE (e)-[:InReport]->(d)
        """, val=domain_value, eid=event_id, fs=first_seen, ls=last_seen, nx=has_nxdomain)

        for record in dns_records:
            addr = record.get('address')
            rtype = record.get('record_type')
            if rtype in ['A', 'AAAA'] and addr and addr != 'NXDOMAIN':
                session.run("""
                    MATCH (d:domain {value: $dom})
                    MERGE (i:IP {value: $ip})
                    MERGE (d)-[:RESOLVES_TO]->(i)
                """, dom=domain_value, ip=addr)

    def _process_url(self, session, event_id, ioc_data):
        url_value = ioc_data['ioc']
        hostname = ioc_data.get('hostname')
        if not hostname:
            try:
                from urllib.parse import urlparse
                hostname = urlparse(url_value).netloc.split(':')[0]
            except:
                pass

        session.run("""
            MERGE (u:URL {value: $val})
            SET u.hostname = $host, u.server = $srv, u.http_code = $code,
                u.filetype = $ft, u.encoding = $enc, u.type = 'URL'
            WITH u
            MATCH (e:EVENT {id: $eid})
            MERGE (e)-[:InReport]->(u)
        """, val=url_value, eid=event_id, host=hostname, srv=ioc_data.get('server'),
           code=ioc_data.get('http_code'), ft=ioc_data.get('filetype'),
           enc=ioc_data.get('encoding'))

        if hostname:
            session.run("""
                MATCH (u:URL {value: $val})
                MERGE (d:domain {value: $host})
                MERGE (u)-[:HostedOn]->(d)
            """, val=url_value, host=hostname)

        if ioc_data.get('ip'):
            session.run("""
                MATCH (u:URL {value: $val})
                MERGE (i:IP {value: $ip})
                MERGE (u)-[:RESOLVES_TO]->(i)
            """, val=url_value, ip=ioc_data.get('ip'))

    def _compare_bucket(self, list1, list2, processed_pairs, edges, threshold):
        is_same_bucket = (list1 is list2)
        for i in range(len(list1)):
            start_j = i + 1 if is_same_bucket else 0
            for j in range(start_j, len(list2)):
                f1, f2 = list1[i], list2[j]
                pair_key = tuple(sorted((f1['sha256'], f2['sha256'])))
                if pair_key in processed_pairs:
                    continue
                try:
                    s = ppdeep.compare(f1['ssdeep'], f2['ssdeep'])
                    if s >= threshold:
                        processed_pairs.add(pair_key)
                        edges.append({
                            'sha1': f1['sha256'], 'sha2': f2['sha256'],
                            'score': s, 'reason': f"SSDEEP={s}"
                        })
                except:
                    pass

    def _run_tlsh_fallback(self, files, processed_pairs, edges, threshold):
        n = len(files)
        for i in range(n):
            for j in range(i + 1, n):
                f1, f2 = files[i], files[j]
                pair_key = tuple(sorted((f1['sha256'], f2['sha256'])))
                if pair_key in processed_pairs:
                    continue
                if f1['tlsh'] and f2['tlsh']:
                    try:
                        d = tlsh.diff(f1['tlsh'], f2['tlsh'])
                        if d <= threshold:
                            similarity_score = max(0, 100 - d)
                            edges.append({
                                'sha1': f1['sha256'], 'sha2': f2['sha256'],
                                'score': similarity_score, 'reason': f"TLSH={d}"
                            })
                            processed_pairs.add(pair_key)
                    except:
                        pass


def main():
    """增量更新示例"""
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neo4j123"
    BASE_DIR = "/root/PythonProject/Trident/src"

    # 新数据路径（示例）
    NEW_THREAT_DATA_DIR = os.path.join(BASE_DIR, "output_incremental/ioc/")
    NEW_EVENT_TTPS_FILE = os.path.join(BASE_DIR, "build_dataset/incremental_ttp_data.json")
    NEW_CVE_DATA_DIR = os.path.join(BASE_DIR, "output_incremental/cve/")
    NEW_FILE_CSV_PATH = os.path.join(BASE_DIR, "output_incremental/incremental_file_hashes.csv")

    updater = TrailNeo4jIncrementalUpdater(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        # 1. 检查数据库状态
        updater.check_database_health()

        # 2. 增量导入新事件和IOCs
        if os.path.exists(NEW_THREAT_DATA_DIR):
            updater.incremental_import_events(NEW_THREAT_DATA_DIR)

        # 3. 增量导入MITRE标签
        if os.path.exists(NEW_EVENT_TTPS_FILE):
            updater.incremental_import_mitre_tags(NEW_EVENT_TTPS_FILE)

        # 4. 增量导入CVE
        if os.path.exists(NEW_CVE_DATA_DIR):
            updater.incremental_import_cve(NEW_CVE_DATA_DIR)

        # 5. 增量导入文件
        if os.path.exists(NEW_FILE_CSV_PATH):
            updater.incremental_import_files(NEW_FILE_CSV_PATH)

        # 6. 增量更新文件相似度
        updater.incremental_update_similarity()

        # 7. 最终统计
        print("\n[增量更新完成] 最终数据库状态:")
        updater.check_database_health()

    finally:
        updater.close()


if __name__ == '__main__':
    main()
