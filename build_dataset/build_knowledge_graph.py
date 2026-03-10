import json
import glob
import os
import csv
import math
from tqdm import tqdm
from neo4j import GraphDatabase
from urllib.parse import urlparse
import ppdeep
import tlsh


class TrailNeo4jBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"[*] 已连接到 Neo4j: {uri}")

    def close(self):
        self.driver.close()

    def clear_database(self):
        """清空数据库所有数据、动态清理旧约束和索引"""
        with self.driver.session() as session:
            print("[!] 正在清空数据库所有节点和关系...")
            session.run("MATCH (n) DETACH DELETE n")
            print("[*] 数据库数据已清空")

            print("[!] 正在动态清理旧的约束和索引...")
            try:
                # 动态获取并删除所有约束
                constraints = session.run("SHOW CONSTRAINTS YIELD name")
                for record in constraints:
                    session.run(f"DROP CONSTRAINT {record['name']}")
                
                # 动态获取并删除所有自定义索引 (跳过 Neo4j 默认的 LOOKUP 索引)
                indexes = session.run("SHOW INDEXES YIELD name, type")
                for record in indexes:
                    if "LOOKUP" not in record.get("type", "").upper():
                        try:
                            session.run(f"DROP INDEX {record['name']}")
                        except Exception:
                            pass
                print("[*] 旧约束和索引已彻底清理")
            except Exception as e:
                print(f"[*] 动态清理约束/索引时提示 (可能由于 Neo4j 版本不支持): {e}")

    def create_constraints(self):
        print("\n[*] 创建全量约束和索引...")

        constraints = [
            # 基础实体
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:EVENT) REQUIRE e.id IS UNIQUE",

            # 网络 IOCs
            "CREATE CONSTRAINT ip_value IF NOT EXISTS FOR (i:IP) REQUIRE i.value IS UNIQUE",
            "CREATE CONSTRAINT domain_value IF NOT EXISTS FOR (d:domain) REQUIRE d.value IS UNIQUE",
            "CREATE CONSTRAINT url_value IF NOT EXISTS FOR (u:URL) REQUIRE u.value IS UNIQUE",
            "CREATE CONSTRAINT asn_value IF NOT EXISTS FOR (a:ASN) REQUIRE a.value IS UNIQUE",

            # 文件与漏洞
            "CREATE CONSTRAINT file_sha256 IF NOT EXISTS FOR (f:File) REQUIRE f.sha256 IS UNIQUE",
            "CREATE CONSTRAINT cve_id IF NOT EXISTS FOR (c:CVE) REQUIRE c.id IS UNIQUE",

            # TTP双层节点
            "CREATE CONSTRAINT technique_id IF NOT EXISTS FOR (t:Technique) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT tactic_id IF NOT EXISTS FOR (t:Tactic) REQUIRE t.id IS UNIQUE",
        ]

        # 索引加速查询
        indexes = [
            "CREATE INDEX file_apt_tag IF NOT EXISTS FOR (f:File) ON (f.apt_tag)",
            "CREATE INDEX file_ssdeep IF NOT EXISTS FOR (f:File) ON (f.ssdeep)",
            "CREATE INDEX file_tlsh IF NOT EXISTS FOR (f:File) ON (f.tlsh)",
            "CREATE INDEX event_label IF NOT EXISTS FOR (e:EVENT) ON (e.label)",
            # 增加 tags 索引以便查询技术点
            "CREATE INDEX event_tags IF NOT EXISTS FOR (e:EVENT) ON (e.tags)"
        ]

        with self.driver.session() as session:
            for c in constraints:
                try:
                    session.run(c)
                except Exception as e:
                    print(f"[!] 约束创建失败: {c} -> {e}")
                    
            for idx in indexes:
                try:
                    session.run(idx)
                except Exception as e:
                    print(f"[!] 索引创建失败: {idx} -> {e}")
                    
        print("[*] 约束和新索引创建完成")

    # =========================================================================
    # PART 1: 导入事件与富化 IOC (Integrated Rich Attributes) - 批量优化版本
    # =========================================================================
    def import_threat_events(self, data_dir):
        print(f"\n[Step 1] 导入事件与网络 IOC: {data_dir}")
        if not os.path.exists(data_dir):
            print(f"[!] 目录不存在: {data_dir}")
            return

        apt_dirs = [d for d in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(d)]

        # ============ 阶段1: 收集所有数据到内存 ============
        print("    -> 收集数据到内存...")
        events_batch = []
        ips_batch = []
        domains_batch = []
        urls_batch = []
        ip_asn_batch = []
        ip_dns_batch = []
        domain_dns_batch = []

        for apt_dir in tqdm(apt_dirs, desc="读取文件"):
            json_files = glob.glob(os.path.join(apt_dir, '*.json'))
            apt_name = os.path.basename(apt_dir)

            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        event_data = json.load(f)

                    iocs = event_data.get('iocs', [])
                    if not iocs:
                        continue

                    event_id = event_data['event_id']
                    # 提取描述信息（从 details.description 字段）
                    description = None
                    if 'details' in event_data and isinstance(event_data['details'], dict):
                        description = event_data['details'].get('description')
                    events_batch.append({'id': event_id, 'label': apt_name, 'description': description})

                    for ioc_data in iocs:
                        ioc_val = ioc_data.get('ioc')
                        if not ioc_val: continue

                        itype = ioc_data.get('type', '').upper()

                        try:
                            if itype in ['IP', 'IPV4', 'IPV6']:
                                self._collect_ip_data(ioc_data, event_id, ips_batch, ip_asn_batch, ip_dns_batch)
                            elif itype in ['DOMAIN', 'HOSTNAME']:
                                self._collect_domain_data(ioc_data, event_id, domains_batch, domain_dns_batch)
                            elif itype == 'URL':
                                self._collect_url_data(ioc_data, event_id, urls_batch)
                        except Exception as e:
                            # IOC处理错误，跳过该IOC但继续处理其他
                            pass

                except json.JSONDecodeError as e:
                    # JSON解析错误，打印警告
                    tqdm.write(f"[!] JSON解析错误 {json_file}: {e}")
                except Exception as e:
                    # 其他文件处理错误
                    tqdm.write(f"[!] 文件处理错误 {json_file}: {e}")

        print(f"  -> 共收集 {len(events_batch)} 个事件, {len(ips_batch)} 个IP, {len(domains_batch)} 个Domain, {len(urls_batch)} 个URL")

        # ============ 阶段2: 批量写入数据库 ============

        with self.driver.session() as session:
            # 2.1 批量创建EVENT节点
            if events_batch:
                self._batch_create_events(session, events_batch)

            # 2.2 批量创建IP节点和边
            if ips_batch:
                self._batch_create_ips(session, ips_batch)

            # 2.3 批量创建IP-ASN关系
            if ip_asn_batch:
                self._batch_create_ip_asn(session, ip_asn_batch)

            # 2.4 批量创建IP-DNS解析关系
            if ip_dns_batch:
                self._batch_create_ip_dns(session, ip_dns_batch)

            # 2.5 批量创建domain节点和边
            if domains_batch:
                self._batch_create_domains(session, domains_batch)

            # 2.6 批量创建domain-DNS解析关系
            if domain_dns_batch:
                self._batch_create_domain_dns(session, domain_dns_batch)

            # 2.7 批量创建URL节点和边
            if urls_batch:
                self._batch_create_urls(session, urls_batch)

        print("    -> 批量导入完成!")

        # ============ 阶段3: 清理孤立节点 ============
        self.cleanup_orphan_events()

    # =========================================================================
    # 批量处理辅助函数 - 数据收集阶段
    # =========================================================================
    def _collect_ip_data(self, ioc_data, event_id, ips_batch, ip_asn_batch, ip_dns_batch):
        """收集IP数据到批量列表"""
        ip_value = ioc_data['ioc']
        lat = ioc_data.get('latitude')
        lon = ioc_data.get('longitude')
        lat_norm = float(lat) / 90.0 if lat is not None else 0
        lon_norm = float(lon) / 180.0 if lon is not None else 0

        ips_batch.append({
            'value': ip_value,
            'event_id': event_id,
            'country_code': ioc_data.get('country_code'),
            'city': ioc_data.get('city'),
            'region': ioc_data.get('region'),
            'latitude': lat,
            'longitude': lon,
            'lat_norm': lat_norm,
            'lon_norm': lon_norm
        })

        # ASN关系
        asn_str = ioc_data.get('asn')
        if asn_str:
            parts = asn_str.split(' ', 1)
            asn_val = parts[0]
            issuer = parts[1] if len(parts) > 1 else ''
            ip_asn_batch.append({'ip': ip_value, 'asn': asn_val, 'issuer': issuer})

        # DNS解析
        for res in ioc_data.get('resolves_to', []):
            host = res.get('host')
            if host:
                ip_dns_batch.append({'ip': ip_value, 'host': host})

    def _collect_domain_data(self, ioc_data, event_id, domains_batch, domain_dns_batch):
        """收集domain数据到批量列表"""
        domain_value = ioc_data['ioc']
        dns_records = ioc_data.get('dns_records', [])

        first_seen, last_seen, has_nxdomain, lifespan_days, lifespan_log = self._extract_domain_features(dns_records)

        domains_batch.append({
            'value': domain_value,
            'event_id': event_id,
            'first_seen': first_seen,
            'last_seen': last_seen,
            'has_nxdomain': has_nxdomain,
            'lifespan_days': lifespan_days,
            'lifespan_log': lifespan_log
        })

        # DNS解析记录
        for record in dns_records:
            addr = record.get('address')
            rtype = record.get('record_type')
            if rtype in ['A', 'AAAA'] and addr and addr != 'NXDOMAIN':
                domain_dns_batch.append({'domain': domain_value, 'ip': addr})

    def _collect_url_data(self, ioc_data, event_id, urls_batch):
        """收集URL数据到批量列表"""
        url_value = ioc_data['ioc']
        hostname = ioc_data.get('hostname')
        if not hostname:
            try:
                hostname = urlparse(url_value).netloc.split(':')[0]
            except:
                hostname = None

        urls_batch.append({
            'value': url_value,
            'event_id': event_id,
            'hostname': hostname,
            'server': ioc_data.get('server'),
            'http_code': ioc_data.get('http_code'),
            'filetype': ioc_data.get('filetype'),
            'encoding': ioc_data.get('encoding'),
            'ip': ioc_data.get('ip')
        })

    def _extract_domain_features(self, dns_records):
        """提取域名特征"""
        timestamps = []
        has_nxdomain = False

        for record in dns_records:
            if record.get('address') == 'NXDOMAIN':
                has_nxdomain = True
            if record.get('first'):
                timestamps.append(record['first'])
            if record.get('last'):
                timestamps.append(record['last'])

        first_seen = None
        last_seen = None
        if timestamps:
            try:
                sorted_ts = sorted(timestamps)
                first_seen = sorted_ts[0]
                last_seen = sorted_ts[-1]
            except:
                pass

        lifespan_days = 0
        if first_seen and last_seen:
            try:
                from datetime import datetime
                fs = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
                ls = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                days = (ls - fs).days
                lifespan_days = max(0, days)
            except:
                pass

        lifespan_log = math.log1p(lifespan_days) if lifespan_days > 0 else 0
        return first_seen, last_seen, has_nxdomain, lifespan_days, lifespan_log

    # =========================================================================
    # 批量处理辅助函数 - 数据库写入阶段
    # =========================================================================
    def _batch_create_events(self, session, events_batch, batch_size=5000):
        """批量创建EVENT节点"""
        query = """
            UNWIND $batch AS row
            MERGE (e:EVENT {id: row.id})
            SET e.label = row.label,
                e.description = row.description
        """
        for i in range(0, len(events_batch), batch_size):
            batch = events_batch[i:i + batch_size]
            session.run(query, batch=batch)

    def _batch_create_ips(self, session, ips_batch, batch_size=2000):
        """批量创建IP节点和EVENT-IP边"""
        query = """
            UNWIND $batch AS row
            MERGE (i:IP {value: row.value})
            SET i.country_code = row.country_code,
                i.city = row.city,
                i.region = row.region,
                i.latitude = row.latitude,
                i.longitude = row.longitude,
                i.lat_norm = row.lat_norm,
                i.lon_norm = row.lon_norm
            WITH i, row
            MATCH (e:EVENT {id: row.event_id})
            MERGE (e)-[:USES_INFRASTRUCTURE]->(i)
        """
        for i in range(0, len(ips_batch), batch_size):
            batch = ips_batch[i:i + batch_size]
            session.run(query, batch=batch)

    def _batch_create_ip_asn(self, session, ip_asn_batch, batch_size=2000):
        """批量创建IP-ASN关系"""
        query = """
            UNWIND $batch AS row
            MATCH (i:IP {value: row.ip})
            MERGE (a:ASN {value: row.asn})
            SET a.issuer = row.issuer
            MERGE (i)-[:BELONGS_TO_NETWORK]->(a)
        """
        for i in range(0, len(ip_asn_batch), batch_size):
            batch = ip_asn_batch[i:i + batch_size]
            session.run(query, batch=batch)

    def _batch_create_ip_dns(self, session, ip_dns_batch, batch_size=2000):
        """批量创建IP-domain DNS解析关系"""
        query = """
            UNWIND $batch AS row
            MATCH (i:IP {value: row.ip})
            MERGE (d:domain {value: row.host})
            MERGE (i)-[:RESOLVES_TO]->(d)
        """
        for i in range(0, len(ip_dns_batch), batch_size):
            batch = ip_dns_batch[i:i + batch_size]
            session.run(query, batch=batch)

    def _batch_create_domains(self, session, domains_batch, batch_size=2000):
        """批量创建domain节点和EVENT-domain边"""
        query = """
            UNWIND $batch AS row
            MERGE (d:domain {value: row.value})
            SET d.first_seen = row.first_seen,
                d.last_seen = row.last_seen,
                d.has_nxdomain = row.has_nxdomain,
                d.lifespan_days = row.lifespan_days,
                d.lifespan_log = row.lifespan_log
            WITH d, row
            MATCH (e:EVENT {id: row.event_id})
            MERGE (e)-[:USES_DOMAIN]->(d)
        """
        for i in range(0, len(domains_batch), batch_size):
            batch = domains_batch[i:i + batch_size]
            session.run(query, batch=batch)

    def _batch_create_domain_dns(self, session, domain_dns_batch, batch_size=2000):
        """批量创建domain-IP双向DNS解析关系"""
        query = """
            UNWIND $batch AS row
            MATCH (d:domain {value: row.domain})
            MERGE (i:IP {value: row.ip})
            MERGE (d)-[:RESOLVES_TO]->(i)
            MERGE (i)-[:RESOLVES_FROM]->(d)
        """
        for i in range(0, len(domain_dns_batch), batch_size):
            batch = domain_dns_batch[i:i + batch_size]
            session.run(query, batch=batch)

    def _batch_create_urls(self, session, urls_batch, batch_size=2000):
        """批量创建URL节点和边关系"""
        # 创建URL节点和EVENT-URL边
        query1 = """
            UNWIND $batch AS row
            MERGE (u:URL {value: row.value})
            SET u.hostname = row.hostname,
                u.server = row.server,
                u.http_code = row.http_code,
                u.filetype = row.filetype,
                u.encoding = row.encoding,
                u.type = 'URL'
            WITH u, row
            MATCH (e:EVENT {id: row.event_id})
            MERGE (e)-[:DELIVERS_VIA_URL]->(u)
        """
        # URL-domain关系
        query2 = """
            UNWIND $batch AS row
            MATCH (u:URL {value: row.value})
            MATCH (e:EVENT {id: row.event_id})
            WHERE row.hostname IS NOT NULL
            MERGE (d:domain {value: row.hostname})
            MERGE (u)-[:HOSTED_ON_DOMAIN]->(d)
        """
        # URL-IP关系
        query3 = """
            UNWIND $batch AS row
            MATCH (u:URL {value: row.value})
            WHERE row.ip IS NOT NULL
            MERGE (i:IP {value: row.ip})
            MERGE (u)-[:RESOLVES_TO_IP]->(i)
        """

        # 过滤有hostname的记录
        with_hostname = [r for r in urls_batch if r.get('hostname')]
        with_ip = [r for r in urls_batch if r.get('ip')]

        for i in range(0, len(urls_batch), batch_size):
            batch = urls_batch[i:i + batch_size]
            session.run(query1, batch=batch)

        if with_hostname:
            for i in range(0, len(with_hostname), batch_size):
                batch = with_hostname[i:i + batch_size]
                session.run(query2, batch=batch)

        if with_ip:
            for i in range(0, len(with_ip), batch_size):
                batch = with_ip[i:i + batch_size]
                session.run(query3, batch=batch)

    # =========================================================================
    # PART 2.5: TTP 节点导入（双层结构 Technique + Tactic）
    # =========================================================================
    def import_ttp_nodes(self, ttp_features_file):
        """
        导入双层TTP节点：Technique（技术） + Tactic（战术）

        图结构:
        EVENT -[:USES_TECHNIQUE]-> Technique -[:BELONGS_TO]-> Tactic
        """
        print(f"\n[Step 2.5] 导入双层TTP节点: {ttp_features_file}")

        if not os.path.exists(ttp_features_file):
            print(f"[!] 文件不存在: {ttp_features_file}")
            return

        with open(ttp_features_file, 'r', encoding='utf-8') as f:
            ttp_data = json.load(f)

        # 收集所有唯一的Tactic和Technique
        all_tactics = {}  # {tactic_id: {name, description}}
        all_techniques = {}  # {technique_id: {name, description}}
        event_techniques = []  # [(event_id, technique_id)]
        technique_tactics = []  # [(technique_id, tactic_id)]

        for org_name, pulse_list in ttp_data.items():

            for pulse in pulse_list:
                event_id = pulse.get('pulse_id')
                tactics = pulse.get('tactics', {})

                if not event_id:
                    continue

                # 遍历战术
                for tactic_id, tactic_data in tactics.items():
                    # 收集Tactic
                    if tactic_id not in all_tactics:
                        all_tactics[tactic_id] = {
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

                        # 收集Technique
                        if tech_id not in all_techniques:
                            all_techniques[tech_id] = {
                                'id': tech_id,
                                'name': tech.get('name', ''),
                                'description': tech.get('description', '')
                            }

                        # EVENT-Technique关系
                        event_techniques.append((event_id, tech_id))

                        # Technique-Tactic关系
                        technique_tactics.append((tech_id, tactic_id))

        print(f"    发现 {len(all_tactics)} 个唯一Tactic")
        print(f"    发现 {len(all_techniques)} 个唯一Technique")
        print(f"    发现 {len(event_techniques)} 个EVENT-Technique关系")

        if not all_techniques or not all_tactics:
            print("    [!] 没有找到有效的TTP数据，跳过")
            return

        batch_size = 1000
        with self.driver.session() as session:
            # Step 1: 创建Tactic节点
            print(f"    -> 创建Tactic节点 ({len(all_tactics)} 个)...")
            tactic_list = list(all_tactics.values())
            create_tactic_query = """
                UNWIND $batch AS row
                MERGE (t:Tactic {id: row.id})
                SET t.name = row.name,
                    t.description = row.description
            """
            for i in range(0, len(tactic_list), batch_size):
                batch = tactic_list[i:i + batch_size]
                session.run(create_tactic_query, batch=batch)

            # Step 2: 创建Technique节点
            print(f"    -> 创建Technique节点 ({len(all_techniques)} 个)...")
            tech_list = list(all_techniques.values())
            create_tech_query = """
                UNWIND $batch AS row
                MERGE (t:Technique {id: row.id})
                SET t.name = row.name,
                    t.description = row.description
            """
            for i in range(0, len(tech_list), batch_size):
                batch = tech_list[i:i + batch_size]
                session.run(create_tech_query, batch=batch)

            # Step 3: 创建EVENT-Technique关系
            print(f"    -> 创建EVENT-USES_TECHNIQUE关系 ({len(event_techniques)} 条)...")
            # 去重
            event_techniques = list(set(event_techniques))
            create_event_tech_query = """
                UNWIND $batch AS row
                MATCH (e:EVENT {id: row.event_id})
                MATCH (t:Technique {id: row.tech_id})
                MERGE (e)-[:USES_TECHNIQUE]->(t)
            """
            for i in range(0, len(event_techniques), batch_size):
                batch = [{'event_id': e, 'tech_id': t} for e, t in event_techniques[i:i + batch_size]]
                session.run(create_event_tech_query, batch=batch)

            # Step 4: 创建Technique-Tactic关系
            print(f"    -> 创建Technique-BELONGS_TO-Tactic关系 ({len(technique_tactics)} 条)...")
            # 去重
            technique_tactics = list(set(technique_tactics))
            create_tech_tactic_query = """
                UNWIND $batch AS row
                MATCH (t:Technique {id: row.tech_id})
                MATCH (tac:Tactic {id: row.tactic_id})
                MERGE (t)-[:BELONGS_TO]->(tac)
            """
            for i in range(0, len(technique_tactics), batch_size):
                batch = [{'tech_id': t, 'tactic_id': tac} for t, tac in technique_tactics[i:i + batch_size]]
                session.run(create_tech_tactic_query, batch=batch)

        print(f"    -> TTP导入完成! Tactic: {len(all_tactics)}, Technique: {len(all_techniques)}")

    # =========================================================================
    # PART 3 & 4: CVE 和 File
    # =========================================================================
    def import_cve_data(self, cve_dir):
        print(f"\n[Step 3] 导入 CVE 数据: {cve_dir}")
        if not os.path.exists(cve_dir): return
        json_files = glob.glob(os.path.join(cve_dir, '**', '*.json'), recursive=True)

        # 修改：只对已存在的EVENT添加CVE边，不创建新EVENT
        query = """
            MATCH (e:EVENT {id: $eid})
            WITH e
            UNWIND $cves as cve_id
            MERGE (c:CVE {id: cve_id})
            SET c.year = CASE
                WHEN cve_id =~ 'CVE-\\\d{4}-.*' THEN toInteger(substring(cve_id, 4, 4))
                ELSE NULL
            END,
            c.year_norm = CASE
                WHEN cve_id =~ 'CVE-\\\d{4}-.*' THEN (toInteger(substring(cve_id, 4, 4)) - 1999.0) / 30.0
                ELSE NULL
            END
            MERGE (e)-[:EXPLOITS_VULN]->(c)
        """
        with self.driver.session() as session:
            for json_file in tqdm(json_files, desc="CVE 导入"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    eid = data.get('event_id')
                    cves = data.get('indicators', [])
                    if not cves or not eid: continue
                    session.run(query, eid=eid, cves=cves)
                except json.JSONDecodeError:
                    pass  # 跳过无效的JSON文件
                except Exception as e:
                    tqdm.write(f"[!] CVE文件处理错误 {json_file}: {e}")

    def import_file_csv(self, csv_path):
        print(f"\n[Step 4] 导入恶意文件 Hash: {csv_path}")
        if not os.path.exists(csv_path): return
        # 只对已存在的EVENT添加File边，不创建新EVENT
        query = """
            MATCH (e:EVENT {id: row.event_id})
            MERGE (f:File {sha256: row.sha256})
            SET f.signature = row.signature,
                f.imphash = row.imphash,
                f.ssdeep = row.ssdeep,
                f.tlsh = row.tlsh,
                f.apt_tag = row.apt,
                f.type = 'File'
            MERGE (e)-[:DROPS_MALWARE]->(f)
             """
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            batch = []
            with self.driver.session() as session:
                for row in tqdm(reader, desc="File 导入"):
                    if not row.get('event_id') or row['event_id'] == "Unknown": continue
                    batch.append(row)
                    if len(batch) >= 2000:
                        session.run(f"UNWIND $batch as row {query}", batch=batch)
                        batch = []
                if batch: session.run(f"UNWIND $batch as row {query}", batch=batch)

    # =========================================================================
    # PART 5: 相似度计算（优化版本）
    # =========================================================================
    def build_file_similarity_edges(self):
        print("\n[Step 5] 开始极速相似度计算 (优先级: Imphash > SSDeep > TLSH)...")

        print("    -> 正在读取文件特征...")
        fetch_query = """
                MATCH (f:File)
                WHERE f.sha256 IS NOT NULL
                RETURN f.sha256 as sha256, f.apt_tag as apt,
                       f.imphash as imphash, f.ssdeep as ssdeep, f.tlsh as tlsh
                """

        apt_groups = {}
        with self.driver.session() as session:
            results = session.run(fetch_query)
            for r in results:
                apt = r['apt']
                if apt not in apt_groups: apt_groups[apt] = []
                apt_groups[apt].append(dict(r))

        similarity_edges = []
        TH_SSDEEP = 80
        TH_TLSH = 50
        total_files = sum(len(files) for files in apt_groups.values())
        print(f"    -> 共 {total_files} 个文件，分 {len(apt_groups)} 个APT组计算")

        for apt, files in tqdm(apt_groups.items(), desc="APT组内计算"):
            n = len(files)
            if n < 2: continue
            processed_pairs = set()

            # --- 阶段 1: Imphash (精确匹配，最快) ---
            imphash_map = {}
            for f in files:
                imp = f['imphash']
                if imp: imphash_map.setdefault(imp, []).append(f)

            for imp, group in imphash_map.items():
                if len(group) > 1:
                    # 优化：使用组合而非嵌套循环
                    for i in range(len(group)):
                        for j in range(i + 1, len(group)):
                            f1, f2 = group[i], group[j]
                            pair_key = tuple(sorted((f1['sha256'], f2['sha256'])))
                            processed_pairs.add(pair_key)
                            similarity_edges.append({
                                'sha1': f1['sha256'], 'sha2': f2['sha256'],
                                'score': 100, 'reason': "Imphash_Match"
                            })

            # --- 阶段 2: SSDeep (分桶比较) ---
            ssdeep_buckets = {}
            for f in files:
                ssd = f['ssdeep']
                if ssd:
                    try:
                        bs = int(ssd.split(':')[0])
                        ssdeep_buckets.setdefault(bs, []).append(f)
                    except (ValueError, IndexError, AttributeError):
                        # SSDeep格式错误，跳过该文件
                        continue

            block_sizes = sorted(ssdeep_buckets.keys())
            for i, bs in enumerate(block_sizes):
                current_bucket = ssdeep_buckets[bs]
                self._compare_bucket(current_bucket, current_bucket, processed_pairs, similarity_edges, TH_SSDEEP)
                if i + 1 < len(block_sizes):
                    next_bs = block_sizes[i + 1]
                    # 只比较相邻的块大小（block_size 和 2*block_size）
                    if next_bs == bs * 2:
                        next_bucket = ssdeep_buckets[next_bs]
                        self._compare_bucket(current_bucket, next_bucket, processed_pairs, similarity_edges, TH_SSDEEP)

            # --- 阶段 3: TLSH (限制使用，避免O(n²)爆炸) ---
            # 优化：只对小规模组（<1000）或高价值样本启用TLSH
            if n < 1000:
                self._run_tlsh_fallback(files, processed_pairs, similarity_edges, TH_TLSH)
            elif n >= 1000 and n < 5000:
                # 对大规模组，只对有TLSH值的文件进行采样比较
                tlsh_files = [f for f in files if f.get('tlsh')]
                if len(tlsh_files) < 1000:  # 只在有TLSH的文件较少时比较
                    self._run_tlsh_fallback(tlsh_files, processed_pairs, similarity_edges, TH_TLSH)

        if not similarity_edges:
            print("  未发现相似文件。")
            return

        print(f" 计算完成，发现 {len(similarity_edges)} 条唯一相似边，正在写入...")
        write_query = """
                    MATCH (a:File {sha256: row.sha1})
                    MATCH (b:File {sha256: row.sha2})
                    MERGE (a)-[r:SIMILAR_TO]-(b)
                    SET r.score = row.score, r.reason = row.reason
                """
        with self.driver.session() as session:
            batch_size = 5000  # 增大批量大小提高写入效率
            for i in tqdm(range(0, len(similarity_edges), batch_size), desc="写入Neo4j"):
                batch = similarity_edges[i:i + batch_size]
                session.run(f"UNWIND $batch as row {write_query}", batch=batch)

        print(f"    -> 相似边写入完成: {len(similarity_edges)} 条")

    def _compare_bucket(self, list1, list2, processed_pairs, edges, threshold):
        is_same_bucket = (list1 is list2)
        for i in range(len(list1)):
            start_j = i + 1 if is_same_bucket else 0
            for j in range(start_j, len(list2)):
                f1, f2 = list1[i], list2[j]
                pair_key = tuple(sorted((f1['sha256'], f2['sha256'])))
                if pair_key in processed_pairs: continue
                try:
                    s = ppdeep.compare(f1['ssdeep'], f2['ssdeep'])
                    if s >= threshold:
                        processed_pairs.add(pair_key)
                        edges.append({'sha1': f1['sha256'], 'sha2': f2['sha256'], 'score': s, 'reason': f"SSDEEP={s}"})
                except (ppdeep.ppdeep.Error, ValueError):
                    # SSDeep比较失败，跳过这对文件
                    pass

    def _run_tlsh_fallback(self, files, processed_pairs, edges, threshold):
        n = len(files)
        for i in range(n):
            for j in range(i + 1, n):
                f1, f2 = files[i], files[j]
                pair_key = tuple(sorted((f1['sha256'], f2['sha256'])))
                if pair_key in processed_pairs: continue
                if f1['tlsh'] and f2['tlsh']:
                    try:
                        d = tlsh.diff(f1['tlsh'], f2['tlsh'])
                        if d <= threshold:
                            similarity_score = max(0, 100 - d)
                            edges.append({'sha1': f1['sha256'], 'sha2': f2['sha256'], 'score': similarity_score,
                                          'reason': f"TLSH={d}"})
                            processed_pairs.add(pair_key)
                    except (ValueError, tlsh.TlshError):
                        # TLSH比较失败，跳过这对文件
                        pass

    def get_statistics(self):
        print("\n=== 最终图谱统计 (Graph Statistics) ===")
        node_types = [
            ('Event', 'EVENT'),
            ('IP', 'IP'),
            ('Domain', 'domain'),
            ('URL', 'URL'),
            ('File', 'File'),
            ('ASN', 'ASN'),
            ('CVE', 'CVE'),
            ('Technique', 'Technique'),
            ('Tactic', 'Tactic')
        ]

        with self.driver.session() as session:
            total_nodes = 0
            for name, label in node_types:
                try:
                    query = f"MATCH (n:{label}) RETURN count(n)"
                    count = session.run(query).single()[0]
                    print(f"  {name:<15}: {count:,}")
                    total_nodes += count
                except Exception as e:
                    print(f"  {name:<15}: 0 (未找到或标签不匹配)")

            print("-" * 30)
            print(f"  {'Total Nodes':<15}: {total_nodes:,}")

            try:
                orphan_query = "MATCH (n) WHERE NOT (n)--() RETURN count(n)"
                orphans = session.run(orphan_query).single()[0]
                if orphans > 0:
                    print(f"\n  [!] 警告: 发现 {orphans} 个孤立节点 (Orphan Nodes)")
                else:
                    print(f"\n  [√] 质量检查通过: 无孤立节点")
            except:
                pass

            # 统计TTP相关关系
            try:
                event_tech_query = "MATCH (:EVENT)-[r:USES_TECHNIQUE]->(:Technique) RETURN count(r)"
                event_tech_rels = session.run(event_tech_query).single()[0]
                print(f"  EVENT-Technique关系: {event_tech_rels:,}")
            except:
                pass

            try:
                tech_tactic_query = "MATCH (:Technique)-[r:BELONGS_TO]->(:Tactic) RETURN count(r)"
                tech_tactic_rels = session.run(tech_tactic_query).single()[0]
                print(f"  Technique-Tactic关系: {tech_tactic_rels:,}")
            except:
                pass

    def cleanup_orphan_events(self):
        """
        清理孤立EVENT节点（没有任何连接的EVENT）
        在所有数据导入完成后调用
        """
        print("\n[清理] 删除孤立EVENT节点...")

        # 统计孤立节点数量
        count_query = """
        MATCH (e:EVENT)
        WHERE NOT (e)--()
        RETURN count(e) as orphan_count
        """

        # 删除孤立节点
        delete_query = """
        MATCH (e:EVENT)
        WHERE NOT (e)--()
        DETACH DELETE e
        RETURN count(e) as deleted_count
        """

        with self.driver.session() as session:
            # 先统计
            result = session.run(count_query)
            orphan_count = result.single()['orphan_count']
            print(f"  发现 {orphan_count} 个孤立EVENT节点")

            # 删除
            if orphan_count > 0:
                result = session.run(delete_query)
                deleted_count = result.single()['deleted_count']
                print(f"  已删除 {deleted_count} 个孤立EVENT节点")
            else:
                print(f"  没有孤立EVENT节点")


def main():
    NEO4J_URI = "bolt://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neo4j123"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    THREAT_DATA_DIR = os.path.join(BASE_DIR, "src/output/")
    EVENT_TTPS_FILE = os.path.join(BASE_DIR, "src/test_dataset/cti/ttp_features_by_organization.json")
    CVE_DATA_DIR = os.path.join(BASE_DIR, "src/test_dataset/cve/dataset/")
    FILE_CSV_PATH = os.path.join(BASE_DIR, "src/test_dataset/apt_filtered.csv")

    print("\n" + "="*60)
    print("使用过滤后的数据构建知识图谱")
    print("="*60)
    print(f"  IOC数据: {THREAT_DATA_DIR}")
    print(f"  TTP数据: {EVENT_TTPS_FILE}")
    print(f"  CVE数据: {CVE_DATA_DIR}")
    print(f"  File数据: {FILE_CSV_PATH}")
    print("="*60 + "\n")

    builder = TrailNeo4jBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        builder.clear_database()
        builder.create_constraints()
        builder.import_threat_events(THREAT_DATA_DIR)

        # TTP节点导入
        if os.path.exists(EVENT_TTPS_FILE):
            builder.import_ttp_nodes(EVENT_TTPS_FILE)

        if os.path.exists(CVE_DATA_DIR): builder.import_cve_data(CVE_DATA_DIR)
        if os.path.exists(FILE_CSV_PATH):
            builder.import_file_csv(FILE_CSV_PATH)
            builder.build_file_similarity_edges()

        builder.get_statistics()
    finally:
        builder.close()


if __name__ == '__main__':
    main()