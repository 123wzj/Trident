import argparse
import json
import glob
import os
import csv
import math
from pathlib import Path
from tqdm import tqdm
from neo4j import GraphDatabase
from urllib.parse import urlparse
import ppdeep
import tlsh

try:
    from .config import add_neo4j_args, require_password
except ImportError:
    from config import add_neo4j_args, require_password


class TrailNeo4jBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            try:
                constraints = session.run("SHOW CONSTRAINTS YIELD name")
                for record in constraints:
                    session.run(f"DROP CONSTRAINT {record['name']}")

                indexes = session.run("SHOW INDEXES YIELD name, type")
                for record in indexes:
                    if "LOOKUP" not in record.get("type", "").upper():
                        try:
                            session.run(f"DROP INDEX {record['name']}")
                        except Exception:
                            pass
            except Exception as e:
                print(f"[WARN] Schema cleanup skipped: {e}")

    def create_constraints(self):

        constraints = [
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:EVENT) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT ip_value IF NOT EXISTS FOR (i:IP) REQUIRE i.value IS UNIQUE",
            "CREATE CONSTRAINT domain_value IF NOT EXISTS FOR (d:domain) REQUIRE d.value IS UNIQUE",
            "CREATE CONSTRAINT url_value IF NOT EXISTS FOR (u:URL) REQUIRE u.value IS UNIQUE",
            "CREATE CONSTRAINT asn_value IF NOT EXISTS FOR (a:ASN) REQUIRE a.value IS UNIQUE",
            "CREATE CONSTRAINT file_sha256 IF NOT EXISTS FOR (f:File) REQUIRE f.sha256 IS UNIQUE",
            "CREATE CONSTRAINT cve_id IF NOT EXISTS FOR (c:CVE) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT technique_id IF NOT EXISTS FOR (t:Technique) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT tactic_id IF NOT EXISTS FOR (t:Tactic) REQUIRE t.id IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX file_apt_tag IF NOT EXISTS FOR (f:File) ON (f.apt_tag)",
            "CREATE INDEX file_ssdeep IF NOT EXISTS FOR (f:File) ON (f.ssdeep)",
            "CREATE INDEX file_tlsh IF NOT EXISTS FOR (f:File) ON (f.tlsh)",
            "CREATE INDEX event_label IF NOT EXISTS FOR (e:EVENT) ON (e.label)",
            "CREATE INDEX event_tags IF NOT EXISTS FOR (e:EVENT) ON (e.tags)",
        ]

        with self.driver.session() as session:
            for c in constraints:
                try:
                    session.run(c)
                except Exception as e:
                    print(f"[WARN] Constraint failed: {e}")

            for idx in indexes:
                try:
                    session.run(idx)
                except Exception as e:
                    print(f"[WARN] Index failed: {e}")

        print("Schema is ready")

    def import_threat_events(self, data_dir):
        if not os.path.exists(data_dir):
            return

        apt_dirs = [
            d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)
        ]
        events_batch = []
        ips_batch = []
        domains_batch = []
        urls_batch = []
        ip_asn_batch = []
        ip_dns_batch = []
        domain_dns_batch = []

        for apt_dir in tqdm(apt_dirs, desc="Importing IOC events"):
            json_files = glob.glob(os.path.join(apt_dir, "*.json"))
            apt_name = os.path.basename(apt_dir)

            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        event_data = json.load(f)

                    iocs = event_data.get("iocs", [])
                    if not iocs:
                        continue

                    event_id = event_data["event_id"]

                    description = None
                    if "details" in event_data and isinstance(
                        event_data["details"], dict
                    ):
                        description = event_data["details"].get("description")
                    events_batch.append(
                        {"id": event_id, "label": apt_name, "description": description}
                    )

                    for ioc_data in iocs:
                        ioc_val = ioc_data.get("ioc")
                        if not ioc_val:
                            continue

                        itype = ioc_data.get("type", "").upper()

                        try:
                            if itype in ["IP", "IPV4", "IPV6"]:
                                self._collect_ip_data(
                                    ioc_data,
                                    event_id,
                                    ips_batch,
                                    ip_asn_batch,
                                    ip_dns_batch,
                                )
                            elif itype in ["DOMAIN", "HOSTNAME"]:
                                self._collect_domain_data(
                                    ioc_data, event_id, domains_batch, domain_dns_batch
                                )
                            elif itype == "URL":
                                self._collect_url_data(ioc_data, event_id, urls_batch)
                        except Exception as e:
                            pass

                except json.JSONDecodeError as e:
                    tqdm.write(f"[WARN] Invalid JSON in {json_file}: {e}")
                except Exception as e:
                    tqdm.write(f"[WARN] Failed to parse {json_file}: {e}")

        with self.driver.session() as session:
            if events_batch:
                self._batch_create_events(session, events_batch)

            if ips_batch:
                self._batch_create_ips(session, ips_batch)

            if ip_asn_batch:
                self._batch_create_ip_asn(session, ip_asn_batch)

            if ip_dns_batch:
                self._batch_create_ip_dns(session, ip_dns_batch)

            if domains_batch:
                self._batch_create_domains(session, domains_batch)

            if domain_dns_batch:
                self._batch_create_domain_dns(session, domain_dns_batch)

            if urls_batch:
                self._batch_create_urls(session, urls_batch)

        self.cleanup_orphan_events()

    def _collect_ip_data(
        self, ioc_data, event_id, ips_batch, ip_asn_batch, ip_dns_batch
    ):
        ip_value = ioc_data["ioc"]
        lat = ioc_data.get("latitude")
        lon = ioc_data.get("longitude")
        lat_norm = float(lat) / 90.0 if lat is not None else 0
        lon_norm = float(lon) / 180.0 if lon is not None else 0

        ips_batch.append(
            {
                "value": ip_value,
                "event_id": event_id,
                "country_code": ioc_data.get("country_code"),
                "city": ioc_data.get("city"),
                "region": ioc_data.get("region"),
                "latitude": lat,
                "longitude": lon,
                "lat_norm": lat_norm,
                "lon_norm": lon_norm,
            }
        )

        asn_str = ioc_data.get("asn")
        if asn_str:
            parts = asn_str.split(" ", 1)
            asn_val = parts[0]
            issuer = parts[1] if len(parts) > 1 else ""
            ip_asn_batch.append({"ip": ip_value, "asn": asn_val, "issuer": issuer})

        for res in ioc_data.get("resolves_to", []):
            host = res.get("host")
            if host:
                ip_dns_batch.append({"ip": ip_value, "host": host})

    def _collect_domain_data(self, ioc_data, event_id, domains_batch, domain_dns_batch):
        domain_value = ioc_data["ioc"]
        dns_records = ioc_data.get("dns_records", [])

        (
            first_seen,
            last_seen,
            has_nxdomain,
            lifespan_days,
            lifespan_log,
        ) = self._extract_domain_features(dns_records)

        domains_batch.append(
            {
                "value": domain_value,
                "event_id": event_id,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "has_nxdomain": has_nxdomain,
                "lifespan_days": lifespan_days,
                "lifespan_log": lifespan_log,
            }
        )

        for record in dns_records:
            addr = record.get("address")
            rtype = record.get("record_type")
            if rtype in ["A", "AAAA"] and addr and addr != "NXDOMAIN":
                domain_dns_batch.append({"domain": domain_value, "ip": addr})

    def _collect_url_data(self, ioc_data, event_id, urls_batch):
        url_value = ioc_data["ioc"]
        hostname = ioc_data.get("hostname")
        if not hostname:
            try:
                hostname = urlparse(url_value).netloc.split(":")[0]
            except:
                hostname = None

        urls_batch.append(
            {
                "value": url_value,
                "event_id": event_id,
                "hostname": hostname,
                "server": ioc_data.get("server"),
                "http_code": ioc_data.get("http_code"),
                "filetype": ioc_data.get("filetype"),
                "encoding": ioc_data.get("encoding"),
                "ip": ioc_data.get("ip"),
            }
        )

    def _extract_domain_features(self, dns_records):
        timestamps = []
        has_nxdomain = False

        for record in dns_records:
            if record.get("address") == "NXDOMAIN":
                has_nxdomain = True
            if record.get("first"):
                timestamps.append(record["first"])
            if record.get("last"):
                timestamps.append(record["last"])

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

                fs = datetime.fromisoformat(first_seen.replace("Z", "+00:00"))
                ls = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
                days = (ls - fs).days
                lifespan_days = max(0, days)
            except:
                pass

        lifespan_log = math.log1p(lifespan_days) if lifespan_days > 0 else 0
        return first_seen, last_seen, has_nxdomain, lifespan_days, lifespan_log

    def _batch_create_events(self, session, events_batch, batch_size=5000):
        query = """
            UNWIND $batch AS row
            MERGE (e:EVENT {id: row.id})
            SET e.label = row.label,
                e.description = row.description
        """
        for i in range(0, len(events_batch), batch_size):
            batch = events_batch[i : i + batch_size]
            session.run(query, batch=batch)

    def _batch_create_ips(self, session, ips_batch, batch_size=2000):
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
            batch = ips_batch[i : i + batch_size]
            session.run(query, batch=batch)

    def _batch_create_ip_asn(self, session, ip_asn_batch, batch_size=2000):
        query = """
            UNWIND $batch AS row
            MATCH (i:IP {value: row.ip})
            MERGE (a:ASN {value: row.asn})
            SET a.issuer = row.issuer
            MERGE (i)-[:BELONGS_TO_NETWORK]->(a)
        """
        for i in range(0, len(ip_asn_batch), batch_size):
            batch = ip_asn_batch[i : i + batch_size]
            session.run(query, batch=batch)

    def _batch_create_ip_dns(self, session, ip_dns_batch, batch_size=2000):
        query = """
            UNWIND $batch AS row
            MATCH (i:IP {value: row.ip})
            MERGE (d:domain {value: row.host})
            MERGE (i)-[:RESOLVES_TO]->(d)
        """
        for i in range(0, len(ip_dns_batch), batch_size):
            batch = ip_dns_batch[i : i + batch_size]
            session.run(query, batch=batch)

    def _batch_create_domains(self, session, domains_batch, batch_size=2000):
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
            batch = domains_batch[i : i + batch_size]
            session.run(query, batch=batch)

    def _batch_create_domain_dns(self, session, domain_dns_batch, batch_size=2000):
        query = """
            UNWIND $batch AS row
            MATCH (d:domain {value: row.domain})
            MERGE (i:IP {value: row.ip})
            MERGE (d)-[:RESOLVES_TO]->(i)
            MERGE (i)-[:RESOLVES_FROM]->(d)
        """
        for i in range(0, len(domain_dns_batch), batch_size):
            batch = domain_dns_batch[i : i + batch_size]
            session.run(query, batch=batch)

    def _batch_create_urls(self, session, urls_batch, batch_size=2000):
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

        query2 = """
            UNWIND $batch AS row
            MATCH (u:URL {value: row.value})
            MATCH (e:EVENT {id: row.event_id})
            WHERE row.hostname IS NOT NULL
            MERGE (d:domain {value: row.hostname})
            MERGE (u)-[:HOSTED_ON_DOMAIN]->(d)
        """

        query3 = """
            UNWIND $batch AS row
            MATCH (u:URL {value: row.value})
            WHERE row.ip IS NOT NULL
            MERGE (i:IP {value: row.ip})
            MERGE (u)-[:RESOLVES_TO_IP]->(i)
        """

        with_hostname = [r for r in urls_batch if r.get("hostname")]
        with_ip = [r for r in urls_batch if r.get("ip")]

        for i in range(0, len(urls_batch), batch_size):
            batch = urls_batch[i : i + batch_size]
            session.run(query1, batch=batch)

        if with_hostname:
            for i in range(0, len(with_hostname), batch_size):
                batch = with_hostname[i : i + batch_size]
                session.run(query2, batch=batch)

        if with_ip:
            for i in range(0, len(with_ip), batch_size):
                batch = with_ip[i : i + batch_size]
                session.run(query3, batch=batch)

    def import_ttp_nodes(self, ttp_features_file):

        if not os.path.exists(ttp_features_file):
            return

        with open(ttp_features_file, "r", encoding="utf-8") as f:
            ttp_data = json.load(f)

        all_tactics = {}
        all_techniques = {}
        event_techniques = []
        technique_tactics = []

        for org_name, pulse_list in ttp_data.items():
            for pulse in pulse_list:
                event_id = pulse.get("pulse_id")
                tactics = pulse.get("tactics", {})

                if not event_id:
                    continue

                for tactic_id, tactic_data in tactics.items():
                    if tactic_id not in all_tactics:
                        all_tactics[tactic_id] = {
                            "id": tactic_id,
                            "name": tactic_data.get("name", ""),
                            "description": tactic_data.get("description", ""),
                        }

                    techniques = tactic_data.get("techniques", [])
                    for tech in techniques:
                        tech_id = tech.get("id")
                        if not tech_id:
                            continue

                        if tech_id not in all_techniques:
                            all_techniques[tech_id] = {
                                "id": tech_id,
                                "name": tech.get("name", ""),
                                "description": tech.get("description", ""),
                            }

                        event_techniques.append((event_id, tech_id))

                        technique_tactics.append((tech_id, tactic_id))

        if not all_techniques or not all_tactics:
            print("[WARN] No TTP nodes found; skipping TTP import")
            return

        batch_size = 1000
        with self.driver.session() as session:
            tactic_list = list(all_tactics.values())
            create_tactic_query = """
                UNWIND $batch AS row
                MERGE (t:Tactic {id: row.id})
                SET t.name = row.name,
                    t.description = row.description
            """
            for i in range(0, len(tactic_list), batch_size):
                batch = tactic_list[i : i + batch_size]
                session.run(create_tactic_query, batch=batch)

            tech_list = list(all_techniques.values())
            create_tech_query = """
                UNWIND $batch AS row
                MERGE (t:Technique {id: row.id})
                SET t.name = row.name,
                    t.description = row.description
            """
            for i in range(0, len(tech_list), batch_size):
                batch = tech_list[i : i + batch_size]
                session.run(create_tech_query, batch=batch)

            event_techniques = list(set(event_techniques))
            create_event_tech_query = """
                UNWIND $batch AS row
                MATCH (e:EVENT {id: row.event_id})
                MATCH (t:Technique {id: row.tech_id})
                MERGE (e)-[:USES_TECHNIQUE]->(t)
            """
            for i in range(0, len(event_techniques), batch_size):
                batch = [
                    {"event_id": e, "tech_id": t}
                    for e, t in event_techniques[i : i + batch_size]
                ]
                session.run(create_event_tech_query, batch=batch)
            technique_tactics = list(set(technique_tactics))
            create_tech_tactic_query = """
                UNWIND $batch AS row
                MATCH (t:Technique {id: row.tech_id})
                MATCH (tac:Tactic {id: row.tactic_id})
                MERGE (t)-[:BELONGS_TO]->(tac)
            """
            for i in range(0, len(technique_tactics), batch_size):
                batch = [
                    {"tech_id": t, "tactic_id": tac}
                    for t, tac in technique_tactics[i : i + batch_size]
                ]
                session.run(create_tech_tactic_query, batch=batch)

    def import_cve_data(self, cve_dir):
        if not os.path.exists(cve_dir):
            print(f"[WARN] CVE directory not found: {cve_dir}")
            return
        json_files = glob.glob(os.path.join(cve_dir, "**", "*.json"), recursive=True)

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
            for json_file in tqdm(json_files, desc="Importing CVEs"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    eid = data.get("event_id")
                    cves = data.get("indicators", [])
                    if not cves or not eid:
                        continue
                    session.run(query, eid=eid, cves=cves)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    tqdm.write(f"[WARN] Failed to import CVEs from {json_file}: {e}")

    def import_file_csv(self, csv_path):
        if not os.path.exists(csv_path):
            print(f"[WARN] File metadata CSV not found: {csv_path}")
            return

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
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            batch = []
            with self.driver.session() as session:
                for row in tqdm(reader, desc="Importing files"):
                    if not row.get("event_id") or row["event_id"] == "Unknown":
                        continue
                    batch.append(row)
                    if len(batch) >= 2000:
                        session.run(f"UNWIND $batch as row {query}", batch=batch)
                        batch = []
                if batch:
                    session.run(f"UNWIND $batch as row {query}", batch=batch)

    def build_file_similarity_edges(self):
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
                apt = r["apt"]
                if apt not in apt_groups:
                    apt_groups[apt] = []
                apt_groups[apt].append(dict(r))

        similarity_edges = []
        TH_SSDEEP = 80
        TH_TLSH = 50
        total_files = sum(len(files) for files in apt_groups.values())

        for apt, files in tqdm(apt_groups.items(), desc="Comparing file groups"):
            n = len(files)
            if n < 2:
                continue
            processed_pairs = set()

            imphash_map = {}
            for f in files:
                imp = f["imphash"]
                if imp:
                    imphash_map.setdefault(imp, []).append(f)

            for imp, group in imphash_map.items():
                if len(group) > 1:
                    for i in range(len(group)):
                        for j in range(i + 1, len(group)):
                            f1, f2 = group[i], group[j]
                            pair_key = tuple(sorted((f1["sha256"], f2["sha256"])))
                            processed_pairs.add(pair_key)
                            similarity_edges.append(
                                {
                                    "sha1": f1["sha256"],
                                    "sha2": f2["sha256"],
                                    "score": 100,
                                    "reason": "Imphash_Match",
                                }
                            )

            ssdeep_buckets = {}
            for f in files:
                ssd = f["ssdeep"]
                if ssd:
                    try:
                        bs = int(ssd.split(":")[0])
                        ssdeep_buckets.setdefault(bs, []).append(f)
                    except (ValueError, IndexError, AttributeError):
                        continue

            block_sizes = sorted(ssdeep_buckets.keys())
            for i, bs in enumerate(block_sizes):
                current_bucket = ssdeep_buckets[bs]
                self._compare_bucket(
                    current_bucket,
                    current_bucket,
                    processed_pairs,
                    similarity_edges,
                    TH_SSDEEP,
                )
                if i + 1 < len(block_sizes):
                    next_bs = block_sizes[i + 1]

                    if next_bs == bs * 2:
                        next_bucket = ssdeep_buckets[next_bs]
                        self._compare_bucket(
                            current_bucket,
                            next_bucket,
                            processed_pairs,
                            similarity_edges,
                            TH_SSDEEP,
                        )

            if n < 1000:
                self._run_tlsh_fallback(
                    files, processed_pairs, similarity_edges, TH_TLSH
                )
            elif n >= 1000 and n < 5000:
                tlsh_files = [f for f in files if f.get("tlsh")]
                if len(tlsh_files) < 1000:
                    self._run_tlsh_fallback(
                        tlsh_files, processed_pairs, similarity_edges, TH_TLSH
                    )

        if not similarity_edges:
            print("No file similarity edges found")
            return
        write_query = """
                    MATCH (a:File {sha256: row.sha1})
                    MATCH (b:File {sha256: row.sha2})
                    MERGE (a)-[r:SIMILAR_TO]-(b)
                    SET r.score = row.score, r.reason = row.reason
                """
        with self.driver.session() as session:
            batch_size = 5000
            for i in tqdm(range(0, len(similarity_edges), batch_size), desc="Writing similarity edges"):
                batch = similarity_edges[i : i + batch_size]
                session.run(f"UNWIND $batch as row {write_query}", batch=batch)

    def _compare_bucket(self, list1, list2, processed_pairs, edges, threshold):
        is_same_bucket = list1 is list2
        for i in range(len(list1)):
            start_j = i + 1 if is_same_bucket else 0
            for j in range(start_j, len(list2)):
                f1, f2 = list1[i], list2[j]
                pair_key = tuple(sorted((f1["sha256"], f2["sha256"])))
                if pair_key in processed_pairs:
                    continue
                try:
                    s = ppdeep.compare(f1["ssdeep"], f2["ssdeep"])
                    if s >= threshold:
                        processed_pairs.add(pair_key)
                        edges.append(
                            {
                                "sha1": f1["sha256"],
                                "sha2": f2["sha256"],
                                "score": s,
                                "reason": f"SSDEEP={s}",
                            }
                        )
                except (ppdeep.ppdeep.Error, ValueError):
                    pass

    def _run_tlsh_fallback(self, files, processed_pairs, edges, threshold):
        n = len(files)
        for i in range(n):
            for j in range(i + 1, n):
                f1, f2 = files[i], files[j]
                pair_key = tuple(sorted((f1["sha256"], f2["sha256"])))
                if pair_key in processed_pairs:
                    continue
                if f1["tlsh"] and f2["tlsh"]:
                    try:
                        d = tlsh.diff(f1["tlsh"], f2["tlsh"])
                        if d <= threshold:
                            similarity_score = max(0, 100 - d)
                            edges.append(
                                {
                                    "sha1": f1["sha256"],
                                    "sha2": f2["sha256"],
                                    "score": similarity_score,
                                    "reason": f"TLSH={d}",
                                }
                            )
                            processed_pairs.add(pair_key)
                    except (ValueError, tlsh.TlshError):
                        pass

    def get_statistics(self):
        node_types = [
            ("Event", "EVENT"),
            ("IP", "IP"),
            ("Domain", "domain"),
            ("URL", "URL"),
            ("File", "File"),
            ("ASN", "ASN"),
            ("CVE", "CVE"),
            ("Technique", "Technique"),
            ("Tactic", "Tactic"),
        ]

        with self.driver.session() as session:
            total_nodes = 0
            for name, label in node_types:
                try:
                    query = f"MATCH (n:{label}) RETURN count(n)"
                    count = session.run(query).single()[0]
                    total_nodes += count
                except Exception as e:
                    print(f"  {name:<15}: unavailable ({e})")
            try:
                orphan_query = "MATCH (n) WHERE NOT (n)--() RETURN count(n)"
                orphans = session.run(orphan_query).single()[0]
                if orphans > 0:
                    print(f"  Orphan Nodes    : {orphans:,}")
                else:
                    print("  Orphan Nodes    : 0")
            except:
                pass

            try:
                event_tech_query = (
                    "MATCH (:EVENT)-[r:USES_TECHNIQUE]->(:Technique) RETURN count(r)"
                )
                event_tech_rels = session.run(event_tech_query).single()[0]
            except:
                pass

            try:
                tech_tactic_query = (
                    "MATCH (:Technique)-[r:BELONGS_TO]->(:Tactic) RETURN count(r)"
                )
                tech_tactic_rels = session.run(tech_tactic_query).single()[0]
            except:
                pass

    def cleanup_orphan_events(self):
        count_query = """
        MATCH (e:EVENT)
        WHERE NOT (e)--()
        RETURN count(e) as orphan_count
        """

        delete_query = """
        MATCH (e:EVENT)
        WHERE NOT (e)--()
        DETACH DELETE e
        RETURN count(e) as deleted_count
        """

        with self.driver.session() as session:
            result = session.run(count_query)
            orphan_count = result.single()["orphan_count"]

            if orphan_count > 0:
                result = session.run(delete_query)
                deleted_count = result.single()["deleted_count"]
            else:
                print("No orphan events to delete")


def main():
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Build the Neo4j knowledge graph.")
    add_neo4j_args(parser)
    parser.add_argument(
        "--threat-data-dir",
        default=str(project_root / "src" / "build" / "output_filtered"),
    )
    parser.add_argument(
        "--ttp-file",
        default=str(project_root / "src" / "build" / "ttp_features_filtered.json"),
    )
    parser.add_argument(
        "--cve-data-dir",
        default=str(project_root / "src" / "build" / "cve_filtered"),
    )
    parser.add_argument(
        "--file-csv",
        default=str(project_root / "src" / "build" / "apt_filtered_filtered.csv"),
    )
    parser.add_argument("--skip-clear", action="store_true")
    parser.add_argument("--skip-file-similarity", action="store_true")
    args = parser.parse_args()

    builder = TrailNeo4jBuilder(args.uri, args.user, require_password(args.password))
    try:
        if not args.skip_clear:
            builder.clear_database()
        builder.create_constraints()
        builder.import_threat_events(args.threat_data_dir)

        if os.path.exists(args.ttp_file):
            builder.import_ttp_nodes(args.ttp_file)

        if os.path.exists(args.cve_data_dir):
            builder.import_cve_data(args.cve_data_dir)
        if os.path.exists(args.file_csv):
            builder.import_file_csv(args.file_csv)
            if not args.skip_file_similarity:
                builder.build_file_similarity_edges()

        builder.get_statistics()
    finally:
        builder.close()


if __name__ == "__main__":
    main()
