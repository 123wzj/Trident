"""Incrementally update the Neo4j knowledge graph."""
import json
import glob
import os
from tqdm import tqdm
from neo4j import GraphDatabase

try:
    import ppdeep
except ImportError:
    ppdeep = None

try:
    import tlsh
except ImportError:
    tlsh = None


# Neo4j import helpers.
class TrailNeo4jIncrementalUpdater:
    """Incremental Neo4j graph updater."""

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"[*] Connected to Neo4j: {uri}")

    def cleanup_isolated_events(self):
        """Remove EVENT nodes without IOC or TTP edges."""
        print("\n[Cleanup] Checking isolated EVENT nodes...")

        with self.driver.session() as session:
            query = """
                MATCH (e:EVENT)
                WHERE NOT (e)-[:USES_INFRASTRUCTURE|USES_DOMAIN|DELIVERS_VIA_URL|EXPLOITS_VULN|DROPS_MALWARE|USES_TECHNIQUE]->()
                RETURN count(e) as isolated_count
            """
            result = session.run(query).single()
            isolated_count = result["isolated_count"]

            if isolated_count == 0:
                print("  [OK] No isolated EVENT nodes found.")
                return 0

            print(f"  Found {isolated_count} isolated EVENT nodes. Deleting...")

            delete_query = """
                MATCH (e:EVENT)
                WHERE NOT (e)-[:USES_INFRASTRUCTURE|USES_DOMAIN|DELIVERS_VIA_URL|EXPLOITS_VULN|DROPS_MALWARE|USES_TECHNIQUE]->()
                DETACH DELETE e
                RETURN count(e) as deleted_count
            """
            result = session.run(delete_query).single()
            deleted_count = result["deleted_count"]

            print(f"  [OK] Deleted {deleted_count} isolated EVENT nodes.")
            return deleted_count

    def close(self):
        self.driver.close()

    def check_database_health(self):
        """Print basic Neo4j node statistics."""
        with self.driver.session() as session:
            stats = session.run(
                """
                MATCH (n)
                RETURN labels(n) as label, count(n) as count
                ORDER BY count DESC
            """
            ).data()

            print("\n[DB] Node statistics:")
            for stat in stats:
                print(
                    f"  {stat['label'][0] if stat['label'] else 'Unknown'}: {stat['count']:,}"
                )

            return stats

    def incremental_import_events(
        self, new_data_dir: str, batch_size: int = 100, valid_event_ids=None
    ):
        """Import IOC event data for existing TTP events."""
        print(f"\n[Import] Event IOC data: {new_data_dir}")

        if valid_event_ids is not None:
            print(f"  [Filter] TTP events: {len(valid_event_ids)}")

        if not os.path.exists(new_data_dir):
            print(f"[!] Missing directory: {new_data_dir}")
            return {"success": 0, "failed": 0, "skipped": 0}

        apt_dirs = [
            d for d in glob.glob(os.path.join(new_data_dir, "*")) if os.path.isdir(d)
        ]

        stats = {"success": 0, "failed": 0, "skipped": 0, "filtered": 0}
        valid_event_set = set(valid_event_ids) if valid_event_ids is not None else None

        with self.driver.session() as session:
            for apt_dir in tqdm(apt_dirs, desc="Event"):
                apt_name = os.path.basename(apt_dir)
                json_files = glob.glob(os.path.join(apt_dir, "*.json"))

                for json_file in json_files:
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            event_data = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"\n[!] JSON {json_file}: {e}")
                        stats["failed"] += 1
                        continue
                    except Exception as e:
                        print(f"\n[!] File processing failed {json_file}: {e}")
                        stats["failed"] += 1
                        continue

                    try:
                        event_id = event_data["event_id"]

                        if (
                            valid_event_set is not None
                            and event_id not in valid_event_set
                        ):
                            stats["filtered"] += 1
                            continue

                        iocs = event_data.get("iocs", [])
                        if not iocs:
                            stats["skipped"] += 1
                            continue

                        existing = session.run(
                            "MATCH (e:EVENT {id: $eid}) RETURN count(e) as count",
                            eid=event_id,
                        ).single()

                        event_exists = existing["count"] > 0

                        if event_exists:
                            connected_count = 0
                            for ioc_data in iocs:
                                ioc_val = ioc_data.get("ioc")
                                if not ioc_val:
                                    continue

                                itype = ioc_data.get("type", "").upper()
                                try:
                                    if itype in ["IP", "IPV4", "IPV6"]:
                                        self._process_ip(session, event_id, ioc_data)
                                        connected_count += 1
                                    elif itype in ["DOMAIN", "HOSTNAME"]:
                                        self._process_domain(
                                            session, event_id, ioc_data
                                        )
                                        connected_count += 1
                                    elif itype == "URL":
                                        self._process_url(session, event_id, ioc_data)
                                        connected_count += 1
                                except Exception as e:
                                    pass

                            if connected_count > 0:
                                stats["success"] += 1
                            else:
                                stats["skipped"] += 1
                        else:
                            if "missing_event" not in stats:
                                stats["missing_event"] = 0
                            stats["missing_event"] += 1

                            if stats["missing_event"] == 1:
                                print(
                                    "\n  [Warn] EVENT is listed in TTP data but missing from Neo4j; skipped."
                                )
                                print(
                                    "         Check the TTP import step for failed EVENT creation."
                                )

                    except Exception as e:
                        error_msg = str(e)
                        if (
                            "Property value is too large to index" in error_msg
                            or "property size" in error_msg
                        ):
                            print(
                                f"\n[!] tags too large for index {json_file}; retrying without tags."
                            )
                            try:
                                session.run(
                                    """
                                    MERGE (e:EVENT {id: $eid})
                                    SET e.label = $apt,
                                        e.source = $src,
                                        e.name = $name,
                                        e.description = $desc,
                                        e.created_at = datetime()
                                """,
                                    eid=event_id,
                                    apt=apt_name,
                                    src=event_data.get("source", "OTX"),
                                    name=event_data.get("details", {}).get("name", ""),
                                    desc=event_data.get("details", {}).get(
                                        "description", ""
                                    ),
                                )
                                stats["success"] += 1
                            except:
                                print(
                                    f"\n[!] Import still failed without tags: {json_file}"
                                )
                                stats["failed"] += 1
                        else:
                            print(f"\n[!] File processing failed {json_file}: {e}")
                            stats["failed"] += 1

        msg = f"\n[Import] Event IOC import done: success={stats['success']}, skipped={stats['skipped']}, failed={stats['failed']}"
        if stats.get("filtered", 0) > 0:
            msg += f", filtered_no_ttp={stats['filtered']}"
        if stats.get("missing_event", 0) > 0:
            msg += f", missing_event={stats['missing_event']}"
        print(msg)
        return stats

    def incremental_import_ttp(self, ttp_file: str, create_missing_nodes=True):
        """Import TTP nodes and relationships."""
        print(f"\n[Import] TTP data: {ttp_file}")

        if not os.path.exists(ttp_file):
            print(f"[!] Missing file: {ttp_file}")
            return []

        with open(ttp_file, "r", encoding="utf-8") as f:
            ttp_data = json.load(f)

        event_techniques = []  # [(event_id, technique_id)]
        technique_tactics = []  # [(technique_id, tactic_id)]
        valid_event_ids = set()

        technique_details = {}  # {tech_id: {name, description}}
        tactic_details = {}  # {tactic_id: {name, description}}

        event_details = {}  # {event_id: {label (org_name), source}}

        all_techniques = set()
        all_tactics = set()

        for org_name, pulse_list in ttp_data.items():
            for pulse in pulse_list:
                event_id = pulse.get("pulse_id")
                tactics = pulse.get("tactics", {})

                if not event_id:
                    continue

                valid_event_ids.add(event_id)

                if event_id not in event_details:
                    event_details[event_id] = {
                        "id": event_id,
                        "label": org_name,
                        "source": pulse.get("source", "TTP"),
                    }

                for tactic_id, tactic_data in tactics.items():
                    all_tactics.add(tactic_id)

                    if tactic_id not in tactic_details:
                        tactic_details[tactic_id] = {
                            "id": tactic_id,
                            "name": tactic_data.get("name", ""),
                            "description": tactic_data.get("description", ""),
                        }

                    techniques = tactic_data.get("techniques", [])
                    for tech in techniques:
                        tech_id = tech.get("id")
                        if not tech_id:
                            continue

                        all_techniques.add(tech_id)

                        if tech_id not in technique_details:
                            technique_details[tech_id] = {
                                "id": tech_id,
                                "name": tech.get("name", ""),
                                "description": tech.get("description", ""),
                            }

                        event_techniques.append((event_id, tech_id))

                        technique_tactics.append((tech_id, tactic_id))

        print(f"  TTP events: {len(valid_event_ids)}")
        print(f"  Unique tactics: {len(all_tactics)}")
        print(f"  Unique techniques: {len(all_techniques)}")
        print(f"  EVENT-Technique pairs: {len(event_techniques)}")

        if not event_techniques:
            print("  [!] No valid TTP data found.")
            return []

        batch_size = 1000
        stats = {
            "event_technique_rels": 0,
            "technique_tactic_rels": 0,
            "created_technique_nodes": 0,
            "created_tactic_nodes": 0,
            "created_event_nodes": 0,
            "missing_technique_nodes": 0,
            "missing_tactic_nodes": 0,
        }

        with self.driver.session() as session:
            print("  -> Ensure EVENT nodes...")
            existing_events = session.run(
                """
                MATCH (e:EVENT)
                WHERE e.id IN $event_ids
                RETURN e.id as id
            """,
                event_ids=list(valid_event_ids),
            )

            existing_event_ids = set()
            for record in existing_events:
                existing_event_ids.add(record["id"])

            missing_events = valid_event_ids - existing_event_ids
            if missing_events:
                print(f"     Creating missing EVENT nodes: {len(missing_events)}")
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
                    batch = [
                        event_details[eid] for eid in event_list[i : i + batch_size]
                    ]
                    result = session.run(create_events_query, batch=batch).single()
                    stats["created_event_nodes"] += result["count"]
                print(
                    f"     EVENT: {len(existing_event_ids)} + {stats['created_event_nodes']} = {len(valid_event_ids)} "
                )
            else:
                print(f"     All EVENT nodes already exist: {len(valid_event_ids)}")

            print("  -> Ensure Tactic nodes...")
            tactic_list = list(all_tactics)
            create_tactics_query = """
                UNWIND $batch AS row
                MERGE (t:Tactic {id: row.id})
                SET t.name = row.name,
                    t.description = row.description
                RETURN count(t) as count
            """
            for i in range(0, len(tactic_list), batch_size):
                batch = [tactic_details[tid] for tid in tactic_list[i : i + batch_size]]
                result = session.run(create_tactics_query, batch=batch).single()
                stats["created_tactic_nodes"] += result["count"]
            print(f"     Tactic nodes touched: {stats['created_tactic_nodes']}")

            print("  -> Ensure Technique nodes...")
            existing_techniques = session.run(
                """
                MATCH (t:Technique)
                WHERE t.id IN $tech_ids
                RETURN t.id as id
            """,
                tech_ids=list(all_techniques),
            )

            existing_tech_ids = set()
            for record in existing_techniques:
                existing_tech_ids.add(record["id"])

            missing_techniques = all_techniques - existing_tech_ids
            if missing_techniques and create_missing_nodes:
                print(
                    f"     Creating missing Technique nodes: {len(missing_techniques)}"
                )
                create_techniques_query = """
                    UNWIND $batch AS row
                    MERGE (t:Technique {id: row.id})
                    SET t.name = row.name,
                        t.description = row.description
                    RETURN count(t) as count
                """
                tech_list = list(missing_techniques)
                for i in range(0, len(tech_list), batch_size):
                    batch = [
                        technique_details[tid] for tid in tech_list[i : i + batch_size]
                    ]
                    result = session.run(create_techniques_query, batch=batch).single()
                    stats["created_technique_nodes"] += result["count"]
                print(
                    f"     Technique: {len(existing_tech_ids)} + {stats['created_technique_nodes']} = {len(all_techniques)} "
                )
            elif missing_techniques:
                print(f"     [Warn] missing techniques: {len(missing_techniques)}")
                stats["missing_technique_nodes"] = len(missing_techniques)

            print(
                f"  -> Create EVENT-USES_TECHNIQUE relationships: {len(event_techniques)}"
            )
            event_techniques = list(set(event_techniques))

            create_event_tech_query = """
                UNWIND $batch AS row
                MATCH (e:EVENT {id: row.event_id})
                MATCH (t:Technique {id: row.tech_id})
                MERGE (e)-[:USES_TECHNIQUE]->(t)
                RETURN count(*) as count
            """

            for i in range(0, len(event_techniques), batch_size):
                batch = [
                    {"event_id": e, "tech_id": t}
                    for e, t in event_techniques[i : i + batch_size]
                ]
                result = session.run(create_event_tech_query, batch=batch).single()
                stats["event_technique_rels"] += result["count"]

            print(
                f"     EVENT-Technique relationships: {stats['event_technique_rels']}"
            )

            print("  -> Create Technique-BELONGS_TO-Tactic relationships...")
            technique_tactics = list(set(technique_tactics))
            create_tech_tactic_query = """
                UNWIND $batch AS row
                MATCH (t:Technique {id: row.tech_id})
                MATCH (tac:Tactic {id: row.tactic_id})
                MERGE (t)-[:BELONGS_TO]->(tac)
                RETURN count(*) as count
            """

            for i in range(0, len(technique_tactics), batch_size):
                batch = [
                    {"tech_id": t, "tactic_id": tac}
                    for t, tac in technique_tactics[i : i + batch_size]
                ]
                result = session.run(create_tech_tactic_query, batch=batch).single()
                stats["technique_tactic_rels"] += result["count"]

        print("\n  [TTP] Import summary")
        print(f"     EVENT: {stats['created_event_nodes']}")
        print(f"     EVENT-Technique: {stats['event_technique_rels']}")
        print(f"     Technique: {stats['created_technique_nodes']}")
        print(f"     Tactic: {stats['created_tactic_nodes']}")
        print(f"     Technique-Tactic: {stats['technique_tactic_rels']}")
        print(f"     Valid events: {len(valid_event_ids)}")

        return list(valid_event_ids)

    def incremental_import_mitre_tags(self, new_tags_file: str):
        """Deprecated MITRE tag import helper."""
        print(f"\n[Import] MITRE tags: {new_tags_file}")
        print(f"  [Warn] Deprecated. Use incremental_import_ttp().")

        if not os.path.exists(new_tags_file):
            print(f"[!] Missing file: {new_tags_file}")
            return

        with open(new_tags_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        query = """
        MATCH (e:EVENT {id: $pulse_id})
        SET e.tags = COALESCE(e.tags, []) + [t IN $new_tags WHERE NOT t IN e.tags]
        """

        stats = {"updated": 0, "not_found": 0}

        with self.driver.session() as session:
            for org_name, event_list in tqdm(data.items(), desc="TTPs"):
                if not isinstance(event_list, list):
                    continue

                for event_item in event_list:
                    pulse_id = event_item.get("id")
                    if not pulse_id:
                        continue

                    existing = session.run(
                        "MATCH (e:EVENT {id: $eid}) RETURN count(e) as count",
                        eid=pulse_id,
                    ).single()

                    if existing["count"] == 0:
                        stats["not_found"] += 1
                        continue

                    raw_attack_ids = event_item.get("attack_ids", [])
                    if isinstance(raw_attack_ids, list) and raw_attack_ids:
                        tech_ids_to_add = list(
                            set([str(tid) for tid in raw_attack_ids if tid])
                        )
                        if tech_ids_to_add:
                            session.run(
                                query, pulse_id=pulse_id, new_tags=tech_ids_to_add
                            )
                            stats["updated"] += 1

        print(f"[MITRE] updated={stats['updated']}, not_found={stats['not_found']}")
        return stats

    def incremental_import_cve(self, new_cve_dir: str, valid_event_ids=None):
        """Import CVE relationships for existing events."""
        print(f"\n[Import] CVE data: {new_cve_dir}")

        if not os.path.exists(new_cve_dir):
            return

        valid_event_set = set(valid_event_ids) if valid_event_ids is not None else None
        if valid_event_set is not None:
            print(f"  [Filter] TTP events for CVE: {len(valid_event_set)}")

        json_files = glob.glob(
            os.path.join(new_cve_dir, "**", "*.json"), recursive=True
        )
        query = """
            MATCH (e:EVENT {id: $eid})
            UNWIND $cves as cve_id
            MERGE (c:CVE {id: cve_id})
            MERGE (e)-[:EXPLOITS]->(c)
        """

        stats = {
            "success": 0,
            "failed": 0,
            "skipped_no_event": 0,
            "skipped_no_cve": 0,
            "filtered_no_ttp": 0,
        }

        with self.driver.session() as session:
            for json_file in tqdm(json_files, desc="CVE"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    eid = data.get("event_id")
                    cves = data.get("indicators", [])

                    if not eid:
                        stats["skipped_no_event"] += 1
                        continue

                    if not cves:
                        stats["skipped_no_cve"] += 1
                        continue

                    if valid_event_set is not None and eid not in valid_event_set:
                        stats["filtered_no_ttp"] += 1
                        continue

                    existing = session.run(
                        "MATCH (e:EVENT {id: $eid}) RETURN count(e) as count", eid=eid
                    ).single()

                    if existing["count"] > 0:
                        session.run(query, eid=eid, cves=cves)
                        stats["success"] += 1
                    else:
                        stats["skipped_no_event"] += 1

                except Exception as e:
                    stats["failed"] += 1

        msg = f"[CVE] success={stats['success']}, failed={stats['failed']}, "
        msg += f"skipped_no_event={stats['skipped_no_event']}, skipped_no_cve={stats['skipped_no_cve']}"
        if stats.get("filtered_no_ttp", 0) > 0:
            msg += f", filtered_no_ttp={stats['filtered_no_ttp']}"
        print(msg)
        return stats

    def incremental_import_files(self, new_csv_path: str, valid_event_ids=None):
        """Import file hash relationships for existing events."""
        print(f"\n[Import] File hashes: {new_csv_path}")

        if not os.path.exists(new_csv_path):
            return

        valid_event_set = set(valid_event_ids) if valid_event_ids is not None else None
        if valid_event_set is not None:
            print(f"  [Filter] TTP events for files: {len(valid_event_set)}")

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

        stats = {
            "success": 0,
            "failed": 0,
            "skipped_no_event": 0,
            "skipped_invalid": 0,
            "filtered_no_ttp": 0,
        }

        import csv

        with open(new_csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            batch = []

            with self.driver.session() as session:
                for row in tqdm(reader, desc="File"):
                    eid = row.get("event_id")

                    if not eid or eid == "Unknown":
                        stats["skipped_invalid"] += 1
                        continue

                    if valid_event_set is not None and eid not in valid_event_set:
                        stats["filtered_no_ttp"] += 1
                        continue

                    existing = session.run(
                        "MATCH (e:EVENT {id: $eid}) RETURN count(e) as count", eid=eid
                    ).single()

                    if existing["count"] > 0:
                        batch.append(row)
                        if len(batch) >= 2000:
                            session.run(f"UNWIND $batch as row {query}", batch=batch)
                            stats["success"] += len(batch)
                            batch = []
                    else:
                        stats["skipped_no_event"] += 1

                if batch:
                    session.run(f"UNWIND $batch as row {query}", batch=batch)
                    stats["success"] += len(batch)

        msg = f"[File] success={stats['success']}, failed={stats['failed']}, "
        msg += f"skipped_no_event={stats['skipped_no_event']}, skipped_invalid={stats['skipped_invalid']}"
        if stats.get("filtered_no_ttp", 0) > 0:
            msg += f", filtered_no_ttp={stats['filtered_no_ttp']}"
        print(msg)
        return stats

    def incremental_update_similarity(self):
        """Incremental update similarity."""
        print("\n[Update] Calculating file similarity...")

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
                apt = r["apt"]
                if apt not in apt_groups:
                    apt_groups[apt] = []
                apt_groups[apt].append(dict(r))

        if not apt_groups:
            print("  No pending items.")
            return

        similarity_edges = []
        TH_SSDEEP = 80
        TH_TLSH = 50

        for apt, files in tqdm(apt_groups.items(), desc="APT"):
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
                    except:
                        pass

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

            if n < 5000:
                self._run_tlsh_fallback(
                    files, processed_pairs, similarity_edges, TH_TLSH
                )

        if not similarity_edges:
            print("  No pending items.")
            return

        print(f"  Writing similarity edges: {len(similarity_edges)}")
        write_query = """
            MATCH (a:File {sha256: row.sha1})
            MATCH (b:File {sha256: row.sha2})
            MERGE (a)-[r:SIMILAR]-(b)
            SET r.score = row.score, r.reason = row.reason
        """

        with self.driver.session() as session:
            batch_size = 2000
            for i in tqdm(range(0, len(similarity_edges), batch_size), desc="Neo4j"):
                batch = similarity_edges[i : i + batch_size]
                session.run(f"UNWIND $batch as row {write_query}", batch=batch)

        print(f"  Similarity update done: {len(similarity_edges)} edges")

    def _process_ip(self, session, event_id, ioc_data):
        ip_value = ioc_data["ioc"]
        session.run(
            """
            MERGE (i:IP {value: $val})
            SET i.country_code = $cc, i.city = $city, i.region = $region,
                i.latitude = $lat, i.longitude = $lon
            WITH i
            MATCH (e:EVENT {id: $eid})
            MERGE (e)-[:InReport]->(i)
        """,
            val=ip_value,
            eid=event_id,
            cc=ioc_data.get("country_code"),
            city=ioc_data.get("city"),
            region=ioc_data.get("region"),
            lat=ioc_data.get("latitude"),
            lon=ioc_data.get("longitude"),
        )

        asn_str = ioc_data.get("asn")
        if asn_str:
            parts = asn_str.split(" ", 1)
            asn_val = parts[0]
            issuer = parts[1] if len(parts) > 1 else ""
            session.run(
                """
                MATCH (i:IP {value: $val})
                MERGE (a:ASN {value: $asn}) SET a.issuer = $iss
                MERGE (i)-[:InGroup]->(a)
            """,
                val=ip_value,
                asn=asn_val,
                iss=issuer,
            )

        for res in ioc_data.get("resolves_to", []):
            host = res.get("host")
            if host:
                session.run(
                    """
                    MATCH (i:IP {value: $val})
                    MERGE (d:domain {value: $host})
                    MERGE (i)-[:RESOLVES_TO]->(d)
                """,
                    val=ip_value,
                    host=host,
                )

    def _process_domain(self, session, event_id, ioc_data):
        domain_value = ioc_data["ioc"]
        dns_records = ioc_data.get("dns_records", [])
        first_seen, last_seen = None, None
        timestamps = []
        has_nxdomain = False

        for record in dns_records:
            if record.get("address") == "NXDOMAIN":
                has_nxdomain = True
            if record.get("first"):
                timestamps.append(record["first"])
            if record.get("last"):
                timestamps.append(record["last"])

        if timestamps:
            try:
                sorted_ts = sorted(timestamps)
                first_seen = sorted_ts[0]
                last_seen = sorted_ts[-1]
            except:
                pass

        session.run(
            """
            MERGE (d:domain {value: $val})
            SET d.first_seen = $fs, d.last_seen = $ls, d.has_nxdomain = $nx, d.type = 'Domain'
            WITH d
            MATCH (e:EVENT {id: $eid})
            MERGE (e)-[:InReport]->(d)
        """,
            val=domain_value,
            eid=event_id,
            fs=first_seen,
            ls=last_seen,
            nx=has_nxdomain,
        )

        for record in dns_records:
            addr = record.get("address")
            rtype = record.get("record_type")
            if rtype in ["A", "AAAA"] and addr and addr != "NXDOMAIN":
                session.run(
                    """
                    MATCH (d:domain {value: $dom})
                    MERGE (i:IP {value: $ip})
                    MERGE (d)-[:RESOLVES_TO]->(i)
                """,
                    dom=domain_value,
                    ip=addr,
                )

    def _process_url(self, session, event_id, ioc_data):
        url_value = ioc_data["ioc"]
        hostname = ioc_data.get("hostname")
        if not hostname:
            try:
                from urllib.parse import urlparse

                hostname = urlparse(url_value).netloc.split(":")[0]
            except:
                pass

        session.run(
            """
            MERGE (u:URL {value: $val})
            SET u.hostname = $host, u.server = $srv, u.http_code = $code,
                u.filetype = $ft, u.encoding = $enc, u.type = 'URL'
            WITH u
            MATCH (e:EVENT {id: $eid})
            MERGE (e)-[:InReport]->(u)
        """,
            val=url_value,
            eid=event_id,
            host=hostname,
            srv=ioc_data.get("server"),
            code=ioc_data.get("http_code"),
            ft=ioc_data.get("filetype"),
            enc=ioc_data.get("encoding"),
        )

        if hostname:
            session.run(
                """
                MATCH (u:URL {value: $val})
                MERGE (d:domain {value: $host})
                MERGE (u)-[:HostedOn]->(d)
            """,
                val=url_value,
                host=hostname,
            )

        if ioc_data.get("ip"):
            session.run(
                """
                MATCH (u:URL {value: $val})
                MERGE (i:IP {value: $ip})
                MERGE (u)-[:RESOLVES_TO]->(i)
            """,
                val=url_value,
                ip=ioc_data.get("ip"),
            )

    def _compare_bucket(self, list1, list2, processed_pairs, edges, threshold):
        if ppdeep is None:
            return
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
                except:
                    pass

    def _run_tlsh_fallback(self, files, processed_pairs, edges, threshold):
        if tlsh is None:
            return
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
                    except:
                        pass


def main():
    """Run the command-line workflow."""
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neo4j123"
    BASE_DIR = "/root/PythonProject/Trail-main/src"

    NEW_THREAT_DATA_DIR = os.path.join(BASE_DIR, "output_incremental/ioc/")
    NEW_EVENT_TTPS_FILE = os.path.join(
        BASE_DIR, "build_dataset/incremental_ttp_data.json"
    )
    NEW_CVE_DATA_DIR = os.path.join(BASE_DIR, "output_incremental/cve/")
    NEW_FILE_CSV_PATH = os.path.join(
        BASE_DIR, "output_incremental/incremental_file_hashes.csv"
    )

    updater = TrailNeo4jIncrementalUpdater(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        updater.check_database_health()

        if os.path.exists(NEW_THREAT_DATA_DIR):
            updater.incremental_import_events(NEW_THREAT_DATA_DIR)

        if os.path.exists(NEW_EVENT_TTPS_FILE):
            updater.incremental_import_mitre_tags(NEW_EVENT_TTPS_FILE)

        if os.path.exists(NEW_CVE_DATA_DIR):
            updater.incremental_import_cve(NEW_CVE_DATA_DIR)

        if os.path.exists(NEW_FILE_CSV_PATH):
            updater.incremental_import_files(NEW_FILE_CSV_PATH)

        updater.incremental_update_similarity()

        print("\n[DB] Final node statistics:")
        updater.check_database_health()

    finally:
        updater.close()


if __name__ == "__main__":
    main()
