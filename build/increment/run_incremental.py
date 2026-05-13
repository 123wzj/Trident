"""Run the incremental update and export pipeline."""

import argparse
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# Local output and incremental input paths.
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "incremental_results"
PROJECT_ROOT = SCRIPT_DIR.parent.parent
INCREMENTAL_DATA_DIR = PROJECT_ROOT / "src" / "output_incremental"
TTP_FILE = (
    PROJECT_ROOT / "src" / "build_dataset" / "incremental_ttp_data_converted.json"
)
NEO4J_MODULE_DIR = PROJECT_ROOT / "build" / "neo4j"
for module_dir in [SCRIPT_DIR, NEO4J_MODULE_DIR]:
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

DEPENDENCY_HINTS = {
    "neo4j": "pip install neo4j",
    "torch": "Install PyTorch for your CUDA or CPU environment.",
    "torch_geometric": "Install PyTorch Geometric for your PyTorch version.",
    "sklearn": "pip install scikit-learn",
    "pandas": "pip install pandas",
    "numpy": "pip install numpy",
}


def raise_dependency_error(exc):
    """Raise a clear message for missing runtime dependencies."""
    hint = DEPENDENCY_HINTS.get(exc.name)
    if hint:
        raise ModuleNotFoundError(f"Missing dependency '{exc.name}'. {hint}") from exc
    raise exc


def print_header(title, blank_before=True):
    """Print a compact console section header."""
    if blank_before:
        print()
    print("=" * 80)
    print(title.center(80))
    print("=" * 80)


def load_graph_exporter():
    """Load the Neo4j-to-PyG exporter with a clear dependency error."""
    try:
        from neo4jpytorch_embedding import ImprovedGraphExporter
    except ModuleNotFoundError as exc:
        if exc.name == "neo4jpytorch_embedding":
            raise ModuleNotFoundError(
                f"Cannot find neo4jpytorch_embedding.py in {NEO4J_MODULE_DIR}."
            ) from exc
        raise_dependency_error(exc)
    return ImprovedGraphExporter


def load_graph_database():
    """Load Neo4j driver lazily so import errors are easier to diagnose."""
    try:
        from neo4j import GraphDatabase
    except ModuleNotFoundError as exc:
        raise_dependency_error(exc)
    return GraphDatabase


def load_incremental_updater():
    """Load the Neo4j incremental updater with clear path/dependency errors."""
    try:
        from incremental_update import TrailNeo4jIncrementalUpdater
    except ModuleNotFoundError as exc:
        if exc.name == "incremental_update":
            raise ModuleNotFoundError(
                f"Cannot find incremental_update.py in {SCRIPT_DIR}."
            ) from exc
        raise_dependency_error(exc)
    return TrailNeo4jIncrementalUpdater


def load_torch():
    """Load torch lazily for data validation and tensor filtering."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise_dependency_error(exc)
    return torch


def step1_export_old_data():
    """Export the baseline IOC and TTP data."""
    print_header(
        "Step 1: Export baseline data (apt_kg_ioc_old.pt + apt_kg_ttp_old.pt)"
    )

    try:
        ImprovedGraphExporter = load_graph_exporter()
        torch = load_torch()
        print("\n  [Export] Connecting Neo4j and exporting data...")
        exporter = ImprovedGraphExporter(
            uri=NEO4J_URI, user=NEO4J_USER, pwd=NEO4J_PASSWORD
        )

        exporter.export_dual_subgraphs(output_dir=str(OUTPUT_DIR))
        exporter.close()

        ioc_old_path = OUTPUT_DIR / "apt_kg_ioc_old.pt"
        ttp_old_path = OUTPUT_DIR / "apt_kg_ttp_old.pt"

        ioc_src = OUTPUT_DIR / "apt_kg_ioc.pt"
        ttp_src = OUTPUT_DIR / "apt_kg_ttp.pt"

        if ioc_src.exists():
            shutil.copy(ioc_src, ioc_old_path)
            print(f"\n  [IOC] Saved: {ioc_old_path}")

        if ttp_src.exists():
            shutil.copy(ttp_src, ttp_old_path)
            print(f"  [TTP] Saved: {ttp_old_path}")

        if ioc_old_path.exists() and ttp_old_path.exists():
            print("\n  [Check] Export succeeded.")

            ioc_data = torch.load(ioc_old_path, weights_only=False)
            print(f"    [IOC] size: {ioc_old_path.stat().st_size / 1024 / 1024:.2f} MB")
            print(f"    [IOC] node types: {list(ioc_data.node_types)}")
            for node_type in ioc_data.node_types:
                num_nodes = ioc_data[node_type].num_nodes
                print(f"      - {node_type}: {num_nodes:,} nodes")

            ttp_data = torch.load(ttp_old_path, weights_only=False)
            print(f"    [TTP] size: {ttp_old_path.stat().st_size / 1024 / 1024:.2f} MB")
            print(f"    [TTP] sequences: {len(ttp_data['causal_sequences'])}")
            print(f"    [TTP] labels: {ttp_data['labels'].shape[0]}")
            print(f"    [TTP] classes: {ttp_data['num_classes']}")

            print("\n  [OK] Step 1 completed.")
            return True
        else:
            print("\n  [ERROR] Required files were not generated.")
            return False

    except Exception as e:
        print(f"\n  [ERROR] Step 1 failed: {e}")
        traceback.print_exc()
        return False


def step2_incremental_update():
    """Apply incremental updates to Neo4j."""
    print_header("Step 2: Update Neo4j")

    updater = None
    try:
        TrailNeo4jIncrementalUpdater = load_incremental_updater()
        updater = TrailNeo4jIncrementalUpdater(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        print("\n  [Check] Database status:")
        updater.check_database_health()

        valid_event_ids = None

        if TTP_FILE.exists():
            print(f"\n  [Import] TTP data: {TTP_FILE}")
            valid_event_ids = updater.incremental_import_ttp(str(TTP_FILE))
            print(f"    - valid events: {len(valid_event_ids)}")
        else:
            print(f"  [!] TTP file not found: {TTP_FILE}")
            print(f"  [Stop] No valid TTP events.")
            return False

        if not valid_event_ids:
            print("  [Stop] No valid TTP events.")
            return False

        ioc_dir = INCREMENTAL_DATA_DIR / "ioc"
        if ioc_dir.exists():
            print(f"\n  [Import] IOC data: {ioc_dir}")
            stats_ioc = updater.incremental_import_events(
                str(ioc_dir), valid_event_ids=valid_event_ids
            )
            print(
                f"    - success={stats_ioc['success']}, skipped={stats_ioc['skipped']}, failed={stats_ioc['failed']}",
                end="",
            )
            if stats_ioc.get("filtered", 0) > 0:
                print(f", filtered_no_ttp={stats_ioc['filtered']}")
            else:
                print()
        else:
            print(f"  [!] IOC directory not found: {ioc_dir}")

        cve_dir = INCREMENTAL_DATA_DIR / "cve"
        if cve_dir.exists():
            print(f"\n  [Import] CVE data: {cve_dir}")
            updater.incremental_import_cve(
                str(cve_dir), valid_event_ids=valid_event_ids
            )
        else:
            print(f"  [!] CVE directory not found: {cve_dir}")

        file_csv = INCREMENTAL_DATA_DIR / "file_hashes" / "incremental_file_hashes.csv"
        if file_csv.exists():
            print(f"\n  [Import] File hash data: {file_csv}")
            updater.incremental_import_files(
                str(file_csv), valid_event_ids=valid_event_ids
            )
        else:
            print(f"  [!] File hash CSV not found: {file_csv}")

        print("\n  [Check] Database status:")
        updater.check_database_health()

        updater.cleanup_isolated_events()

        print("\n  [OK] Step 2 completed: database updated for TTP-backed events.")
        return True

    except Exception as e:
        print(f"\n  [ERROR] Step 2 failed: {e}")
        traceback.print_exc()
        return False
    finally:
        if updater is not None:
            updater.close()


def step3_export_updated_data():
    """Export the updated IOC and TTP data."""
    print_header(
        "Step 3: Export updated data (apt_kg_ioc_updated.pt + apt_kg_ttp_updated.pt)"
    )

    try:
        GraphDatabase = load_graph_database()
        print("\n  [Check] Database status:")
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            total_events = session.run("MATCH (e:EVENT) RETURN count(e) as c").single()[
                "c"
            ]
            ttp_events = session.run(
                "MATCH (e:EVENT)-[:USES_TECHNIQUE]->(:Technique) RETURN count(DISTINCT e) as c"
            ).single()["c"]
            node_stats = session.run(
                """
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
            """
            ).data()

            print(f"    - EVENT: {total_events:,}")
            ttp_ratio = ttp_events / total_events * 100 if total_events else 0.0
            print(f"    - EVENT with TTP: {ttp_events:,} ({ttp_ratio:.1f}%)")
            print("    - node types:")
            for stat in node_stats[:8]:
                print(f"        {stat['label']}: {stat['count']:,}")
        driver.close()

        ImprovedGraphExporter = load_graph_exporter()
        old_ioc_path = OUTPUT_DIR / "apt_kg_ioc_old.pt"
        old_ttp_path = OUTPUT_DIR / "apt_kg_ttp_old.pt"

        if not old_ioc_path.exists() or not old_ttp_path.exists():
            print("\n  [ERROR] Required files were not generated.")
            return False

        print("\n  [Export] Connecting Neo4j and exporting updated data...")
        exporter = ImprovedGraphExporter(
            uri=NEO4J_URI, user=NEO4J_USER, pwd=NEO4J_PASSWORD
        )

        exporter.export_dual_subgraphs(output_dir=str(OUTPUT_DIR))
        exporter.close()

        ioc_updated_path = OUTPUT_DIR / "apt_kg_ioc_updated.pt"
        ttp_updated_path = OUTPUT_DIR / "apt_kg_ttp_updated.pt"

        ioc_src = OUTPUT_DIR / "apt_kg_ioc.pt"
        ttp_src = OUTPUT_DIR / "apt_kg_ttp.pt"

        if ioc_src.exists():
            shutil.copy(ioc_src, ioc_updated_path)
            print(f"\n  [IOC] Saved: {ioc_updated_path}")

        if ttp_src.exists():
            shutil.copy(ttp_src, ttp_updated_path)
            print(f"  [TTP] Saved: {ttp_updated_path}")

        if ioc_updated_path.exists() and ttp_updated_path.exists():
            print("\n  [Check] Export succeeded.")

            torch = load_torch()
            old_ioc_data = torch.load(old_ioc_path, weights_only=False)
            new_ioc_data = torch.load(ioc_updated_path, weights_only=False)

            print("\n  [Compare] IOC data:")
            for node_type in new_ioc_data.node_types:
                old_num = (
                    old_ioc_data[node_type].num_nodes
                    if node_type in old_ioc_data.node_types
                    else 0
                )
                new_num = new_ioc_data[node_type].num_nodes
                delta = new_num - old_num
                if delta > 0:
                    print(f"    - {node_type}: {old_num:,} -> {new_num:,} (+{delta:,})")
                else:
                    print(f"    - {node_type}: {old_num:,} -> {new_num:,}")

            old_ttp_data = torch.load(old_ttp_path, weights_only=False)
            new_ttp_data = torch.load(ttp_updated_path, weights_only=False)

            print("\n  [Compare] TTP data:")
            print(
                f"    - sequences: {len(old_ttp_data['causal_sequences']):,} -> {len(new_ttp_data['causal_sequences']):,}"
            )
            print(
                f"    - labels: {old_ttp_data['labels'].shape[0]:,} -> {new_ttp_data['labels'].shape[0]:,}"
            )

            print("\n  [OK] Step 3 completed.")
            return True
        else:
            print("\n  [ERROR] Required files were not generated.")
            return False

    except Exception as e:
        print(f"\n  [ERROR] Step 3 failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run the command-line workflow."""
    global INCREMENTAL_DATA_DIR
    global NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
    global OUTPUT_DIR, TTP_FILE

    parser = argparse.ArgumentParser(description="Run the incremental pipeline")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "1", "2", "3"],
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument("--neo4j-uri", default=NEO4J_URI)
    parser.add_argument("--neo4j-user", default=NEO4J_USER)
    parser.add_argument("--neo4j-password", default=NEO4J_PASSWORD)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--incremental-data-dir", type=Path, default=INCREMENTAL_DATA_DIR
    )
    parser.add_argument("--ttp-file", type=Path, default=TTP_FILE)

    args = parser.parse_args()
    NEO4J_URI = args.neo4j_uri
    NEO4J_USER = args.neo4j_user
    NEO4J_PASSWORD = args.neo4j_password
    if not NEO4J_PASSWORD:
        parser.error(
            "Neo4j password is required. Set NEO4J_PASSWORD or pass --neo4j-password."
        )

    OUTPUT_DIR = args.output_dir.resolve()
    INCREMENTAL_DATA_DIR = args.incremental_data_dir.resolve()
    TTP_FILE = args.ttp_file.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    print_header("Incremental learning pipeline", blank_before=False)
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Incremental data: {INCREMENTAL_DATA_DIR}")
    print(f"TTP file: {TTP_FILE}")

    results = {}

    if args.step in ["all", "1"]:
        results["step1"] = step1_export_old_data()
    else:
        print("\n[Skip] Step 1")

    if args.step in ["all", "2"]:
        results["step2"] = step2_incremental_update()
    else:
        print("\n[Skip] Step 2")

    if args.step in ["all", "3"]:
        results["step3"] = step3_export_updated_data()
    else:
        print("\n[Skip] Step 3")

    elapsed = (datetime.now() - start_time).total_seconds()

    print_header("Incremental learning pipeline")

    for step, passed in results.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {step}")

    print(f"\nElapsed: {elapsed / 60:.1f} minutes")

    all_passed = all(results.values())
    if all_passed:
        print_header("All steps completed.")
    else:
        print_header("Some steps failed.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
