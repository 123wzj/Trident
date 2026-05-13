# Trident

Trident is a research-oriented APT attribution pipeline. It extracts CTI evidence, builds a Neo4j knowledge graph, exports graph and sequence data, trains a dual-branch model that combines IOC graph evidence with TTP sequence evidence, and supports incremental graph updates with replay-based training experiments.

## Project Structure

```text
Trident/
+-- cti_extrac/              # CTI text extraction and normalization
+-- build/
    +-- dataset/             # Dataset aggregation and enrichment scripts
    +-- neo4j/               # Neo4j graph construction and PyTorch export
    +-- train/               # Dual-branch attribution training
    +-- increment/           # Incremental graph update and replay training
```

## Main Components

- `cti_extrac/main.py`: extracts IOC and TTP information from CTI reports.
- `build/dataset/pipeline.py`: aggregates file hashes, enriches them with MalwareBazaar metadata, and filters dataset records.
- `build/neo4j/build_knowledge_graph.py`: imports CTI, TTP, CVE, and file-hash data into Neo4j.
- `build/neo4j/neo4jpytorch_embedding.py`: exports Neo4j data into PyTorch Geometric datasets.
- `build/train/train.py`: trains the IOC graph branch and TTP sequence branch with entropy-aware decision fusion.
- `build/increment/run_incremental.py`: runs the incremental update pipeline.
- `build/increment/replay_train.py`: trains with replay data for incremental learning experiments.

## Requirements

The project uses Python 3.10+ and depends on packages such as:

- `torch`
- `torch_geometric`
- `scikit-learn`
- `numpy`
- `pandas`
- `neo4j`
- `aiohttp`
- `tqdm`
- `langchain-core`
- `langgraph`
