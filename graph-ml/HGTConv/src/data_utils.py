import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any
from venv import logger
from src.config import Config
import torch
from torch_geometric.data import HeteroData

config=Config()

@dataclass
class DatasetArtifacts:
    technique2id: Dict[str, int]
    id2technique: Dict[int, str]
    tactic2id: Dict[str, int]
    id2tactic: Dict[int, str]
    chain_seq: List[dict]


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json_data(data_path: str | Path) -> dict:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_sequences(raw_data: dict) -> List[dict]:
    """
    out:
    [
        {
            "group": "APT1",
            "intrusion_set_id": "...",
            "sequence": [...]
        },
        ...
    ]
    """
    records = []
    for group_name, payload in raw_data.items():
        seq = payload.get("sequence", [])
        if not seq:
            continue

        records.append(
            {
                "group": group_name,
                "intrusion_set_id": payload.get("intrusion_set_id", group_name),
                "sequence": seq,
                "sequence_data_with_tactics": payload.get("sequence_data_with_tactics", []),
            }
        )
    return records


def build_vocab(records: List[dict]) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    technique_counter = Counter()
    tactic_counter = Counter()
    logger.info(f"${technique_counter},${tactic_counter}")
    for rec in records:
        for item in rec["sequence"]:
            tech = item.get("technique_id")
            tactic = item.get("tactic_name")
            if tech and item.get("is_valid_technique", True):
                technique_counter[tech] += 1
            if tactic:
                tactic_counter[tactic] += 1

    technique_list = sorted(technique_counter.keys())
    tactic_list = sorted(tactic_counter.keys())

    technique2id = {t: i for i, t in enumerate(technique_list)}
    id2technique = {i: t for t, i in technique2id.items()}

    tactic2id = {t: i for i, t in enumerate(tactic_list)}
    id2tactic = {i: t for t, i in tactic2id.items()}

    return technique2id, id2technique, tactic2id, id2tactic


def build_sequence_chain_seq(records: List[dict], technique2id: Dict[str, int], max_seq_len: int = config.MAX_SEQ_LEN) -> List[dict]:
    """
    Each sequence is converted into several supervised examples.
    [t1, t2, t3] -> input=[t1, t2], target=t3
    """
    chain_seq = []

    for rec in records:
        seq = [
            item["technique_id"]
            for item in rec["sequence"]
            if item.get("is_valid_technique", True) and item.get("technique_id") in technique2id
        ]

        if len(seq) < 2:
            continue

        for i in range(1, len(seq)):
            input_seq = seq[max(0, i - max_seq_len):i]
            target = seq[i]
            chain_seq.append(
                {
                    "group": rec["group"],
                    "input_seq": input_seq,
                    "target": target,
                    "target_id": technique2id[target],
                }
            )

    return chain_seq


def sequence_to_heterodata(
    input_seq: List[str],
    technique2id: Dict[str, int],
    tactic_map: Dict[str, str] | None = None
) -> HeteroData:
    """
    sequence -> graph:
    - node type: technique
    - edge type: follows
    """
    data = HeteroData()

    node_ids = [technique2id[t] for t in input_seq if t in technique2id]

    if len(node_ids) == 0:
        node_ids = [0]

    x = torch.tensor(node_ids, dtype=torch.long)
    data["technique"].x = x

    if len(node_ids) > 1:
        src = torch.arange(0, len(node_ids) - 1, dtype=torch.long)
        dst = torch.arange(1, len(node_ids), dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data["technique", "follows", "technique"].edge_index = edge_index

    return data


def split_chain_seq(chain_seq: List[dict], test_ratio=0.2, val_ratio=0.1, seed=42):
    random.Random(seed).shuffle(chain_seq)
    n = len(chain_seq)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_chain_seq = chain_seq[:n_test]
    val_chain_seq = chain_seq[n_test:n_test + n_val]
    train_chain_seq = chain_seq[n_test + n_val:]

    return train_chain_seq, val_chain_seq, test_chain_seq
