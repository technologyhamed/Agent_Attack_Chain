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


import json
from typing import List, Dict

def map_techniques_to_tactics(technique_ids: List[str], 
                              tactic_techniques_path: str = "tactic_techniques.json",
                              tactic_order_path: str = "tactic_order.json") -> List[str]:
    """
    نگاشت لیست تکنیک‌ها به نام تاکتیک‌های مربوطه
    
    Args:
        technique_ids: لیست external_id تکنیک‌ها (مثل ['T1583', 'T1078', ...])
        tactic_techniques_path: مسیر فایل JSON نگاشت تاکتیک به تکنیک
        tactic_order_path: مسیر فایل JSON ترتیب تاکتیک‌ها
    
    Returns:
        class_names: لیست نام تاکتیک‌ها به ترتیب تکنیک‌های ورودی
    
    Example:
        >>> techniques = ['T1583', 'T1078', 'T1566']
        >>> class_names = map_techniques_to_tactics(techniques)
        >>> print(class_names)
        ['resource-development', 'initial-access', 'initial-access']
    """
    # بارگذاری فایل‌ها
    with open(tactic_techniques_path, 'r', encoding='utf-8') as f:
        tactic_techniques = json.load(f)
    
    # ساخت دیکشنری نگاشت: external_id -> tactic_name
    technique_to_tactic = {}
    for tactic_name, techniques in tactic_techniques.items():
        for tech in techniques:
            technique_to_tactic[tech['external_id']] = tactic_name
    
    # نگاشت تکنیک‌های ورودی به تاکتیک‌ها
    class_names = []
    for tech_id in technique_ids:
        if tech_id in technique_to_tactic:
            class_names.append(technique_to_tactic[tech_id])
        else:
            # اگر تکنیک پیدا نشد، از خود تکنیک استفاده کن
            print(f"Warning: Technique '{tech_id}' not found in tactic_techniques.json")
            class_names.append(tech_id)
    
    return class_names


def map_techniques_to_tactics_with_labels(technique_ids: List[str],
                                          tactic_techniques_path: str = "tactic_techniques.json",
                                          format_type: str = "name") -> List[str]:
    """
    نگاشت تکنیک‌ها به برچسب‌های قابل خواندن (نام کامل یا نام فارسی)
    
    Args:
        technique_ids: لیست external_id تکنیک‌ها
        tactic_techniques_path: مسیر فایل JSON
        format_type: نوع فرمت خروجی
            - "name": نام انگلیسی تاکتیک (مثل 'resource-development')
            - "readable": نام قابل خواندن (مثل 'Resource Development')
            - "persian": نام فارسی (نیاز به دیکشنری ترجمه)
    
    Returns:
        class_names: لیست برچسب‌های فرمت شده
    """
    # دیکشنری ترجمه (اختیاری)
    tactic_translations = {
        'resource-development': 'توسعه منابع',
        'initial-access': 'دسترسی اولیه',
        'execution': 'اجرا',
        'persistence': 'پایداری',
        'privilege-escalation': 'افزایش سطح دسترسی',
        'defense-evasion': 'فرار از دفاع',
        'credential-access': 'دسترسی به اعتبارنامه',
        'discovery': 'کشف',
        'lateral-movement': 'حرکت جانبی',
        'collection': 'جمع‌آوری',
        'command-and-control': 'فرمان و کنترل',
        'exfiltration': 'استخراج داده',
        'impact': 'تأثیر'
    }
    
    # نگاشت اولیه
    tactic_names = map_techniques_to_tactics(technique_ids, tactic_techniques_path)
    
    # فرمت‌بندی بر اساس نوع
    if format_type == "readable":
        return [name.replace('-', ' ').title() for name in tactic_names]
    elif format_type == "persian":
        return [tactic_translations.get(name, name) for name in tactic_names]
    else:  # "name"
        return tactic_names


def get_unique_tactics_from_techniques(technique_ids: List[str],
                                      tactic_techniques_path: str = "tactic_techniques.json",
                                      tactic_order_path: str = "tactic_order.json") -> List[str]:
    """
    استخراج لیست یکتای تاکتیک‌ها از تکنیک‌های ورودی (به ترتیب tactic_order)
    
    Args:
        technique_ids: لیست external_id تکنیک‌ها
        tactic_techniques_path: مسیر فایل JSON نگاشت
        tactic_order_path: مسیر فایل JSON ترتیب
    
    Returns:
        unique_tactics: لیست یکتای تاکتیک‌ها به ترتیب
    
    Example:
        >>> techniques = ['T1583', 'T1078', 'T1566', 'T1584']
        >>> unique_tactics = get_unique_tactics_from_techniques(techniques)
        >>> print(unique_tactics)
        ['resource-development', 'initial-access']
    """
    # بارگذاری ترتیب تاکتیک‌ها
    with open(tactic_order_path, 'r', encoding='utf-8') as f:
        tactic_order = json.load(f)
    
    # نگاشت تکنیک‌ها به تاکتیک‌ها
    tactic_names = map_techniques_to_tactics(technique_ids, tactic_techniques_path)
    
    # استخراج یکتاها و مرتب‌سازی
    unique_tactics = list(dict.fromkeys(tactic_names))  # حفظ ترتیب
    unique_tactics.sort(key=lambda x: tactic_order.get(x, 999))
    
    return unique_tactics


import numpy as np
from typing import List, Dict, Union

def convert_indices_to_techniques(y_indices: Union[np.ndarray, List[int]], 
                                  id2technique: Dict[int, str]) -> List[str]:
    """
    تبدیل آرایه ایندکس‌های عددی به لیست external_id تکنیک‌ها
    
    Args:
        y_indices: آرایه numpy یا لیست شامل ایندکس‌های عددی تکنیک‌ها
        id2technique: دیکشنری نگاشت از ایندکس به external_id تکنیک
    
    Returns:
        techniques: لیست external_id تکنیک‌ها (مثل ['T1003.001', 'T1078', ...])
    
    Example:
        >>> y_true = np.array([5, 270, 65, 5, 185])
        >>> id2technique = {5: 'T1003.001', 270: 'T1078', 65: 'T1041', 185: 'T1566'}
        >>> techniques = convert_indices_to_techniques(y_true, id2technique)
        >>> print(techniques)
        ['T1003.001', 'T1078', 'T1041', 'T1003.001', 'T1566']
    """
    # تبدیل numpy array به لیست در صورت نیاز
    if isinstance(y_indices, np.ndarray):
        y_indices = y_indices.tolist()
    
    # نگاشت ایندکس‌ها به تکنیک‌ها
    techniques = []
    for idx in y_indices:
        if idx in id2technique:
            techniques.append(id2technique[idx])
        else:
            print(f"Warning: Index {idx} not found in id2technique mapping")
            techniques.append(f"UNKNOWN_{idx}")
    
    return techniques


def convert_indices_to_techniques_batch(y_true: Union[np.ndarray, List[int]],
                                        y_pred: Union[np.ndarray, List[int]],
                                        id2technique: Dict[int, str]) -> tuple:
    """
    تبدیل همزمان y_true و y_pred به لیست تکنیک‌ها
    
    Args:
        y_true: آرایه ایندکس‌های واقعی
        y_pred: آرایه ایندکس‌های پیش‌بینی شده
        id2technique: دیکشنری نگاشت
    
    Returns:
        (y_true_techniques, y_pred_techniques): تاپل دو لیست تکنیک
    
    Example:
        >>> y_true = np.array([5, 270, 65])
        >>> y_pred = np.array([5, 270, 190])
        >>> y_true_tech, y_pred_tech = convert_indices_to_techniques_batch(
        ...     y_true, y_pred, id2technique
        ... )
    """
    y_true_techniques = convert_indices_to_techniques(y_true, id2technique)
    y_pred_techniques = convert_indices_to_techniques(y_pred, id2technique)
    
    return y_true_techniques, y_pred_techniques


def get_unique_techniques_from_indices(y_indices: Union[np.ndarray, List[int]],
                                      id2technique: Dict[int, str],
                                      preserve_order: bool = True) -> List[str]:
    """
    استخراج لیست یکتای تکنیک‌ها از آرایه ایندکس‌ها
    
    Args:
        y_indices: آرایه ایندکس‌های عددی
        id2technique: دیکشنری نگاشت
        preserve_order: حفظ ترتیب ظهور اولین بار (True) یا مرتب‌سازی (False)
    
    Returns:
        unique_techniques: لیست یکتای تکنیک‌ها
    
    Example:
        >>> y_true = np.array([5, 270, 65, 5, 185, 270])
        >>> unique = get_unique_techniques_from_indices(y_true, id2technique)
        >>> print(unique)
        ['T1003.001', 'T1078', 'T1041', 'T1566']
    """
    techniques = convert_indices_to_techniques(y_indices, id2technique)
    
    if preserve_order:
        # حفظ ترتیب ظهور
        unique_techniques = list(dict.fromkeys(techniques))
    else:
        # مرتب‌سازی الفبایی
        unique_techniques = sorted(set(techniques))
    
    return unique_techniques


def create_technique_to_index_mapping(id2technique: Dict[int, str]) -> Dict[str, int]:
    """
    ساخت نگاشت معکوس: external_id -> index
    
    Args:
        id2technique: دیکشنری نگاشت از ایندکس به تکنیک
    
    Returns:
        technique2id: دیکشنری نگاشت از تکنیک به ایندکس
    
    Example:
        >>> id2technique = {5: 'T1003.001', 270: 'T1078'}
        >>> technique2id = create_technique_to_index_mapping(id2technique)
        >>> print(technique2id)
        {'T1003.001': 5, 'T1078': 270}
    """
    return {tech: idx for idx, tech in id2technique.items()}
