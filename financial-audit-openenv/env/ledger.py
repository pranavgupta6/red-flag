import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from env.models import Transaction


CATEGORIES = ["Office Supplies", "Consulting", "Travel", "Software", "Marketing", "Legal", "Utilities"]
DEPARTMENTS = ["Finance", "HR", "Engineering", "Sales", "Legal", "Operations"]
VENDORS = [
    ("V001", "Acme Corp"), ("V002", "Global Tech"), ("V003", "Summit Consulting"),
    ("V004", "Apex Solutions"), ("V005", "Horizon Ltd"), ("V006", "Pinnacle Services"),
    ("V007", "Delta Systems"), ("V008", "Omega Group"), ("V009", "Zenith Partners"),
    ("V010", "Nexus Corp"),
]


def _random_datetime(start: datetime, end: datetime) -> datetime:
    delta = end - start
    seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=seconds)


def _make_normal_transaction(txn_id: str, base_date: datetime) -> Transaction:
    vendor_id, vendor_name = random.choice(VENDORS)
    amount = round(random.uniform(100, 8000), 2)
    ts = _random_datetime(base_date - timedelta(days=365), base_date)
    # Force normal hours (6am–10pm)
    ts = ts.replace(hour=random.randint(6, 22), minute=random.randint(0, 59))
    return Transaction(
        id=txn_id,
        vendor_id=vendor_id,
        vendor_name=vendor_name,
        amount=amount,
        timestamp=ts,
        category=random.choice(CATEGORIES),
        department=random.choice(DEPARTMENTS),
        approver_id=f"APR_{random.randint(1, 10):02d}",
        is_fraud=False,
        fraud_type=None,
    )


def generate_easy_ledger(seed: int = 42) -> Tuple[List[Transaction], set, Dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    base_date = datetime(2024, 1, 1)
    transactions = []
    fraud_ids = set()

    # 85 normal transactions
    for i in range(85):
        transactions.append(_make_normal_transaction(f"TXN_{i:04d}", base_date))

    # Fraud type 1: round dollar amounts over $10,000
    for i, amount in enumerate([10000.00, 15000.00, 20000.00, 12000.00, 25000.00]):
        txn_id = f"TXN_{85 + i:04d}"
        vendor_id, vendor_name = random.choice(VENDORS)
        ts = _random_datetime(base_date - timedelta(days=365), base_date)
        ts = ts.replace(hour=random.randint(6, 22))
        t = Transaction(
            id=txn_id, vendor_id=vendor_id, vendor_name=vendor_name,
            amount=amount, timestamp=ts, category=random.choice(CATEGORIES),
            department=random.choice(DEPARTMENTS), approver_id=f"APR_{random.randint(1,10):02d}",
            is_fraud=True, fraud_type="round_amount"
        )
        transactions.append(t)
        fraud_ids.add(txn_id)

    # Fraud type 2: duplicate transactions
    base_txn = transactions[0]
    for i in range(3):
        txn_id = f"TXN_{90 + i:04d}"
        t = Transaction(
            id=txn_id, vendor_id=base_txn.vendor_id, vendor_name=base_txn.vendor_name,
            amount=base_txn.amount, timestamp=base_txn.timestamp, category=base_txn.category,
            department=base_txn.department, approver_id=base_txn.approver_id,
            is_fraud=True, fraud_type="duplicate"
        )
        transactions.append(t)
        fraud_ids.add(txn_id)

    # Fraud type 3: late night transactions (2am-4am)
    for i in range(4):
        txn_id = f"TXN_{93 + i:04d}"
        vendor_id, vendor_name = random.choice(VENDORS)
        ts = _random_datetime(base_date - timedelta(days=365), base_date)
        ts = ts.replace(hour=random.randint(2, 3), minute=random.randint(0, 59))
        t = Transaction(
            id=txn_id, vendor_id=vendor_id, vendor_name=vendor_name,
            amount=round(random.uniform(100, 8000), 2), timestamp=ts,
            category=random.choice(CATEGORIES), department=random.choice(DEPARTMENTS),
            approver_id=f"APR_{random.randint(1,10):02d}",
            is_fraud=True, fraud_type="late_night"
        )
        transactions.append(t)
        fraud_ids.add(txn_id)

    random.shuffle(transactions)
    metadata = {"total": len(transactions), "fraud_count": len(fraud_ids), "budget": 15}
    return transactions, fraud_ids, metadata


def generate_medium_ledger(seed: int = 42) -> Tuple[List[Transaction], set, Dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    base_date = datetime(2024, 1, 1)
    transactions = []
    fraud_ids = set()
    vendor_stats = {}

    for vendor_id, vendor_name in VENDORS:
        mean = random.uniform(500, 5000)
        std = random.uniform(100, 800)
        count = random.randint(5, 20)
        vendor_stats[vendor_id] = {
            "mean": round(mean, 2),
            "std": round(std, 2),
            "normal_monthly_count": count,
            "name": vendor_name
        }

    idx = 0
    for _ in range(170):
        vendor_id, vendor_name = random.choice(VENDORS)
        stats = vendor_stats[vendor_id]
        amount = round(max(10, np.random.normal(stats["mean"], stats["std"])), 2)
        ts = _random_datetime(base_date - timedelta(days=365), base_date)
        ts = ts.replace(hour=random.randint(6, 22))
        transactions.append(Transaction(
            id=f"TXN_{idx:04d}", vendor_id=vendor_id, vendor_name=vendor_name,
            amount=amount, timestamp=ts, category=random.choice(CATEGORIES),
            department=random.choice(DEPARTMENTS),
            approver_id=f"APR_{random.randint(1,10):02d}",
            is_fraud=False, fraud_type=None
        ))
        idx += 1

    # Fraud: amounts > 3.5 std above vendor mean — one per vendor for all 10 vendors
    for vendor_id, vendor_name in VENDORS:
        stats = vendor_stats[vendor_id]
        amount = round(stats["mean"] + 3.5 * stats["std"], 2)
        ts = _random_datetime(base_date - timedelta(days=365), base_date)
        ts = ts.replace(hour=random.randint(6, 22))
        txn_id = f"TXN_{idx:04d}"
        transactions.append(Transaction(
            id=txn_id, vendor_id=vendor_id, vendor_name=vendor_name,
            amount=amount, timestamp=ts, category=random.choice(CATEGORIES),
            department=random.choice(DEPARTMENTS),
            approver_id=f"APR_{random.randint(1,10):02d}",
            is_fraud=True, fraud_type="statistical_outlier"
        ))
        fraud_ids.add(txn_id)
        idx += 1

    random.shuffle(transactions)
    metadata = {
        "total": len(transactions),
        "fraud_count": len(fraud_ids),
        "budget": 20,
        "vendor_stats": vendor_stats
    }
    return transactions, fraud_ids, metadata


def generate_hard_ledger(seed: int = 42) -> Tuple[List[Transaction], set, Dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    base_date = datetime(2024, 1, 1)
    transactions = []
    fraud_ids = set()
    clusters = []

    # 270 normal transactions
    idx = 0
    for _ in range(270):
        transactions.append(_make_normal_transaction(f"TXN_{idx:04d}", base_date))
        idx += 1

    # 3 structuring clusters
    recipient = "V001"
    for cluster_num in range(3):
        cluster = set()
        cluster_start = base_date - timedelta(days=random.randint(10, 300))
        for j in range(8):
            txn_id = f"TXN_{idx:04d}"
            amount = round(random.uniform(9200, 9900), 2)
            ts = cluster_start + timedelta(hours=random.randint(0, 72))
            shell_name = f"Shell Co {cluster_num}-{j}"
            shell_id = f"SC_{cluster_num}_{j}"
            transactions.append(Transaction(
                id=txn_id, vendor_id=shell_id, vendor_name=shell_name,
                amount=amount, timestamp=ts, category="Consulting",
                department=random.choice(DEPARTMENTS), approver_id=f"APR_{random.randint(1,10):02d}",
                is_fraud=True, fraud_type="structuring"
            ))
            fraud_ids.add(txn_id)
            cluster.add(txn_id)
            idx += 1
        clusters.append(cluster)

    random.shuffle(transactions)
    metadata = {"total": len(transactions), "fraud_count": len(fraud_ids), "budget": 20, "clusters": [list(c) for c in clusters]}
    return transactions, fraud_ids, metadata
