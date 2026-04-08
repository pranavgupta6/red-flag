from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class Transaction(BaseModel):
    """Internal model — includes ground truth. Never sent to agent."""
    id: str
    vendor_id: str
    vendor_name: str
    amount: float
    timestamp: datetime
    category: str
    department: str
    approver_id: str
    is_fraud: bool
    fraud_type: Optional[str] = None


class TransactionView(BaseModel):
    """What the agent sees — no ground truth labels."""
    id: str
    vendor_id: str
    vendor_name: str
    amount: float
    timestamp: datetime
    category: str
    department: str
    approver_id: str


class AuditObservation(BaseModel):
    task: str
    step: int
    budget_remaining: int
    transactions: List[TransactionView]
    flagged_so_far: List[str]
    vendor_history: Dict[str, Any]
    message: Optional[str] = None


class AuditAction(BaseModel):
    flag_ids: List[str]
    reasoning: Optional[str] = None


class AuditReward(BaseModel):
    value: float
    cumulative: float
    breakdown: Dict[str, float]
