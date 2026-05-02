from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class LogEvent:

    timestamp: Optional[str] = None
    ruleName: Optional[str] = None
    severity: Optional[str] = None
    riskScore: Optional[float] = None
    hostname: Optional[str] = None
    userName: Optional[str] = None
    processId: Optional[int] = None
    processName: Optional[str] = None
    fileName: Optional[str] = None
    socketIn: Optional[str] = None
    socketOut: Optional[str] = None
    techniqueId: Optional[str] = None
    techniqueName: Optional[str] = None
    tacticId: Optional[str] = None
    tacticName: Optional[str] = None

@dataclass
class AttackChain:
    
    """
    Data class for a complete attack chain, containing an ID and a list of events.
    """
    id: str
    events: List[LogEvent] = field(default_factory=list)