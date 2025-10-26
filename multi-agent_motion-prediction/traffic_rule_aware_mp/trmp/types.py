from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np

@dataclass
class AgentHistory:
    xy: np.ndarray           # [T,2]
    yaw: np.ndarray          # [T]
    type: str = ""
    size: List[float] = field(default_factory=list)

UnifiedSample = Dict[str, Any]
