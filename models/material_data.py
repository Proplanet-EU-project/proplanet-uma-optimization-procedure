from pydantic import BaseModel
from typing import Dict, Any, Optional

class MaterialData(BaseModel):
    name: str
    formula: str
    cas: str
    category: str
    totalCarbonAtoms: int
    molecularWeight: float
    ecosystemToxicity: Dict[str, Any]
    humanToxicity: Dict[str, Any]
    rinaTest: Optional[Dict[str, Any]] = None
    id: str
    createdAt: str
    lastUpdatedAt: str