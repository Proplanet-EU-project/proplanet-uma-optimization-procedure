from pydantic import BaseModel
from typing import List,Dict
from models.material_data import MaterialData

class MaterialOptimizationInput(BaseModel):
    substances: List[MaterialData]
    environmental_media: str
    scenario: str
    upperbound: List[int]
    lung_model: str
    lung_respiratory_volume_rate: str
    lung_exposure_duration: float
    lung_deposition_flags: Dict[str, bool]
    base_concentration: float
