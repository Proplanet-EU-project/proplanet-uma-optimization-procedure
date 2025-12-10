from pydantic import BaseModel
from typing import List, Dict
from models.material_data import MaterialData

class CoatingsOptimizationInput(BaseModel):
    substances: List[MaterialData]
    environmental_media: str
    scenario: str
    low_mols: List[int]
    upp_mols: List[int]
    upperbound: List[int]
    lung_model: str
    lung_respiratory_volume_rate: str
    lung_exposure_duration: int
    wca_surface: int
    hca_surface: int
    sfe_surface: int
    lung_deposition_flags: Dict[str, bool]
    performance_flags: Dict[str, bool]
    base_concentration: float
