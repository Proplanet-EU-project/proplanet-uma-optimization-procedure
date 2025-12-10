from fastapi import APIRouter
from fastapi.responses import JSONResponse
from uma_resources.Governing_equation_UMA_materials import optimization_procedure as MOP
from uma_resources.Governing_equation_UMA_coating import coatings_optimization_procedure as COP
from models.material_optimization_input import MaterialOptimizationInput
from models.coatings_optimization_input import CoatingsOptimizationInput

router = APIRouter(prefix="/api/optimization", tags=["UMA'S API"])

@router.post("/material_procedure", summary="Material optimization procedure")
def material_optimization_procedure(input_data: MaterialOptimizationInput):
    try:
        results = MOP(input_data.substances, input_data.environmental_media, input_data.scenario, input_data.upperbound,
                       input_data.lung_model, input_data.lung_respiratory_volume_rate, input_data.lung_exposure_duration,
                       input_data.lung_deposition_flags,input_data.base_concentration)
        
        return JSONResponse(results)
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )
    

@router.post("/coatings_procedure", summary="Coatings optimization procedure")
def coatings_optimization_procedure(input_data: CoatingsOptimizationInput):
    
    try:
        results = COP(input_data.substances, input_data.environmental_media, input_data.scenario, input_data.low_mols,
            input_data.upp_mols, input_data.upperbound, input_data.lung_model, input_data.lung_respiratory_volume_rate,
            input_data.lung_exposure_duration, input_data.wca_surface, input_data.hca_surface, input_data.sfe_surface,
            input_data.lung_deposition_flags,input_data.performance_flags,input_data.base_concentration)
        return JSONResponse(results)
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )
        
    