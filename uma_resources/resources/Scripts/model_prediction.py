import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
import joblib

training_columns = ['pKa', 'Molweight', 'Tm',
       'Pvap25', 'Sol25', 'Kaw', 'Kow', 'Ksw', 'kdeg(air)', 'kdeg(water)',
       'kdeg(sed)', 'kdeg(soil)',
       'e_aR', 'e_w0R', 'e_w1R', 'e_w2R', 'e_s1R', 'e_s2R', 'e_s3R',
       'arealand_R', 'areasea_R','fraclake_R', 'fracfresh_R', 'fracnatsoil_R', 'fracagsoil_R',
       'fracothersoil_R', 'rainrate_R', 'temp_R', 'windspeed_R', 'depthlake_R',
       'depthfreshwater_R', 'fracrun_R', 'fracinf_R', 'erosion_R']

emissions_columns = ['e_aR', 'e_w0R', 'e_w1R', 'e_w2R', 'e_s1R', 'e_s2R', 'e_s3R']
PCA_columns = ['pKa', 'Molweight', 'Tm',
       'Pvap25', 'Sol25', 'Kaw', 'Kow', 'Ksw', 'kdeg(air)', 'kdeg(water)',
       'kdeg(sed)', 'kdeg(soil)',
       'arealand_R', 'areasea_R','fraclake_R', 'fracfresh_R', 'fracnatsoil_R', 'fracagsoil_R',
       'fracothersoil_R', 'rainrate_R', 'temp_R', 'windspeed_R', 'depthlake_R',
       'depthfreshwater_R', 'fracrun_R', 'fracinf_R', 'erosion_R']

label_columns = ['Fresh water sediment', 'Other soil', 'Marine sediment','Fresh water', 'Fresh water lake', 'Natural soil', 'Agricultural soil','Air', 'Surface sea']

val_columns = ['Fresh water sediment val', 'Other soil val', 'Marine sediment val',
       'Fresh water val', 'Fresh water lake val', 'Natural soilval', 'Agricultural soil val',
       'Air val', 'Surface sea/ocean water val']
ident_columns = ['Nr', 'material', 'CAS', 'ChemClass','scenario']
not_used_columns = [ 'Use', 'EF(air)', 'EF(water)', 'EF(soil)']


def get_scenario_properties(scenario):
    scenarios = pd.read_csv("Data/scenarios_list.csv")
    #return scenarios[scenarios["scenario"]==scenario]
    return scenarios

def get_material_properties(material):
    materials = pd.read_csv("Data/NOVA_substances_standardised.csv")
    #return materials[materials["material"]==material].iloc[0]
    return materials

def load_apply_models(data):
    # Load the models
    models = {
        "fresh_water_sediment": xgb.Booster(),
        "other_soil": xgb.Booster(),
        "marine_sediment": xgb.Booster(),
        "fresh_water": xgb.Booster(),
        "fresh_water_lake": xgb.Booster(),
        "natural_soil": xgb.Booster(),
        "agricultural_soil": xgb.Booster(),
        "air": xgb.Booster(),
        "surface_sea": xgb.Booster()
    }

    # Load each model from its respective file
    models["fresh_water_sediment"].load_model("Data/Models/xgb/results/models/Fresh water sediment.h5")
    models["other_soil"].load_model("Data/Models/xgb/results/models/Other soil.h5")
    models["marine_sediment"].load_model("Data/Models/xgb/results/models/Marine sediment.h5")
    models["fresh_water"].load_model("Data/Models/xgb/results/models/Fresh water.h5")
    models["fresh_water_lake"].load_model("Data/Models/xgb/results/models/Fresh water lake.h5")
    models["natural_soil"].load_model("Data/Models/xgb/results/models/Natural soil.h5")
    models["agricultural_soil"].load_model("Data/Models/xgb/results/models/Agricultural soil.h5")
    models["air"].load_model("Data/Models/xgb/results/models/Air.h5")
    models["surface_sea"].load_model("Data/Models/xgb/results/models/Surface sea.h5")

    results = pd.DataFrame()
    # results = pd.DataFrame()
    for model_name, model in models.items():
        # Predict using the model
        results[model_name] = model.predict(data)/10000000000
    
    return results


def load_models():
    # Load the models
    models = {
        "fresh_water_sediment": xgb.Booster(),
        "other_soil": xgb.Booster(),
        "marine_sediment": xgb.Booster(),
        "fresh_water": xgb.Booster(),
        "fresh_water_lake": xgb.Booster(),
        "natural_soil": xgb.Booster(),
        "agricultural_soil": xgb.Booster(),
        "air": xgb.Booster(),
        "surface_sea": xgb.Booster()
    }

    # Load each model from its respective file
    models["fresh_water_sediment"].load_model("Data/Models/xgb/results/models/Fresh water sediment.h5")
    models["other_soil"].load_model("Data/Models/xgb/results/models/Other soil.h5")
    models["marine_sediment"].load_model("Data/Models/xgb/results/models/Marine sediment.h5")
    models["fresh_water"].load_model("Data/Models/xgb/results/models/Fresh water.h5")
    models["fresh_water_lake"].load_model("Data/Models/xgb/results/models/Fresh water lake.h5")
    models["natural_soil"].load_model("Data/Models/xgb/results/models/Natural soil.h5")
    models["agricultural_soil"].load_model("Data/Models/xgb/results/models/Agricultural soil.h5")
    models["air"].load_model("Data/Models/xgb/results/models/Air.h5")
    models["surface_sea"].load_model("Data/Models/xgb/results/models/Surface sea.h5")

    
    return models
    
    
def apply_models(models, data):
    results = pd.DataFrame()
    # results = pd.DataFrame()
    for model_name, model in models.items():
        # Predict using the model
        results[model_name] = model.predict(data)/10000000000
    
    return results
    



def prediction(material, scenario, emissions):
    # Load the data for the transaltion of material and scenarios
    material_properties = get_material_properties(material)
    scenario_properties = get_scenario_properties(scenario)

    #Load the scaler
    scalerX = joblib.load("Data/Models/xgb/results/results-scalerX.h5")

    # Generate the input dataframe
    n = len(emissions)
    data = pd.DataFrame({
        "material": [material] * n,
        "scenario": [scenario] * n,
    })

    for i, col in enumerate(emissions_columns):
        data[col] = emissions[:, i]

    # Merge the master_df with the scenarios
    data  = pd.merge(data, material_properties, on='material', how='inner')
    data  = pd.merge(data, scenario_properties, on='scenario', how='inner')

    data = scalerX.transform(data[training_columns])
    data = xgb.DMatrix(data)

    results = load_apply_models(data)
    
    return results


def prediction_models(models, material, scenario, emissions):
    # Load the data for the transaltion of material and scenarios
    material_properties = get_material_properties(material)
    scenario_properties = get_scenario_properties(scenario)

    #Load the scaler
    scalerX = joblib.load("Data/Models/xgb/results/results-scalerX.h5")

    # Generate the input dataframe
    n = len(emissions)
    data = pd.DataFrame({
        "material": [material] * n,
        "scenario": [scenario] * n,
    })

    for i, col in enumerate(emissions_columns):
        data[col] = emissions[:, i]

    # Merge the master_df with the scenarios
    data  = pd.merge(data, material_properties, on='material', how='inner')
    data  = pd.merge(data, scenario_properties, on='scenario', how='inner')

    data = scalerX.transform(data[training_columns])
    data = xgb.DMatrix(data)

    results = apply_models(models, data)
    
    return results