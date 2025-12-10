from Scripts.helpers import *
from Scripts.helpers import thredsholds
import requests
from requests.exceptions import HTTPError
import pandas as pd 
import json
from requests.exceptions import RequestException
import os
from config import PROPLANET_API_URL

headers = {
"accept": "application/json",
'Content-Type': 'application/json' 
    }  

def governing_equation(material, scenario, emissions):
    '''
    **Inputs**
        - Material: String 
        - Scenario; String
        - Emissions: List

    ** Outputs**
        - One dictionary per group: 
    '''
    #Get all the information from the PROPLANET DB
    try:
        response = requests.get(f'{PROPLANET_API_URL}materials/{material}', headers=headers)
        response_json = response.json()

    # Check if the response status code is not 200
        if response.status_code != 200:
            # If not 200, raise an HTTPError
            response.raise_for_status()
    
    except HTTPError as http_err:
    # Handle HTTP errors that occur because of non-200 status codes
        print(f'Material: {material}')
        print(f'--HTTP error occurred: {http_err}')
        print(f'--Status code: {response.status_code}')
        print(f'--Status text: {response.text}\n')
    except Exception as err:
    # Handle other possible exceptions
        print(f'Material: {material}')
        print(f'Other error occurred: {err}')
        print(f'--Status text: {response.text}\n')
    else:
        molweight = response_json["molecular_weight"]
        total_carbon_atoms = response_json["total_carbon_atoms"]
        eco_tox = response_json["ecosystem_toxicity"]
        human_tox = response_json["human_toxicity"]

        #concentration = parse_SP4P_data(material, scenario, emissions, molweight)
        resources_folder = os.path.join(".", "uma_resources", "resources","coefficients",scenario,"")
        concentration = evaluate(material, scenario, emissions, resources_folder)
        properties = RINA_equations(total_carbon_atoms)
        toxicity = nilus_thredsholds(concentration, eco_tox, human_tox)

        return concentration, properties, toxicity


# New function that represents the same as the governing equations but including the coefficients of the SB4P to reduce the time cost. It has the dictionary with the properties of the
# materials involved as input.
def governing_equation_response(material, scenario, emissions,response_json):
    '''
    **Inputs**
        - Material: String 
        - Scenario; String
        - Emissions: List
        - response_json: json with the information of the material

    ** Outputs**
        - One dictionary per group: 
    '''
    
    molweight = response_json["molecular_weight"]
    total_carbon_atoms = response_json["total_carbon_atoms"]
    eco_tox = response_json["ecosystem_toxicity"]
    human_tox = response_json["human_toxicity"]

        #concentration = parse_SP4P_data(material, scenario, emissions, molweight)
    resources_folder = os.path.join(".", "uma_resources", "resources","coefficients",scenario,"")
    concentration = evaluate(material, scenario, emissions, resources_folder)
    properties = RINA_equations(total_carbon_atoms)
    toxicity = nilus_thredsholds(concentration, eco_tox, human_tox)

    return concentration, properties, toxicity


def parse_SP4P_data(material, scenario, emissions, molweight):
    '''
    This method calls the SB4P tool and returns the information needed for the governing equations

    ** Inputs**
    String material
    String scenario 
    List[Double] emissions
    '''

    # Get the data from the different APIs
    #molecular_weight = requests.
    try:
        if type(emissions) is list:
            data = create_json_nova_list(material, molweight, scenario, emissions)
        else:
             data = create_json_nova_dict(material, molweight, scenario, emissions)
        response = requests.post("https://www.enaloscloud.novamechanics.com/proplanet/apis/sb4p/results", json=data, verify = False)
        if response.status_code != 200:
                    # If not 200, raise an HTTPError
                    response.raise_for_status()

    except HTTPError as http_err:
    # Handle HTTP errors that occur because of non-200 status codes
        print(f'Error with SB4P: {material}')
        print(f'--HTTP error occurred: {http_err}')
        print(f'--Status code: {response.status_code}')
        print(f'--Status text: {response.text}\n')
    except Exception as err:
    # Handle other possible exceptions
        print(f'Error with SB4P: {material}')
        print(f'Material: {material}')
        print(f'Other error occurred: {err}')
        print(f'--Status text: {response.text}\n')   

    else:
        # Transformation of the SB4P response to get the continental concentrations
        response_json = response.json()
        concentration_u = response_json["results"]["Steady-state Concentrations, Fugacities, Emissions and Mass"]
        keys = concentration_u.keys()
        matching_cont_data = {}
        for key in concentration_u.keys():
            matching_cont_data[key] = float(concentration_u[key][key]["Concentration Reg"])
        
        matching_cont_data["AEROSOL- and CLOUD PHASES"] = float(concentration_u["Air"][" * AEROSOL- and CLOUD PHASES"]["Concentration Reg"])
        matching_cont_data.pop("Total")
        matching_cont_data.pop("Deep sea/ocean water")
        return matching_cont_data

def evaluate(material, scenario, emissions, resources_folder):
    #file_name = fr'./resources/coefficients/{scenario}/{material}_{scenario}.csv'
    #path = os.path.join(".","resources","coefficients",scenario,"")
    file_name = fr'{resources_folder}{material}_{scenario}.csv'

    try:
        # Load the CSV file as a dataframe
        df = pd.read_csv(file_name)      
        
        # Create a dictionary for the results
        results = {}

        # Lista de títulos
        titles = ['Fresh water sediment', 'Other soil', 'Marine sediment', 'Fresh water', 
           'Fresh water lake', 'Natural soil', 'Agricultural soil', 'Air', 'Surface sea/ocean water','AEROSOL- and CLOUD PHASES']

        for index, row in df.iterrows():
            output = row['Intercept']  
            for i, coef in enumerate(emissions):
                output += coef * row[f'input{i+1}']
            results[titles[index]] = output
        return results

    except FileNotFoundError:
        print(f"Error: the file {file_name} does not exist.")
        return None


def RINA_equations(carbon_atoms):
    WCA = 89.5364808871647 + 1.40425915103617 * carbon_atoms
    HCA = 18.91250983 + 1.00177098152491 * carbon_atoms
    SFE = 29.1234533438759 - 0.41019156996163 * carbon_atoms

    return {"WCA":WCA, "HCA":HCA, "SFE":SFE}

def nilus_thredsholds(emissions, eco_tox, human_tox):
    # air
    air_eco = thredsholds(emissions,eco_tox, em_name="Air",eco_name="air")

    # air
    air_systematic_effect = thredsholds(emissions,human_tox, em_name="Air",eco_name="p_inhalation_systemic_long")

    # air
    air_local_effect = thredsholds(emissions,human_tox, em_name="Air",eco_name="p_inhalation_local_long")

    # fresh_water
    fresh_water = thredsholds(emissions,eco_tox, em_name='Fresh water',eco_name='aquatic_freshwater')

    # fresh_water lake
    fresh_water_lake = thredsholds(emissions,eco_tox, em_name='Fresh water lake',eco_name='aquatic_freshwater')

    #ocean_water 
    ocean_water = thredsholds(emissions, eco_tox, em_name='Surface sea/ocean water', eco_name='aquatic_marinewater')

    #marine_sediment 
    marine_sediment = thredsholds(emissions, eco_tox, em_name='Marine sediment', eco_name='aquatic_sediment_marinewater')

    #fresh_sediment
    fresh_sediment = thredsholds(emissions, eco_tox, em_name='Fresh water sediment', eco_name = 'aquatic_sediment_freshwater')
    
    #natural_soil
    natural_soil = thredsholds(emissions, eco_tox, em_name = 'Natural soil', eco_name = 'terrestrial_soil')

    #natural_soil
    agricult_soil = thredsholds(emissions, eco_tox, em_name = 'Agricultural soil', eco_name = 'terrestrial_soil')

    #other_soil
    other_soil = thredsholds(emissions, eco_tox, em_name = 'Other soil', eco_name = 'terrestrial_soil')
    
    return {"air":air_eco, "air_systematic_effect":air_systematic_effect,"air_local_effect":air_local_effect, "fresh_water":fresh_water, "fresh_water_lake":fresh_water_lake, "ocean_water":ocean_water, 
            "marine_sediment":marine_sediment, "fresh_sediment":fresh_sediment,
            "natural_soil":natural_soil, "agricultural_soil":agricult_soil, "other_soil":other_soil}

def lung_deposition_model(model, scenario, concentration, respiratoryVolumeRate, exposureDuration):
    attempt = 0
    max_attempts = 3
    session = requests.Session()
    if scenario is None:
        return {'acuteDoseHA': None,
            'acuteDoseTB': None,
            'acuteDoseAL': None}
    else:
        while attempt < max_attempts:
            try:
                data = {
                    "model": model,
                    "scenario": scenario,
                    "concentration": concentration,
                    "respiratoryVolumeRate": respiratoryVolumeRate,
                    "exposureDuration": exposureDuration
                }
                response = session.post("https://www.enaloscloud.novamechanics.com/proplanet/apis/lungdeposition", json=data, verify = False, timeout=10)
                # Check if the response was successful
                if response.status_code == 200:
                    return response.json()
                    break
                else:
                    print(f"Request failed with status code {response.status_code}")
                    attempt += 1

            except RequestException as e:
                print(f"Request failed: {e}")
                attempt += 1

            else:
                return response.json()
        if attempt == max_attempts:
            print("Max attempts reached, moving to the next item")


def create_vectors():
    ## CREATE THE EMISSION VECTOR with variability
    # Define the ranges for each vector
    e_aR_values = range(0, 151, 50)   # [0, 50, 100, 150]
    e_w0R_values = [0, 50]
    e_w1R_values = range(0, 151, 50)  # [0, 50, 100, 150]
    e_w2R_values = [0, 50]
    e_s1R_values = [0, 50]
    e_s2R_values = [0, 50]
    e_s3R_values = [0, 50]

    # Function to add variability of ±20%
    def add_variability(value):
        if value == 0:
            return 0  # No variability for zero values
        variability = value * 0.20
        return round(value + random.uniform(-variability, +variability), 2)

    # Generate all combinations
    combinations = list(itertools.product(e_aR_values, e_w0R_values, e_w1R_values, e_w2R_values, e_s1R_values, e_s2R_values, e_s3R_values))

    # Apply variability and create the list of dictionaries
    keys = ["e_aR", "e_w0R", "e_w1R", "e_w2R", "e_s1R", "e_s2R", "e_s3R"]
    vector_dicts = [dict(zip(keys, [add_variability(value) for value in vector])) for vector in combinations]

    return vector_dicts

def generate_data(material_dict):
    ## calls and saves the SB4P results
    conc = pd.DataFrame(columns=['material','scenario',"e_aR", "e_w0R", "e_w1R", "e_w2R", "e_s1R", "e_s2R", "e_s3R",'Fresh water sediment','Other soil','Marine sediment','Fresh water','Fresh water lake','Natural soil','Agricultural soil','Air','Surface sea/ocean water'])
    
    #Get the list on emissions
    vectors_dict = create_vectors()

    #Get the list of scenarios
    scenarios = requests.get('https://www.enaloscloud.novamechanics.com/proplanet/apis/sb4p/scenarios', headers=headers, verify=False)
    scenarios_json = scenarios.json()
    scenarions_list = pd.DataFrame.from_dict(scenarios_json)["scenario"].tolist()

    for material, molweight in material_dict.items():
        for scenario in scenarions_list:
            for vector_dict in vectors_dict:
                concentration_aux = parse_SP4P_data(material, scenario, vector_dict, molweight)
                dict_aux = {"material":material, "scenario":scenario} | vector_dict | concentration_aux
                conc = pd.concat([conc,pd.DataFrame([dict_aux])])
                conc.to_csv("test.csv", index = False)

def generate_data_per_material(material, molweight):
    ## calls and saves the SB4P results
    conc = pd.DataFrame(columns=['material','scenario',"e_aR", "e_w0R", "e_w1R", "e_w2R", "e_s1R", "e_s2R", "e_s3R",'Fresh water sediment','Other soil','Marine sediment','Fresh water','Fresh water lake','Natural soil','Agricultural soil','Air','Surface sea/ocean water'])
    
    #Get the list on emissions
    vectors_dict = create_vectors()

    #Get the list of scenarios
    scenarios = requests.get('https://www.enaloscloud.novamechanics.com/proplanet/apis/sb4p/scenarios', headers=headers, verify = False)
    scenarios_json = scenarios.json()
    scenarions_list = pd.DataFrame.from_dict(scenarios_json)["scenario"].tolist()

    for scenario in scenarions_list:
        for vector_dict in vectors_dict:
            concentration_aux = parse_SP4P_data(material, scenario, vector_dict, molweight)
            dict_aux = {"material":material, "scenario":scenario} | vector_dict | concentration_aux
            conc = pd.concat([conc,pd.DataFrame([dict_aux])])
            conc.to_csv(f"{material}.csv", index = False)
