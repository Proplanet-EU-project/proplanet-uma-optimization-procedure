import requests
from requests.exceptions import HTTPError
import pandas as pd 
import json
import itertools
import random
from config import PROPLANET_API_URL

headers = {
"accept": "application/json",
'Content-Type': 'application/json' 
    } 

def thredsholds(emissions, eco_tox, em_name, eco_name):
    '''
    CODING:
         0 : No hazard detected
        -1 : means no ECHA data
        -2 : low hazard w/o thredshold -> toxicity to 1
        -3 : medium hazard w/o thredshold -> toxicity to 1.5
        -4 : high hazard w/o thredshold -> toxicity to 2
    '''
    _eco = eco_tox[eco_name]   
    if _eco > 0:
        res= emissions[em_name]/_eco
    elif _eco == -1:
        res = -1
    elif _eco == -2:
        res = 1
    elif _eco == -3:
        res = 1.5
    elif _eco == -4:
        res = 2
    else:
        res = 0
    
    return res

def delete_all_materials():
    r = requests.get(f'{PROPLANET_API_URL}materials', headers=headers)
    r = r.json()
    names = [it["name"] for it in r]
    for mat in r:
        requests.delete(f'{PROPLANET_API_URL}materials/{mat["name"]}',  headers=headers)


def get_materials_list():
    r = requests.get(f'{PROPLANET_API_URL}materials', headers=headers)
    r = r.json()
    return {it["name"]:it["molecular_weight"] for it in r}

def import_materials_excel(excel_path):
    # import excel file
    materials = pd.read_excel(excel_path)
    materials[["name","formula","cas","category"]] = materials[["name","formula","cas","category"]].astype(str)
    materials[['total_carbon_atoms',
        'molecular_weight', 'w_inhalation_systemic_long',
        'w_inhalation_systemic_short', 'w_inhalation_local_long',
        'w_inhalation_local_short', 'w_dermal_systemic_long',
        'w_dermal_systemic_short', 'w_dermal_local_long',
        'w_dermal_local_short', 'w_eyes_local', 'p_inhalation_systemic_long',
        'p_inhalation_systemic_short', 'p_inhalation_local_long',
        'p_inhalation_local_short', 'p_dermal_systemic_long',
        'p_dermal_systemic_short', 'p_dermal_local_long',
        'p_dermal_local_short', 'p_eyes_local', 'p_oral_systemic_long',
        'p_oral_systemic_short', 'aquatic_freshwater', 'aquatic_marinewater',
        'aquatic_stp', 'aquatic_sediment_freshwater',
        'aquatic_sediment_marinewater', 'air', 'terrestrial_soil',
        'predators_oral_poisoning']] = materials[['total_carbon_atoms',
        'molecular_weight', 'w_inhalation_systemic_long',
        'w_inhalation_systemic_short', 'w_inhalation_local_long',
        'w_inhalation_local_short', 'w_dermal_systemic_long',
        'w_dermal_systemic_short', 'w_dermal_local_long',
        'w_dermal_local_short', 'w_eyes_local', 'p_inhalation_systemic_long',
        'p_inhalation_systemic_short', 'p_inhalation_local_long',
        'p_inhalation_local_short', 'p_dermal_systemic_long',
        'p_dermal_systemic_short', 'p_dermal_local_long',
        'p_dermal_local_short', 'p_eyes_local', 'p_oral_systemic_long',
        'p_oral_systemic_short', 'aquatic_freshwater', 'aquatic_marinewater',
        'aquatic_stp', 'aquatic_sediment_freshwater',
        'aquatic_sediment_marinewater', 'air', 'terrestrial_soil',
        'predators_oral_poisoning']].astype(float)
    
 
    for _, material in materials.iterrows():
        try:
            data_material = create_json_import_material(material)
            response = requests.post(f'{PROPLANET_API_URL}materials', json = data_material, headers = headers)

            # Check if the response status code is not 200
            if response.status_code != 200:
                # If not 200, raise an HTTPError
                response.raise_for_status()

        except HTTPError as http_err:
        # Handle HTTP errors that occur because of non-200 status codes
            print(f'Material: {material["name"]}')
            print(f'--HTTP error occurred: {http_err}')
            print(f'--Status code: {response.status_code}')
            print(f'--Status text: {response.text}\n')
        except Exception as err:
        # Handle other possible exceptions
            print(f'Material: {material["name"]}')
            print(f'Other error occurred: {err}')
            print(f'--Status text: {response.text}\n')
        else:
            data_human = create_json_import_human_tox(material)
            data_ecotox = create_json_import_ecotox(material)

            requests.post(f'{PROPLANET_API_URL}human_toxicities/{material["name"]}', json = data_human, headers = headers)
            requests.post(f'{PROPLANET_API_URL}ecosystem_toxicities/{material["name"]}', json = data_ecotox, headers = headers)


### JSONs for the different API calls

def create_json_import_material(material):
    return {
  "name": material["name"] ,
  "formula": material["formula"] ,
  "cas": material["cas"] ,
  "category":"test",
  "total_carbon_atoms":material["total_carbon_atoms"] ,
  "molecular_weight": material["molecular_weight"],
}

def create_json_import_human_tox(material):
    return {
      "w_inhalation_systemic_long": material["w_inhalation_systemic_long"],
      "w_inhalation_systemic_short":material["w_inhalation_systemic_short"] ,
      "w_inhalation_local_long": material["w_inhalation_local_long"],
      "w_inhalation_local_short":material["w_inhalation_local_short"] ,
      "w_dermal_systemic_long": material["w_dermal_systemic_long"],
      "w_dermal_systemic_short":material["w_dermal_systemic_short"] ,
      "w_dermal_local_long": material["w_dermal_local_long"],
      "w_dermal_local_short": material["w_dermal_local_short"],
      "w_eyes_local": material["w_eyes_local"],
      "p_inhalation_systemic_long": material["p_inhalation_systemic_long"],
      "p_inhalation_systemic_short": material["p_inhalation_systemic_short"],
      "p_inhalation_local_long": material["p_inhalation_local_long"],
      "p_inhalation_local_short": material["p_inhalation_local_short"],
      "p_dermal_systemic_long": material["p_dermal_systemic_long"],
      "p_dermal_systemic_short": material["p_dermal_systemic_short"],
      "p_dermal_local_long": material["p_dermal_local_long"],
      "p_dermal_local_short": material["p_dermal_local_short"],
      "p_eyes_local": material["p_eyes_local"],
      "p_oral_systemic_long": material["p_oral_systemic_long"],
      "p_oral_systemic_short": material["p_oral_systemic_short"]

  }

def create_json_import_ecotox(material):
    return{
      "aquatic_freshwater": material["aquatic_freshwater"],
      "aquatic_marinewater": material["aquatic_marinewater"],
      "aquatic_stp": material["aquatic_stp"],
      "aquatic_sediment_freshwater":material["aquatic_sediment_freshwater"] ,
      "aquatic_sediment_marinewater":material["aquatic_sediment_marinewater"] ,
      "air": material["air"],
      "terrestrial_soil": material["terrestrial_soil"],
      "predators_oral_poisoning":material["predators_oral_poisoning"] }




def create_json_nova_list(material,  molweight, scenario, emissions):
    return  {"scenario": scenario,
        "substance": material,
        "molweight":  molweight,
        "e_aR": emissions[0],
        "e_w0R": emissions[1],
        "e_w1R": emissions[2],
        "e_w2R": emissions[3],
        "e_s1R": emissions[4],
        "e_s2R": emissions[5],
        "e_s3R": emissions[6],
        "e_aC": 0,
        "e_w0C": 0,
        "e_w1C": 0,
        "e_w2C": 0,
        "e_s1C": 0,
        "e_s2C": 0,
        "e_s3C": 0,
        "e_aA": 0,
        "e_w2A": 0,
        "e_sA": 0,
        "e_aT": 0,
        "e_w2T": 0,
        "e_sT": 0,
        "e_aM": 0,
        "e_w2M": 0,
        "e_sM": 0,
        "filePath": "string"
        }


def create_json_nova_dict(material,  molweight, scenario, emissions):
    return  {"scenario": scenario,
        "substance": material,
        "molweight":  molweight,
        "e_aR": emissions["e_aR"],
        "e_w0R": emissions["e_w0R"],
        "e_w1R": emissions["e_w1R"],
        "e_w2R": emissions["e_w2R"],
        "e_s1R": emissions["e_s1R"],
        "e_s2R": emissions["e_s2R"],
        "e_s3R": emissions["e_s3R"],
        "e_aC": 0,
        "e_w0C": 0,
        "e_w1C": 0,
        "e_w2C": 0,
        "e_s1C": 0,
        "e_s2C": 0,
        "e_s3C": 0,
        "e_aA": 0,
        "e_w2A": 0,
        "e_sA": 0,
        "e_aT": 0,
        "e_w2T": 0,
        "e_sT": 0,
        "e_aM": 0,
        "e_w2M": 0,
        "e_sM": 0,
        "filePath": "string"
}
