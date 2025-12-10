import pandas as pd

# This function create a dicctionary given an index
def create_dict_from_index(df, index, selected_columns):
    if index < 0 or index >= len(df):
        raise IndexError("The index is out of the bounds of the DataFrame.")
    row = df.iloc[index]
    dict_content = {col: row[col] for col in selected_columns}
    return dict_content

# Fuction to color solutions
def get_color(toxicity):
    if toxicity == -1:
        return 'blue'
    elif toxicity == 0:
        return 'darkgreen'
    elif toxicity < 0.8:
        return 'lightgreen'
    #I choose 1.1 but you can modify the limits
    elif 0.8 <= toxicity <= 1.1:
        return 'yellow'
    else:
        return 'red'

def get_risk(toxicity):
    if toxicity == -1:
        return 'NO IDENTIFIED RISK'
    elif toxicity == 0:
        return 'NO RISK'
    elif toxicity < 0.8:
        return 'LOW RISK'
    #I choose 1.1 but you can modify the limits
    elif 0.8 <= toxicity <= 1.1:
        return 'MEDIUM RISK'
    else:
        return 'HIGH RISK'
    
def get_risk_color(toxicity):

    if not isinstance(toxicity, str):
        return 'no string'  # O manejar el caso de manera adecuada
    if toxicity == 'blue':
        return 'NO IDENTIFIED RISK'
    if toxicity == 'darkgreen':
        return 'NO RISK'
    elif toxicity == 'lightgreen':
        return 'LOW RISK'
    #I choose 1.1 but you can modify the limits
    elif  toxicity == 'yellow':
        return 'MEDIUM RISK'
    else:
        return 'HIGH RISK'
    
# This function approximate the results given by SB4P API. It essentially read the Excel file associated to the material and scenario
#chosen and then compute the result
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

