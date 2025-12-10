import pandas as pd
import requests

# Load the CSV file of NOVA substances
nova_subs = pd.read_csv("NOVA_substances_standardised.csv")

# Load the different CSV files for the materials to be considered
mat1 = pd.read_csv("acquired_data/2-Octenylsuccinic anhydride.csv")
mat2 = pd.read_csv("acquired_data/Acetic acid.csv")
mat3 = pd.read_csv("acquired_data/Chitosan.csv")
mat4 = pd.read_csv("acquired_data/Dodecyltriethoxysilane.csv")
mat5 = pd.read_csv("acquired_data/Glycerol.csv")
mat6 = pd.read_csv("acquired_data/Hexadecyltrimethoxysilane.csv")
mat7 = pd.read_csv("acquired_data/Hexamethyldisiloxane.csv")
mat8 = pd.read_csv("acquired_data/Methyltrimethoxysilane.csv")
mat9 = pd.read_csv("acquired_data/Polysiloxanes, di-Me (Silicon oil).csv")
mat10 = pd.read_csv("acquired_data/Polysiloxanes, di-Me, hydroxy-terminated.csv")
mat11 = pd.read_csv("acquired_data/Sodium alginate.csv")
mat12 = pd.read_csv("acquired_data/Starch (Corn starch).csv")
mat13 = pd.read_csv("acquired_data/Trimethoxyphenylsilane.csv")

# Now, concatenate the different dfs
mats = []
mats.append(mat1)
mats.append(mat2)
mats.append(mat3)
mats.append(mat4)
mats.append(mat5)
mats.append(mat6)
mats.append(mat7)
mats.append(mat8)
mats.append(mat9)
mats.append(mat10)
mats.append(mat11)
mats.append(mat12)
mats.append(mat13)

mats_concat = pd.concat(mats, ignore_index=True)

# Merge the two dfs
master_df = pd.merge(nova_subs, mats_concat, on='material', how='inner')

# Load the scenarios data
headers = {
"accept": "application/json",
'Content-Type': 'application/json'
    }
scenarios = requests.get('https://www.enaloscloud.novamechanics.com/proplanet/apis/sb4p/scenarios', headers=headers)
scenarios_json = scenarios.json()
df_scenarios = pd.DataFrame.from_dict(scenarios_json)

# Get rid of the emission rates provided with the scenarios
drop = ['e_aR', 'e_w0R', 'e_w1R', 'e_w2R', 'e_s1R', 'e_s2R', 'e_s3R', 'e_aC', 'e_w0C', 'e_w1C', 'e_w2C', 'e_s1C',
        'e_s2C', 'e_s3C', 'e_aA', 'e_w2A', 'e_sA', 'e_aT', 'e_w2T', 'e_sT', 'e_aM', 'e_w2M', 'e_sM']
df_scenarios = df_scenarios.drop(drop, axis=1)

# Merge the master_df with the scenarios
master_df = pd.merge(master_df, df_scenarios, on='scenario', how='inner')

# Save the merged data
master_df.to_csv('acquired_data.csv', index=False)