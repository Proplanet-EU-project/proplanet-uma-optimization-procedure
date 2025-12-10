import pandas as pd
import numpy as np
from io import StringIO
import time
import os
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go

from uma_resources.resources import gov_equations as gov_equations
from uma_resources.resources import helpers_UMA as UMA

# This is the final function to reproduce the idea of the Scenario B proposed by IDENER.
# As inputs we need a list of materials(It could be whatever the user wants, but only that materials that we could
# find in the API of NoVamechanics)
# Also we have a toxicity parameter fix, an scenerio and an upperbound o the emissions
# The toxicitiy parameter must be written like 'natural_soil_toxicity'
def optimization_procedure(materials_list, toxicity_parameter, scenario, upperbound, model, respiratoryVolumeRate, exposureDuration, lung_deposition_flags,base_concentration):

    resources_folder = os.path.join(".", "uma_resources", "resources","coefficients",scenario,"")
    dict_sb4p_lung2 = {
        'Hexadecyltrimethoxysilane': 'Silane_based_Motzkus_et_al_2011',
        'Hexamethyldisiloxane': 'Siloxane_based_McDonagh_and_Byrne_2014',
        'Trimethoxyphenylsilane': 'Silane_based_Motzkus_et_al_2011',
        'Polysiloxanes, di-Me, hydroxy-terminated': 'Siloxane_based_McDonagh_and_Byrne_2014',
        'Polysiloxanes, di-Me (Silicon oil)': 'Siloxane_based_McDonagh_and_Byrne_2014',
        'Acetic acid': 'Acetic_Acid_Zhang_et_al_2019',
        'Starch (Corn starch)': 'Corn_Starch_Fuentes_et_al_2022',
        'Zinc oxide (ZnO)': 'ZnO_Monse_et_al_2021',
        'Sodium alginate': 'Sodium_alginate_Santa-Maria_et_al_2012',
        'Glycerol': 'Glycerol_Guzman_2024',
        'Chitosan': 'Chitosan_Patil_and_Sawant_2011',
        '2-Octenylsuccinic anhydride': '2-Octenylsuccinic_anhydride_Wang_et_al_2020',
        'Dodecyltriethoxysilane': 'Silane_based_Motzkus_et_al_2011',
        'Methyltrimethoxysilane': 'Silane_based_Motzkus_et_al_2011',
        'Octyltrimethoxysilane': 'Silane_based_Motzkus_et_al_2011'
    }
    toxicity_name_mapping = {
        'air': 'air toxicity',
        'air_systematic_effect': 'air systematic toxicity',
        'air_local_effect': 'air local toxicity',
        'fresh_water': 'fresh water toxicity',
        'ocean_water': 'ocean water toxicity',
        'marine_sediment': 'marine sediment toxicity',
        'fresh_sediment': 'fresh sediment toxicity',
        'natural_soil': 'natural soil toxicity',
        'agricultural_soil': 'agricultural soil toxicity',
        'other_soil': 'other soil toxicity'
    }

    toxicity_parameter_final = toxicity_name_mapping.get(toxicity_parameter, toxicity_parameter)
    # Pre-calculate lung deposition factors once per material.
    lung_deposition_factors = {}
    for material in materials_list:
        material_name = material.dict()['name']
        if lung_deposition_flags.get(material_name, False):
            lung_dict = gov_equations.lung_deposition_model(
                model,
                dict_sb4p_lung2[material_name],
                base_concentration,
                respiratoryVolumeRate,
                exposureDuration
            )
            lung_deposition_factors[material_name] = {
                'acuteDoseHA_factor': float(lung_dict['acuteDoseHA']),
                'acuteDoseTB_factor': float(lung_dict['acuteDoseTB']),
                'acuteDoseAL_factor': float(lung_dict['acuteDoseAL'])
            }
    # First we define the optimization problem
    class Optimizer(Problem):

        def __init__(self, material):
            super().__init__(n_var=7, n_obj=2, xl=0, xu=upperbound)
            self.material = material
            
        def _evaluate(self, x, out, *args, **kwargs):
            # List to keep the API values for each initial solution.
            api_values = []

            # Iterate over each solution in the population and call the API to retrieve the associated values
            for emissions in x:
                retry_count = 0
                max_retries = 5
                success = False
                while retry_count < max_retries and not success:
                    try:
                        api_result = UMA.evaluate(self.material, scenario, emissions, resources_folder)
                        if api_result is not None:
                            keys_except_last = list(api_result.keys())[:-1]
                            api_result2 = {key: api_result[key] for key in keys_except_last}
                            api_values.append(api_result2)
                            success = True
                        else:
                            raise ValueError("API result is None")
                    except Exception as e:
                        print(f"Error calling API: {e}. Retrying in 5 seconds...")
                        print(f"Retry count: {retry_count + 1}/{max_retries}")
                        retry_count += 1
                        time.sleep(5)
                        if retry_count == max_retries:
                            api_values.append({k: np.nan for k in range(9)})
                            break
                if(retry_count == max_retries):
                    print(f"Max retries reached for emissions: {emissions}. Skipping this solution.")
                    break
                    
            
            # Convert the list of values returned by the API into a NumPy array
            api_values_matrix = np.array([list(d.values()) for d in api_values])

            # Calculate the first objective: maximization of the sum of the variables (convert to minimization)
            f1 = -np.sum(x, axis=1)
            # Create a copy of the array to modify

            modified_values = np.copy(api_values_matrix)

            # Multiply all elements by 1000 except the seventh element, which is air.(change of units to unify all of them)

            modified_values[:, :7] *= 1000
            modified_values[:, 8:] *= 1000
            
            # Sum the values of each row
            f2 = np.sum(modified_values, axis=1)
            # Calculate the second objective: minimization of the sum of the API outputs        

            # Save the result in the output dictionary
            out["F"] = np.column_stack([f1, f2])
    
    all_df_combined = pd.DataFrame() # All data combined
    all_df_ecotox = pd.DataFrame()   # Data associated to toxicitity_parameter

    #Now we will solve the Optimization problem finding the respective pareto fronts for each material
    for material in materials_list:
        
        material_name = material.dict()['name']
        
        print(f"Optimizing for material: {material_name}")
        problem = Optimizer(material_name)

        algorithm = NSGA2(pop_size=100)

        Solutions = minimize(problem,
                             algorithm,
                             ('n_gen', 25),
                             seed=1,
                             save_history=True,
                             verbose=True)

        print(f"Non-dominated solutions for {material_name}:")
        print(Solutions.X)
        print(f"Objective values for these solutions for {material_name}:")
        print(Solutions.F)

        last_population = Solutions.history[-1].pop
        last_population_emission_rates = np.array([ind.X for ind in last_population])
        last_population_function_values = np.array([ind.F for ind in last_population])
        
        # Converts the solutions and their objectives values in Dataframes
        df_optimal_emission_rates = pd.DataFrame(Solutions.X, columns=[f'x{i+1}' for i in range(Solutions.X.shape[1])])
        # df_optimal_function_value = pd.DataFrame(Solutions.F, columns=[f'f{i+1}' for i in range(Solutions.F.shape[1])])

        df_lastpopulation = pd.DataFrame(last_population_emission_rates, columns=[f'x{i+1}' for i in range(last_population_emission_rates.shape[1])])

        # Now we compute the toxicity for each non-dominate solutions individually
        
        new_columns = ['Fresh water sediment', 'Other soil', 'Marine sediment', 'Fresh water', 'Fresh water lake', 'Natural soil', 'Agricultural soil', 'Air', 'Surface sea/ocean water']
        lung_columns = ['acuteDoseHA', 'acuteDoseTB', 'acuteDoseAL']
        for col in new_columns:
            df_optimal_emission_rates[col] = np.nan
        for col_Lung in lung_columns:
            df_optimal_emission_rates[col_Lung] = np.nan
        
        for index, row in df_optimal_emission_rates.iterrows():
        
            emission_rates = row.loc['x1':'x7'].tolist()
        
            # We applied our aproximation function to each non-dominate solution
            api_results_optimal_emission_rates = UMA.evaluate(material_name, scenario, emission_rates, resources_folder)
            if lung_deposition_flags.get(material_name, False):

                concentration = api_results_optimal_emission_rates['AEROSOL- and CLOUD PHASES']*1e12
                material_factors = lung_deposition_factors[material_name]
                lung_values = {
                    'acuteDoseHA': material_factors['acuteDoseHA_factor'] * concentration,
                    'acuteDoseTB': material_factors['acuteDoseTB_factor'] * concentration,
                    'acuteDoseAL': material_factors['acuteDoseAL_factor'] * concentration
                }
            else:
                lung_values = {
                    'acuteDoseHA': np.nan,
                    'acuteDoseTB': np.nan,
                    'acuteDoseAL': np.nan
                }
            
            for col in new_columns:
                df_optimal_emission_rates.at[index, col] = api_results_optimal_emission_rates.get(col)
            for col_Lung in lung_columns:
                df_optimal_emission_rates.at[index, col_Lung] = lung_values.get(col_Lung)

        for index2, row2 in df_lastpopulation.iterrows():
        
            all_emission_rates = row2.loc['x1':'x7'].tolist()
        
            api_results_all_emission_rates = UMA.evaluate(material_name, scenario, all_emission_rates, resources_folder)

            for col in new_columns:
                df_lastpopulation.at[index2, col] = api_results_all_emission_rates.get(col)
                
        #Now we will include new columns associated to toxicity parameters
        df_toxicity_all_solutions = df_lastpopulation.iloc[:, 7:]

        df_toxicity_optimal_solutions = df_optimal_emission_rates.iloc[:,7:]

        #df_ecotox_main = df_toxicity_all_solutions
        
        df_ecotox_optimal_solutions = df_toxicity_optimal_solutions
        
        # With that code you can show the dominate solutions too
        #df_ecotox = pd.concat([df_ecotox_main, df_ecotox_optimal_solutions])

        df_ecotox = df_ecotox_optimal_solutions
        df_ecotox['Material'] = material_name

        all_df_ecotox = pd.concat([all_df_ecotox, df_ecotox], ignore_index=True)
        
        # Unified the units to sum different emissions rates
        columns_multiply = ['Fresh water sediment', 'Other soil', 'Marine sediment', 'Fresh water', 'Fresh water lake', 'Natural soil', 'Agricultural soil', 'Surface sea/ocean water']

        df_toxicity_all_solutions[columns_multiply] = df_toxicity_all_solutions[columns_multiply]*1000
        df_toxicity_optimal_solutions[columns_multiply] = df_toxicity_optimal_solutions[columns_multiply]*1000

        objectives_all_solutions = last_population_function_values
        objectives_optimal_solutions = Solutions.F

        df_all_solutions = pd.DataFrame(objectives_all_solutions, columns=[f'f{i+1}' for i in range(2)])
        df_all_solutions = pd.concat([df_all_solutions,df_toxicity_all_solutions], axis=1)
        df_optimal_solutions = pd.DataFrame(objectives_optimal_solutions, columns=[f'f{i+1}' for i in range(2)])
        df_optimal_solutions = pd.concat([df_optimal_solutions,df_toxicity_optimal_solutions], axis=1)
        df_all_solutions['f1'] *= -1
        df_optimal_solutions['f1'] *= -1
        df_all_solutions['type'] = 'All solutions'
        df_optimal_solutions['type'] = 'Non dominate solutions'
        # Shows all solutions in the graph 
        #df_combined = pd.concat([df_all_solutions, df_optimal_solutions], ignore_index=True)

        df_combined = df_optimal_solutions
        df_combined['Material'] = material_name
        
        df_combined.rename(columns={
            'f1': 'Total Emission rates (t/y)',
            'f2' : 'Total Concentration levels (ppm)'
            
        }, inplace=True)

        # Adding the properties obtained using the model of RINA to the dataframe
        total_carbon_atoms = material.dict()["totalCarbonAtoms"]

        Rina_values = gov_equations.RINA_equations(total_carbon_atoms)

        if total_carbon_atoms == 0:
            for key, value in Rina_values.items():
                
                df_combined[key] = -1
        else:
            for key, value in Rina_values.items():
                
                df_combined[key] = value
        all_df_combined = pd.concat([all_df_combined, df_combined], ignore_index=True)
        print(all_df_combined)
        
    # Adding the toxicity results obtained using Nilu tresholds
    data = []
    for index, row in all_df_ecotox.iterrows():
            
        material = row['Material']

        selected_columns = ['Fresh water sediment', 'Other soil', 'Marine sediment', 'Fresh water', 'Fresh water lake', 'Natural soil', 'Agricultural soil', 'Air', 'Surface sea/ocean water']      
        emissions_dic = UMA.create_dict_from_index(all_df_ecotox,index,selected_columns)
        print(emissions_dic)
        material_data = next(m for m in materials_list if m.name == material)
        
        molecular_weight = material_data.molecularWeight
        total_carbon_atoms = material_data.totalCarbonAtoms
        eco_tox = material_data.ecosystemToxicity
        human_tox = material_data.humanToxicity
        
        data.append(gov_equations.nilus_thredsholds(emissions_dic,eco_tox,human_tox))
            
    toxicity_df = pd.DataFrame(data)
    
    toxicity_df.rename(columns={
            'air': 'air toxicity',
            'air_systematic_effect' : 'air systematic toxicity',
            'air_local_effect' : 'air local toxicity',
            'fresh_water': 'fresh water toxicity',
            'ocean_water': 'ocean water toxicity',
            'marine_sediment': 'marine sediment toxicity',
            'fresh_sediment': 'fresh sediment toxicity',
            'natural_soil' : 'natural soil toxicity',
            'agricultural_soil' : 'agricultural soil toxicity',
            'other_soil': 'other soil toxicity'
        }, inplace=True)
    
    all_df_combined2 = pd.concat([all_df_combined, toxicity_df], axis=1)
    
    # Create risk and color columns for ALL toxicity parameters, not just the selected one.
    # This pre-calculates the colors for the interactive dropdown.
    all_toxicity_parameters = list(toxicity_name_mapping.values())
    for tox_param in all_toxicity_parameters:
        all_df_combined2[f'risk_{tox_param}'] = all_df_combined2[tox_param].apply(UMA.get_risk)

    risk_order = ['NO IDENTIFIED RISK', 'NO RISK', 'LOW RISK', 'MEDIUM RISK', 'HIGH RISK']
    risk_categorical_type = pd.CategoricalDtype(categories=risk_order, ordered=True)

    risk_columns = [f'risk_{tox_param}' for tox_param in all_toxicity_parameters]

    for col in risk_columns:
        all_df_combined2[col] = all_df_combined2[col].astype(risk_categorical_type)

    all_df_combined2['risk_All'] = all_df_combined2[risk_columns].max(axis=1)

    df_optimal_emission_rates_necessary = df_optimal_emission_rates.iloc[:, :7]
    df_optimal_emission_rates_necessary.columns = ['emission_rates_to_air','emission_rates_to_lake_water', 'emission_rates_to_fresh_water', 'emission_rates_to_sea_water', 'emission_rates_to_natural_soil', 'emission_rates_to_agricultural_soil', 'emission_rates_to_other_soil']
    all_df_combined3 = pd.concat([all_df_combined2, df_optimal_emission_rates_necessary], axis = 1)

    color_discrete_map = {
        'NO RISK': 'darkgreen',
        'LOW RISK': 'lightgreen',
        'MEDIUM RISK': 'yellow',
        'HIGH RISK': 'red',
        'NO IDENTIFIED RISK': 'blue'
    }
    
    material_names = [material.name for material in materials_list]
    symbol_sequence = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'star']
    # Use Plotly Graph Objects for more control
    fig_Optimization = go.Figure()

    # Create a trace for each material
    for i,material_name in enumerate(all_df_combined3['Material'].unique()):
        df_material = all_df_combined3[all_df_combined3['Material'] == material_name]
        
        initial_colors = df_material['risk_All'].astype(str).map(color_discrete_map)

        fig_Optimization.add_trace(
            go.Scatter(
                x=df_material['Total Emission rates (t/y)'],
                y=df_material['Total Concentration levels (ppm)'],
                mode='markers',
                marker=dict(
                    color=initial_colors,
                    symbol=symbol_sequence[i % len(symbol_sequence)],
                ),
                name=material_name,
                customdata=df_material[[
                    'air toxicity', 'air systematic toxicity', 'air local toxicity', 
                    'fresh water toxicity', 'ocean water toxicity', 'marine sediment toxicity', 
                    'fresh sediment toxicity', 'natural soil toxicity', 'agricultural soil toxicity', 
                    'other soil toxicity', 'acuteDoseHA', 'acuteDoseTB', 'acuteDoseAL'
                ]],
                hovertemplate=(
                    "<b>Objective functions:</b><br>"
                    "Total Emission rates (t/y): %{x:.5e}<br>"
                    "Total Concentration levels (ppm): %{y:.5e}<br>"
                    "<br>"
                    "<b>Concentration ratio:</b><br>"
                    "air toxicity: %{customdata[0]:.5e}<br>"
                    "air systematic toxicity: %{customdata[1]:.5e}<br>"
                    "air local toxicity: %{customdata[2]:.5e}<br>"
                    "fresh water toxicity: %{customdata[3]:.5e}<br>"
                    "ocean water toxicity: %{customdata[4]:.5e}<br>"
                    "marine sediment toxicity: %{customdata[5]:.5e}<br>"
                    "fresh sediment toxicity: %{customdata[6]:.5e}<br>"
                    "natural soil toxicity: %{customdata[7]:.5e}<br>"
                    "agricultural soil toxicity: %{customdata[8]:.5e}<br>"
                    "other soil toxicity: %{customdata[9]:.5e}<br>"
                    "<br>"
                    "<b>Lung exposure values:</b><br>"
                    "Head Airway lung region (pg): %{customdata[10]:.5e}<br>"
                    "Tracheobronchial lung region (pg): %{customdata[11]:.5e}<br>"
                    "Alveolar lung region (pg): %{customdata[12]:.5e}<br>"
                    "<extra></extra>" 
                )
            )
        )

    buttons = []
    all_colors_per_trace = []
    for trace in fig_Optimization.data:
        material_name = trace.name
        df_material = all_df_combined3[all_df_combined3['Material'] == material_name]
        colors = df_material['risk_All'].astype(str).map(color_discrete_map).tolist()
        all_colors_per_trace.append(colors)

    buttons.append(dict(
        method='update',
        label='All (Most Restrictive)',
        args=[{'marker.color': all_colors_per_trace},
              {'title': f"Optimal Solutions: {scenario} {', '.join(material_names)}<br><br><sup>Colored according to Most Restrictive Risk</sup>"}]
    ))

    for tox_param in all_toxicity_parameters:
        new_colors_per_trace = []
        for trace in fig_Optimization.data:
            material_name = trace.name
            df_material = all_df_combined3[all_df_combined3['Material'] == material_name]
            colors = df_material[f'risk_{tox_param}'].astype(str).map(color_discrete_map).tolist()
            new_colors_per_trace.append(colors)

        buttons.append(dict(
            method='update',
            label=tox_param.replace(" toxicity", "").replace("_", " ").title(),
            args=[{'marker.color': new_colors_per_trace},
                  {'title': f"Optimal Solutions: {scenario} {', '.join(material_names)}<br><br><sup>Colored according to {tox_param}</sup>"}]
        ))

    # Update layout with the dropdown menu
    fig_Optimization.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.0,
            xanchor="right",
            y=1.25,
            yanchor="top"
        )],
        title=f"Optimal_Solutions: {scenario} {', '.join(material_names)}<br><br><sup>Colored according to Most Restrictive Risk</sup>",
        xaxis_title="Total Emission rates (t/y)",
        yaxis_title="Total Concentration levels (ppm)",
        legend_title="Materials"
    )

    new_df = all_df_combined2[['Material', 'WCA','HCA','SFE']]
    df_long = pd.melt(new_df, id_vars=['Material'], value_vars=['WCA', 'HCA', 'SFE'],
                      var_name='Parameter', value_name='Value')
    df_long['RINA'] = df_long['Value'].apply(lambda x: "Not modelled by RINA" if x == -1 else "modelled by RINA")
    
    # Create the properties graph
    fig_properties = px.scatter(df_long,
                                x='Parameter',
                                y='Value',
                                color='Material',
                                title="Values of WCA, HCA and SFE",
                                labels={'Value': 'Value', 'Parameter': 'Properties'},
                                hover_data={'Material': True, 'RINA': True})
    
    # Create new plot for Lung Data
    df_lung_plot_data = all_df_combined3.dropna(subset=lung_columns)

    df_lung_melted = df_lung_plot_data.melt(
        id_vars=['Material'], 
        value_vars=lung_columns, 
        var_name='Lung Region', 
        value_name='Dose (pg)'
    )
    legend_labels = {
    'acuteDoseHA': 'Head Airways (HA)',
    'acuteDoseTB': 'Tracheobronchial Region (TB)',
    'acuteDoseAL': 'Alveolar Region (AL)'
    }   
    df_lung_melted['Lung Region'] = df_lung_melted['Lung Region'].map(legend_labels)
    fig_lung = px.box(
        df_lung_melted,
        x='Material',
        y='Dose (pg)',
        color='Lung Region',
        title='Lung Exposure Distribution per Material',
        labels={'Dose (pg)': 'Acute Dose (pg)'}
    )
    
    optimization_html = fig_Optimization.to_html(full_html=False, include_plotlyjs='cdn')
    properties_html = fig_properties.to_html(full_html=False, include_plotlyjs=False)
    lung_plot_html = fig_lung.to_html(full_html=False, include_plotlyjs=False)

    csv_buffer = StringIO()
    all_df_combined3.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    return {
        "html_content": {
            "optimization_plot": optimization_html,
            "properties_plot": properties_html,
            "lung_plot": lung_plot_html
        },
        "csv_data": csv_content
    }