import pandas as pd
import numpy as np
from io import StringIO
import os

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
import plotly.express as px
import plotly.graph_objects as go

from uma_resources.resources import gov_equations as gov_equations
from uma_resources.resources import helpers_UMA as UMA

# This is the final function to reproduce the idea of the Scenario C proposed by IDENER, in which we are optimizing the coating.
# As inputs we need a list of materials that forms the desired coating(It could be whatever the user wants, but only that materials that we could
# find in the API of NoVamechanics)
# Also we have lower bounds and upperbound for the proportions (in mols) of each material
# Also we have a toxicity parameter fix, an scenario and an upperbound of the emissions
# The toxicitiy parameter must be written like 'natural_soil' 
# Moreover, to have information about the Lung exposure we need as input the 'model', 'respiratoryVolumeRate'. 'exposureDuration'
# To understand how the proportions modify the final properties, it's necessary to give a base value of the wca of the initial substrate of the coating.
def coatings_optimization_procedure(materials_list, toxicity_parameter, scenario, l_ranges, u_ranges, upperbound, model, respiratoryVolumeRate, exposureDuration,
                                     wca_surface, hca_surface, sfe_surface, lung_deposition_flags, performance_flags,base_concentration):

    # First we create our list of dictionaries that will contain one dictionary with all the information for each material on the material list
    dicts_material = [m.dict() for m in materials_list]
    material_names = [material.name for material in materials_list]
    
    # List that identifies the relation between the materials of SB4P and the scenarios of the Lung model.
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
        'Octyltrimethoxysilane': 'Silane_based_Motzkus_et_al_2011',
        'Water (H2O)' : 'H2O_Ito_et_al_2021',
        'Diethoxydimethyl silane (DEDMS)' : 'Silane_based_Motzkus_et_al_2011',
        'Trimethoxymethylsilane (TMMS)' : 'Silane_based_Motzkus_et_al_2011',
        'n-Propyltriethoxysilane (C3)' : 'Silane_based_Motzkus_et_al_2011',
        'n-Hexiltrietoxisilane (C6)' : 'Silane_based_Motzkus_et_al_2011',
        'Ethanol (EtOH)': "Ethanol_Dahm_et_al_2019" 
    }
    
    lung_deposition_factors = {}
    for material_name in material_names:
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
        
    # Governing equations proportionate by IDENER
    def full_gov_eqx_reponse(material_list: list, mols_list: list, emissions_list: list, scenario: str,
                            lung_model:str, lung_respiratory_volume_rate: str,lung_exposure_duration:int,
                            wca_surface:float, hca_surface: float, sfe_surface: float, dicts_materials: list,
                            lung_deposition_flags: dict, performance_flags: dict):
        
        if len(material_list) != len(mols_list) or len(material_list) != len(emissions_list):
            raise ValueError("The lists must have the same length")
        else:
            concentrations_df=pd.DataFrame()
            properties_df = pd.DataFrame()
            toxicities_df = pd.DataFrame()
            lung_model_df = pd.DataFrame()
            total_weight = sum(mols_list)
            normalized_weights = [w / total_weight for w in mols_list]
            for i in range(len(material_list)):
                material = material_list[i]
                dict_material = dicts_materials[i]
                concentration_dict, property_dict, toxicity_dict = gov_equations.governing_equation_response(
                    material=material, scenario=scenario, emissions= np.array(emissions_list[i])*normalized_weights[i], response_json=dict_material)
                
                concentration = concentration_dict["AEROSOL- and CLOUD PHASES"] * 1e12
                if lung_deposition_flags.get(material, False):
                    lung_result = {
                        'acuteDoseHA': (lung_deposition_factors[material]['acuteDoseHA_factor']) * concentration,
                        'acuteDoseTB': (lung_deposition_factors[material]['acuteDoseTB_factor']) * concentration,
                        'acuteDoseAL': (lung_deposition_factors[material]['acuteDoseAL_factor']) * concentration
                    }
                else:
                    lung_result = {
                        'acuteDoseHA': np.nan,
                        'acuteDoseTB': np.nan,
                        'acuteDoseAL': np.nan
                    }
                
                concentrations_df = pd.concat([concentrations_df,pd.DataFrame([concentration_dict])])
                properties_df = pd.concat([properties_df,pd.DataFrame([property_dict])])
                toxicities_df = pd.concat([toxicities_df,pd.DataFrame([toxicity_dict])])
                lung_model_df = pd.concat([lung_model_df,pd.DataFrame([lung_result])])

            lung_model_df["compound"] = material_list
            diffusion_coating = concentrations_df.sum(axis=0).drop(["AEROSOL- and CLOUD PHASES"])
            emission_coating = diffusion_coating.to_dict()
            normalized_weights = [w if w >= 0.3 else 0 for w in normalized_weights]
            WCA_coating = wca_surface + sum(weight*(wca - wca_surface) for weight, wca in zip(normalized_weights, properties_df["WCA"]))
            HCA_coating = hca_surface + sum(weight*(hca - hca_surface) for weight, hca in zip(normalized_weights, properties_df["HCA"]))
            SFE_coating = sfe_surface + sum(weight*(sfe - sfe_surface) for weight, sfe in zip(normalized_weights, properties_df["SFE"]))
            return emission_coating, WCA_coating, HCA_coating, SFE_coating, concentrations_df, properties_df, toxicities_df, lung_model_df
    
    # Function that determines the final color of each solution, depending on the toxicity color associated to each compound of the coating
    def determine_color(row,number_materials):
        colors = {row[f'col_{i}'] for i in range(number_materials)}
        if 'red' in colors:
            return 'red'
        elif 'yellow' in colors:
            return 'yellow'
        elif 'blue' in colors:
            return 'blue'
        elif 'lightgreen' in colors:
            return 'lightgreen'
        else:
            return 'darkgreen'

    # Define the Callback to stop when the number of solutions converge
    class ConvergenceCallback(Callback):
        def __init__(self, patience=10):
            super().__init__()
            self.patience = patience
            self.no_change_count = 0
            self.last_n_nds = None

        def notify(self, algorithm):
            n_nds = len(algorithm.opt)
            if self.last_n_nds == n_nds:
                self.no_change_count += 1
            else:
                self.no_change_count = 0
            self.last_n_nds = n_nds
            if self.no_change_count >= self.patience:
                print(f"Convergence reached: n_nds = {n_nds} remained stable for {self.patience} generations.")
                algorithm.termination.force_termination = True
    
    # First we define the optimization problem
    class Optimizer(Problem):
        def __init__(self, materials_list, l_ranges, u_ranges, upperbound, scenario, model, respiratoryVolumeRate, exposureDuration, wca_surface, hca_surface, sfe_surface, dict_material,lung_deposition_flags, performance_flags):
            # First we define the bounds that interest us
            # Note that it's supposed that for all materials the upperbound for the emissions rates will be the same.
            # Two possibilities, give a list of upperbounds that will be the upperbound for each one of the materials
            # Give only a number, in this case that number is the upperboun for all the materials
            lower_bounds = np.concatenate([np.zeros(7 * len(materials_list)), l_ranges])
            if isinstance(upperbound, (int, float)):
                upperbounds = np.concatenate([np.array([upperbound] * (7 * len(materials_list))), u_ranges])
            else:
                upperbounds = np.concatenate([np.tile(upperbound, len(materials_list)), u_ranges])
            
            # We define now the problem and the number of objectives
            super().__init__(n_var=7 * len(materials_list) + len(materials_list), n_obj=2, xl=lower_bounds, xu=upperbounds)
            
            # We save on self the parameters that we need for the optimization.
            self.material = materials_list
            self.scenario = scenario
            self.model = model
            self.respiratoryVolumeRate = respiratoryVolumeRate
            self.exposureDuration = exposureDuration
            self.wca_surface = wca_surface
            self.hca_surface = hca_surface
            self.sfe_surface = sfe_surface
            self.dict_material = dict_material
            self.lung_deposition_flags = lung_deposition_flags
            self.performance_flags = performance_flags
        
        def _evaluate(self, x, out, *args, **kwargs):
            api_values = []
            emissions_values = []
            # We iterate on each possible solution of our problem
            for emissions in x:
                emissions_rates = emissions[:(7 * len(self.material))]
                emissions_rates = [emissions_rates[i:i + 7] for i in range(0, len(emissions_rates), 7)]
                emissions_rates = [arr.tolist() for arr in emissions_rates]
                emissions_values.append(-np.sum(emissions_rates))
                
                proportions = emissions[-len(self.material):]
                
                # We call SB4P to obtain the data to compare the solutions
                emissions_compound, _, _, _, _, _, _, _ = full_gov_eqx_reponse(
                    self.material, proportions, emissions_rates,
                    self.scenario, self.model, self.respiratoryVolumeRate, self.exposureDuration, self.wca_surface,self.hca_surface, self.sfe_surface, self.dict_material,
                    self.lung_deposition_flags, self.performance_flags
                )
                
                if emissions_compound is not None:
                    api_values.append(emissions_compound)
                else:
                    raise ValueError("API result is None")
            
            # Convert the values into a Matrix to compute the final objective function.
            api_values_matrix = np.array([list(d.values()) for d in api_values])
            
            # First objective: maximize the sum of all the emission rates (the production), we have assumed that the production of the coating is the sum of the production of all the
            # materials that form it.
            f1 = np.array(emissions_values)
            
            # Now we compute the second objective that is to minimize the concentration levels(toxicity) of the coating
            modified_values = np.copy(api_values_matrix)
            modified_values[:, :7] *= 1000
            modified_values[:, 8:] *= 1000
            f2 = np.sum(modified_values, axis=1)
            
            # Then we save that two objective because we need both to compare the solutions during the optimization
            out["F"] = np.column_stack([f1, f2])
    
    # We first create an instance of the problem
    problem = Optimizer(material_names, l_ranges, u_ranges, upperbound, scenario, model, respiratoryVolumeRate, exposureDuration, wca_surface,hca_surface, sfe_surface, dicts_material,lung_deposition_flags, performance_flags)
    # We configurate the NSGA2 algorithm
    # In that part you could modify the pop_size regarding to the number of final solutions you want to obtain in the last Front
    algorithm = NSGA2(pop_size=100)
    #We code the convergence criteria
    callback = ConvergenceCallback(patience=10)
    # In that point we decided the number of generations needed for the algorithm to end up on the Pareto Front
    # You could modify the n_gen number if you want, but you will not be sure if it end up to the final front at this number of generations.
    Solutions = minimize(problem,
                         algorithm,
                         ('n_gen', 80),
                         seed=1,
                         save_history=True,
                         verbose=True,
                         callback=callback)

    print(f"Non-dominated solutions for coating {material_names}:")
    print(Solutions.X)
    print(f"Objective values for these solutions for coating {material_names}:")
    print(Solutions.F)

    df_optimal_function_value = pd.DataFrame(Solutions.F, columns=[f'f{i+1}' for i in range(Solutions.F.shape[1])])
    df_optimal_emission_rates = pd.DataFrame(Solutions.X, columns=[f'x{i+1}' for i in range(Solutions.X.shape[1])])
    df_optimal_function_value['f1'] *= -1

    # We recall our full_gov function again with the optimal_emission_rates to obtain the additional information that we need for each material
    #We rename de columns of the optimal emission rates for each material, in order to identify the proportions obtained for each non-dominate solution, and also the emission rates needed
    new_columns, new_proportions = [], []
    for material in material_names:
        for i in range(7):
            new_columns.append(f'{material}_x{i+1}')
        new_proportions.append(f'{material}')
    new_names = new_columns + new_proportions
    df_optimal_emission_rates.columns = new_names

    # We initialize the toxicity and WCA dataframes to save the values obtained in this two outputs of the governing equation
    df_toxicity = pd.DataFrame()
    df_lung = pd.DataFrame()
    df_wca = pd.DataFrame(columns=['WCA_coating'])
    df_hca = pd.DataFrame(columns=['HCA_coating'])
    df_sfe = pd.DataFrame(columns=['SFE_coating'])

    for index, row in df_optimal_emission_rates.iterrows():
        emissions_list = [row[7*i:7*(i+1)].to_list() for i in range(len(material_names))]
        emission_coating, wca_coating, HCA_coating_val, SFE_coating_val, concentrations_df, properties_df, toxicities_df, lung_model_df = full_gov_eqx_reponse(
            material_names, row[-len(material_names):], emissions_list, scenario, model, respiratoryVolumeRate, exposureDuration, wca_surface, hca_surface, sfe_surface, dicts_material,
            lung_deposition_flags, performance_flags
        )
        
        df_wca.loc[index] = wca_coating
        df_hca.loc[index] = HCA_coating_val
        df_sfe.loc[index] = SFE_coating_val
        
        all_toxicity_keys = toxicities_df.columns.tolist()
        values = toxicities_df.values.flatten()
        column_names = [f'{col}_{material_names[i]}' for i in range(len(material_names)) for col in all_toxicity_keys]
        df_toxicity_row = pd.DataFrame([values], columns=column_names)
        df_toxicity = pd.concat([df_toxicity, df_toxicity_row], axis=0, ignore_index=True)
        
        lung_model_df_Important = lung_model_df.iloc[:,:3]
        values_lung = lung_model_df_Important.values.flatten()
        column_names_lung = [f'{col}_{material_names[i]}' for i in range(len(material_names)) for col in lung_model_df_Important.columns]
        df_lung_row = pd.DataFrame([values_lung], columns=column_names_lung)
        df_lung = pd.concat([df_lung, df_lung_row], axis=0, ignore_index=True)

    # We initialize our final Dataframe and start to concatenate all the information obtained before
    df_combined = pd.DataFrame()
    df_combined = pd.concat([
        df_optimal_function_value,
        df_optimal_emission_rates,
        df_toxicity,
        df_lung,
        df_wca,
        df_hca,
        df_sfe
    ], axis=1)
    lung_cols_to_drop = []
    for material in material_names:
        if not lung_deposition_flags.get(material, False):
            lung_cols_to_drop.extend([f'acuteDoseHA_{material}', f'acuteDoseTB_{material}', f'acuteDoseAL_{material}'])
    
    existing_cols_to_drop = [col for col in lung_cols_to_drop if col in df_combined.columns]
    df_combined.drop(columns=existing_cols_to_drop, inplace=True)

    all_toxicity_parameters = all_toxicity_keys
    
    for tox_param in all_toxicity_parameters:
        temp_color_df = pd.DataFrame()
        for i, material in enumerate(material_names):
            color_col_name = f'color_{tox_param}_{material}'
            tox_col_name = f'{tox_param}_{material}'
            df_combined[color_col_name] = df_combined[tox_col_name].apply(UMA.get_color)
            temp_color_df[f'col_{i}'] = df_combined[color_col_name]
        
        final_color_col = f'Final_Color_{tox_param}'
        df_combined[final_color_col] = temp_color_df.apply(lambda row: determine_color(row, len(material_names)), axis=1)
        
        risk_col = f'RISK_{tox_param}'
        df_combined[risk_col] = df_combined[final_color_col].apply(UMA.get_risk_color)

    color_order = {'darkgreen': 0, 'lightgreen': 1, 'blue': 2, 'yellow': 3, 'red': 4}
    inv_color_order = {v: k for k, v in color_order.items()}
    final_color_cols = [f'Final_Color_{tox_param}' for tox_param in all_toxicity_parameters]
    
    df_combined['Final_Color_All'] = df_combined[final_color_cols].apply(lambda row: row.map(color_order)).max(axis=1).map(inv_color_order)
    df_combined['RISK_All'] = df_combined['Final_Color_All'].apply(UMA.get_risk_color)

    df_combined.rename(columns={
        'f1': 'Total Emissions rates (t/y)',
        'f2' : 'Total Concentration levels (ppm)'
    }, inplace=True)

    maximum_wca, minimum_wca = df_combined['WCA_coating'].max(), df_combined['WCA_coating'].min()
    df_combined['normalization_wca'] = (df_combined['WCA_coating'] - minimum_wca) / (maximum_wca - minimum_wca) if (maximum_wca - minimum_wca) != 0 else 0
    maximum_hca, minimum_hca = df_combined['HCA_coating'].max(), df_combined['HCA_coating'].min()
    df_combined['normalization_hca'] = (df_combined['HCA_coating'] - minimum_hca) / (maximum_hca - minimum_hca) if (maximum_hca - minimum_hca) != 0 else 0
    maximum_sfe, minimum_sfe = df_combined['SFE_coating'].max(), df_combined['SFE_coating'].min()
    df_combined['normalization_sfe'] = (df_combined['SFE_coating'] - minimum_sfe) / (maximum_sfe - minimum_sfe) if (maximum_sfe - minimum_sfe) != 0 else 0

    color_discrete_map = {
        'NO RISK': 'darkgreen',
        'LOW RISK': 'lightgreen',
        'MEDIUM RISK': 'yellow',
        'HIGH RISK': 'red',
        'NO IDENTIFIED RISK': 'blue'
    }

    fig_Optimization = go.Figure()

    fig_Optimization.add_trace(go.Scatter(
        x=df_combined['Total Emissions rates (t/y)'],
        y=df_combined['Total Concentration levels (ppm)'],
        mode='markers',
        marker=dict(
           
            color=df_combined['RISK_All'].map(color_discrete_map),
            size=10
        ),
        customdata=df_combined[['WCA_coating', 'HCA_coating', 'SFE_coating']],
        hovertemplate=(
            "<b>Objective functions:</b><br>"
            "Total Emissions rates (t/y): %{x:.5e}<br>"
            "Total Concentration levels (ppm): %{y:.5e}<br><br>"
            "<b>Coating Properties:</b><br>"
            "WCA: %{customdata[0]:.2f}<br>"
            "HCA: %{customdata[1]:.2f}<br>"
            "SFE: %{customdata[2]:.2f}<br>"
            "<extra></extra>"
        )
    ))

    buttons = []
    
    buttons.append(dict(
        method='update',
        label='All (Most Restrictive)',
        args=[{'marker.color': [df_combined['RISK_All'].map(color_discrete_map)]},
              {'title.text': f"Optimal Solutions: {scenario} {', '.join(material_names)}<br><br><sup>Colored according to Most Restrictive Risk</sup>"}]
    ))

    for tox_param in all_toxicity_parameters:
        buttons.append(dict(
            method='update',
            label=tox_param.replace("_", " ").title(),
            args=[{'marker.color': [df_combined[f'RISK_{tox_param}'].map(color_discrete_map)]},
                  {'title.text': f"Optimal Solutions: {scenario} {', '.join(material_names)}<br><br><sup>Colored according to {tox_param.replace('_', ' ')}</sup>"}]
        ))

    fig_Optimization.update_layout(
        title=f"Optimal Solutions: {scenario} {', '.join(material_names)}<br><br><sup>Colored according to Most Restrictive Risk</sup>",
        xaxis_title="Total Emission rates (t/y)",
        yaxis_title="Total Concentration levels (ppm)",
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
        )]
    )
    
    # Now we have all the information needed so we procced to save and show the results obtained
    df_combined_order = df_combined.sort_values(by='Total Emissions rates (t/y)', ascending=False)
    df_combined_order2 = df_combined.sort_values(by='Total Concentration levels (ppm)', ascending=False)

    df_proportions_production_volume = df_combined_order[new_proportions]
    df_proportions_concentration = df_combined_order2[new_proportions]

    df_proportions_production_volume['Total Emission Rates (t/y)'] = df_combined_order['Total Emissions rates (t/y)'].round(5)
    df_melted = df_proportions_production_volume.melt(id_vars='Total Emission Rates (t/y)',var_name='Material', value_name='Proportion')

    df_proportions_concentration['Total Concentration levels (ppm)'] = df_combined_order2['Total Concentration levels (ppm)']
    df_proportions_concentration['Total Concentration levels (ppm)'] = df_proportions_concentration['Total Concentration levels (ppm)'].apply(lambda x: '{:.4e}'.format(x))
    df_melted2 = df_proportions_concentration.melt(id_vars='Total Concentration levels (ppm)',var_name='Material', value_name='Proportion')

    # We generate the second and third plot
    figure_proportions_production_volume = px.strip(df_melted, x='Material', y='Proportion', color='Total Emission Rates (t/y)', stripmode='overlay',
                                                    title=f"Proportions in descending order by Total Emission rates: {', '.join(material_names)}")
    
    figure_proportions_concentration = px.strip(df_melted2, x='Material', y='Proportion', color='Total Concentration levels (ppm)', stripmode='overlay',
                                                 title=f"Proportions in descending order by Total Concentration level: {', '.join(material_names)}")
    
    optimization_html = fig_Optimization.to_html(full_html=False, include_plotlyjs='cdn')
    proportions_html = figure_proportions_production_volume.to_html(full_html=False, include_plotlyjs=False)
    concentration_html = figure_proportions_concentration.to_html(full_html=False, include_plotlyjs=False)

    # Create new plot for Lung Data
    lung_cols = [col for col in df_combined.columns if col.startswith('acuteDose')]
    df_lung_melted = df_combined.melt(value_vars=lung_cols, var_name='Variable', value_name='Dose (pg)')
    if not df_lung_melted.empty:
        df_lung_melted[['Lung Region', 'Material']] = df_lung_melted['Variable'].str.rsplit('_', n=1, expand=True)
        legend_labels = {
        'acuteDoseHA': 'Head Airways (HA)',
        'acuteDoseTB': 'Tracheobronchial Region (TB)',
        'acuteDoseAL': 'Alveolar Region (AL)'
        }
        df_lung_melted['Lung Region'] = df_lung_melted['Lung Region'].map(legend_labels)
    lung_cols = [col for col in df_combined.columns if col.startswith('acuteDose')]
    lung_plot_html = ""

    if lung_cols: 
        df_lung_melted = df_combined.melt(value_vars=lung_cols, var_name='Variable', value_name='Dose (pg)')
        if not df_lung_melted.empty:
            df_lung_melted[['Lung Region', 'Material']] = df_lung_melted['Variable'].str.rsplit('_', n=1, expand=True)
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
                title='Lung Exposure Distribution per Coating Component',
                labels={'Dose (pg)': 'Acute Dose (pg)'}
            )
            lung_plot_html = fig_lung.to_html(full_html=False, include_plotlyjs=False)

    csv_buffer = StringIO()
    df_combined.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    return {
        "html_content": {
            "optimization_plot": optimization_html,
            "proportions_plot": proportions_html,
            "concentration_plot": concentration_html,
            "lung_plot": lung_plot_html
        },
        "csv_data": csv_content
    }