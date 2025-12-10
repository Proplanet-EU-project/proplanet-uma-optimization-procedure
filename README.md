# Optimization Procedure

Here you will find the final software delivery (**Deliverable D5.5**), structured into **4 interconnected repositories** that compose the complete solution:

* **[Replication Tool Core](https://github.com/Proplanet-EU-project/proplanet-replication-tool):** The main web application.
* **[SimpleBox4Planet](https://github.com/NovaMechanicsOpenSource/SimpleBox4Planet):** Environmental fate module.
* **[Lung Deposition](https://github.com/NovaMechanicsOpenSource/lungdeposition):** Exposure assessment module.
* **Optimization Engine:** Optimization algorithms (UMA).

Code for optimizing from UMA side

## Setup

This project uses a virtual environment named "myenv". Follow these steps to set up and run the project:

### Prerequisites

- Python 3.x (specify the version you're using)
- pip (usually comes with Python)

### Installation

1. Download or copy the project files to your local machine.

2. Open a terminal/command prompt and navigate to the project directory.

3. Create the virtual environment:
```
python -m venv myenv
```

4. Activate the virtual environment:
- On Windows:
  ```
  myenv\Scripts\activate
  ```
- On macOS and Linux:
  ```
  source myenv/bin/activate
  ```

5. Install the required packages:
```
pip install -r requirements.txt
```
### Usage

Code for Materials

inputs: 
- list_of_substances
- environmental_media
- scenario 
- bound_of_emission_rates
- output_folder
- model
- scenario_lung
- respiratoryVolumeRate
- exposureDuration.

outputs: 

- Two graphics that shows the properties of the materials considered and the optimal solutions of the problem colored according to the toxicity on the environmental media choosen respectively.
- A .csv file with the detailed information obtained after solving the optimisation problem.

Code for Coatings

inputs:
- list_of_substances
- environmental_media
- scenario
- low_mols_list
- upp_mols_list
- upperbound
- output_folder
- lung_model
- lung_respiratory_volume_rate
- lung_exposure_duration
- wca_surface

outputs:

- One graphic showing the optimal solutions colored according to the toxicity on the environmentla media choosen.
- Two more graphics with the proportions of each material involved on the coating ordered by the toxicity and the volume production respectively
- A .csv file with the detailed information obtained after solving the optimisation problem.

## Release Procedure
To deploy the microservice in the production environment, follow these steps:
1. Confirm that all modifications have been merged into the main branch.
2. Go to the "Releases" section and click on "Draft a new release".
3. In the "Choose a tag" interface, generate a new tag conforming to the semantic versioning convention (vMAJOR.MINOR.PATCH).
4. Publish the release and await the completion of the CI/CD process.
