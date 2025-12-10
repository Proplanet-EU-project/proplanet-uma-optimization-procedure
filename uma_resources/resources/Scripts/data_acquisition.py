import requests
import pandas as pd
import itertools
import numpy as np
import itertools
import random
import helpers
import gov_equations

headers = {
"accept": "application/json",
'Content-Type': 'application/json'
    }

# Get the list of all materials that can be used to call the Nova API
material_dict = helpers.get_materials_list()
keys_to_delete = ["Zinc oxide (ZnO)","Octyltrimethoxysilane"]
for key in keys_to_delete:
    del material_dict[key]

# Generate the data and save it in a .csv file (test.csv)
# helpers.generate_data(material_dict)
