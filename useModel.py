#!/home/abcsim/miniconda3/bin/python

import os
import numpy as np
import pandas as pd
import cclib
from HessianTools import *
import nablachem.alchemy
import matplotlib.pyplot as plt
from generate_groups import generate_unique_porphyrin_structures as gen
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Import tqdm for the progress bar
import matplotlib.pyplot as plt
import cupy as cp

# Read all the .out and .hess
text_files = [
    f for f in os.listdir(os.getcwd()) 
    if f.endswith(".out")
]

file_path = 'output.csv'

print("Reading all .out and .hess files")

if os.path.exists(file_path):

    print(f"The file '{file_path}' already exists.")

else:
     
    dataframes = []
    # Create the file since it does not exist
    for filename in text_files:
        #load the output
        data = cclib.io.ccread(filename)

        #Create a list of 0 (carbons)
        mol_indexes = np.zeros(20)
    
        #Change the B(5) to -1 and N(7) to 1
        for i in range(20):
            if data.atomnos[i] == 5:
                mol_indexes[i] = -1
            if data.atomnos[i] == 7:
                mol_indexes[i] = 1
    
        #Create a DataFrame for the current file and add to the list
        df = pd.DataFrame([mol_indexes], columns=[f'z{i}' for i in range(20)])

        # Add the SCF Energies column
        scfEnergy = data.scfenergies
        df["Energy"] = scfEnergy

        # Add the HOMO and LUMO columns
        homo_value = data.moenergies[0][data.homos[0]]
        lumo_value = data.moenergies[0][data.homos[0] + 1]
        df["HOMO"] = homo_value
        df["LUMO"] = lumo_value

        # Load the last gradient matrix and flatten it
        grad = data.grads[-1]
        grad_flat = data.grads.flatten()

        # Create the gradient column names
        grad_column = [f"grad[{i}][{j}]" for i in range(data.natom) for j in ['x','y','z']]
        grad_df = pd.DataFrame([grad_flat], columns=grad_column)
        df = pd.concat([df, grad_df], axis=1)    

        # Load the Hessian matrix and flatten it 
        hess = HessianTools(f"{filename[:-4]}.hess")
        hessian_flat = hess.hessian.flatten()
    
        # Create Hessian column names
        hessian_column = [f"hess[{i}][{j}]" for i in range(data.natom*3) for j in range(data.natom*3)]
        hessian_df = pd.DataFrame([hessian_flat], columns=hessian_column)
        df = pd.concat([df, hessian_df], axis=1)    
    
        df.index = [filename]  # Set the filename as the index
        dataframes.append(df)
    
    all_data = pd.concat(dataframes)

    print("Saving dataframe")
    all_data.to_csv("output.csv", index=True)  # Use index=True to include the filename index in the CSV

df = pd.read_csv("output.csv")
df = df.drop(columns=['Unnamed: 0'])
column_tuple = tuple(df.columns)

print("Dataframe loaded")

# Assuming 'Energy', 'HOMO', 'LUMO' are outputs specified when initializing MultiTaylor
mt = nablachem.alchemy.MultiTaylor(df, outputs=column_tuple[20:])

print("Training the model")

# Set the center using ONLY input columns (z0-z19)
mt.reset_center(
    z0=0, z1=0, z2=0, z3=0,
    z4=0, z5=0, z6=0, z7=0,
    z8=0, z9=0, z10=0, z11=0,
    z12=0, z13=0, z14=0, z15=0,
    z16=0, z17=0, z18=0, z19=0
)
mt.build_model(1)

print("Model Trained")

# Generate structures from x to y gen(x,y)
# Accepting input from the user
total_charge = input("Please enter the total charge for 3 to 20 substitutions: ")
# Displaying the input back to the user

print("Generating structures")
structures = gen(int(total_charge))
print(f"Total unique structures generated: {len(structures)}")

# Initialize to aggregate counts and gaps
gap_data = {}
homos = []
lumos = []
energies = []
gaps = []
indices = []

# Your GPU code here
def process_structure_gpu(j, structures):
    # Prepare parameters for the query
    params = {f'z{i}': val for i, val in enumerate(structures[j])}
    result = mt.query(**params)  # Assuming mt.query is GPU-compatible

    # Calculate the HOMO-LUMO gap on GPU
    energy = cp.asarray(result["Energy"])
    homo = cp.asarray(result["HOMO"])
    lumo = cp.asarray(result["LUMO"])
    gaps = cp.asarray(lumos) - cp.asarray(homos)

    return j, energy, homo, lumo, gaps

# Execute processing in parallel on GPU
structures = cp.asarray(structures)  # Ensure structures are on GPU
futures = []

print("Starting data accessing")

for j in tqdm(range(len(structures)), desc="Processing structures"):
    j, energy, homo, lumo, gap = process_structure_gpu(j, structures)
    futures.append((j, energy, homo, lumo, gap))
    energies.append(cp.asnumpy(energy))
    homos.append(cp.asnumpy(homo))
    lumos.append(cp.asnumpy(lumo))
    gaps.append(cp.asnumpy(gap))
    indices.append(j)
    gap_data[j] = cp.asnumpy(gap)


#def process_structure(j):
#    # Prepare parameters for the query
#    params = {f'z{i}': val for i, val in enumerate(structures[j])}
#    result = mt.query(**params)  # Assuming mt.query is defined
#
#    # Calculate the HOMO-LUMO gap
#    energy = result["Energy"]
#    homo = result["HOMO"]
#    lumo = result["LUMO"]
#    gap = lumo - homo
#
#    return j, energy, homo, lumo, gap
#
## Parallel execution with a progress bar
#with ThreadPoolExecutor() as executor:
#    futures = [executor.submit(process_structure_gpu, j) for j in range(len(structures))]
#
#    # Use tqdm to display progress
#    for future in tqdm(futures, desc="Processing structures"):
#        j, energy, homo, lumo, gap = future.result()
#        energies.append(energy)
#        homos.append(homo)
#        lumos.append(lumo)
#        gaps.append(gap)
#        indices.append(j)
#        gap_data[j].append(gap)
## Access the stored gap values later

