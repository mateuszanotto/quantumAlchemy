# src/chem_workflow/data_processor.py

import os
import pandas as pd
import numpy as np
import cclib
from .hessian import *

def create_training_set(results_dir, output_csv_path="data/final_training_set.csv"):
    """
    Parses computational chemistry output files (.out) to create a training set.
    """
    print(f"\n--- Starting Data Processing from '{results_dir}' ---")
    
    try:
        text_files = [f for f in os.listdir(results_dir) if f.endswith(".out")]
        if not text_files:
            print("Warning: No '.out' files found to process.")
            return None
    except FileNotFoundError:
        print(f"Error: Results directory not found at '{results_dir}'")
        return None

    all_dataframes = []
    for filename in text_files:
        # Load the output
        data = cclib.io.ccread(filename)

        # Create a list of 0 (carbons)
        mol_indexes = np.zeros(20)

        #Change the B(5) to -1 and N(7) to 1
        for i in range(20):
            if data.atomnos[i] == 5:
                mol_indexes[i] = -1
            if data.atomnos[i] == 7:
                mol_indexes[i] = 1

        # Create a DataFrame for the current file and add to the list
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


    if not all_dataframes:
        print("No data was processed.")
        return None
        
    final_dataframe = pd.concat(all_dataframes)
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_dataframe.to_csv(output_csv_path, index=True)
    
    print(f"--- Data Processing Complete ---")
    print(f"Training data saved to '{output_csv_path}'")
    return final_dataframe

