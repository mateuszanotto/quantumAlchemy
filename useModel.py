#!/home/abcsim/miniconda3/bin/python

import os
import numpy as np
import pandas as pd
import cclib
from HessianTools import *
import nablachem.alchemy
from generate_groups import generate_unique_porphyrin_structures as gen
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import logging

# Suppress logging for cleaner output
logging.basicConfig(level=logging.WARNING)

# Read and process data
def load_data():
    file_path = 'output.csv'
    if not os.path.exists(file_path):
        text_files = [f for f in os.listdir(os.getcwd()) if f.endswith(".out")]
        dataframes = []
        
        for filename in text_files:
            data = cclib.io.ccread(filename)
            
            # Create molecular indexes
            mol_indexes = np.zeros(20)
            mol_indexes[np.where(data.atomnos == 5)] = -1
            mol_indexes[np.where(data.atomnos == 7)] = 1
            
            # Create DataFrame
            df = pd.DataFrame([mol_indexes], columns=[f'z{i}' for i in range(20)])
            df["Energy"] = data.scfenergies[-1]
            
            # Molecular orbitals
            homo = data.moenergies[0][data.homos[0]]
            lumo = data.moenergies[0][data.homos[0] + 1]
            df["HOMO"] = homo
            df["LUMO"] = lumo
            
            # Gradients and Hessian
            grad = data.grads[-1].flatten()
            hess = HessianTools(f"{filename[:-4]}.hess").hessian.flatten()
            
            grad_cols = [f"grad[{i//3}][{['x','y','z'][i%3]}]" for i in range(len(grad))]
            hess_cols = [f"hess[{i//20}][{i%20}]" for i in range(len(hess))]
            
            df = pd.concat([df, pd.DataFrame([grad], columns=grad_cols), 
                         pd.DataFrame([hess], columns=hess_cols)], axis=1)
            df.index = [filename]
            dataframes.append(df)
        
        pd.concat(dataframes).to_csv(file_path, index=True)
    
    return pd.read_csv(file_path).drop(columns=['Unnamed: 0'])

# Initialize model
def init_model(df):
    mt = nablachem.alchemy.MultiTaylor(df, outputs=df.columns[20:])
    mt.reset_center(**{f'z{i}':0 for i in range(20)})
    mt.build_model(1)
    return mt

# Process single structure
def process_structure(args):
    j, structure, mt = args
    params = {f'z{i}': val for i, val in enumerate(structure)}
    result = mt.query(**params)
    return (
        j,
        result["Energy"],
        result["HOMO"],
        result["LUMO"],
        result["LUMO"] - result["HOMO"]
    )

# Maiin execution
if __name__ == "__main__":
    print("Loading data")
    df = load_data()
    
    proceed = input("output.csv is ready. Proceed with structure generation and processing? (y/n) [default: y]: ").strip().lower()
    if proceed not in ('y', 'yes', ''):
        print("Exiting program - output.csv is available for use")
        exit()

    print("Initializing model")
    mt = init_model(df)
    
    total_charge = input("Enter total charge for substitutions (3-20): ")
    structures = gen(int(total_charge))
    print(f"Generated {len(structures)} structures")
    
    print("Processing structures...")
    results = {'Energy': [], 'HOMO': [], 'LUMO': [], 'Gap': [], 'Index': []}
    
    with ProcessPoolExecutor() as executor:
        # Prepare arguments with model reference (if picklable)
        # If mt isn't picklable, consider using initializer for workers
        args = [(j, struct, mt) for j, struct in enumerate(structures)]
        
        # Use chunksize to reduce overhead
        futures = list(tqdm(
            executor.map(process_structure, args, chunksize=100),
            total=len(structures),
            desc="Processing"
        ))
        
    for j, energy, homo, lumo, gap in futures:
        results['Index'].append(j)
        results['Energy'].append(energy)
        results['HOMO'].append(homo)
        results['LUMO'].append(lumo)
        results['Gap'].append(gap)
    
    print("Processing complete")
    df_results = pd.DataFrame(results)
    
    # Add substitution patterns as columns (z0-z19)
    df_structures = pd.DataFrame(structures, columns=[f'z{i}' for i in range(20)])
    
    # Combine structures with results
    df_final = pd.concat([df_structures, df_results], axis=1)
    
    # Save to CSV with structure count in filename
    output_file = f'out_{len(structures)}.csv'
    df_final.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")
    # ======== END NEW CODE ========# Add further analysis or saving here
