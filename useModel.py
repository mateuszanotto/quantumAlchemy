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
import matplotlib.pyplot as plt

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
            

            hess = HessianTools(f"{filename[:-4]}.hess").hessian.flatten()
            
            grad_cols = [f"grad[{i//3}][{['x','y','z'][i%3]}]" for i in range(len(grad))]
            hess_cols = [f"hess[{i//20}][{i%20}]" for i in range(len(hess))]
            
            df = pd.concat([df, pd.DataFrame([grad], columns=grad_cols), 
                         pd.DataFrame([hess], columns=hess_cols)], axis=1)
            df.index = [filename]
            dataframes.append(df)
            
            # Select the first 20 columns of the DataFrame
            subset = df.iloc[:, :20]

            # Calculate atomB as the count of 1s in each row
            atomB = (subset == 1).sum(axis=1).to_numpy()
            # Calculate atomN as the negative count of -1s in each row
            atomN = (-(subset == -1).sum(axis=1)).to_numpy()
            df['atomB'] = atomB
            df['atomN'] = atomN       
            df['Gap'] = df['LUMO']-df["HOMO"]

        pd.concat(dataframes).to_csv(file_path, index=True)
    
    return pd.read_csv(file_path).drop(columns=['Unnamed: 0'])

# Initialize model
def init_model(df):
    mt = nablachem.alchemy.MultiTaylor(df, outputs=df.columns[20:])
    mt.reset_center(**{f'z{i}':0 for i in range(20)})
    mt.build_model(1)
    return mt

def graphs(df,mt):
    file_path = 'output.csv'
    modelE=[]
    trueE=[]
    modelH=[]
    trueH=[]
    modelL=[]
    trueL=[]
    for n in range(len(df)):
        result = mt.query(
            z0=df['z0'][n], z1=df['z1'][n], z2=df['z2'][n], z3=df['z3'][n],
            z4=df['z4'][n], z5=df['z5'][n], z6=df['z6'][n], z7=df['z7'][n],
            z8=df['z8'][n], z9=df['z9'][n], z10=df['z10'][n], z11=df['z11'][n],
            z12=df['z12'][n], z13=df['z13'][n], z14=df['z14'][n], z15=df['z15'][n],
            z16=df['z16'][n], z17=df['z17'][n], z18=df['z18'][n], z19=df['z19'][n]
        )
        modelE.append(result["Energy"])  # Remove [n] here
        trueE.append(df['Energy'][n])    # Directly access the nth row's 'Energy' value  
        modelH.append(result["HOMO"])  # Remove [n] here
        trueH.append(df['HOMO'][n])    # Directly access the nth row's 'Energy' value  
        modelL.append(result["LUMO"])  # Remove [n] here
        trueL.append(df['LUMO'][n])    # Directly access the nth row's 'Energy' value  
    for n in range(len(modelE)):
        if modelE[n]-trueE[n] < -100:
            df2=pd.read_csv(file_path)
            print("CAREFUL! This is an outlier:")
            print(df2['Unnamed: 0'][n])
    
    plt.scatter(modelE, trueE)
    plt.ylabel('Model Energy')
    plt.xlabel('PBE/def2-TZVPP DFT Energy')
    plt.plot(trueE, trueE)
    plt.savefig('ModelEnergy.png', dpi=300)  


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

def save_expansion(results):
    df_results = pd.DataFrame(results)
    
    # Add substitution patterns as columns (z0-z19)
    df_structures = pd.DataFrame(structures, columns=[f'z{i}' for i in range(20)]) 

    # Combine structures with results
    df_final = pd.concat([df_structures, df_results], axis=1)
    
    # Select the first 20 columns of the DataFrame
    subset = df.iloc[:, :20]
    # Calculate atomB as the count of 1s in each row
    atomB = (subset == 1).sum(axis=1).to_numpy()
    # Calculate atomN as the negative count of -1s in each row
    atomN = (-(subset == -1).sum(axis=1)).to_numpy()
    
    df_final['atomB'] = atomB
    df_final['atomN'] = atomN
    df_final['Gap'] = df_final['LUMO']-df_final["HOMO"]

    # Save to CSV with structure count in filename
    output_file = f'out_{len(structures)}.csv'
    df_final.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")
    # ======== END NEW CODE ========# Add further analysis or saving here

# Main execution
if __name__ == "__main__":
    print("Loading data")
    df = load_data()

    print("Initializing model")
    mt = init_model(df)
   
    gen_graphs = input("Generate energy graph with model? (y/n) [default: n]? ").strip().lower()
    if gen_graphs not in ('n','no',''):
        print("Generating energy graph")
        graphs = graphs(df,mt)
    

    proceed = input("output.csv is ready. Proceed with structure generation and processing? (y/n) [default: y]: ").strip().lower()
    if proceed not in ('y', 'yes', ''):
        print("Exiting program - output.csv is available for use")
        exit()
    
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
        
    save = save_expansion(results)
