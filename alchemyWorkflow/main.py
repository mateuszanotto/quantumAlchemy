# src/main.py

import os
import sys
import numpy as np
from ase import Atoms
from ase.io import write

# Import from other modules within the same package
# Add the src directory to the Python path to allow for package imports
# This is a common pattern for top-level scripts.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the chem_workflow package
# Thanks to our __init__.py, we can import these directly.
from src import GeometryGenerator, create_training_set

def main():
    """Main function to orchestrate the entire workflow."""
    print("==============================")
    print("=== ALCHEMY WORKFLOW START ===")
    print("==============================")

    # --- STEP 1: Generate Geometries ---
    os.makedirs("data", exist_ok=True)
    reference_file = "data/porph.xyz" ## Define ref
    
    geom_gen = GeometryGenerator(output_dir="data/raw_geometries")
    geom_gen.run(mol_reference=reference_file, atom_type='C', charge=0)

    # --- STEP 2: Manual Calculation Step ---
    print("\n----------------- MANUAL STEP -----------------")
    print("Run your quantum chemistry calculations on the .xyz files")
    print("in 'data/raw_geometries/' to produce the '.out' files.")
    print("-----------------------------------------------")
    
    results_dir = "data/calc_results"
    os.makedirs(results_dir, exist_ok=True)

    # --- STEP 3: Process Calculation Results ---
    create_training_set(results_dir=results_dir, output_csv_path="data/final_training_set.csv")

    print("\n===========================================")
    print("=== WORKFLOW COMPLETED SUCCESSFULLY ===")
    print("===========================================")

if __name__ == "__main__":
    main()
