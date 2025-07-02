# This is a complete, self-contained example. 
# You can run this file directly.
# In a real project, you would split these classes and functions into separate files.

import os
import pandas as pd
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.data import atomic_numbers, chemical_symbols

# ==============================================================================
# 1. GEOMETRY GENERATION MODULE (was Generate_geometry.py)
# ==============================================================================
# We'll place this in a file like: src/my_project/geometry_generator.py

class GeometryGenerator:
    """
    Generates a set of molecular geometries by substituting atoms
    in a reference molecule.
    """
    def __init__(self, output_dir="data/raw"):
        """
        Initializes the generator.

        Args:
            output_dir (str): The directory where generated .xyz files will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory for geometries: '{self.output_dir}'")

    def _get_indexes(self, mol, atom_type):
        """Finds the indexes of a specific atom type in the molecule."""
        if isinstance(atom_type, str):
            atom_symbol = atom_type.upper()
        else:
            atom_symbol = chemical_symbols[atom_type]
        
        return np.where(mol.get_chemical_symbols() == atom_symbol)[0]

    def run(mol, atom_index, charge, mol_reference):
        #count how many species are being generated
        n_species = 0
    
        try:
            dir_path='data/raw'
            os.makedirs(f'{dir_path}', exist_ok=True)
            print('files created on {dir_path}')
        except OSError as e:
            print(f'Error creating directory: {e}')
    
        # Single substitution
        for i in range(len(atom_index)):
            mol[atom_index[i]].number+=1 # substitute for the next element (C->N)
            charge+=1
            symbol_i = chemical_symbols[mol[atom_index[i]].number]
            write(f"{mol_reference}_{symbol_i}{i}_c{charge}.xyz", mol, format='xyz')
            n_species+=1
            mol[atom_index[i]].number-=1 # go back to original (B->C)
            charge-=1
    
            mol[atom_index[i]].number-=1 # substitute for the previous element (N->B)
            charge-=1
            symbol_i = chemical_symbols[mol[atom_index[i]].number]
            write(f'{mol_reference}_{symbol_i}{i}_c{charge}.xyz', mol, format='xyz')
            n_species+=1
            mol[atom_index[i]].number+=1 # go back to original (B->C)
            charge+=1
    
        # Double substitution
        for i in range(len(atom_index)):
            for j in range(i+1):
                if i==j:
                    None
                else:
                    # up up
                    mol[atom_index[i]].number+=1 # substitute for the next element (C->N)
                    charge+=1
                    mol[atom_index[j]].number+=1 # substitute for the next element (C->N)
                    charge+=1
                    symbol_i = chemical_symbols[mol[atom_index[i]].number]
                    symbol_j = chemical_symbols[mol[atom_index[j]].number]
                    write(f'{mol_reference}_{symbol_i}{i}_{symbol_j}{j}_c{charge}.xyz', mol, format='xyz')
                    n_species+=1
                    mol[atom_index[i]].number-=1 # original
                    mol[atom_index[j]].number-=1 # original
                    charge-=2
                   
                    # down down
                    mol[atom_index[i]].number-=1 # substitute for the previous element (C->B)
                    charge-=1
                    mol[atom_index[j]].number-=1 # substitute for the previous element (C->B)
                    charge-=1
                    symbol_i = chemical_symbols[mol[atom_index[i]].number]
                    symbol_j = chemical_symbols[mol[atom_index[j]].number]
                    write(f'{mol_reference}_{symbol_i}{i}_{symbol_j}{j}_c{charge}.xyz', mol, format='xyz')
                    n_species+=1
                    mol[atom_index[i]].number+=1 # original
                    mol[atom_index[j]].number+=1 # original
                    charge+=2
    
                    # up down
                    mol[atom_index[i]].number+=1 # substitute for the next element (C->N)
                    charge+=1
                    mol[atom_index[j]].number-=1 # substitute for the previous element (C->B)
                    charge-=1
                    symbol_i = chemical_symbols[mol[atom_index[i]].number]
                    symbol_j = chemical_symbols[mol[atom_index[j]].number]
                    write(f'{mol_reference}_{symbol_i}{i}_{symbol_j}{j}_c{charge}.xyz', mol, format='xyz')
                    n_species+=1
                    mol[atom_index[i]].number-=1 # substitute for the next element (C->N)
                    charge-=1
                    mol[atom_index[j]].number+=1 # substitute for the next element (C->N)
                    charge+=1
                    
                    # down up
                    mol[atom_index[i]].number-=1 # substitute for the next element (C->N)
                    charge-=1
                    mol[atom_index[j]].number+=1 # substitute for the next element (C->N)
                    charge+=1
                    symbol_i = chemical_symbols[mol[atom_index[i]].number]
                    symbol_j = chemical_symbols[mol[atom_index[j]].number]
                    write(f'{mol_reference}_{symbol_j}{j}_{symbol_i}{i}_c{charge}.xyz', mol, format='xyz')
                    n_species+=1
                    mol[atom_index[i]].number+=1 # substitute for the next element (C->N)
                    charge+=1
                    mol[atom_index[j]].number-=1 # substitute for the next element (C->N)
                    charge-=1
        print(f'files generates: {n_species} + reference')
                       
