# src/chem_workflow/geometry_generator.py

import os
import itertools
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.data import chemical_symbols

class GeometryGenerator:
    """
    Generates molecular geometries by substituting atoms in a reference molecule.
    Handles single and double substitutions.
    """
    def __init__(self, output_dir="data/raw_geometries"):
        """Initializes the generator and creates the output directory."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory for geometries set to: '{self.output_dir}'")

    def _get_atom_indices(self, mol, atom_type):
        """Finds the indices of a specific atom type in the molecule."""
        target_symbol = chemical_symbols[atom_type] if isinstance(atom_type, int) else atom_type.capitalize()
        return [i for i, symbol in enumerate(mol.get_chemical_symbols()) if symbol == target_symbol]

    def _write_molecule(self, mol, base_name, substitutions, charge):
        """Helper function to create a filename and write the molecule."""
        sub_str = "_".join([f"{chem_sym}{idx}" for idx, chem_sym in substitutions])
        filename = os.path.join(self.output_dir, f"{base_name}_{sub_str}_c{charge}.xyz")
        write(filename, mol)

    def run(self, mol_reference, atom_type, charge=0):
        """
        Generates and saves all substituted molecular geometries.

        Args:
            mol_reference (str): Path to the reference .xyz file.
            atom_type (str or int): The symbol ('C') or atomic number (6) to substitute.
            initial_charge (int): The charge of the reference molecule.
        """
        print(f"\n--- Starting Geometry Generation from '{mol_reference}' ---")
        try:
            mol_original = read(mol_reference)
            mol_ref_name = os.path.splitext(os.path.basename(mol_reference))[0]
        except FileNotFoundError:
            print(f"Error: Reference file not found at {mol_reference}")
            return
        initial_charge = charge
        atom_indices = self._get_atom_indices(mol_original, atom_type)
        if not atom_indices:
            print(f"No atoms of type '{atom_type}' found to substitute.")
            return

        n_species = 0
        original_number = mol_original[atom_indices[0]].number

        # --- Single substitutions ---
        print("Generating single substitutions...")
        for idx in atom_indices:
            # Up-substitution (e.g., C -> N)
            mol_copy = mol_original.copy()
            mol_copy[idx].number += 1
            final_charge = initial_charge + 1
            self._write_molecule(mol_copy, mol_ref_name, [(idx, mol_copy.get_chemical_symbols()[idx])], final_charge)
            n_species += 1
            
            # Down-substitution (e.g., C -> B)
            mol_copy = mol_original.copy()
            mol_copy[idx].number -= 1
            final_charge = initial_charge - 1
            self._write_molecule(mol_copy, mol_ref_name, [(idx, mol_copy.get_chemical_symbols()[idx])], final_charge)
            n_species += 1

        # --- Double substitutions ---
        print("Generating double substitutions...")
        for idx1, idx2 in itertools.combinations(atom_indices, 2):
            # Up-Up
            mol_copy = mol_original.copy()
            mol_copy[idx1].number += 1
            mol_copy[idx2].number += 1
            final_charge = initial_charge + 2
            subs = [(idx1, mol_copy.get_chemical_symbols()[idx1]), (idx2, mol_copy.get_chemical_symbols()[idx2])]
            self._write_molecule(mol_copy, mol_ref_name, subs, final_charge)
            n_species += 1

            # Down-Down
            mol_copy = mol_original.copy()
            mol_copy[idx1].number -= 1
            mol_copy[idx2].number -= 1
            final_charge = initial_charge - 2
            subs = [(idx1, mol_copy.get_chemical_symbols()[idx1]), (idx2, mol_copy.get_chemical_symbols()[idx2])]
            self._write_molecule(mol_copy, mol_ref_name, subs, final_charge)
            n_species += 1

            # Up-Down
            mol_copy = mol_original.copy()
            mol_copy[idx1].number += 1
            mol_copy[idx2].number -= 1
            final_charge = initial_charge # +1 and -1 cancel out
            subs = [(idx1, mol_copy.get_chemical_symbols()[idx1]), (idx2, mol_copy.get_chemical_symbols()[idx2])]
            self._write_molecule(mol_copy, mol_ref_name, subs, final_charge)
            n_species += 1
            
            # Down-up
            mol_copy = mol_original.copy()
            mol_copy[idx1].number -= 1
            mol_copy[idx2].number += 1
            final_charge = initial_charge # -1 and +1 cancel out
            subs = [(idx1, mol_copy.get_chemical_symbols()[idx1]), (idx2, mol_copy.get_chemical_symbols()[idx2])]
            self._write_molecule(mol_copy, mol_ref_name, subs, final_charge)
            n_species += 1

        print(f"Generated {n_species} new molecular species.")
        print("--- Geometry Generation Complete ---")

#### Data Processor (Unchanged)
#**File Location:** `src/chem_workflow/data_processor.py`
#*(Content is the same as the previous version)*

#### Requirements File (Unchanged)
#**File Location:** `requirements.txt`
#*(Content is the same as the previous version)*

### How to Run Your New Project

#The instructions for running the project remain the same. From the root `chem_workflow` directory, run:
#`python src/main.py`

