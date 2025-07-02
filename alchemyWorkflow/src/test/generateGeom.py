from ase import Atoms
from ase.io import read, write
from ase.data import atomic_numbers, chemical_symbols

import os

def gen_geom(mol_reference, atom_type, charge):
    mol=read(f'{mol_reference}.xyz')
    atom_index = get_indexes(mol, atom_type)
    gen_subs(mol, atom_index, charge, mol_reference)

def get_indexes(mol, atom_type):
    # xyz file with no changes

    # accepts either atomic number or symbol
    if isinstance(atom_type, str):
        atom_number = atomic_numbers[upper(atom_type)]
        atom_symbol = atom_type
    else:
        atom_symbol = chemical_symbols[atom_type]
        atom_number = atom_type

    # get indexes for desired atom_type substitution
    atom_index = mol.symbols.search(atom_symbol)
    return atom_index

def gen_subs(mol, atom_index, charge, mol_reference):
    #count how many species are being generated
    n_species = 0
    
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
    print(f'files generates: {n_species}')
                   
