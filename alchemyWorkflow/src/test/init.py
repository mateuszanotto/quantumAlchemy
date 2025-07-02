from generateGeom import gen_geom

## gen_geom(mol_reference, atom_type, charge)
#       mol_reference = file name in .xyz format (without the .xyz)
#       atom_type = atom that will be changed to +1 and -1 (can accept number or string)
#       charge = total charge of reference molecule

a = gen_geom('porph', 6,0)
