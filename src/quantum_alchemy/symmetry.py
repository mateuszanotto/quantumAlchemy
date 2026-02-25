import numpy as np
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

def get_permutations(xyz_file, target_atom_indices):
    """
    Identifies the point group of the molecule and returns the atom permutations
    for the specified target indices under each symmetry operation.
    """
    mol = Molecule.from_file(xyz_file)
    pga = PointGroupAnalyzer(mol)
    
    print(f"Detected Point Group: {pga.get_pointgroup()}")
    
    symm_ops = pga.get_symmetry_operations()
    print(f"Found {len(symm_ops)} symmetry operations.")
    
    target_coords = mol.cart_coords[target_atom_indices]
    
    permutations = []
    
    for op in symm_ops:
        # Apply symmetry operation to target coordinates
        transformed_coords = op.operate_multi(target_coords)
        
        # Find which index each transformed coordinate matches in the original target_coords
        perm = []
        for t_coord in transformed_coords:
            # Find index in target_coords that is closest to t_coord
            diffs = target_coords - t_coord
            dists = np.linalg.norm(diffs, axis=1)
            match_idx = np.argmin(dists)
            
            if dists[match_idx] > 0.1:
                raise ValueError(f"Symmetry operation {op} mapping failed. Min dist: {dists[match_idx]}")
            
            perm.append(match_idx)
        
        permutations.append(perm)
        
    # Filter unique permutations (sometimes different ops lead to same permutation if points are special)
    unique_perms = []
    seen = set()
    for p in permutations:
        p_tuple = tuple(p)
        if p_tuple not in seen:
            unique_perms.append(p)
            seen.add(p_tuple)
            
    print(f"Unique permutations: {len(unique_perms)}")
    
    # Generate inverse permutations for lookup
    p_invs = []
    for p in unique_perms:
        inv = [0] * len(p)
        for i, val in enumerate(p):
            inv[val] = i
        p_invs.append(inv)
        
    return unique_perms, p_invs, pga.get_pointgroup().sch_symbol

def get_cycle_lengths(perm):
    """Returns the lengths of all cycles in a permutation."""
    visited = [False] * len(perm)
    lengths = []
    for i in range(len(perm)):
        if not visited[i]:
            length = 0
            curr = i
            while not visited[curr]:
                visited[curr] = True
                curr = perm[curr]
                length += 1
            lengths.append(length)
    return lengths

def get_pet_counts_by_k(perms, num_targets, max_k):
    """
    Calculates the number of unique colorings for each k substitutions (where k is the number of non-reference atoms).
    Assumes 2 possible non-reference colors (e.g., B and N).
    """
    import numpy as np
    
    # Polynomial for each permutation: product of (1 + 2*x^L) for each cycle of length L
    total_poly = np.poly1d([0])
    
    for p in perms:
        lengths = get_cycle_lengths(p)
        poly = np.poly1d([1])
        for L in lengths:
            # (1 + 2*x^L)
            term_coeffs = [0] * (L + 1)
            term_coeffs[0] = 2 # 2*x^L
            term_coeffs[L] = 1 # 1
            poly *= np.poly1d(term_coeffs)
        total_poly += poly
    
    # Divide coefficients by |G|
    final_coeffs = total_poly.coeffs / len(perms)
    # The poly coefficients are in order [x^max, ..., x^0]
    # We want [x^0, ..., x^max_k]
    counts = {}
    for i, coeff in enumerate(reversed(final_coeffs)):
        if i <= max_k:
            counts[i] = int(round(coeff))
            
    return counts

def get_pet_count(perms, num_colors):
    """Calculates total unique colorings using Polya Enumeration Theorem."""
    total = 0
    for p in perms:
        total += num_colors ** len(get_cycle_lengths(p))
    return total // len(perms)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python symmetry_utils.py <xyz_file>")
        sys.exit(1)
        
    xyz_path = sys.argv[1]
    # For porphyrin in the example, carbons are 0-19
    target_indices = list(range(20))
    perms, pg = get_permutations(xyz_path, target_indices)
    print(f"Point Group: {pg}")
    print(f"Permutations: {len(perms)}")
    for i, p in enumerate(perms):
        print(f"Op {i}: {p}")
