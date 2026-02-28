import numpy as np
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

def _deduplicate_and_invert(permutations):
    """Helper: deduplicate permutations and compute their inverses."""
    unique_perms = []
    seen = set()
    for p in permutations:
        p_tuple = tuple(p)
        if p_tuple not in seen:
            unique_perms.append(p)
            seen.add(p_tuple)

    p_invs = []
    for p in unique_perms:
        inv = [0] * len(p)
        for i, val in enumerate(p):
            inv[val] = i
        p_invs.append(inv)

    return unique_perms, p_invs


def get_permutations_all_atoms(xyz_file, target_atom_indices, tolerance=1.0):
    """
    Returns permutations of the target atoms under the molecule's full symmetry
    group, considering ALL atoms when matching. Operations that map any target
    atom onto a non-target position are skipped.

    Use this for total unique-structure counting (Pólya enumeration).
    """
    mol = Molecule.from_file(xyz_file)
    pga = PointGroupAnalyzer(mol)

    print(f"Detected Point Group: {pga.get_pointgroup()}")

    symm_ops = pga.get_symmetry_operations()
    print(f"Found {len(symm_ops)} symmetry operations.")

    all_coords = mol.cart_coords
    target_set = set(target_atom_indices)
    # Map global index -> position in target list
    global_to_target = {g: t for t, g in enumerate(target_atom_indices)}

    permutations = []
    skipped = 0

    for op in symm_ops:
        transformed_all = op.operate_multi(all_coords)

        perm = []
        valid = True
        for t_idx in target_atom_indices:
            t_coord = transformed_all[t_idx]
            diffs = all_coords - t_coord
            dists = np.linalg.norm(diffs, axis=1)
            match_global = np.argmin(dists)

            if dists[match_global] > tolerance:
                valid = False
                break

            if match_global not in target_set:
                # This op maps a target atom to a non-target position — skip
                valid = False
                break

            perm.append(global_to_target[match_global])

        if valid:
            permutations.append(perm)
        else:
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} ops that don't preserve the target subset.")

    unique_perms, p_invs = _deduplicate_and_invert(permutations)
    print(f"Unique permutations (all-atom): {len(unique_perms)}")

    return unique_perms, p_invs, pga.get_pointgroup().sch_symbol


def get_permutations_target_atoms(xyz_file, target_atom_indices, tolerance=1.0):
    """
    Returns permutations of the target atoms by applying symmetry operations
    ONLY to the target atom coordinates and matching within that subset.

    Use this for training-set generation and geometry extrapolation, where
    only the symmetry of the target sub-lattice matters.
    """
    mol = Molecule.from_file(xyz_file)
    pga = PointGroupAnalyzer(mol)

    print(f"Detected Point Group: {pga.get_pointgroup()}")

    symm_ops = pga.get_symmetry_operations()
    print(f"Found {len(symm_ops)} symmetry operations.")

    target_coords = mol.cart_coords[target_atom_indices]

    permutations = []
    skipped = 0

    for op in symm_ops:
        transformed_coords = op.operate_multi(target_coords)

        perm = []
        valid = True
        for t_coord in transformed_coords:
            diffs = target_coords - t_coord
            dists = np.linalg.norm(diffs, axis=1)
            match_idx = np.argmin(dists)

            if dists[match_idx] > tolerance:
                valid = False
                break

            perm.append(match_idx)

        if valid:
            permutations.append(perm)
        else:
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} ops that don't map within target subset.")

    unique_perms, p_invs = _deduplicate_and_invert(permutations)
    print(f"Unique permutations (target-only): {len(unique_perms)}")

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
    perms, p_invs, pg = get_permutations_all_atoms(xyz_path, target_indices)
    print(f"Point Group: {pg}")
    print(f"Permutations: {len(perms)}")
    for i, p in enumerate(perms):
        print(f"Op {i}: {p}")
