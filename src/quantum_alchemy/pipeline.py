#!/home/abcsim/miniconda3/bin/python
import os
import sys
import argparse
import json
import itertools
import numpy as np
import polars as pl
import pickle
import nablachem.alchemy
from tqdm import tqdm
from multiprocessing import Pool
from .symmetry import get_permutations, get_pet_count, get_pet_counts_by_k
from .results import extract_all
from pymatgen.core import Molecule

def get_atom_types(xyz_file):
    mol = Molecule.from_file(xyz_file)
    species = [str(s) for s in mol.species]
    unique_types = sorted(list(set(species)))
    return unique_types, species

def save_xyz(coords, species, filename, comment=""):
    with open(filename, "w") as f:
        f.write(f"{len(species)}\n")
        f.write(f"{comment}\n")
        for s, (x, y, z) in zip(species, coords):
            f.write(f"{s:2} {x:12.6f} {y:12.6f} {z:12.6f}\n")

def get_atomic_number_map():
    periodic_table = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
        "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30
    }
    inv_map = {v: k for k, v in periodic_table.items()}
    return periodic_table, inv_map

def is_orbit_representative(structure, group_ops):
    structure_tuple = tuple(structure)
    for perm in group_ops:
        transformed = tuple(structure[i] for i in perm)
        if transformed < structure_tuple:
            return False
    return True

def is_partial_canonical(pos, current_struct, p_invs):
    """
    Checks if the partial structure current_struct[0:pos+1] is canonical.
    Lexicographical order: -1 < 0 < 1
    """
    for p_inv in p_invs:
        for j in range(pos + 1):
            source_idx = p_inv[j]
            if source_idx > pos:
                # Image at j depends on an atom not yet assigned.
                # Cannot prove non-canonicity yet.
                break
            
            val_image = current_struct[source_idx]
            val_orig = current_struct[j]
            
            if val_image < val_orig:
                return False # Found a smaller image!
            if val_image > val_orig:
                break # This permutation makes the image larger, so it's safe.
    return True

def generate_prediction_set(config, perms, p_invs, max_subs):
    """Generates all unique structures up to max_subs using backtracking and pruning."""
    print(f"Generating prediction set (up to {max_subs} substitutions) using backtracking...")
    num_targets = config["num_target_atoms"]
    output_temp = "dataset_temp.csv"
    z_cols = [f"z{i}" for i in range(num_targets)]
    
    theoretical_counts = get_pet_counts_by_k(perms, num_targets, max_subs)
    total_theoretical = sum(theoretical_counts.values()) + 1 # +1 for reference k=0
    
    # Check if dataset.feather exists
    output_feather = "dataset.feather"
    if os.path.exists(output_feather):
        try:
            count = pl.scan_ipc(output_feather).select(pl.len()).collect().item()
            if count > 0:
                print(f"{output_feather} found with {count} structures. Skipping generation (theoretical max: {total_theoretical}).")
                if os.path.exists(output_temp):
                    os.remove(output_temp)
                    print(f"Removed temporary file {output_temp}.")
                return
        except Exception as e:
            print(f"Error checking {output_feather}: {e}")

    import csv
    existing_counts = {}
    if os.path.exists(output_temp):
        print(f"Found existing {output_temp}. Checking for resume point (lazy scan)...")
        try:
            # Use lazy scan to get counts per k without loading whole file
            q = (
                pl.scan_csv(output_temp)
                .select(pl.sum_horizontal(pl.all().abs()).alias("k"))
                .group_by("k")
                .len()
            )
            res_df = q.collect()
            if len(res_df) > 0:
                res_dict = res_df.to_dict(as_series=False)
                existing_counts = dict(zip(res_dict["k"], res_dict["len"]))
        except Exception as e:
            print(f"Error checking resume point: {e}")

    # Open CSV in append mode
    file_exists = os.path.exists(output_temp)
    csv_file = open(output_temp, "a", newline='')
    writer = csv.DictWriter(csv_file, fieldnames=z_cols)
    if not file_exists or 0 not in existing_counts:
        writer.writeheader()
        # Reference (k=0)
        writer.writerow({f"z{i}": 0 for i in range(num_targets)})
        existing_counts[0] = 1
        csv_file.flush()

    total_unique_found = sum(existing_counts.values())
    
    # We use a recursive generator for each k
    def backtrack_recursive(idx, current_struct, k_rem):
        # Base case: we've assigned all atoms
        if idx == num_targets:
            if k_rem == 0:
                yield tuple(current_struct)
            return

        # Pruning: if we need more substitutions than remaining slots, prune
        if k_rem > (num_targets - idx):
            return

        # Try colors -1, 0, 1 in that order (lexicographical minimum)
        for color in [-1, 0, 1]:
            if color != 0:
                if k_rem == 0: continue
                current_struct[idx] = color
                if is_partial_canonical(idx, current_struct, p_invs):
                    yield from backtrack_recursive(idx + 1, current_struct, k_rem - 1)
            else:
                # color 0
                current_struct[idx] = 0
                if is_partial_canonical(idx, current_struct, p_invs):
                    yield from backtrack_recursive(idx + 1, current_struct, k_rem)
        
        # Reset for backtracking
        current_struct[idx] = 0

    # Calculate total theoretical unique structures across all k
    total_target = sum(theoretical_counts.get(k, 0) for k in range(1, max_subs + 1))
    pbar = tqdm(total=total_target, desc="Generating Prediction Set", unit="struct")
    
    for k in range(1, max_subs + 1):
        target_k = theoretical_counts.get(k, 0)
        found_k = existing_counts.get(k, 0)
        
        # Update progress bar with current k
        pbar.set_description(f"k={k}/{max_subs} ({target_k} exp.)")
        
        if found_k >= target_k and target_k > 0:
            pbar.update(target_k)
            continue
            
        # To avoid duplicates if resuming within k, load only k-level structures
        existing_in_k = set()
        if found_k > 0:
            try:
                k_structs = (
                    pl.scan_csv(output_temp)
                    .filter(pl.sum_horizontal(pl.all().abs()) == k)
                    .collect()
                )
                for row_vals in k_structs.iter_rows():
                    existing_in_k.add(tuple(row_vals))
            except Exception as e:
                print(f"Error loading existing structures for k={k}: {e}")
            pbar.update(found_k)

        k_count = found_k
        current_struct = [0] * num_targets
        
        for struct in backtrack_recursive(0, current_struct, k):
            if struct not in existing_in_k:
                writer.writerow({f"z{i}": val for i, val in enumerate(struct)})
                k_count += 1
                total_unique_found += 1
                pbar.update(1)
                if k_count % 1000 == 0: csv_file.flush()
                
    pbar.close()
    csv_file.close()
    print(f"Total unique structures in dataset: {total_unique_found}")
    print(f"Converting to dataset.feather (streaming)...")
    pl.scan_csv(output_temp).sink_ipc("dataset.feather")
    print("Done.")
    
    # Cleanup temporary CSV
    try:
        os.remove(output_temp)
        print(f"Removed temporary file {output_temp}.")
    except Exception as e:
        print(f"Warning: Could not remove {output_temp}: {e}")

def convert_huge_csv_to_feather(csv_path, feather_path):
    print(f"Iniciando conversão via Streaming: {csv_path} -> {feather_path}")
    import time
    start_time = time.time()
    (
        pl.scan_csv(csv_path)
        .sink_ipc(feather_path, compression="zstd")
    )
    end_time = time.time()
    csv_size = os.path.getsize(csv_path) / (1024**3)
    feather_size = os.path.getsize(feather_path) / (1024**3)
    print("-" * 30)
    print(f"Conversão concluída em {end_time - start_time:.2f} segundos")
    print(f"Tamanho CSV: {csv_size:.2f} GB")
    print(f"Tamanho Feather: {feather_size:.2f} GB")
    print("-" * 30)

def run_prediction(training_feather, dataset_feather, config, properties):
    """Loads training data, builds MultiTaylor model, and predicts for dataset."""
    print(f"Building MultiTaylor model from {training_feather}...")
    num_targets = config["num_target_atoms"]
    z_cols = [f"z{i}" for i in range(num_targets)]
    
    df_train = pl.read_ipc(training_feather).select(z_cols + properties).to_pandas()
    
    mt = nablachem.alchemy.MultiTaylor(df_train, outputs=properties)
    mt.reset_center(**{f'z{i}': 0 for i in range(num_targets)})
    mt.build_model(2)
    print("Model built successfully.")
    
    print(f"Predicting for structures in {dataset_feather} (vectorized)...")
    
    exprs = []
    for output in properties:
        poly_expr = pl.lit(0.0)
        for monomial in mt._monomials[output]:
            coeff = monomial.prefactor()
            if not monomial._powers:
                poly_expr = poly_expr + coeff
            else:
                term_expr = pl.lit(coeff)
                for col, power in monomial._powers.items():
                    center_val = mt._center[col]
                    term_expr = term_expr * ((pl.col(col) - center_val) ** power)
                poly_expr = poly_expr + term_expr
        exprs.append(poly_expr.alias(output))
    
    output_feather = "results_final.feather"
    output_csv = "results_temp.csv"
    
    df_predict = pl.read_ipc(dataset_feather, memory_map=True)
    total_rows = len(df_predict)
    chunk_size = 5_000_000 
    
    # Execute prediction and append to CSV
    with tqdm(total=total_rows, desc="Predicting", unit="struct") as pbar:
        for i, chunk in enumerate(df_predict.iter_slices(n_rows=chunk_size)):
            # Calc and write chunk
            res_chunk = chunk.with_columns(exprs)
            
            if i == 0:
                res_chunk.write_csv(output_csv)
            else:
                # Appending to CSV
                with open(output_csv, "ab") as f:
                    res_chunk.write_csv(f, include_header=False)
            
            pbar.update(len(chunk))
    
    # Convert CSV to Feather
    convert_huge_csv_to_feather(output_csv, output_feather)
    
    # Verification
    print("Verifying results...")
    try:
        count_feather = pl.scan_ipc(output_feather).select(pl.len()).collect().item()
        if count_feather == total_rows:
            print(f"Verification successful: {count_feather} rows in Feather matches dataset.")
            os.remove(output_csv)
            print(f"Removed temporary file {output_csv}.")
        else:
            print(f"Warning: Row count mismatch! Feather: {count_feather}, Expected: {total_rows}")
            print(f"Temporary file {output_csv} retained for inspection.")
    except Exception as e:
        print(f"Error during verification: {e}")

def write_auto_orca_script(directory):
    script_path = os.path.join(directory, "auto_orca.sh")
    content = r"""#!/bin/bash
# ORCA Input Template (Edit this part for different methods/basis sets)
INPUT_HEADER="! B3LYP def2-TZVPP
%PAL NPROCS ${NPROCS} END
%tddft
 nroots 20
end
%maxcore 8000"

# Configuration
BASEDIR=$(pwd)
OUTPUT_DIR="$BASEDIR/../training_outputs"
mkdir -p "$OUTPUT_DIR"

# SLURM NPROCS detection (sets to 1 if not on cluster)
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
    NPROCS=$SLURM_CPUS_PER_TASK
elif [ -n "$SLURM_NTASKS" ]; then
    NPROCS=$SLURM_NTASKS
else
    NPROCS=1
fi

echo "Using NPROCS=$NPROCS"

for geom in *.xyz; do
    if [ "$geom" == "auto_orca.sh" ]; then continue; fi
    geom_name=$(basename "$geom" .xyz)
    
    # Extract charge from 2nd line (e.g. "Charge: 0")
    # This reads the last column of the 2nd line
    carga=$(sed -n '2p' "$geom" | awk '{print $NF}')
    
    # Skip if already being processed or done
    if [ -e "run/$geom_name/$geom_name.out" ]; then
        echo "O arquivo $geom_name já existe ou está rodando. Pulando..."
        continue
    fi

    echo "Preparando conformero: $geom_name (Carga: $carga)"
    mkdir -p "run/$geom_name"
    cp "$geom" "run/$geom_name/"

    # Generate the ORCA input file
    cat <<EOF > "run/$geom_name/$geom_name.inp"
$INPUT_HEADER
* xyzfile $carga 2 ${geom_name}.xyz
EOF

    cd "run/$geom_name" || exit

    # Execute ORCA
    ORCA_EXEC=${ORCA:-orca}
    echo "Executando ORCA para $geom_name..."
    $ORCA_EXEC "${geom_name}.inp" > "${geom_name}.out"

    # Copy output back to training_outputs
    cp "${geom_name}.out" "$OUTPUT_DIR/"

    echo "$geom_name concluído"
    cd "$BASEDIR" || exit
done
"""
    with open(script_path, "w") as f:
        f.write(content)
    os.chmod(script_path, 0o755)

def phase_setup_training(args):
    atom_types, all_species = get_atom_types(args.reference)
    print(f"Atom types present: {', '.join(atom_types)}")
    
    target_type = args.atom
    if not target_type:
         target_type = input(f"Which atom type to substitute? ({'/'.join(atom_types)}): ")
    
    if target_type not in atom_types:
        print(f"Error: {target_type} not in molecule.")
        sys.exit(1)

    target_indices = [i for i, s in enumerate(all_species) if s == target_type]
    print(f"Found {len(target_indices)} atoms of type {target_type}.")

    # Symmetry
    perms, p_invs, pg_symbol = get_permutations(args.reference, target_indices)
    total_unique_all_subs = get_pet_count(perms, 3) 
    print(f"Total theoretically unique structures (all possible substitutions): {total_unique_all_subs}")

    config = {
        "reference_file": args.reference,
        "target_type": target_type,
        "target_indices": target_indices,
        "symmetry_pg": pg_symbol,
        "initial_charge": args.charge,
        "num_target_atoms": len(target_indices)
    }
    
    with open(".config.json", "w") as f:
        json.dump(config, f, indent=4)

    pt, inv_pt = get_atomic_number_map()
    target_z = pt[target_type]
    sub_minus_type = inv_pt[target_z - 1]
    sub_plus_type = inv_pt[target_z + 1]
    
    print(f"Substitutions: {target_type} -> {sub_minus_type}(-1) or {sub_plus_type}(+1)")

    os.makedirs("training_inputs", exist_ok=True)
    mol = Molecule.from_file(args.reference)
    coords = mol.cart_coords
    
    unique_training_structs = []
    unique_training_structs.append(([0] * len(target_indices), 0)) # Reference
    
    for k in range(1, args.subs + 1):
        k_count = 0
        for num_minus in range(k + 1):
            num_plus = k - num_minus
            charge_delta = num_plus - num_minus
            
            for minus_idxs in itertools.combinations(range(len(target_indices)), num_minus):
                remaining = [i for i in range(len(target_indices)) if i not in minus_idxs]
                for plus_idxs in itertools.combinations(remaining, num_plus):
                    struct = [0] * len(target_indices)
                    for i in minus_idxs: struct[i] = -1
                    for i in plus_idxs: struct[i] = 1
                    
                    if is_orbit_representative(struct, perms):
                        unique_training_structs.append((struct, charge_delta))
                        k_count += 1
        print(f"k={k}: {k_count} unique structures.")

    for idx, (struct, charge_delta) in enumerate(unique_training_structs):
        new_species = list(all_species)
        for i, val in enumerate(struct):
            if val == -1: new_species[target_indices[i]] = sub_minus_type
            elif val == 1: new_species[target_indices[i]] = sub_plus_type
        
        final_charge = args.charge + charge_delta
        filename = f"training_inputs/struct_{idx:04d}_q{final_charge}.xyz"
        save_xyz(coords, new_species, filename, comment=f"Charge: {final_charge}")

    write_auto_orca_script("training_inputs")
    print(f"Done. {len(unique_training_structs)} training files generated in 'training_inputs/'.")
    print("Next: Run 'training_inputs/auto_orca.sh' and then run this script again.")

def phase_extract_predict(config, perms, p_invs, args):
    if args.structures:
        generate_prediction_set(config, perms, p_invs, config.get("num_target_atoms", 20))
        print("Workflow: Structure generation complete.")
        return True

    if not os.path.exists("training_inputs"):
        print("Workflow: 'training_inputs/' directory not found. Re-generating training set...")
        # We need to re-run the setup phase logic or similar. 
        # For simplicity, let's signal to main to proceed to setup.
        return False

    num_inputs = len([f for f in os.listdir("training_inputs") if f.endswith(".xyz")])
    if num_inputs == 0:
        print("Workflow: 'training_inputs/' is empty. Re-generating training set...")
        return False

    if not os.path.exists("training_outputs"):
        print(f"Workflow: 'training_outputs/' not found. Please run calculations first.")
        return True
        
    num_outputs = len([f for f in os.listdir("training_outputs") if f.endswith(".out")])
    print(f"Workflow: {num_outputs}/{num_inputs} calculations complete.")
    
    if num_outputs < num_inputs:
        print("Workflow: Waiting for all calculations to complete.")
        return

    print("Workflow: All training calculations complete. Proceeding to dataset generation.")
    generate_prediction_set(config, perms, p_invs, config.get("num_target_atoms", 20)) 
    
    print("Workflow: Proceeding to property extraction.")
    props = input("Properties to extract (space separated, default: Energy_DFT): ")
    if not props: props = ["Energy_DFT"]
    else: props = props.split()
    
    from .results import extract_all
    df_train = extract_all("training_outputs", "training.feather", config, props)
    
    if df_train is not None:
        if args.recalculate or not os.path.exists("results_final.feather"):
            run_prediction("training.feather", "dataset.feather", config, props)
            print("Workflow: Extraction and prediction complete.")
        else:
            print("Workflow: Prediction already exists. Use -r to recalculate.")

def main():
    parser = argparse.ArgumentParser(description="Quantum Alchemy Pipeline")
    parser.add_argument("reference", help="Reference XYZ file")
    parser.add_argument("-c", "--charge", type=int, default=0, help="Initial charge of the molecule")
    parser.add_argument("-a", "--atom", help="Atom type to substitute (e.g., C)")
    parser.add_argument("-p", "--pipeline", action="store_true", help="Run the full pipeline (default)")
    parser.add_argument("-s", "--structures", action="store_true", help="Generate unique structures only")
    parser.add_argument("-r", "--recalculate", action="store_true", help="Force recalculate results (debug)")
    parser.add_argument("-k", "--subs", type=int, default=2, help="Max substitutions for training (default: 2)")
    args = parser.parse_args()

    if not os.path.exists(args.reference):
        print(f"Error: {args.reference} not found.")
        sys.exit(1)

    if os.path.exists(".config.json"):
        with open(".config.json", "r") as f:
            config = json.load(f)
        
        if config["reference_file"] == args.reference:
            target_indices = config["target_indices"]
            perms, p_invs, pg = get_permutations(args.reference, target_indices)
            if phase_extract_predict(config, perms, p_invs, args):
                return

    # Phase 1: Setup Training
    phase_setup_training(args)

if __name__ == "__main__":
    main()
