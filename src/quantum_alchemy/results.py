import os
import polars as pl
import cclib
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def process_file_optimized(args):
    filename, target_indices, properties = args
    try:
        data = cclib.io.ccread(filename)
    except Exception as e:
        return {"status": "error", "filename": filename, "error": str(e)}

    # Atom-type encoding (assuming C->0, B->-1, N->1 as in original)
    # This needs to be consistent with the pipeline's target_indices and atom substitutions
    # For now, let's make it generic enough
    
    row = {"filename": os.path.basename(filename)}
    
    # We need to know which atoms were changed. 
    # The filename might help, but ideally we check the atomnos
    # In training_inputs, we know which indices were targets
    atomnos = data.atomnos
    
    # Extract properties
    if "Energy_DFT" in properties:
        row["Energy_DFT"] = data.scfenergies[-1] if hasattr(data, 'scfenergies') else None
    if "HOMO" in properties:
        row["HOMO"] = data.moenergies[0][data.homos[0]] if hasattr(data, 'moenergies') else None
    if "LUMO" in properties:
        row["LUMO"] = data.moenergies[0][data.homos[0] + 1] if hasattr(data, 'moenergies') else None
    
    # Extract atom substitutions specifically for target_indices
    # This requires knowing what the reference atom was. 
    # Let's assume the pipeline will handle the mapping from atomic numbers to z-scores.
    
    return {"status": "ok", "filename": filename, "row": row, "atomnos": atomnos}

def extract_all(directory, output_feather, config, properties=["Energy_DFT"]):
    out_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".out")]
    if not out_files:
        print("No .out files found.")
        return None

    target_indices = config["target_indices"]
    
    work_items = [(f, target_indices, properties) for f in out_files]
    
    rows = []
    skipped = []
    
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_file_optimized, work_items), total=len(work_items), desc="Extracting results"))
        
        for res in results:
            if res["status"] == "ok":
                # Add z-scores based on atomnos and reference
                # We need to know the mapping. 
                # C(6) -> 0, B(5) -> -1, N(7) -> 1
                row = res["row"]
                atomnos = res["atomnos"]
                for i, idx in enumerate(target_indices):
                    z = 0
                    if atomnos[idx] == 5: z = -1
                    elif atomnos[idx] == 7: z = 1
                    row[f"z{i}"] = z
                rows.append(row)
            else:
                skipped.append((res["filename"], res["error"]))

    if skipped:
        print(f"Skipped {len(skipped)} files due to errors.")
        for f, e in skipped[:5]: print(f"  {f}: {e}")

    df = pl.DataFrame(rows)
    
    # Reorder columns: z0..zn first, then others
    z_cols = [f"z{i}" for i in range(len(target_indices))]
    other_cols = [c for c in df.columns if c not in z_cols]
    df = df.select(z_cols + other_cols)
    
    df.write_ipc(output_feather)
    print(f"Saved {len(rows)} results to {output_feather}")
    return df

def main():
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="training_outputs")
    parser.add_argument("--out", default="training.feather")
    parser.add_argument("--props", nargs="+", default=["Energy_DFT"])
    args = parser.parse_args()
    
    if os.path.exists(".config.json"):
        with open(".config.json", "r") as f:
            config = json.load(f)
        extract_all(args.dir, args.out, config, args.props)
    else:
        print("Error: .config.json not found. Run the pipeline first.")

if __name__ == "__main__":
    main()
