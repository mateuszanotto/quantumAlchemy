import cclib
from HessianTools import *

# get all file names ending with .out
text_files = [
    f for f in os.listdir(os.getcwd()) 
    if f.endswith(".out")
]

dataframes = []
for filename in text_files:
    #load the output
    data = cclib.io.ccread(filename)

    #Create a list of 0 (carbons)
    mol_indexes = np.zeros(20)

    #Change the B(5) to -1 and N(7) to 1
    for i in range(20):
        if data.atomnos[i] == 5:
            mol_indexes[i] = -1
        if data.atomnos[i] == 7:
            mol_indexes[i] = 1

    # Create a DataFrame for the current file and add to the list
    df = pd.DataFrame([mol_indexes], columns=[f'z{i}' for i in range(20)])

    # Add the SCF Energies column
    scfEnergy = data.scfenergies
    df["Energy"] = scfEnergy

    # Add the HOMO and LUMO columns
    homo_value = data.moenergies[0][data.homos[0]]
    lumo_value = data.moenergies[0][data.homos[0] + 1]
    df["HOMO"] = homo_value
    df["LUMO"] = lumo_value

    # Load the last gradient matrix and flatten it
    grad = data.grads[-1]
    grad_flat = data.grads.flatten()

    # Create the gradient column names
    grad_column = [f"grad[{i}][{j}]" for i in range(data.natom) for j in ['x','y','z']]
    grad_df = pd.DataFrame([grad_flat], columns=grad_column)
    df = pd.concat([df, grad_df], axis=1)

    # Load the Hessian matrix and flatten it
    hess = HessianTools(f"{filename[:-4]}.hess")
    hessian_flat = hess.hessian.flatten()

    # Create Hessian column names
    hessian_column = [f"hess[{i}][{j}]" for i in range(data.natom*3) for j in range(data.natom*3)]
    hessian_df = pd.DataFrame([hessian_flat], columns=hessian_column)
    df = pd.concat([df, hessian_df], axis=1)

    df.index = [filename]  # Set the filename as the index
    dataframes.append(df)

all_data = pd.concat(dataframes)
all_data.to_csv("output.csv", index=True)  # Use index=True to include the filename index in the CSV
