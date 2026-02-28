# Quantum Alchemy

Molecular Alchemy Pipeline for generating and predicting properties of substituted molecules. The properties expansion is calculated by the Quantum Alchemy algorithm Nablachem (https://github.com/NablaChem/nablachem) 

## Installation

Install the package using pip:

```bash
pip install .
```

For development mode (where changes to the code are reflected immediately):

```bash
pip install -e .
```

## Usage

After installation, use the `quantumAlchemy` command:

```bash
quantumAlchemy reference.xyz [options]
```

### Options

*   `-h`, `--help`: Show the help message and exit.
*   `-a`, `--atom`: Atom type to substitute (e.g., `C`).
*   `-c`, `--charge`: Initial charge of the molecule (default: `0`).
*   `-p`, `--pipeline`: Run the full pipeline (default behavior).
*   `-s`, `--structures`: Generate the unique structures dataset only.
*   `-r`, `--recalculate`: Force recalculation of results (debug mode).

### Examples

**1. Setup Training Files**
```bash
quantumAlchemy benzene.xyz -a C
```

**2. Generate Unique Structures Dataset Only**
```bash
quantumAlchemy benzene.xyz -s
```

**3. Run Full Pipeline and Prediction**
```bash
quantumAlchemy benzene.xyz
```

### Workflow

**1. Create the training structures for DFT calculation**
The training structures are composed by the single and double atomic charge perturbations for every possibility.


**2. Read the reference symmetry and extrapolate**
Generate a .feather list containing the atom changes for triple to n perturbations.

**3. Generate Dataset**
Retrieve the Quantum Alchemy model properties and save to a .feather
