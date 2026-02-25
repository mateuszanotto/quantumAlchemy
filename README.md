# Quantum Alchemy

Molecular Alchemy Pipeline for generating and predicting properties of substituted molecules.

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
