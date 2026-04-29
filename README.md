# GOLD: Graph Anomaly-based Logic Locking Deciphering

This is the implementation for Graph Anomaly-based Logic Locking Deciphering.

## Project Structure

```
.
├── data/                  # Original circuit datasets
│   ├── TRLL/              # TRLL locked circuit data
│   ├── TRLL+/             # TRLL+ locked circuit data
│   └── TroMUX/            # TroMUX locked circuit data
├── data_part/             # Datasets with standard features only
├── rewire/                # Circuit rewiring module based on Ollivier-Ricci curvature
│   ├── netlists/          # Original netlist circuit data
│   ├── GraphRicciCurvature/  # Ricci curvature computation library
│   ├── preprocessing/     # Data preprocessing scripts
│   └── run_netlists_rewire.py  # Main rewiring script
├── main.py                # Main program entry
├── BWGNN.py               # Beta Wavelet Graph Neural Network implementation
├── bonGNN.py              # Bernstein polynomial GNN implementation
├── cheGNN.py              # Chebyshev polynomial GNN implementation
└── dataset.py             # Dataset loading and processing module
```

## Data Format

Each circuit dataset (e.g., `data/TRLL/c1355/0/`) contains the following files:
- `edges.txt`: Graph edge connections
- `feat.txt`: Node feature matrix
- `keys.txt`: Indices of key nodes
- `label.txt`: Node labels (0 for normal nodes, 1 for key nodes)

## File Descriptions

### main.py
Main program entry responsible for loading data, constructing graphs, training models, and evaluating results. Includes command-line argument parsing, data preprocessing, model training, and evaluation.

### BWGNN.py
Implements Beta Wavelet Graph Neural Network, using Beta wavelet basis functions as graph filters, effectively capturing both local and global structural information in graphs.

### bonGNN.py
Implements Bernstein polynomial-based Graph Neural Network, using Bernstein polynomials as graph filters, suitable for processing non-stationary graph signals.

### cheGNN.py
Implements Chebyshev polynomial-based Graph Neural Network, approximating spectral graph convolutions with Chebyshev polynomials for more efficient graph signal processing.

### dataset.py
Dataset loading and processing module, supporting various dataset formats.

### rewire/
Circuit rewiring module based on Ollivier-Ricci curvature. This module performs graph rewiring on circuit netlists to enhance robustness against logic locking attacks. Key features include:
- **Ollivier-Ricci Curvature**: Computes curvature metrics for graph edges
- **Circuit Support**: Supports various benchmark circuits (b11, b14, b17, c1355, c1908, c2670, c3540, c6288, etc.)
- **Data Processing**: Converts between different graph data formats and applies rewiring transformations

For detailed usage, see [rewire/README_netlists.md](rewire/README_netlists.md).

## Dependencies

### Core Model
- pytorch 1.9.0
- dgl 0.8.1
- sympy
- argparse
- sklearn

### Rewire Module
- pytorch
- torch-geometric
- networkx
- scipy
- numpy

## How to Run

### Main Model (GOLD)

```bash
python main.py --dataset amazon --train_ratio 0.4 --hid_dim 64 \
--order 2 --homo 1 --epoch 100 --run 1 --lockedName TRLL --circuit c1355 \
--test_start 449 --val_start 400 --lenss 450 --dataSetType data
```

#### Parameter Descriptions

- `--dataset`: Dataset name, e.g., amazon, yelp
- `--train_ratio`: Training set ratio, default 0.4
- `--hid_dim`: Hidden layer dimension, default 64
- `--order`: Order C parameter in Beta Wavelet, default 2
- `--homo`: 1 for BWGNN(Homo), 0 for BWGNN(Hetero)
- `--epoch`: Maximum number of training epochs, default 100
- `--run`: Number of runs, default 1
- `--lockedName`: Name of the lock, default "TRLL"
- `--circuit`: Circuit name, default "c1355"
- `--test_start`: Starting index for test set, required
- `--val_start`: Starting index for validation set, required
- `--lenss`: Total data length, required
- `--dataSetType`: Dataset type directory, required


### Circuit Rewiring (Rewire Module)

```bash
cd rewire
python run_netlists_rewire.py \
    --circuit_type c1355 \
    --num_circuits 10 \
    --num_iterations 3 \
    --borf_batch_add 20 \
    --borf_batch_remove 10
```

#### Rewire Parameter Descriptions

- `--circuit_type`: Circuit type name (e.g., b11, b14, c1355, c1908, c2670, c3540, c6288)
- `--num_circuits`: Number of circuit instances to process (starting from 0)
- `--num_iterations`: Number of rewiring iterations
- `--borf_batch_add`: Number of edges to add per iteration
- `--borf_batch_remove`: Number of edges to remove per iteration
- `--netlists_dir`: Input netlists directory (default: `netlists`)
- `--output_dir`: Output directory (default: `graphData/rewired`)

## Acknowledgement

The band-pass filter implementation in GOLD utilizes the beta-wavelet codes from the following paper: J. Tang, J. Li, Z. Gao, and J. Li, Rethinking graph neural networks for anomaly detection, ICML 2022. Thanks to the authors for making their code available.

