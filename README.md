# PFS-GNN-bipartite

The data are too large for github. Only two example graphs (one for each case) are uploaded here.

To run the code, choose one of the config file, then run "python train.py". The graphs are constructed for gpu so you need a gpu to use them.

# PyTorch Installation

Create a virtual environment: 

```
python3 -m venv .venv
source .venv/bin/activate
```

Install a CPU-only version of PyTorch: 
```
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

In a Python interpreter, execute `import torch; print(torch.__version__)` to note your version of PyTorch. Then install the lower-level packages: 

```
pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.x.0+cpu.html

pip install torch-sparse \
  -f https://data.pyg.org/whl/torch-2.x.0+cpu.html

pip install torch-cluster \
  -f https://data.pyg.org/whl/torch-2.x.0+cpu.html

pip install torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.x.0+cpu.html

pip install torch-geometric
```

where `2.x.0` should be your version of PyTorch, e.g. `2.1.0`. 

# Brief Descriptions of the Files

gnn.py : Define the bipartite graph and the GNN.

train.py : The main file to run the code.

/config : Contain the config files for both cases. To use them, copy the config file to the same directory as the train.py and rename it as config.py

brute.py : GD method.

construct_graph.py : Construct the graph. The connectivity and the initialization of the graph will depend on the problem.

load.py: Load the graphs. Currently we only support batchsize = 1. To do a batched training, merge multiple graphs into a single one in the graph construction step.


