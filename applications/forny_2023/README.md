# Forny et al., 2023 Multiomics of Methylmalonyl-CoA Mutase Deficiency

## Environment Setup

```bash
uv venv --python 3.11
source .venv/bin/activate

uv pip install napistu'[scverse]'
# some functionality is bundled into a personal utility package https://github.com/shackett/shackett-utils
uv pip install "git+https://github.com/shackett/shackett-utils.git[all]" 
# for excel support in pandas
uv pip install openpyxl
# imputation
uv pip install scikit-learn
# multiomic factor analysis
uv pip install mofapy2
# ipynb support
uv pip install ipykernel
python -m ipykernel install --user --name=forny-2023
```

## Analyses

### 1 - `forny.ipynb` - Summarizes molecular changes with disease



### 2 - `napistu_forny.ipynb` - Find molecular subnetworks enriched in disease



### 3 - `network_vis.qmd` - Visualizes disease subgraphs





```bash
quarto render network_vis.qmd --verbose
```