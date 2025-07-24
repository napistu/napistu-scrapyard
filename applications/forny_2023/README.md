# Forny et al., 2023 Multiomics of Methylmalonyl-CoA Mutase Deficiency

```bash
uv venv --python 3.11
uv pip install napistu
# for excel support
uv pip install openpyxl

# playing around with R <> python data formats
uv pip install mudata
# DESeq2 for Python with anndata
# uv pip install pydeseq2
# regression
uv pip install scikit-learn
# EDA
uv pip install seaborn

# multiomic factor analysis
uv pip install muon
uv pip install mofapy2

uv pip install ipykernel
python -m ipykernel install --user --name=forny-2023
```

## Environment Setup

```bash
uv venv --python 3.11
uv pip install napistu[scverse]

# some functionality is bundled into a personal utility package https://github.com/shackett/shackett-utils
uv pip install shackett-utils[all]
```

## Analyses

### 1 - `forny.ipynb` - Summarizes molecular changes with disease



### 2 - `napistu_forny.ipynb` - Find molecular subnetworks enriched in disease



### 3 - `network_vis.qmd` - Visualizes disease subgraphs





```bash
quarto render network_vis.qmd --verbose
```