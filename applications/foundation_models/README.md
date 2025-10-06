This directory contains ETL workflows for pulling Napistu-relevant summaries out of expression foundation models.

Each model has a separate environment setup process, a model-specific utility .py file, and there is a shared .py file for common cross-model logic and definitions.


## scGPT

venv's didn't work because the torchtext wheel is not compatible with torch, so i went with conda. 

### Download model

Following the instructions on the [scGPT](https://github.com/bowang-lab/scGPT?tab=readme-ov-file#pretrained-scgpt-model-zoo) Github, download the model for GoogleDrive to a local directory (it should be named `scGPT_bc`)

### Setup Python environment

```bash
# conda setup since torchtext is brittle
brew install miniforge
conda init zsh
mamba create -n scgpt python=3.11
mamba activate scgpt

# Install PyTorch and torchtext (from conda-forge, NOT Anaconda)
mamba install pytorch torchtext==0.18.0 -c pytorch -c conda-forge

# Install the rest
pip install scgpt wandb gseapy ipykernel napistu
python -m ipykernel install --user --name=scGPT
```
