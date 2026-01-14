This directory contains ETL workflows for pulling Napistu-relevant summaries out of expression foundation models.

Most models have their own environment setup process (due to cross-model version incompatibilities). For details see below.

These notebook's functions are organized in the *Napistu-Torch*'s `load` subpackage.
- foundation_model_etl.py contains *process_* functions which format models weights as  `napistu_torch.load.foundation_model.FoundationModel` instances
- model-specific dependencies are handled with decorators and runtime requirements


## Foundation model ingestion

This notebook demonstrates how to ingest the gene embeddings and attention patterns from multiple Virtual Cell Foundation Models into a common format. The models considered here are:

- AIDOcell (3 versions)
- scPRINT (3 versions)
- scFoundation ()
- scGPT (scGPT.ipynb)





## scGPT

### Download model

Following the instructions on the [scGPT](https://github.com/bowang-lab/scGPT?tab=readme-ov-file#pretrained-scgpt-model-zoo) Github, download the model for GoogleDrive to a local directory (it should be named `scGPT_bc`)

### Setup Python environment

**Quick install (pip):**
```bash
pip install scgpt wandb gseapy ipykernel "napistu-torch>=0.3.8"
```

**Note:** If you encounter issues with `torchtext` compatibility (common with PyTorch), use the conda/mamba setup below.

**Detailed setup (conda/mamba - recommended for torchtext compatibility):**

venv's didn't work because the torchtext wheel is not compatible with torch, so conda/mamba is recommended. 

```bash
# conda setup since torchtext is brittle
# brew install miniforge
# conda init zsh
# mamba shell init --shell zsh
mamba create -n scgpt python=3.11
mamba activate scgpt

# Install PyTorch and torchtext (from conda-forge, NOT Anaconda)
mamba install pytorch torchtext==0.18.0 -c pytorch -c conda-forge

# Install the rest
pip install scgpt wandb gseapy ipykernel "napistu-torch>=0.3.8"
python -m ipykernel install --user --name=scGPT 
```

## GenBio-AI (AIDO.Cell / scFoundation)

https://github.com/genbio-ai/ModelGenerator is a package for downloading and working with the AIDO models

```bash
uv venv .genbio
source .genbio/bin/activate

uv pip install modelgenerator ipykernel "napistu-torch>=0.3.8"
python -m ipykernel install --user --name=genbio --display-name="GenBio-AI (scFoundation/AIDOCell)"
```

Here,
- "modelgenerator" is a package from *Genbio-AI* which is used by `AIDOCell` and `scFoundation`

## scPRINT

https://virtualcellmodels.cziscience.com/model/scprint
https://github.com/cantinilab/scPRINT

**Required dependencies:**
- `scprint` - Core scPRINT package
- `bionty` - For gene annotation lookups (optional but recommended)

```bash
uv venv .scprint --python 3.11
source .scprint/bin/activate

uv pip install scprint ipykernel "napistu-torch>=0.3.8"
python -m ipykernel install --user --name=scPRINT

# Initialize lamin database for gene annotations (optional but recommended)
lamin init --storage data/lamin_db --name scPRINT_lamin --modules bionty
```

**Note:** The `populate_lamin_db()` function requires `bionty`. Gene annotations will still work without it, but gene symbols may not be populated.

Here,
- "scprint" is an `scPRINT`-specific package
- The `lamin` call initializes a database for gene annotations which is needed to add Ensembl gene IDs to `scPRINT` results.


## Notes on version incompatibilities

Ideally, these models could be processed in a common environment, but there are incompatible dependencies that prevent this. 

First, scGPT depends on `torchtext`, which is most reliably set up using mamba/conda due to compatibility issues with PyTorch when installed via pip. However, the other models (scPRINT and AIDOCell/scFoundation) work fine with simple venv/pip installations and don't require conda/mamba. 

Second, there is a fundamental conflict between scPRINT and AIDOCell/scFoundation: AIDOCell requires `modelgenerator==0.1.2` (a specific pinned version), but when this version is installed alongside scPRINT's dependencies (`scprint`, `scdataloader`, `bionty`, `lamindb`), it causes Django model registration errors. Specifically, bionty's Django models conflict with lamindb's base classes, resulting in a `FieldError` where the `Source` model's `id` field clashes with the `Record` base class. This error occurs even during `lamin init`, indicating it's a fundamental package incompatibility rather than a database schema issue. Pinning to `modelgenerator==0.1.2` breaks scPRINT's lamin database setup, while using a different version breaks AIDOCell. 

Due to these issues, its easier to maintain a separate environments for each model's initial ingestion. Because once ingested working with core `FoundationModel` data no longer requires `scgpt`, `scprint`, `lamindb`, etc.

