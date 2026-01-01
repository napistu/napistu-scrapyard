```bash
uv venv --python 3.11
source .venv/bin/activate

uv pip install napistu
uv pip install torch==2.8.0
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
uv pip install 'napistu-torch[pyg]'
uv pip install ipykernel
python -m ipykernel install --name napistu_torch_scrapyard --display-name "Napistu - PyTorch scrapyard" --user
```
