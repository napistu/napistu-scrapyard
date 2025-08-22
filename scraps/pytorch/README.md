```bash
uv venv --python 3.11
source .venv/bin/activate

uv pip install napistu
uv pip install "git+https://github.com/napistu/napistu-torch.git[pyg]" 
uv pip install ipykernel
python -m ipykernel install --name napistu_torch_scrapyard --display-name "Napistu - PyTorch scrapyard" --user
```
