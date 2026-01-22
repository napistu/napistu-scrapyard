To run DiffDock, build the image with:

```zsh
./build_diffdock.sh
```

Then you can use it with:

```zsh
docker run --rm \
  -v $(pwd):/data:ro \
  -v $(pwd)/diffdock_results:/results:rw \
  diffdock:cpu \
  --protein_path /data/1CRN.pdb \
  --ligand_description "CC(=O)C(=O)[O-]" \
  --out_dir /results \
  --samples_per_complex 10
```

Protein path can either be a PDB ID or an absolute path within the Docker container.