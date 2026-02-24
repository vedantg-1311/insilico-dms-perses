# Barnase–Barstar in-silico DMS (Perses + OpenMM)

## Setup
1. Create environment:
   - `conda env create -f environment.yml`
   - `conda activate barnase-barstar-dms`

2. Inputs live in `data/raw/` and are never edited.

## Configs
- `configs/system.yaml`: inputs + chains + solvent box/ionic strength
- `configs/protocol.yaml`: NEQ protocol, platform, replicates

## Runs
All results go under `runs/barnase_barstar/<mutation>/<seed>/`.
Each run folder contains:
- resolved config
- metadata (env/platform)
- forward/reverse works
- logs
