# 10_623_final_project


To make virtual environment for dependenceies:
```
python -m venv ./venv
source ./venv/bin/activate
```

To install developmer requirements into active environment (e.g., for training and making dataset)
```
pip install -e .[dev]
```

The project contains several scripts which are built at CLI commands:
- optimize_molecules: Runs energy minimization on the dataset using MACE
- train_model: Trains model given config and dataset
    - `train_model --config="./training/configs/config.yml"`


