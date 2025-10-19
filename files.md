prt_finance/
│
├── data/
│   ├── raw/
│   │   ├── RELIANCE.csv
│   │   ├── TCS.csv
│   │   └── NIFTY50.csv
│   ├── processed/
│   │   └── features.csv
│   └── README.md
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_vanilla_transformer_training.ipynb
│   ├── 03_prt_model_training.ipynb
│   └── 04_results_visualization.ipynb
│
├── src/
│   ├── __init__.py
│   ├── base_transformer.py                 # Vanilla Transformer (from scratch)
│   ├── prototype_routing_transformer.py    # PRT model (custom attention)
│   ├── data_loader.py                      # Data fetching & preprocessing utils
│   ├── train_utils.py                      # Training loop, evaluation, logging
│   ├── visualize.py                        # Charts & prototype visualization
│   └── config.py                           # Config file for hyperparameters
│
├── results/
│   ├── logs/
│   │   ├── base_transformer.log
│   │   └── prt_model.log
│   ├── models/
│   │   ├── base_transformer.pth
│   │   └── prt_model.pth
│   ├── plots/
│   │   ├── predictions_reliance.png
│   │   ├── prototypes_tsne.png
│   │   └── loss_curve.png
│   └── metrics.json
│
├── requirements.txt
├── README.md
└── main.py
