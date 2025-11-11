# TPA
개요 (추후 작성)

## Directory Structure
```
.
├── experiments/
│   ├── __init__.py
│   └── __main__.py   # Main execution script 
├── models/
│   ├── cnn_backbone.py
│   ├── lstm_backbone.py
│   ├── tpa_modules.py  # TPA Modules
│   ├── transformer_backbone.py
│   └── __init__.py
├── transition_generator/ # transition dataset generation script
│   ├── pamap2_transition_generator.py
│   ├── uci-har_transition_generator.py   
│   └── wisdm_transition_generator.py
├── utils/
│   ├── config.py       # Configuration file management
│   ├── dataset.py      # Dataset loading and processing script
│   ├── train.py        # Contains training logic
│   └── __init__.py
├── __init__.py
└── requirements.txt
```

## Download Datasets
**https://drive.google.com/drive/folders/1_SfYaRQ4B7xLBtiUS5BEAEq7rhuNRKa6**

## Requirements
All experiments and result validations for this project were completed using the Python 3.12.12 environment.
```
pip install -r requirements.txt
```

## Execution (Terminal Commands)


**Example Command:**
```
python -m experiments UCI CNN ./data_folder ./results
```


**Arguments:**

| Argument | Description | Possible Values |
| :--- | :--- | :--- |
| `[1] DATASET` | Target dataset type. | UCI, WISDM, PAMAP2 |
| `[2] BACKBONE` | Neural network backbone architecture. | CNN, BILSTM, TRANSFORMER |
| `[3] DATA_DIR` | Absolute or relative path to the folder containing the generated transition dataset folders (e.g., ./datasets/uci_datasets). | Path string |
| `[4] SAVE_DIR` | Directory where the JSON result file should be saved. | Path string |

## Citation

## Contact
Dahyun Kang: dadang6842@gmail.com

## License

## License
