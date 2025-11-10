# Transition Dataset Generator for HAR Benchmarks

This folder contains the dataset generation scripts used to create synthetic transition segments for **Human Activity Recognition (HAR)** benchmark datasets (UCI-HAR, WISDM, and PAMAP2) using the Sequence Splitting method.


***

## Reproducibility Notice (Random Seed)

The original results presented in the paper were obtained using data generated without a fixed random seed.

To guarantee that the datasets created by these scripts are identical across all executions, the generation process is fixed with 'seed=42'.

To reproduce the quantitative results(performance metrics) stated in the final paper, please use the **pre-generated original datasets** provided via the download link below.

**https://drive.google.com/drive/folders/1_SfYaRQ4B7xLBtiUS5BEAEq7rhuNRKa6?usp=sharing**

***

## Execution (Terminal Commands)

Run the scripts from your terminal, providing the path to the original dataset file/folder, the desired output directory, and the generation parameters.

**Example Command:**
```
python uci-har_transition_generator.py ./UCI_HAR_Dataset ./uci_datasets 0.10 42
```


**Arguments:**

| Argument | Description | Default Value (if omitted) |
| :--- | :--- | :--- |
| `[1] DATA_PATH` | Path to the original dataset file/folder. | (Required) |
| `[2] OUTPUT_DIR` | Directory to save the generated datasets. | `./{Dataset}_datasets` |
| `[3] AUG_RATIO` | Ratio of transition segments to augment (0.01 to 0.99). | `0.10` |
| `[4] SEED` | Random seed for generation. | `42` |
