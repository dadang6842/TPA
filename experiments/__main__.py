import os, json, sys
import numpy as np
import torch
import random
from utils import Config, load_and_preprocess_dataset, train_model, SEED
from models.tpa_modules import GAPModel, TPAModel, GatedTPAModel
from models.cnn_backbone import MultiPathCNN
from models.lstm_backbone import BiLSTMBackbone
from models.transformer_backbone import TransformerBackbone

def create_model_with_backbone(model_name: str, backbone_type: str, cfg: Config, in_ch: int, max_len: int):
    tpa_config = {
        'num_prototypes': cfg.tpa_num_prototypes, 
        'heads': cfg.tpa_heads,
        'dropout': cfg.tpa_dropout, 
        'temperature': cfg.tpa_temperature,
    }

    if backbone_type == "CNN":
        backbone = MultiPathCNN(in_ch=in_ch, d_model=cfg.d_model)
    elif backbone_type == "BILSTM":
        backbone = BiLSTMBackbone(in_ch=in_ch, d_model=cfg.d_model)
    elif backbone_type == "TRANSFORMER":
        backbone = TransformerBackbone(in_channels=in_ch, d_model=cfg.d_model, max_seq_len=max_len)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

    if model_name == "GAP":
        model = GAPModel(backbone, cfg.d_model, cfg.num_classes)
    elif model_name == "TPA":
        model = TPAModel(backbone, cfg.d_model, cfg.num_classes, tpa_config)
    elif model_name == "Gated-TPA":
        model = GatedTPAModel(backbone, cfg.d_model, cfg.num_classes, tpa_config)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model.to(cfg.device).float()

def get_dataset_config_and_list(dataset_type: str):
    cfg = Config(dataset_type=dataset_type)

    if dataset_type == 'UCI':
        cfg.num_classes = 6
        cfg.in_channels = 9
        cfg.max_seq_len = 128
        transition_list = ["STANDING_TO_SITTING", "STANDING_TO_LAYING", "SITTING_TO_LAYING", 
                           "SITTING_TO_STANDING", "LAYING_TO_SITTING", "LAYING_TO_STANDING",
                           "WALKING_TO_WALKING_UPSTAIRS", "WALKING_TO_WALKING_DOWNSTAIRS",
                           "WALKING_UPSTAIRS_TO_WALKING", "WALKING_DOWNSTAIRS_TO_WALKING", "STANDING_TO_WALKING", "WALKING_TO_STANDING"]
    elif dataset_type == 'WISDM':
        cfg.num_classes = 6
        cfg.in_channels = 3
        cfg.max_seq_len = 200
        transition_list = ["STANDING_TO_SITTING", "SITTING_TO_STANDING", "STANDING_TO_WALKING", 
                           "WALKING_TO_STANDING", "WALKING_TO_JOGGING", "JOGGING_TO_WALKING",
                           "WALKING_TO_UPSTAIRS", "WALKING_TO_DOWNSTAIRS", 
                           "UPSTAIRS_TO_WALKING", "DOWNSTAIRS_TO_WALKING"]
    elif dataset_type == 'PAMAP2':
        cfg.num_classes = 12
        cfg.in_channels = 27
        cfg.max_seq_len = 100
        transition_list = ['Standing_TO_Lying', 'Lying_TO_Standing', 'Standing_TO_Walking', 
                           'Walking_TO_Standing', 'Walking_TO_Running', 'Running_TO_Walking', 
                           'Walking_TO_Ascending_stairs', 'Walking_TO_Descending_stairs', 
                           'Ascending_stairs_TO_Walking', 'Descending_stairs_TO_Walking',
                           'Sitting_TO_Standing', 'Lying_TO_Sitting', 'Standing_TO_Lying', 'Lying_TO_Standing']
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Choose UCI, WISDM, or PAMAP2.")

    mix_pcts = [10, 20, 30, 40]
    datasets = ["ORIGINAL"]
    for transition in transition_list:
        for pct in mix_pcts:
            datasets.append(f"{transition}_{pct}{'PCT' if dataset_type == 'PAMAP2' else 'pct'}")
    
    return cfg, datasets

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python run_experiment.py <DATASET_TYPE> <BACKBONE_TYPE> <DATA_DIR> <SAVE_DIR>")
        print("Example: python run_experiment.py UCI BILSTM ./data_folder ./results")
        sys.exit(1)

    DATASET_TYPE = sys.argv[1].upper()
    BACKBONE_TYPE = sys.argv[2].upper()
    DATA_DIR_INPUT = sys.argv[3]
    SAVE_DIR_INPUT = sys.argv[4]
    
    if BACKBONE_TYPE not in ["CNN", "BILSTM", "TRANSFORMER"]:
        print(f"Unsupported backbone type: {BACKBONE_TYPE}. Choose CNN, BILSTM, or TRANSFORMER.")
        sys.exit(1)

    cfg, datasets = get_dataset_config_and_list(DATASET_TYPE)
    
    cfg.data_dir = DATA_DIR_INPUT
    cfg.save_dir = SAVE_DIR_INPUT
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    RESULTS_JSON_PATH = os.path.join(cfg.save_dir, f"{DATASET_TYPE.lower()}_{BACKBONE_TYPE.lower()}_results.json")
    
    print("="*80)
    print(f"UNIFIED EXPERIMENT: {DATASET_TYPE} - {BACKBONE_TYPE} BACKBONE")
    print(f"Total Datasets to Test: {len(datasets)}")
    print("="*80)

    all_results = []
    model_names = ["GAP", "TPA", "Gated-TPA"]

    for i, dataset_name in enumerate(datasets, 1):
        print(f"\n[Progress: {i}/{len(datasets)}]")
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {dataset_name}")
        print(f"{'='*80}")
        
        train_loader, val_loader, test_loader, in_ch, max_len = load_and_preprocess_dataset(
            cfg.data_dir, dataset_name, cfg.dataset_type, cfg
        )
        
        for model_name in model_names:
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            model = create_model_with_backbone(model_name, BACKBONE_TYPE, cfg, in_ch, max_len)
            
            results = train_model(model, train_loader, val_loader, test_loader, cfg, model_name)
            
            all_results.append({
                'Dataset_Type': DATASET_TYPE,
                'Backbone': BACKBONE_TYPE,
                'Model': model_name,
                'Dataset': dataset_name,
                **results
            })

    with open(RESULTS_JSON_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print(f"EXPERIMENT COMPLETE. Results saved to {RESULTS_JSON_PATH}")
    print("="*80)