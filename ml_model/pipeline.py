# see the readme for more information on how to run this pipeline!

import os
import sys
import yaml
import argparse
from pathlib import Path

def setup_environment(config):
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    gpu_id = config['gpu']['cuda_visible_devices']
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"Environment configured: GPU={gpu_id}")


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_adult_training(config):
    from adult_train import train_adult_model
    
    print("\n" + "="*60)
    print("STAGE 1: Training Adult Ear Model (55 landmarks)")
    print("="*60)
    
    train_adult_model(config['adult_training'])
    print("✓ Adult training completed")


def run_55_to_22_transfer(config):
    import torch
    import adult_model
    
    print("\n" + "="*60)
    print("STAGE 2: Transferring 55 to 22 Landmarks")
    print("="*60)
    
    cfg = config['transfer_55_to_22']
    
    old_state = torch.load(cfg['input_checkpoint'], map_location="cpu")
    
    new_model = adult_model.get_model(cfg['num_landmarks_new'], cfg['num_stages'])
    new_state = new_model.state_dict()
    
    old_indices = [cfg['landmark_mapping'][i] for i in range(cfg['num_landmarks_new'])]
    
    transferred = []
    kept_as_is = []
    
    for key in old_state:
        if key not in new_state:
            continue
        
        old_tensor = old_state[key]
        new_tensor = new_state[key]
        
        if old_tensor.shape == new_tensor.shape:
            new_state[key] = old_tensor.clone()
            kept_as_is.append(key)
        
        # Output layer: landmark dimension in shape[0]
        elif old_tensor.shape[0] == cfg['num_landmarks_old'] and new_tensor.shape[0] == cfg['num_landmarks_new']:
            new_state[key] = old_tensor[old_indices].clone()
            transferred.append(key)
        
        # Input layer: landmark dimension in shape[1]
        elif old_tensor.shape[1] == cfg['num_landmarks_old'] and new_tensor.shape[1] == cfg['num_landmarks_new']:
            n_features = old_tensor.shape[1] - cfg['num_landmarks_old']
            feature_weights = old_tensor[:, :n_features, ...].clone()
            landmark_weights = old_tensor[:, n_features:, ...][:, old_indices, ...].clone()
            new_state[key] = torch.cat([feature_weights, landmark_weights], dim=1)
            transferred.append(key)
    
    new_model.load_state_dict(new_state)
    torch.save(new_model.state_dict(), cfg['output_checkpoint'])
    
    print(f"Kept as-is: {len(kept_as_is)} tensors")
    print(f"Transferred (mapped): {len(transferred)} tensors")
    print(f"✓ Saved to {cfg['output_checkpoint']}")


def run_22_to_23_transfer(config):
    import torch
    import adult_model
    
    print("\n" + "="*60)
    print("STAGE 3: Expanding from 22 to 23 Landmarks")
    print("="*60)
    
    cfg = config['transfer_22_to_23']
    
    old_state = torch.load(cfg['input_checkpoint'], map_location="cpu")
    
    new_model = adult_model.get_model(cfg['num_landmarks_new'], cfg['num_stages'])
    new_state = new_model.state_dict()
    
    transferred = []
    kept_as_is = []
    
    for key in old_state:
        if key not in new_state:
            continue
        
        old_tensor = old_state[key]
        new_tensor = new_state[key]
        
        if old_tensor.shape == new_tensor.shape:
            new_state[key] = old_tensor.clone()
            kept_as_is.append(key)
        
        # Output layer: copy old rows, leave 23rd as random init
        elif old_tensor.shape[0] == cfg['num_landmarks_old'] and new_tensor.shape[0] == cfg['num_landmarks_new']:
            new_state[key][:cfg['num_landmarks_old']] = old_tensor.clone()
            transferred.append(key)
        
        # Input layer: copy old channels, leave 23rd as random init
        elif old_tensor.shape[1] == cfg['num_landmarks_old'] and new_tensor.shape[1] == cfg['num_landmarks_new']:
            new_state[key][:, :cfg['num_landmarks_old'], ...] = old_tensor.clone()
            transferred.append(key)
    
    new_model.load_state_dict(new_state)
    torch.save(new_model.state_dict(), cfg['output_checkpoint'])
    
    print(f"Kept as-is: {len(kept_as_is)} tensors")
    print(f"Transferred (expanded): {len(transferred)} tensors")
    print(f"✓ Saved to {cfg['output_checkpoint']}")


def run_infant_training(config):
    from infant_train import train_infant_model
    
    print("\n" + "="*60)
    print("STAGE 4: Training Infant Ear Model (23 landmarks)")
    print("="*60)
    
    train_infant_model(config['infant_training'])
    print("✓ Infant training completed")


def main():
    parser = argparse.ArgumentParser(
        description='Ear Landmark Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--stages', nargs='+',
                        choices=['adult', 'transfer_55_22', 'transfer_22_23', 'infant'],
                        help='Specific stages to run (default: all enabled stages)')
    parser.add_argument('--skip', nargs='+',
                        choices=['adult', 'transfer_55_22', 'transfer_22_23', 'infant'],
                        help='Stages to skip')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found")
        sys.exit(1)
    
    config = load_config(args.config)
    
    setup_environment(config)
    
    stage_map = {
        'adult': ('adult_training', run_adult_training),
        'transfer_55_22': ('transfer_55_to_22', run_55_to_22_transfer),
        'transfer_22_23': ('transfer_22_to_23', run_22_to_23_transfer),
        'infant': ('infant_training', run_infant_training),
    }
    
    if args.stages:
        stages_to_run = args.stages
    else:
        stages_to_run = [
            stage for stage, (cfg_key, _) in stage_map.items()
            if config[cfg_key]['enabled']
        ]
    
    if args.skip:
        stages_to_run = [s for s in stages_to_run if s not in args.skip]
    
    if not stages_to_run:
        print("No stages to run. Check your configuration or command-line arguments.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("EAR LANDMARK TRAINING PIPELINE")
    print("="*60)
    print(f"Stages to execute: {', '.join(stages_to_run)}")
    print("="*60)
    
    for stage in stages_to_run:
        cfg_key, run_func = stage_map[stage]
        try:
            run_func(config)
        except Exception as e:
            print(f"\n✗ Error in stage '{stage}': {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()
