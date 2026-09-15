import os
import sys
import yaml
import argparse
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.loader import GarbageDataset
from models.vit import build_vit_model
from training.cross_validation import cross_validate_model
from evaluation.metrics import save_results


def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: 5-Fold CV for ViT")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger = setup_logging(config['paths']['logs_dir'])
    
    logger.info("="*60)
    logger.info("PHASE 1: 5-Fold Cross-Validation - ViT")
    logger.info("="*60)
    logger.info(f"Config: {args.config}")
    
    # Extract config values
    dataset_config = config['dataset']
    model_config = config['model']
    training_config = config['training']
    paths_config = config['paths']
    
    # Override if provided
    if args.epochs is not None:
        training_config['epochs'] = args.epochs
    if args.batch_size is not None:
        training_config['batch_size'] = args.batch_size
    
    logger.info(f"Image size: {dataset_config['image_size']}")
    logger.info(f"Batch size: {training_config['batch_size']}")
    logger.info(f"Epochs: {training_config['epochs']}")
    logger.info(f"Folds: {training_config['num_folds']}")
    
    # 1. Load dataset
    logger.info("\n" + "-"*40)
    logger.info("Loading dataset...")
    logger.info("-"*40)
    
    dataset = GarbageDataset(config)
    dataset.load_data().load_paths()
    train_paths, test_paths, train_labels, test_labels = dataset.split()
    dataset.save_test_set(test_paths, test_labels, paths_config['splits_file'])
    
    # 2. Define model builder
    def vit_model_builder():
        return build_vit_model(
            num_classes=dataset_config['num_classes'],
            image_size=dataset_config['image_size'],
            patch_size=model_config['patch_size'],
            embed_dim=model_config['embed_dim'],
            num_heads=model_config['num_heads'],
            mlp_dim=model_config['mlp_dim'],
            num_layers=model_config['num_layers'],
            learning_rate=model_config['learning_rate'],
            weight_decay=model_config['weight_decay']
        )
    
    # 3. Run cross-validation
    logger.info("\n" + "-"*40)
    logger.info("Running 5-Fold Cross-Validation...")
    logger.info("-"*40)
    
    vit_results = cross_validate_model(
        model_builder=vit_model_builder,
        model_name="ViT",
        train_paths=train_paths,
        train_labels=train_labels,
        batch_size=training_config['batch_size'],
        data_dir=dataset.data_dir,
        image_size=dataset_config['image_size'],
        num_classes=dataset_config['num_classes'],
        epochs=training_config['epochs'],
        is_vit=True,
        num_folds=training_config['num_folds'],
        early_stopping_patience=training_config['early_stopping_patience'],
        reduce_lr_patience=training_config['reduce_lr_patience'],
        reduce_lr_factor=training_config['reduce_lr_factor']
    )
    
    # 4. Save results
    logger.info("\n" + "-"*40)
    logger.info("Saving results...")
    logger.info("-"*40)
    
    results_df = save_results(vit_results, paths_config['results_dir'])
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 1 COMPLETE!")
    logger.info("="*60)
    logger.info(f"Results saved to: {paths_config['results_dir']}")
    
    return vit_results


if __name__ == "__main__":
    main()