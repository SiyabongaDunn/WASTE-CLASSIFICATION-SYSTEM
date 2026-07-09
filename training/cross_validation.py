import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import tensorflow as tf

from dataset.loader import create_dataset
from training.callbacks import get_callbacks


def cross_validate_model(
    model_builder,
    model_name,
    train_paths,
    train_labels,
    batch_size,
    data_dir,
    image_size,
    num_classes,
    epochs=20,
    is_vit=False,
    num_folds=5,
    early_stopping_patience=5,
    reduce_lr_patience=3,
    reduce_lr_factor=0.5
):
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    fold_metrics = {
        'macro_f1': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'latency_ms': [],
        'history': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_paths, train_labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{num_folds} - {model_name}")
        print(f"{'='*50}")
        
        # Split data
        fold_train_paths = [train_paths[i] for i in train_idx]
        fold_train_labels = [train_labels[i] for i in train_idx]
        fold_val_paths = [train_paths[i] for i in val_idx]
        fold_val_labels = [train_labels[i] for i in val_idx]
        
        print(f"Train: {len(fold_train_paths)}, Val: {len(fold_val_paths)}")
        
        # Create datasets
        train_ds = create_dataset(
            fold_train_paths, fold_train_labels, batch_size,
            data_dir, image_size, num_classes,
            augment=True, is_vit=is_vit
        )
        val_ds = create_dataset(
            fold_val_paths, fold_val_labels, batch_size,
            data_dir, image_size, num_classes,
            augment=False, is_vit=is_vit
        )
        
        # Build model
        model = model_builder()
        
        # Callbacks
        callbacks = get_callbacks(
            patience_early_stop=early_stopping_patience,
            patience_reduce_lr=reduce_lr_patience,
            reduce_lr_factor=reduce_lr_factor
        )
        
        # Measure inference latency
        def measure_latency(model, dataset, num_samples=100):
            latencies = []
            for images, _ in dataset.take(1):
                images = images[:min(num_samples, len(images))]
                for i in range(len(images)):
                    single_img = images[i:i+1]
                    start = time.perf_counter()
                    _ = model.predict(single_img, verbose=0)
                    latencies.append((time.perf_counter() - start) * 1000)
            return np.mean(latencies)
        
        # Train
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        y_true, y_pred = [], []
        for images, labels in val_ds:
            preds = model.predict(images, verbose=0)
            if is_vit:
                y_true.extend(labels.numpy())
                y_pred.extend(np.argmax(preds, axis=1))
            else:
                y_true.extend(np.argmax(labels.numpy(), axis=1))
                y_pred.extend(np.argmax(preds, axis=1))
        
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        latency = measure_latency(model, val_ds)
        
        fold_metrics['macro_f1'].append(macro_f1)
        fold_metrics['accuracy'].append(accuracy)
        fold_metrics['precision'].append(precision)
        fold_metrics['recall'].append(recall)
        fold_metrics['latency_ms'].append(latency)
        fold_metrics['history'].append(history)
        
        print(f"Fold {fold+1} - F1: {macro_f1:.4f}, Acc: {accuracy:.4f}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"{model_name} - {num_folds}-Fold CV Results")
    print(f"{'='*50}")
    print(f"Macro F1:  {np.mean(fold_metrics['macro_f1']):.4f} ± {np.std(fold_metrics['macro_f1']):.4f}")
    print(f"Accuracy:  {np.mean(fold_metrics['accuracy']):.4f} ± {np.std(fold_metrics['accuracy']):.4f}")
    print(f"Precision: {np.mean(fold_metrics['precision']):.4f} ± {np.std(fold_metrics['precision']):.4f}")
    print(f"Recall:    {np.mean(fold_metrics['recall']):.4f} ± {np.std(fold_metrics['recall']):.4f}")
    print(f"Latency:   {np.mean(fold_metrics['latency_ms']):.2f} ± {np.std(fold_metrics['latency_ms']):.2f} ms")
    
    return fold_metrics