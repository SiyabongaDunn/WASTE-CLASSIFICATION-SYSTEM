import os
import pickle
import pandas as pd
import numpy as np


def save_results(vit_results, output_dir="results"):

    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame
    results_df = pd.DataFrame({
        'Model': ['ViT'],
        'Macro F1 (mean±std)': [
            f"{np.mean(vit_results['macro_f1']):.4f} ± {np.std(vit_results['macro_f1']):.4f}"
        ],
        'Accuracy': [
            f"{np.mean(vit_results['accuracy']):.4f}"
        ],
        'Latency (ms)': [
            f"{np.mean(vit_results['latency_ms']):.2f}"
        ]
    })
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'phase1_cv_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n Results saved to '{csv_path}'")
    
    # Save raw metrics
    phase1_raw = {
        'vit': {
            'macro_f1': vit_results['macro_f1'],
            'accuracy': vit_results['accuracy'],
            'precision': vit_results['precision'],
            'recall': vit_results['recall'],
            'latency_ms': vit_results['latency_ms']
        }
    }
    
    pkl_path = os.path.join(output_dir, 'phase1_raw_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(phase1_raw, f)
    print(f" Raw results saved to '{pkl_path}'")
    
    # Print summary
    print("\n" + "="*60)
    print("PHASE 1 - FINAL RESULTS (5-Fold Cross-Validation)")
    print("="*60)
    print(results_df.to_string(index=False))
    
    return results_df