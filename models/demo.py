import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter

# --- STEP 1: DEFINE YOUR EXISTING SPLIT ---

# 1A. Your 2022 Training Dataset (Used for Cross-Validation)
train_classes_2022 = {
    'Glass': 401, 'Paper': 474, 'Plastic': 388, 
    'Metal': 328, 'Cardboard': 321, 'Trash': 110
}
X_train_2022 = []
y_train_2022 = []
for name, count in train_classes_2022.items():
    X_train_2022.extend([f"train/path/{name}_{i}.jpg" for i in range(count)])
    y_train_2022.extend([name] * count)

X_train_2022 = np.array(X_train_2022)
y_train_2022 = np.array(y_train_2022)

# --- STEP 2: RUN STRATIFIED 5-FOLD ONLY ON THE 2022 IMAGES ---

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# The library only splits the 2022 training images
for fold, (cv_train_idx, cv_val_idx) in enumerate(skf.split(X_train_2022, y_train_2022)):
    print(f"\n--- CV FOLD {fold + 1} ---")
    
    # Slice the 2022 data into current training and validation sets
    X_cv_train = X_train_2022[cv_train_idx]  # ~1617 images
    y_cv_train = y_train_2022[cv_train_idx]
    
    X_cv_val = X_train_2022[cv_val_idx]      # ~405 images
    y_cv_val = y_train_2022[cv_val_idx]
    
    print(f"Training on:   {len(X_cv_train)} images")
    print(f"Validating on: {len(X_cv_val)} images")
    
    # --- TRAINING HAPPENS HERE ---
    # model.fit(X_cv_train, y_cv_train)
    # val_score = model.evaluate(X_cv_val, y_cv_val)


# --- STEP 3: FINAL EVALUATION ---
print("\n" + "="*40)
print("CROSS VALIDATION FINISHED!")

