import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

def load_data(manifest_path, base_dir, feature_key='f3'):
    X, y = [], []
    with open(manifest_path, 'r') as f:
        lines = f.readlines()[1:] # skip header
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        npz_rel_path, label = line.split(',')
        npz_full_path = os.path.join(base_dir, npz_rel_path)
        
        if os.path.exists(npz_full_path):
            data = np.load(npz_full_path)
            # Ensure the feature exists
            if feature_key in data:
                X.append(data[feature_key])
                y.append(int(label))
                
    return np.array(X), np.array(y)

def build_model(input_dim=64):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    feature_name = 'Biomechanics'
    feature_key = 'f3'
    
    base_dir = os.path.join('DataSets', 'deepfakes_feature')
    manifest_path = os.path.join(base_dir, 'manifest.csv')
    
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found. Run extraction script first.")
        return
        
    print(f"Loading {feature_name} features ({feature_key})...")
    X, y = load_data(manifest_path, base_dir, feature_key)
    
    if len(X) == 0:
        print("No data found!")
        return
        
    print(f"Loaded {len(X)} samples.")
    
    # Train test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    
    model = build_model(input_dim=X.shape[1])
    
    print(f"\n--- Training {feature_name} System ---")
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
    
    print(f"\n--- Testing {feature_name} System ---")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy for {feature_name} ({feature_key}): {test_acc:.4f}")

if __name__ == '__main__':
    main()
