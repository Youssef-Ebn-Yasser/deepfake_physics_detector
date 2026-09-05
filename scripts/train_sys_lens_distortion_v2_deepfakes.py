import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

def load_data(manifest_path, base_dir, feature_key='f1_v2'):
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

def evaluate_dataset(model, name, manifest_path, base_dir, feature_key, scaler=None):
    print(f"\n--- Evaluating on {name} ---")
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return
        
    X_test, y_test = load_data(manifest_path, base_dir, feature_key)
    if len(X_test) == 0:
        print("No data found!")
        return
        
    if scaler is not None:
        X_test = scaler.transform(X_test)
        
    metrics = model.evaluate(X_test, y_test, verbose=0)
    print(f"Samples: {len(X_test)}")
    print(f"Accuracy: {metrics[1]:.4f}")
    print(f"ROC-AUC : {metrics[2]:.4f}")

def main():
    feature_name = 'Lens Distortion (V2)'
    feature_key = 'f1_v2'
    
    # Train on Deepfakes
    base_dir_train = os.path.join('DataSets', 'f1v2_features', 'deepfakes')
    manifest_path_train = os.path.join(base_dir_train, 'manifest.csv')
    
    if not os.path.exists(manifest_path_train):
        print(f"Error: {manifest_path_train} not found. Run extraction script first.")
        return
        
    print(f"Loading {feature_name} training features from Deepfakes...")
    X, y = load_data(manifest_path_train, base_dir_train, feature_key)
    
    if len(X) == 0:
        print("No data found in training set!")
        return
        
    # We evaluate locally on the training validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    print(f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples.")
    
    model = build_model(input_dim=X.shape[1])
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    
    # Internal validation eval
    evaluate_dataset(model, "Deepfakes (Internal Val)", manifest_path_train, base_dir_train, feature_key, scaler)
    
    # Zero-shot cross-dataset evaluation on Celeb-DF and FF++
    celeb_dir = os.path.join('DataSets', 'f1v2_features', 'celeb_df')
    evaluate_dataset(model, "Celeb-DF-v2 (Zero-shot)", os.path.join(celeb_dir, 'manifest.csv'), celeb_dir, feature_key, scaler)
    
    ff_dir = os.path.join('DataSets', 'f1v2_features', 'ff_plus_plus')
    evaluate_dataset(model, "FaceForensics++ (Zero-shot)", os.path.join(ff_dir, 'manifest.csv'), ff_dir, feature_key, scaler)

if __name__ == '__main__':
    main()
