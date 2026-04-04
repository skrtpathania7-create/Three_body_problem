import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Data 
df = pd.read_csv('orbital_data.csv')
X = df.drop('stable', axis=1)
y = df['stable']

print(f"Dataset loaded: {len(df)} simulations")
print(f"  Stable:   {y.sum()} ({100*y.mean():.1f}%)")
print(f"  Unstable: {(1-y).sum()} ({100*(1-y.mean()):.1f}%)")

#2. Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature Scalin
# Positions range -10 to 10, velocities -0.5 to 0.5
# Without scaling, the network ignores velocity features almost entirely
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use SAME scaler — never fit on test data

# 4. Neural Network 
# Was: (10, 10) — way too small for chaotic physics
# Now: (128, 64, 32) — deeper network that can learn non-linear patterns
model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=2000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)

# MLPClassifier doesn't support class_weight directly —
# we pass sample weights into fit() instead (same effect)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

print("\nTraining neural network...")
model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
print(f"Training stopped after {model.n_iter_} iterations (early stopping)")

# 5. Evaluate 
predictions = model.predict(X_test_scaled)
score = accuracy_score(y_test, predictions)

print(f"\n{'='*45}")
print(f"  MODEL ACCURACY: {score * 100:.2f}%")
print(f"{'='*45}")

print("\nDetailed breakdown:")
print(classification_report(y_test, predictions, target_names=['Unstable', 'Stable']))

cm = confusion_matrix(y_test, predictions)
print("Confusion matrix:")
print(f"  True Stable predicted Stable:     {cm[1][1]}")
print(f"  True Stable predicted Unstable:   {cm[1][0]}  (missed)")
print(f"  True Unstable predicted Unstable: {cm[0][0]}")
print(f"  True Unstable predicted Stable:   {cm[0][1]}  (false alarm)")

#6. Demo Prediction 
print("\n--- Live demo prediction ---")
sample = X_test.iloc[0].values.reshape(1, -1)
sample_scaled = scaler.transform(sample)
guess = model.predict(sample_scaled)
confidence = model.predict_proba(sample_scaled)[0]

print(f"Initial conditions (first 4 of 12 features):")
print(f"  Body 1: x={sample[0][0]:.2f}, y={sample[0][1]:.2f}")
print(f"  Body 2: x={sample[0][2]:.2f}, y={sample[0][3]:.2f}")
result = 'STABLE' if guess[0] == 1 else 'UNSTABLE'
conf_pct = max(confidence) * 100
print(f"\nPrediction: {result}  (confidence: {conf_pct:.1f}%)")















