import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
df = pd.read_csv('orbital_data.csv')
X = df.drop('stable', axis=1) # Features (Starting conditions)
y = df['stable']              # Target (1 = Stayed, 0 = Escaped)

# 2. Split into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Create the "Brain" (Neural Network)
# 2 hidden layers with 10 neurons each
model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)

print("Training the AI on Physics data...")
model.fit(X_train, y_train)

# 4. Test it
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)

print(f"AI Accuracy: {score * 100:.2f}%")

# 5. Make a manual prediction
sample_start = X_test.iloc[0].values.reshape(1, -1)
guess = model.predict(sample_start)
print(f"For a new random launch, the AI predicts: {'STABLE' if guess[0]==1 else 'UNSTABLE'}")

























