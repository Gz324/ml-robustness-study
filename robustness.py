import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Function to add noise
def add_noise(X, noise_level):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

# Noise levels
noise_levels = [0, 0.1, 0.2, 0.3, 0.5]

lr_scores = []
rf_scores = []

for noise in noise_levels:
    X_noisy = add_noise(X, noise)

    X_train, X_test, y_train, y_test = train_test_split(
        X_noisy, y, test_size=0.2, random_state=42
    )

    # Logistic Regression
    lr = LogisticRegression(max_iter=10000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_scores.append(accuracy_score(y_test, lr_pred))

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_scores.append(accuracy_score(y_test, rf_pred))

# Plot results
plt.figure()
plt.plot(noise_levels, lr_scores, marker='o', label='Logistic Regression')
plt.plot(noise_levels, rf_scores, marker='o', label='Random Forest')

plt.xlabel("Noise Level")
plt.ylabel("Accuracy")
plt.title("Model Robustness under Noisy Data")
plt.legend()

plt.savefig("results.png")
plt.show()

# Save results
df = pd.DataFrame({
    "Noise Level": noise_levels,
    "Logistic Regression": lr_scores,
    "Random Forest": rf_scores
})

df.to_csv("results.csv", index=False)

print(df)
