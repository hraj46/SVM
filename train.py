import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("Energy.csv")

# Drop missing values
df = df.dropna()

# Create binary label
df["label"] = df["Appliances"].apply(lambda x: "Normal" if x < 300 else "High")

# Downsampling
normal = df[df["label"] == "Normal"].sample(frac=0.5, random_state=42)
high = df[df["label"] == "High"]
df = pd.concat([normal, high])

# Features
features = ["T1", "RH_1", "T2", "RH_2", "T_out", "RH_out"]
X = df[features]
y = df["label"]

# Encode labels
y = y.map({"Normal": 0, "High": 1})

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# Save model + scaler
pickle.dump((model, scaler), open("model.pkl", "wb"))