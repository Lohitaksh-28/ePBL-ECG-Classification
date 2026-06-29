import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the Data
print("Loading dataset...")
# Ensure these CSV files are in the same directory as this script
train_data = pd.read_csv('mitbih_train.csv', header=None)
test_data = pd.read_csv('mitbih_test.csv', header=None)

# The last column (187) is the label. 0: Normal, 1-4: Abnormalities
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# 2. Signal Processing Visualization (Save this plot for your report)
plt.figure(figsize=(10, 5))
plt.plot(X_train[y_train == 0.0][0], label="Normal Beat (Class 0)", color='blue')
plt.plot(X_train[y_train == 1.0][0], label="Supraventricular Premature (Class 1)", color='red')
plt.title("ECG Signal Waveform Comparison")
plt.xlabel("Time (Samples)")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.grid(True)
plt.savefig("ecg_waveform_comparison.png") # Automatically saves screenshot
print("Saved ECG waveform plot as 'ecg_waveform_comparison.png'")
plt.show()

# 3. Train the Intelligent System (Classifier)
print("Training Random Forest Classifier... (This will take a few seconds)")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 4. Evaluate and Generate Results
print("Evaluating model...")
y_pred = rf_model.predict(X_test)

print("\n--- Classification Report ---")
report = classification_report(y_test, y_pred)
print(report)

# 5. Confusion Matrix Visualization (Save this plot for your report)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.savefig("confusion_matrix.png") # Automatically saves screenshot
print("Saved Confusion Matrix plot as 'confusion_matrix.png'")
plt.show()
