# %%
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
# Number of patients
num_patients = 10000

# Generate data
data = {
    'Age': np.random.randint(20, 80, num_patients),
    'BMI': np.random.uniform(18, 40, num_patients),
    'GlucoseLevel': np.random.uniform(70, 200, num_patients),
    'Insulin': np.random.uniform(2, 300, num_patients),
    'BloodPressure': np.random.uniform(60, 180, num_patients),
    'Diabetes': np.random.choice([0, 1], num_patients, p=[0.7, 0.3])  # 70% no diabetes, 30% diabetes
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Introduce missing values randomly
for col in ['BMI', 'GlucoseLevel', 'Insulin', 'BloodPressure']:
    df.loc[np.random.choice(df.index, size=40, replace=False), col] = np.nan  # 40 NaN values per column

# Display first few rows
print(df.head())


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot for Glucose Level
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['GlucoseLevel'])
plt.title("Boxplot of Glucose Level")
plt.show()

# Boxplot for Insulin
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['Insulin'])
plt.title("Boxplot of Insulin Levels")
plt.show()


# %%
from sklearn.preprocessing import MinMaxScaler

# Fill NaN values with median
df.fillna(df.median(), inplace=True)

# Normalize features
scaler = MinMaxScaler()
df[['Age', 'BMI', 'GlucoseLevel', 'Insulin', 'BloodPressure']] = scaler.fit_transform(
    df[['Age', 'BMI', 'GlucoseLevel', 'Insulin', 'BloodPressure']]
)

print("\nData after normalization:\n", df.head())


# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Splitting dataset into features (X) and target (y)
X = df.drop(columns=['Diabetes'])
y = df['Diabetes']

# Split data into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_logreg = log_reg.predict(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Model Evaluation
print("\n🔹 Logistic Regression Results 🔹")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Classification Report:\n", classification_report(y_test, y_pred_logreg))

print("\n🔹 Random Forest Results 🔹")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))


# %%
import seaborn as sns

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()



