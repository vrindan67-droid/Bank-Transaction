import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("bank_transactions_dataset final.xlsx")

# Display first 5 rows
print(df.head())

# Dataset information
print(df.info())

# Fraud count
print(df['IsFraud'].value_counts())

# Bar chart
df['IsFraud'].value_counts().plot(kind='bar')
plt.title("Fraud vs Non-Fraud")
plt.show()

# Import ML libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Features and target
X = df[['Amount']]
y = df['IsFraud']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Test new transaction
new_transaction = pd.DataFrame([[15000]], columns=['Amount'])

result = model.predict(new_transaction)

if result[0] == 1:
    print("Fraud")
else:
    print("Not Fraud")