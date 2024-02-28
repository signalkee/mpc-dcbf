from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Load the generated data
df = pd.read_csv('simulation_data.csv')

# Extract features and labels
X = df[['x_start', 'y_start', 'theta_start']].values
y = df['collision'].astype(int).values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train an SVM classifier
learned_cbf = SVC(kernel='rbf', C=1.0, gamma='auto')
learned_cbf.fit(X_train_scaled, y_train)

# Evaluate the classifier
y_pred = learned_cbf.predict(X_test_scaled)
print("Accuracy on Test Set: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


from joblib import dump

# Save the trained model
dump(learned_cbf, 'svm_cbf_classifier.joblib')
