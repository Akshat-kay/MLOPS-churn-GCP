import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from google.cloud import storage
from tempfile import NamedTemporaryFile

# Data loading and preprocessing
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/telco-customer-churn.csv")
df = df.dropna()
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

# Feature engineering
y = df["Churn"]
X = pd.get_dummies(df.drop(["customerID", "Churn"], axis=1), drop_first=True)  # Avoid dummy trap

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Save model locally and to GCS
with NamedTemporaryFile(suffix='.joblib') as tmp:
    joblib.dump(model, tmp.name)
    
    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket("mlops-churn-bucket")
    blob = bucket.blob("models/model.joblib")
    blob.upload_from_filename(tmp.name)
    
print(f"Model saved to gs://mlops-churn-bucket/models/model.joblib")
print(f"Model accuracy: {model.score(X_test, y_test):.2f}")
