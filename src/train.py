import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from google.cloud import storage

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/telco-customer-churn.csv")
df = df.dropna()
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

y = df["Churn"]
X = pd.get_dummies(df.drop(["customerID", "Churn"], axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "model.joblib")

client = storage.Client()
bucket = client.bucket("mlops-churn-bucket")
bucket.blob("models/model.joblib").upload_from_filename("model.joblib")

