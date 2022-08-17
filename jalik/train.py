import mlflow
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# setup mlflow for this experiment
mlflow.set_tracking_uri('https://community.mlflow.deploif.ai')
mlflow.set_experiment("Deploifai/Lichess/LichessJalik")

train = pd.read_csv('LichessDataset/train.csv')
test = pd.read_csv('LichessDataset/test.csv')

cols_to_drop = ['opening_eco']
train = train.drop(columns=cols_to_drop)
test = test.drop(columns=cols_to_drop)

X_train, y_train = train.drop(columns='winner'), train['winner']
X_test, y_test = test.drop(columns='winner'), test['winner']

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(preds, y_test)
print(accuracy)

dirname = os.path.dirname(os.path.realpath(__file__))
print(dirname)
dump(model, os.path.join(dirname, "artifacts", "model"))

with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", accuracy)
