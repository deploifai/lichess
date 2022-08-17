#!/usr/bin/env python
import mlflow
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# only if running an experiment outside deploifai (this must come before setting mlflow tracking uri and experiment)
# os.environ["MLFLOW_TRACKING_USERNAME"] = "Deploifai/Lichess/LichessRichard"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "drat-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImNsNnE5b20wczM3MzE2M3MzZXE1NDFjMmNuIiwiaWF0IjoxNjYwNzI0MDgyLCJleHAiOjE2OTIyODE2ODJ9.13eh6Pvne8tvCyPsxTFLUhNDxqzAOwzknGu2rDlRhVE"

# setup mlflow for this experiment
mlflow.set_tracking_uri('https://community.mlflow.deploif.ai')
mlflow.set_experiment("Deploifai/Lichess/LichessRichard")

train = pd.read_csv('/data/LichessDataset/train.csv')
test = pd.read_csv('/data/LichessDataset/test.csv')

train_winner = train['winner']
test_winner = test['winner']

drop_cols = ['winner']
train.drop(columns=drop_cols, inplace=True)
test.drop(columns=drop_cols, inplace=True)

encoder = LabelEncoder()
train_winner = encoder.fit_transform(train_winner)
test_winner = encoder.fit_transform(test_winner)

models = [BernoulliNB(), ComplementNB(), GaussianNB(), MultinomialNB(), KNeighborsClassifier()]
results = []
for model in models:
    model.fit(train, train_winner)
    predict = model.predict(test)
    accuracy = accuracy_score(predict, test_winner)
    results.append(accuracy)

best_accuracy = max(results)
print(best_accuracy)
best_model = models[results.index(best_accuracy)]

dirname = os.path.dirname(os.path.realpath(__file__))
dump(best_model, os.path.join(dirname, "artifacts", "model"))

with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", best_accuracy)
