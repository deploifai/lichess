import os
import pandas as pd
from joblib import load
from sklearn.ensemble import RandomForestClassifier

model_path = os.path.sep.join([os.path.dirname(os.path.realpath(__file__)), "model"])
model: RandomForestClassifier = load(model_path)

def predict(input):
    X = pd.json_normalize(input)
    
    predictions = model.predict(X)
    if predictions[0] == 2:
        return "white"
    elif predictions[0] == 0:
        return "black"
    
    return "draw"

if __name__ == "__main__":
    input = {
        "rated": False,
        "turns": 60,
        "white_rating": 1600,
        "black_rating": 1800,
        "opening_ply": 6,
        "time": 7,
        "increment": 2
    }
    print(predict(input))