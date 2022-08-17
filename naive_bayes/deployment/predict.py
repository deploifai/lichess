import os
import pandas as pd

from joblib import load

base_directory = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(base_directory, "model")
model = load(model_path)


def predict(input):
    df = pd.json_normalize(input)

    predictions = model.predict(df)

    result = predictions[0]

    response = ""

    if result == 0:
        response = 'black'
    elif result == 1:
        response = 'draw'
    elif result == 2:
        response = 'white'
    else:
        pass

    return response


if __name__ == '__main__':
    # f = open('test.json')
    d = {
        "rated": True,
        "turns": 69,
        "white_rating": 1560,
        "black_rating": 15320,
        "opening_ply": 9,
        "time": 10,
        "increment": 5
    }

    print(predict(d))


