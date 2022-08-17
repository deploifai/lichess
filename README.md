The dataset used in this project is a chess dataset. 

The original dataset can be accessed here: https://www.kaggle.com/datasets/datasnaek/chess

There are 2 models, Naive Bayes, and Random Forest with accuracy of 60-70%.

The models are predictors to predict the outcome of a match.

Sample input:
```
{
    "input": {
        "rated": true,
        "turns": 60,
        "white_rating": 3500,
        "black_rating": 2600,
        "opening_ply": 7,
        "time": 20,
        "increment": 2
    }, 
    "input_type": "dict"
}
```

Sample output:
```
{
    "output": "draw"
}
```
