# tweets_hate_speech_detection

In this notebook [tweets_hate_speech_detection](https://github.com/YamenHabib/tweets_hate_speech_detection/blob/main/tweets_hate_speech_detection.ipynb) we are getting to know our dataset, building the model and test it before and after traingin.

In this notebook [ploting_precision_recall_curve](https://github.com/YamenHabib/tweets_hate_speech_detection/blob/main/ploting_precision_recall_curve.ipynb) we are using Tensorbord to plot PR-curve.

[Eval](https://github.com/YamenHabib/tweets_hate_speech_detection/blob/main/eval.py) is an CLI python file to test our model.

``` python
python eval.py "american black"
```
the output would be: 
```
american black
Our Model thinks that your tweet is: a hate speach
```

you can download the train model from [here](https://drive.google.com/file/d/1-ObPKmSgN8Pprmz9S02L02pi1qjCgBUE/view?usp=sharing)
you need to put it in the same directory with the eval.py file.
