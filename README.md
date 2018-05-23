# Sentiment Classifier

This is a sentiment classifier, using tweets to make a text corpus. It has about 6000 tweets with a positive sentiment and 6000 with a negative sentiment. These tweets were chosen from the Sentiment140 dataset.

REST api can be found here:
https://keras-sentiment-analysis.herokuapp.com/predict

### Preprocessing

The script does some preprocessing on the tweets. It'll tokenize the top 10k words and create a dictionary.json file of them. It'll then convert each of these words to a word index, and then convert those indexes to a numpy array.

### Training the model

The script then takes these words and feeds it into a single layer neural network. 

### Prototyping the API

Commandline script called run.py loads the model and gets the user to input a phrase. The script checks the corpus for the word. if it finds it, it'll load the index of it. Otherwise, it'll skip over that word. it'll feed that into the trained model and output a prediciton of positive or negative. It'll also output an "accuracy" float between 0 or 1.

### Flask API

A flask REST api has been created, hosted on heroku. use the model, run a POST request on the url above. here's an example of a request
```
{
    "phrase": "well isn't this stupid"
}
```
The output will look like

```
{
    "accuracy": 0.9215755462646484,
    "polarity": "negative"
}
```
