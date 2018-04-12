# twitter_sentiment_classifier_keras

This is a sentiment classifier, using tweets to make a text corpus. It has about 6000 tweets with a positive sentiment and 6000 with a negative sentiment.

REST api can be found here:
https://keras-sentiment-analysis.herokuapp.com/predict

### Preprocessing

The script does some preprocessing on the tweets. It'll tokenize the top 10k words and create a dictionary.json file of them. It'll then convert each of these words to a word index, and then convert those indexes to a numpy array.

### Training the model

the script then takes these words and trains a single layer neural network. 

### Using the model

commandline script called run.py loads the model and gets the user to input a phrase. the script checks the corpus for the word. if it finds it, it'll load the index of it. otherwise, it'll skip over that word. it'll feed that into the trained model and output a prediciton of positive or negative. it'll also output an "accuracy" between 0 or 1.

### Flask API

A flask REST api has been created, hosted on heroku. use the model, curl the url above and run a POST request, with the json in the format of {"phrase": "this is a test phrase"}
