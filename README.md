# twitter_sentiment_classifier_keras

This is a sentiment classifier, using tweets to make a text corpus. It has about 6000 tweets with a positive sentiment and 6000 with a negative sentiment.

* preprocessing
The script does some preprocessing on the tweets. It'll tokenize the top 10k words and create a dictionary.json file of them. It'll then convert each of these words to a word index, and then convert those indexes to a numpy array.

*training the model
the script then takes these words and trains a single layer neural network. 

* using the model
commandline script called run.py loads the model and gets the user to input a phrase. the script checks the corpus for the word. if it finds it, it'll load the index of it. otherwise, it'll skip over that word. it'll feed that into the trained model and output a prediciton of positive or negative. it'll also output an "accuracy" between 0 or 1.

* current problem
i'm guessing because i used a really simple single layer neural network, it'll output an inaccurate prediction. for the most part it works okay, but if you throw it something like "not happy", it'll predict it as a positive sentiment. I'm guessing using Word2Vec or other word embedding layers would give you better results. 

*next steps
i'm currently building a flask api endpoint for this. the plan is then to throw it on heroku, and build a simple frontend with a JS script that curls that endpoint with some text entry from a user.
