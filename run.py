import json
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json

tokenizer = Tokenizer(num_words=10000)
labels = ['negative', 'positive']

# read in our saved dictionary
with open('training_data/dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            print("'%s' not in training dictionary; ignoring." %(word))
    return wordIndices

#load model from disk
json_file = open('training_data/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('training_data/model.h5')

#use stdin to grab something from the user
while 1:
    evalSentence = input('Input a sentence to be evaluated, or Enter to quit: ')

    if len(evalSentence) == 0:
        break

    # format your input for the neural net
    testArr = convert_text_to_index_array(evalSentence)
    input_eval = tokenizer.sequences_to_matrix([testArr], mode='binary')
    # predict which label the input belongs in
    pred = model.predict(input_eval)
    print(pred)
    print("%s sentiment; %f confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)]))
