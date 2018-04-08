import json
import numpy as np
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
from flask import Flask, request, jsonify

model = None
labels = ['negative', 'positive']


def load_model():
    # initialize flask
    app = Flask(__name__)
    json_file = open('training_data/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # and create a model from that
    global model
    model = model_from_json(loaded_model_json)
    # and weight the nodes with saved values
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights('training_data/model.h5')
    print('model created')
    #return app variable
    return app

# this ensures that the model is loaded when the service starts
app = load_model()


def preprocess_text(input_text):
    tokenizer = Tokenizer(num_words=10000)
    text_array = convert_text_to_index_array(input_text)
    output_text = tokenizer.sequences_to_matrix([text_array], mode='binary')
    return output_text


def convert_text_to_index_array(text):
    with open('training_data/dictionary.json', 'r') as dictionary_file:
        dictionary = json.load(dictionary_file)
    words = kpt.text_to_word_sequence(text)
    word_indices = []
    for word in words:
        if word in dictionary:
            word_indices.append(dictionary[word])
    return word_indices


@app.route('/')
def index():
    return 'yo its lit'


@app.route("/predict", methods=["POST"])
def predict():
    #dictionary (no the other dictionary)
    if request.method == "POST":
        req = request.get_json(force=True)
        output_text = preprocess_text(req["phrase"])
        prediction = model.predict(output_text)
        return jsonify({"polarity": labels[np.argmax(prediction)],
                        "accuracy": float(prediction[0][np.argmax(prediction)])})


if __name__ == "__main__":
    app.run()





