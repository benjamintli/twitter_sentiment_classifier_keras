import json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
print(loaded_model_json)