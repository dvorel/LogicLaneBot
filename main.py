import tensorflow
from tensorflow.keras.models import load_model
import json
import numpy as np
import random
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')

intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('vocab.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_LogicLane.h5')
lemmatizer = WordNetLemmatizer()


def clean_up_sentence(sentence: str) -> list:
    """Function takes a string, it performs tokenization and lemmatization and then returns list of words"""
    sent_words = word_tokenize(sentence)
    sent_words = [lemmatizer.lemmatize(word).lower() for word in sent_words]
    return sent_words

def bag_of_words(sentence: str) -> np.array:
    """Function takes a string, cleans it up and tranforms it in to a vector that can be classified by our model and returns it as np.array"""
    sent_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sent_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence: str) -> list:
    """Function takes string, transforms it into a vector, sends it to the model for prediction and returns list of predictions"""
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    
    results = [[i, r] for i, r in enumerate(res)] 
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability': str(r[1])}) 
    return return_list

def get_response(intents_list: list, intents_json: json) -> str:
    """Function takes list of predictions, induces the intent from predictions and compares with the intents in json file. 
    When it founds a match for the intent than it returns a response in string type """
    tag = intents_list[0]['intent']
    probability = float(intents_list[0]['probability'])
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag and probability > 0.5:
            result=random.choice(i['responses'])
            break
        else:
            result = "Oprosti, ne razumijem. Preformulirajte pitanje pa me opet pitajte."
    return result



file = open('povijest.txt', 'a+', encoding='utf-8')
lista_upita = []

print("Chatbot Online......za kraj razgovora napiši: ajd bok")
while True:
    message = input("") 
    lista_upita.extend("Korisnik: " + message + "\n")
    if message == "ajd bok":
        break   
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("LogicLaneBot: " + res)
    lista_upita.extend("LogicLaneBot: " + res + "\n" + "---------------------------------------\n") 
        
file.writelines(lista_upita)
file.close()
print("DONE!")  