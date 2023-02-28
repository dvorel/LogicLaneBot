import json
import pickle
import nltk
import random
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import sklearn.model_selection as sk
nltk.download('stopwords')


intents = json.loads(open('intents.json', encoding='utf-8').read())
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def create_documents_from_intents(intents: json) -> list:
    """Function takes json file and returns vocab(list of all words), classes(list of all classes)
    and documents(list of tuples showing which pattern is conected to wich class)"""
    vocab = []
    classes = []
    documents = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = word_tokenize(pattern)
            vocab.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    classes = sorted(set(classes))
    return vocab, classes, documents

def lemmatize_vocab(vocab: list, stopwords: list) -> list:
    """Function takes list of words, lemmatizes them and returns list of lemmatized words"""
    vocab_lemmatized = [lemmatizer.lemmatize(word.lower()) for word in vocab if word not in stopwords]
    vocab_lemmatized = sorted(list(set(vocab_lemmatized)))
    return vocab_lemmatized

def create_pickle_file(vocab: list, classes:list) -> None:
    """Function takes lists and makes pickle files"""
    pickle.dump(vocab, open('vocab.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

def create_training_data(documents: list, vocab: list, classes: list) -> np.array:
    """Function takes lists of data for training and returns np.arrays of training data and corelating labels"""
    training = []
    output_empty = [0] * len(classes)
    for document in documents:
        bag = []
        word_pattern = document[0]
        word_pattern = [lemmatizer.lemmatize(word).lower() for word in word_pattern]
        for word in vocab:
            bag.append(1) if word in word_pattern else bag.append(0)
        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append((bag, output_row))
    random.shuffle(training)
    training = np.array(training ,dtype=object)
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    return train_x, train_y

def build_and_train_model(vocab: list, train_x: np.array, train_y: np.array) -> None:
    "Function takes vocabulary, vectors for traning data and labels then saves model after it is trained"
    model = Sequential()
    model.add(Dense(len(vocab), input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(train_y[0]), activation='softmax'))
    sgd = SGD(learning_rate=0.01,  momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    hist = model.fit(np.array(train_x), np.array(train_y), epochs=30, batch_size=4, verbose=1)
    model.save('chatbot_LogicLane.h5', hist)
    print("Done")


vocab, classes, documents = create_documents_from_intents(intents)
vocab_lemm = lemmatize_vocab(vocab,stop_words)
create_pickle_file(vocab_lemm, classes)
train_x, train_y = create_training_data(documents,vocab_lemm,classes)
build_and_train_model(vocab_lemm, train_x, train_y)