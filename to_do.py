import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
import sklearn.datasets as skds
from pathlib import Path


# For reproducibility
np.random.seed(1237)

dataset_path = "/home/samir/Documents/companion/datasets/dataset.tsv"
dataset = pd.read_csv(dataset_path, delimiter = "\t", quoting = 3)

train_size = int(len(dataset) * .8)

train_types = dataset['type'][:train_size]
train_sentence = dataset['sentence'][:train_size]
train_person = dataset['person'][:train_size]

test_types = dataset['type'][train_size:]
test_sentence = dataset['sentence'][train_size:]
test_person = dataset['person'][train_size:]

# 20 news groups
types_count = 10
vocab_size = 15000
batch_size = 10

# define Tokenizer with Vocab Size
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_sentence)

person_tokenizer = Tokenizer(num_words=2)
person_tokenizer.fit_on_texts(train_person)

person_train_column = person_tokenizer.texts_to_matrix(train_person, mode='tfidf')
person_test_column = person_tokenizer.texts_to_matrix(test_person, mode='tfidf')


x_train = np.append(tokenizer.texts_to_matrix(train_sentence, mode='tfidf'), person_train_column, axis=1)

x_test = np.append(tokenizer.texts_to_matrix(test_sentence, mode='tfidf'), person_test_column, axis=1)
 
encoder = LabelBinarizer()
encoder.fit(train_types)
y_train = encoder.transform(train_types)
y_test = encoder.transform(test_types)

#keras model
model = Sequential()
model.add(Dense(512, input_shape=(vocab_size + 2,))) #shape of the input
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(types_count)) #shape of the output
model.add(Activation('softmax'))
model.summary()
 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=30, verbose=1, validation_split=0.1)


score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
 
print('Test accuracy:', score[1])
 
text_labels = encoder.classes_
 
for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction[0])]
    #print(test_files_names.iloc[i])
    print('Actual label:' + test_types.iloc[i])
    print("Predicted label: " + predicted_label)