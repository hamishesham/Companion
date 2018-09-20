from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import imdb


(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz", num_words=None, skip_top=0, maxlen=None, 
													  seed=113, start_char=1, oov_char=2, index_from=3)

#creating model
model = Sequential()

print(x_test)
print(y_train)
#Stacking layers
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

#configure its learning process
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# itritating on training data in batches
model.fit(x_train, y_train, epochs=5, batch_size=32)

#evaluate performance
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

#pridict on new data
classes = model.predict(x_test, batch_size=128)