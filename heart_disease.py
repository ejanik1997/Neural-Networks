from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
import csv

results = []
with open("heart.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        results.append(row)

x_train = np.asarray(results[40:])
x_train = x_train[:, :-1]

y_train = np.asarray(results[40:])
y_train = y_train[:, 13]

x_test = np.asarray(results[0:40])
x_test = x_test[:, :-1]

y_test = np.asarray(results[0:40])
y_test = y_test[:, 13]

x_train /= np.max(x_train, axis=0)
x_test /= np.max(x_test, axis=0)

print("x_train: ")
print(x_train)

model = Sequential()

model.add(Dense(13, input_dim=13, activation='sigmoid'))
model.add(Dropout(0.2, noise_shape=None, seed=None))
model.add(Dense(10, activation='sigmoid'))
model.add(Dropout(0.2, noise_shape=None, seed=None))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=250,
          batch_size=10)

score = model.evaluate(x_test, y_test,)
print('Test loss:', score[0])
print('Test accuracy:', score[1])