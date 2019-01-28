import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def move_accuracy(y_test, y_pred):
    """A predicted move is correct if the largest output is 1 in the test vector."""
    return np.mean(y_test[y_pred == np.max(y_pred, axis=1, keepdims=True)])


np.random.seed(1234)

df = pd.read_csv("tictactoe-data.csv")
print("Scores:", Counter(df["score"]))

# Input is all the board features (2x9 squares) plus the turn.
X = df.iloc[:, list(range(18)) + [-2]]

# Target variables are the possible move squares.
moves = df.iloc[:, list(range(18, 27))]
# To predict score instead, use this as the target:
# score = pd.get_dummies(df['score'])

X_train, X_test, y_train, y_test = train_test_split(X, moves, test_size=0.2)

print("Train/test shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation="relu", input_dim=X.shape[1]))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(moves.shape[1], activation="softmax"))

# For a multi-class classification problem
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print(model.summary())

# This is not needed, but lets you view a lot of useful information using
# > tensorboard --logdir logs
# at your terminal prompt.
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="./logs",
    histogram_freq=1,
    batch_size=100,
    write_graph=True,
    write_grads=True,
    write_images=True,
)
model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_data=[X_test, y_test],
    callbacks=[tensorboard_callback],
)

print("accuracy:", model.evaluate(X_test, y_test))
print("Custom accuracy:", move_accuracy(y_test.values, model.predict(X_test)))

model.save("tictacNET.h5")
