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
