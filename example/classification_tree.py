from lib import ClassificationDecisionTree
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split


def preprocess():
    spam_df = pd.read_csv("../data/spam.txt")
    spam_df.drop(columns=["test"], inplace=True)
    y = spam_df["spam"]
    X = spam_df.drop(columns=["spam"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = preprocess()

    cdt = ClassificationDecisionTree(max_level=10)

    t0 = time.time()
    cdt.fit(X_train.values, y_train.values)
    t1 = time.time()
    print(f'build tree costs {t1 - t0} seconds')

    y_predict = cdt.predict(X_test.values)

    diff = pd.Series(y_predict - y_test.values, index=y_test.index)
    accuracy = (diff == 0).sum() / len(y_test)
    print(f'Predict accuracy: {accuracy}')


if __name__ == '__main__':
    main()
    

    



