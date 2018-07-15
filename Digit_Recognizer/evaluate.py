import numpy as np
import pandas as pd
from sklearn.metrics.scorer import f1_score, accuracy_score

def p2tag(p):
    p = pd.DataFrame(p)
    tag = p.agg(lambda x: x.values.argmax(), axis=1)
    return np.array(tag)

def accu_score(y_pred, y_true):
    '''
    mission 1&2
    :param y_pred:
    :param y_true:
    :return:
    '''
    score = accuracy_score(y_pred=y_pred, y_true=y_true)
    return score


def f1_avg(y_pred, y_true):
    '''
    mission 1&2
    :param y_pred:
    :param y_true:
    :return:
    '''
    f1_micro = f1_score(y_pred=y_pred, y_true=y_true, pos_label=1, average='micro')
    f1_macro = f1_score(y_pred=y_pred, y_true=y_true, pos_label=1, average='macro')
    return (f1_micro + f1_macro) / 2


if __name__ == '__main__':
    print(p2tag(np.array([[1, 0], [0, 1]])))
    print(f1_avg(y_pred=np.array([1, 2]),
                 y_true=np.array([1, 1])))
    print(f1_avg(y_pred=np.array([[1, 0], [0, 1]]),
                 y_true=np.array([[1, 0], [1, 0]])))
    print(accu(y_pred=np.array([1, 2]),
               y_true=np.array([1, 1])))
    print(accu(y_pred=np.array([[1, 0], [0, 1]]),
               y_true=np.array([[1, 0], [1, 0]])))
