from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import KFold
import os

from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from ad_lakncn import ADLAKNCN

def data_load_keel(path_dir):
    """
    address: http://www.keel.es/
    :param path_dir:
    :return:
    """
    data_list=os.listdir(path_dir)
    train_datas=[open(os.path.join(path_dir,data)).readlines() for data in data_list if data.endswith('tra.dat')]
    test_datas=[open(os.path.join(path_dir,data)).readlines() for data in data_list if data.endswith('tst.dat')]
    train_datas = [np.array([line.strip().split(',') for line in train_data if not line.startswith('@')]) for train_data in train_datas]
    test_datas = [np.array([line.strip().split(',') for line in test_data if not line.startswith('@')]) for test_data in test_datas ]
    X_train=[i[:,:-1] for i in train_datas]
    Y_train=[i[:,-1] for i in train_datas]
    X_test = [i[:,:-1] for i in test_datas]
    Y_test = [i[:,-1] for i in test_datas]
    return X_train,X_test,Y_train,Y_test


def data_load_uci(path_dir):
    """
    address: http://archive.ics.uci.edu/ml/datasets.php
    :param path_dir:
    :return:
    """
    data = loadmat(path_dir)
    kf = KFold(n_splits=10, shuffle=True,random_state=1)
    train_Data = list(kf.split(data['data']))
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for train, test in train_Data:
        X_train.append(data['data'][train])
        y_train.append(data['label'].squeeze()[train])
        X_test.append(data['data'][test])
        y_test.append(data['label'].squeeze()[test])
    return X_train,X_test,y_train,y_test


def valid(dataset='UCI'):
    if dataset=='UCI':
        path = './data/uci_data'
    else:
        path='./data/keel_data'
    data_paths = os.listdir(path)
    for data_path in data_paths:
        data_path = os.path.join(path, data_path)
        if dataset == 'UCI':
            X_trains, X_tests, y_trains, y_tests = data_load_uci(data_path)
        else:
            X_trains, X_tests, y_trains, y_tests = data_load_keel(data_path)
        best_scores = []
        for k in range(1, 21):
            scores = []
            for X_train, X_test, y_train, y_test in zip(X_trains, X_tests, y_trains, y_tests):
                #ten-fold cross-validation
                ss = StandardScaler()
                X_train = ss.fit_transform(X_train)
                X_test = ss.transform(X_test)

                ad_lakncn = ADLAKNCN(k)
                ad_lakncn.fit(X_train, y_train)
                result = ad_lakncn.predict(X_test)
                score = accuracy_score(y_test, result)
                scores.append(score)
            print(np.array(scores).mean(), data_path)
            best_scores.append(np.array(scores).mean())

        print("kmax:%s,score:%s,data_path:%s"%(np.argmax(np.array(best_scores)) + 1, np.max(np.array(best_scores)), data_path))

if __name__=='__main__':
    #dataset='UCI' or 'KEEL'
    valid(dataset='KEEL')