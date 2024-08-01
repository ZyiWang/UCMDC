import numpy as np
import torch
import torch.nn.functional as F
from utils import NormalizeFeaTorch, get_Similarity
# from keras_preprocessing import image
from numpy import hstack
from scipy import misc
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import h5py
import random
import warnings
warnings.filterwarnings("ignore")




ALL_data = {
    'Stuart': {1: 'Stuart', 'N': 1728, 'K': 5, 'V': 2, 'n_input': [1000,25], 'n_hid': [256,256], 'n_output': 64},
    # 'Pbmc': {1: 'Pbmc', 'N': 3762, 'K': 16, 'V': 2, 'n_input': [1000,49], 'n_hid': [256,256], 'n_output': 64},
    # 'Gayoso': {1: 'Gayoso', 'N': 6018, 'K': 10, 'V': 2, 'n_input': [1000,112], 'n_hid': [256,256], 'n_output': 64},
    # 'Smage': {1: 'Smage', 'N': 2585, 'K': 14, 'V': 2, 'n_input': [2000,2000], 'n_hid': [256,256], 'n_output': 64},
}


path = "./dataset/"
def load_data(dataset):
    data = h5py.File(path + dataset[1] + ".mat")
    X = []
    Y = []
    Label = np.array(data['Y']).T
    Label = Label.reshape(Label.shape[0])
    # print('Label.shape',Label.shape)
    mm = MinMaxScaler()
    for i in range(data['X'].shape[1]):
        diff_view = data[data['X'][0, i]]
        diff_view = np.array(diff_view, dtype=np.float32).T
        std_view = mm.fit_transform(diff_view)
        ######
        X.append(std_view)
        Y.append(Label)
    size = len(Y[0])  # size=n, X与Y均为nx(d1+d2+···+d_V)
    view_num = len(X)
    index = [i for i in range(size)]

    np.random.shuffle(index)
    for v in range(view_num):
        X[v] = X[v][index]
        Y[v] = Y[v][index]
    for v in range(view_num):
        X[v] = torch.from_numpy(X[v])

    return X, Y





