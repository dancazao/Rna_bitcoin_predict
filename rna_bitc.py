
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

base= pd.read_csv('coin_Bitcoin.csv')
base= base.dropna()

base_treinamento= base.iloc[:,6:7]

#print(base_treinamento)

normalizador= MinMaxScaler(feature_range=(0,1))

