
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

base= pd.read_csv('coin_Bitcoin.csv')
base= base.dropna()

base_treinamento= base.iloc[:2804,6:7]

#print(base_treinamento)

normalizador= MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada= normalizador.fit_transform(base_treinamento)

previsores= []
preco_real=[]

for i in range(90,2804):
    previsores.append(base_treinamento_normalizada[i-90:i,0])
    preco_real.append(base_treinamento_normalizada[i,0])

previsores, preco_real= np.array(previsores), np.array(preco_real)

previsores= np.reshape(previsores,(previsores.shape[0],previsores.shape[1],1))


regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 1, activation = 'linear'))

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])
regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32)

base_teste= base.iloc[2804:,6:7]
base_completa= base.iloc[:,6:7]

entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)


X_teste = []
for i in range(90, 148):
    X_teste.append(entradas[i-90:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))
previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

previsoes.mean()
base_teste.mean()
diferenca=base_teste.mean()-previsoes.mean()

