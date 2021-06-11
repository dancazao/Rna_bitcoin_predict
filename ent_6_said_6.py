from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import time

inicio=time.time()

base= pd.read_csv('coin_Bitcoin.csv')


base_treinamento= base.iloc[:2804,4:10]

normalizador= MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada= normalizador.fit_transform(base_treinamento)
b_t=base_treinamento.iloc[:,2:3]
normalizador_previsao = MinMaxScaler(feature_range=(0,1))
normalizador_previsao.fit_transform(b_t)

previsores= []
preco_real=[]


for i in range(90,2804):
    previsores.append(base_treinamento_normalizada[i-90:i,:])
    preco_real.append(base_treinamento_normalizada[i,:])
    
previsores, preco_real= np.array(previsores), np.array(preco_real)

#previsores= np.reshape(previsores,(previsores.shape[0],previsores.shape[1],previsores.shape[2]))

regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 6, activation = 'linear'))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])
es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
mcp = ModelCheckpoint(filepath = 'pesos.h5', monitor = 'loss', 
                      save_best_only = True, verbose = 1)
regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32,
              callbacks = [es, rlr, mcp])

fim=time.time()
tempo_treino=fim-inicio
print(f"O tempo de treino foi de {tempo_treino} segundos")

inicio_teste=time.time()
base_teste= base.iloc[2804:,4:10]
base_completa= base.iloc[:,4:10]
preco_real_teste = base.iloc[2804:,4:10].values
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
#entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)


X_teste = []
for i in range(90, len(entradas)):
    X_teste.append(entradas[i-90:i, 0:6])
X_teste = np.array(X_teste)

previsoes = regressor.predict(X_teste)
previsoes = normalizador_previsao.inverse_transform(previsoes)

pvm=previsoes.mean()
prm=preco_real_teste.mean()

erro_total=abs(prm-pvm)

print(f"A diferença absoluta da média entre a previsão e o valor real foi de ${erro_total} ")
    
plt.plot(preco_real_teste[:,0], color = 'black', label = 'Preço real de alta')
plt.plot(previsoes[:,0], color = 'green', label = 'Previsão de alta')
plt.plot(preco_real_teste[:,2], color = 'red', label = 'Preço real de entrada')
plt.plot(previsoes[:,2], color = 'blue', label = 'Previsão de entrada')
plt.title('Previsão preço bitcoins 2021')
plt.xlabel('Tempo')
plt.ylabel('Valores bitcoin')
plt.legend()
plt.show()

plt.plot(preco_real_teste[:,2], color = 'red', label = 'Preço real de entrada')
plt.plot(previsoes[:,2], color = 'blue', label = 'Previsão de entrada')
plt.title('Previsão preço bitcoins 2021')
plt.xlabel('Tempo')
plt.ylabel('Valores bitcoin')
plt.legend()
plt.show()


#previsão para 20 anos

entradas_1=base_completa[len(base_completa)- len(base_teste)-90:len(base_completa)- len(base_teste)].values
entradas_1 = entradas_1.reshape(-1, 6)
entradas_1 = normalizador.transform(entradas_1)

Y_teste = []
for i in range(90, 148):
    Y_teste.append(entradas_1[i-90:i, 0:6])
    Z_teste = np.array(Y_teste)
    Z_teste = np.reshape(Z_teste, (Z_teste.shape[0], Z_teste.shape[1], 6))
    previsoes_1 = regressor.predict(Z_teste)
    a = previsoes_1
    a = np.float64(a)
    a = a.reshape(-1, 6)
    entradas_1=np.append(entradas_1,a[i-90])
    entradas_1 = entradas_1.reshape(-1, 6)
    print(i)
previsoes_1 = normalizador.inverse_transform(previsoes_1)

fim_teste=time.time()
tempo_teste=fim_teste-inicio_teste
print(f"O tempo de teste foi de {tempo_teste} segundos")

plt.plot(preco_real_teste[:,0], color = 'black', label = 'Preço real de alta')
plt.plot(previsoes_1[:,0], color = 'green', label = 'Previsão de alta')
plt.plot(preco_real_teste[:,2], color = 'red', label = 'Preço real de entrada')
plt.plot(previsoes_1[:,2], color = 'blue', label = 'Previsão de entrada')
plt.title('Previsão preço bitcoins 2021')
plt.xlabel('Tempo')
plt.ylabel('Valores bitcoin')
plt.legend()
plt.show()

plt.plot(preco_real_teste[:,2], color = 'red', label = 'Preço real de entrada')
plt.plot(previsoes_1[:,2], color = 'blue', label = 'Previsão de entrada')
plt.title('Previsão preço bitcoins 2021')
plt.xlabel('Tempo')
plt.ylabel('Valores bitcoin')
plt.legend()
plt.show()


inicio_testefinal=time.time()

entradas_2=base_completa[len(base_completa)-90:].values
entradas_2 = entradas_2.reshape(-1, 6)
entradas_2 = normalizador.transform(entradas_2)

A_teste = []
ano_20=(20*365)+90
for i in range(90,ano_20):
    A_teste.append(entradas_2[i-90:i, 0:6])
    B_teste = np.array(A_teste)
    B_teste = np.reshape(B_teste, (B_teste.shape[0], B_teste.shape[1], 6))
    previsoes_2 = regressor.predict(B_teste)
    b = previsoes_2
    b = np.float64(b)
    b = b.reshape(-1, 6)
    entradas_2=np.append(entradas_2,b[i-90])
    entradas_2 = entradas_2.reshape(-1, 6)
    print(i)
previsoes_2 = normalizador.inverse_transform(previsoes_2)

fim_testefinal=time.time()
tempo_testefinal=fim_testefinal-inicio_testefinal
print(f"O tempo de teste foi de {tempo_testefinal} segundos")


plt.plot(previsoes_2[:,0], color = 'green', label = 'Alta')
#plt.plot(previsoes_2[:,1], color = 'black', label = 'Baixa')
#plt.plot(previsoes_2[:,2], color = 'blue', label = 'Entrada')
#plt.plot(previsoes_2[:,3], color = 'yellow', label = 'Fechamento')
#plt.plot(previsoes_2[:,4], color = 'red', label = 'Volume')
#plt.plot(previsoes_2[:,4], color = 'orange', label = 'Volumra')
plt.title('Previsão preço bitcoins durante 20 anos')
plt.xlabel('Tempo')
plt.ylabel('Valor bitcoin')
plt.legend()
plt.show()

#plt.plot(preco_real_teste[:,2], color = 'red', label = '')
plt.plot(previsoes_2[:,2], color = 'blue', label = 'Entrada')
plt.title('Previsão preço bitcoins durante 20 anos')
plt.xlabel('Tempo')
plt.ylabel('Valor bitcoin')
plt.legend()
plt.show()

plt.plot(previsoes_2[:,2], color = 'red', label = 'entrada')

plt.title('Previsão preço bitcoins durante 20 anos')
plt.xlabel('Tempo')
plt.ylabel('Valor bitcoin')
plt.legend()
plt.show()
