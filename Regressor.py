import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Importando os dados de teste e de treinamento
data_teste = pd.read_csv()
data_treinamento = pd.read_csv()

# separando a variável que queremos medir no conjunto de teste
resultado = data_teste["preço"]

## Separando a variavl que queremos medir no conjunto de treinamento
x = data_treinamento.drop(columns = ["Colocar aqui o nome das variaveis que você quer descartar"], axis=1)
y = data_treinamento['preço']

## Aplicando One Hot Encoding
x = pd.get_dummies(x, columns = ['Colocar aqui as colunas que precisam de One Hot Encoding'])

## Aplicando Binarização
binarizador = LabelEncoder()
x['Coluna que você quer binarizar'] = binarizador.fit_transform(x['Coluna que você quer binarizar'])

## Arrumando colunas com valores vazios
imputer = SimpleImputer(strategy='most_frequent')
x = imputer.fit_transform(x)

## Arrumando as escalas
StdSc = StandardScaler()
StdSc = StdSc.fit(x)
x = StdSc.transform(x)


##############

## Repetindo os processos acima para o conjunto de teste
## Separando a variavl que queremos medir no conjunto de treinamento
x_teste = data_teste.drop(columns = ["Colocar aqui o nome das variaveis que você quer descartar"], axis=1)

## Aplicando One Hot Encoding
x_teste = pd.get_dummies(x_teste, columns = ['Colocar aqui as colunas que precisam de One Hot Encoding'])

## Aplicando Binarização
x_teste['Coluna que você quer binarizar'] = binarizador.fit_transform(x_teste['Coluna que você quer binarizar'])

## Arrumando colunas com valores vazios
imputer = SimpleImputer(strategy='most_frequent')
x_teste = imputer.fit_transform(x_teste)

## Arrumando as escalas
StdSc = StdSc.fit(x_teste)
x_teste = StdSc.transform(x_teste)

#############
## Treinando o regressor

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


regressorKNN = KNeighborsRegressor(n_neighbors=10, p=1, n_jobs=2, algorithm='kd_tree', weights='distance')
regressorKNN = regressorKNN.fit(x_train,y_train)
knnPredictions = regressorKNN.predict(x_test)
knnError = mean_squared_error(y_test, knnPredictions)
knnScore = r2_score(y_test, knnPredictions)
print("KNN Error was: ", knnError)
print("KNN Score was: ", knnScore)

############
## Prevendo os valores oficiais

resposta = regressorKNN.predict(x_teste)
prediction_file = {"Definir o arquivo final"}
prediction_file = pd.DataFrame(data=prediction_file)

prediction_file.to_csv('Colocar aqui a pasta para onde ess arquivo vai', index=False)
