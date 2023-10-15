import pandas as pd

## Inseridos: accuracy_score, auc, roc_curve,  roc_auc_score,  average_precision_score, v_measure_score, r2_score
## Não Inseridos: confusion_matrix,  multilabel_confusion_matrix, silhouette_score

# Lendo o arquivo de treinamento
df = pd.read_csv('./treinamento_sem_null.csv')

# Aplicando o LabelEncoder
from sklearn.preprocessing import LabelEncoder
var_mod = ['Sexo','Casado','Dependentes','Educacao','Autonomo','AreaPropriedade', 'Concedido']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i].astype(str))

# Separando conjuntos de treino e teste
x = df.drop(['Concedido', 'ID'], axis=1)
y = df['Concedido']

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3) 


#Import models from scikit learn module:
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, X_teste, Y_teste, X_train, Y_train):
  #Fit the model:
  model.fit(X_teste,Y_teste)
  
 #Make predictions on training set:
  predictions = model.predict(X_train)
   
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,Y_train)
  print("Accuracia :{0:.3%}".format(accuracy))
  
  #Print the area under the curve
  #area = metrics.auc(predictions, Y_train)
  #print("Area under the curve :{0:.3%}".format(area))
  
  # ROC
  #roc = metrics.roc_curve(predictions, Y_train)
  #print("ROC :{0:.3%}".format(roc))
  
  #roc_auc_score
  rocAuc = metrics.roc_auc_score(predictions,Y_train, average='macro') 
  print("roc_auc_score :{0:.3%}".format(rocAuc))
  
  #average_precision_score
  avp = metrics.average_precision_score(predictions,Y_train, average='macro') 
  print("average_precision_score :{0:.3%}".format(avp))
  
  #Print v_measure_score
  vms = metrics.v_measure_score(predictions,Y_train) 
  print("v_measure_score :{0:.3%}".format(vms))
  
  #Print r2_score
  r2 = metrics.r2_score(predictions,Y_train) 
  print("r2_score :{0:.3%}".format(r2))


# Testando com alguns classificadores
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

model1 = RandomForestClassifier(n_estimators=100)
print("RandomForest")
classification_model(model1, X_test, y_test, X_train, y_train)

model2 = DecisionTreeClassifier()
print("DecisionTree")
classification_model(model2, X_test, y_test, X_train, y_train)
