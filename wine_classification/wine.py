import pandas as pd

# carregamento dos scv
data = pd.read_csv("/content/sample_data/wine_dataset.csv")
# tratando os dados
data['style'] = data['style'].replace("red",0)
data['style'] = data["style"].replace("white",1)
data['style']

# separando as variaveus entre preditoras e as alvos
y = data['style']
# axis = 1 colunas
x = data.drop('style', axis = 1)


# criando as var de treino e testes

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

x_train.shape

# aplicando algoritmo machine learning

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
# treinando o modelo

model.fit(x_train,y_train)

# analisando se o treinamento foi bom
response = model.score(x_test,y_test)
print("acuracia", response)

# Verificando se o algoritmo tem boas previsões
print(y_test[400:410])
print(x_test[400:410])
predictions = model.predict(x_test[400:410])
print(y_test[400:410])
predictions 