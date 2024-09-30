import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('aluguel_dataset.csv')

colunas_independentes_x = ["Categoria", "Numero de Passageiros", "Capacidade do porta-malas", "Ar Condicionado", "Câmbio"]
colunas_independentes_y = ["Valor"]

dados_x = df[colunas_independentes_x]
dados_y = df[colunas_independentes_y]

modelo = LinearRegression().fit(dados_x, dados_y)

valores_teste = pd.DataFrame({
    "Categoria": [1],  
    "Numero de Passageiros": [5],
    "Capacidade do porta-malas": [2],
    "Ar Condicionado": [1],
    "Câmbio": [0]
})

predicao = modelo.predict(valores_teste)

print(predicao)

