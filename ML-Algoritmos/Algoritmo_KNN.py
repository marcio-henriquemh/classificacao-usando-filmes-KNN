# ============================================================
#   ALGORITMO KNN COM INTERFACE GRÁFICA (TKINTER)
#   Autor: Márcio Henrique Matos de Freitas
# ============================================================

import numpy as np
import pandas as pd
from tkinter import *
from tkinter import messagebox
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


#Carregar Dataset

dados = pd.read_csv("/home/marciohenrique/UFS/ML-Algoritmos/Dados_Filmes.csv")
caracteristicas = ["Violência", "Romance", "Ação", "Comédia"]

# Normalização 
scaler = MinMaxScaler()
dados_normalizados = scaler.fit_transform(dados[caracteristicas])


# Distancia Euclidiana

def distancia_euclidiana(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def knn_manual(df, novo_ponto, k=3):
    distancias = []

    for i, row in df.iterrows():
        ponto = row[caracteristicas].values
        dist = distancia_euclidiana(ponto, novo_ponto)
        distancias.append((row["Filmes"], dist))

    distancias.sort(key=lambda x: x[1])

    return distancias[:k]



#
#Interface Tkinter

janela = Tk()
janela.title("Classificação de Filmes - KNN")
janela.geometry("420x450")
janela.configure(bg="#202020")

titulo = Label(janela, text="Classificação de Filmes (KNN)",
               bg="#202020", fg="white", font=("Arial", 16))
titulo.pack(pady=10)

# Campos de entrada
labels = ["Violência", "Romance", "Ação", "Comédia"]
entradas = {}

for lbl in labels:
    frame = Frame(janela, bg="#202020")
    frame.pack(pady=5)
    Label(frame, text=lbl + ":", bg="#202020",
          fg="white", font=("Arial", 12)).pack(side=LEFT)
    entrada = Entry(frame, width=10, font=("Arial", 12))
    entrada.pack(side=LEFT)
    entradas[lbl] = entrada


# 
# Função de classificação

def classificar():
    try:
        novo = np.array([
            float(entradas["Violência"].get()),
            float(entradas["Romance"].get()),
            float(entradas["Ação"].get()),
            float(entradas["Comédia"].get())
        ])

    except:
        messagebox.showerror("Erro", "Insira valores numéricos válidos.")
        return

    # KNN manual
    resultado_manual = knn_manual(dados, novo, k=3)


    texto_resultado.delete(1.0, END)

    texto_resultado.insert(END, "===  (3 MAIS PRÓXIMOS) ===\n")
    for filme, dist in resultado_manual:
        texto_resultado.insert(END, f"{filme}  |  Distância: {dist:.3f}\n")


# Botão
btn = Button(janela, text="Classificar", command=classificar,
             bg="#0066ff", fg="white", font=("Arial", 14))
btn.pack(pady=10)

# Caixa de resultado
texto_resultado = Text(janela, height=12, width=45,
                       bg="#303030", fg="white", font=("Arial", 12))
texto_resultado.pack(pady=10)

janela.mainloop()

plt.show()
