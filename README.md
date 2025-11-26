# Algoritmo KNN para Classificação de Filmes

Este projeto implementa o algoritmo **KNN (K-Nearest Neighbors)** para classificar filmes com base em suas características: Violência, Romance, Ação e Comédia.  
Também inclui uma **interface gráfica em Tkinter** para facilitar o uso.

---

## 📌 O que é o KNN?

KNN (K-vizinhos mais próximos) é um algoritmo de **classificação baseado em similaridade**.

Ele funciona assim:

1. Você tem um conjunto de dados com entradas (**X**) e classes (**Y**).
2. O usuário fornece um **novo ponto** (um filme com características).
3. O algoritmo calcula a **distância euclidiana** entre o novo ponto e todos os pontos da base.
4. Ele escolhe os **K pontos mais próximos**.
5. A classe mais comum entre esses K vizinhos é atribuída ao novo ponto.

**KNN não aprende, não cria modelos e não traça funções.**  
Ele apenas compara distâncias.

---

## 📂 Arquivos do Projeto
  
- `Dados_Filmes.csv` → Base de filmes utilizada no algoritmo  
- `Algoritmo_KNN.py` → Script de classificação em linha de comando  
- `README.md` → Documentação do projeto  

---

## 📦 Bibliotecas Necessárias

Antes de executar, instale as dependências:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
sudo apt install python3-tk


Como Executar o Script KNN (linha de comando)

python3 Algoritmo_KNN.py











```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install numpy pandas scikit-learn matplotlib seaborn1''
