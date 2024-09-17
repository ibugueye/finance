import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
# Sélection des actifs (actions)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']

# Télécharger les données de prix historiques (8 ans de données)
data = yf.download(tickers, start='2015-01-01', end='2024-01-01')['Adj Close']

# Calcul des rendements quotidiens
returns = data.pct_change().dropna()
 

# Sommaire
st.write("# Sommaire")
st.markdown("""
- [1. Téléchargement des données](#1-téléchargement-des-données)
- [2. Calcul des rendements](#2-calcul-des-rendements)
- [3. Indicateurs clés de performance (KPI)](#3-indicateurs-clés-de-performance-kpi)
- [4. Modèle prédictif (LSTM)](#4-modèle-prédictif-lstm)
- [5. Ajout des indicateurs techniques](#5-ajout-des-indicateurs-techniques)
""")

# 1. Téléchargement des données historiques
st.write("## 1. Téléchargement des données")
st.write("""
# Sélection des actifs (actions)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']

# Télécharger les données de prix historiques (8 ans de données)
data = yf.download(tickers, start='2015-01-01', end='2024-01-01')['Adj Close']

# Calcul des rendements quotidiens
returns = data.pct_change().dropna()
### Théorie :
Les **données de marché** sont une base essentielle pour toutes les analyses financières. Elles incluent les prix ajustés, qui tiennent compte des événements d'entreprise comme les dividendes et les fractionnements d'actions.
""")
st.code("""
import yfinance as yf

 1. Téléchargement des données historiques
st.write("## 1. Téléchargement des Données Historiques")
st.write("### Aperçu des données")
st.dataframe(data.head())

""", language='python')
data
# 2. Calcul des rendements quotidiens et cumulés
st.write("## 2. Calcul des rendements")
st.write("""
### Théorie :
Le **rendement** est utilisé pour mesurer la performance d’un investissement. Le rendement quotidien est la variation des prix d'un jour à l'autre, tandis que le rendement cumulé montre l’évolution globale de l’investissement sur une période donnée.
""")
st.code("""
# Calcul des rendements quotidiens et cumulés
data['Daily Return'] = data['Adj Close'].pct_change()
data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
""", language='python')
st.write("## 2. Calcul des Rendements")
 
cumulative_returns = (1 + returns).cumprod()
# Visualiser les rendements cumulés
st.write("### Visualisation des Rendements Cumulés")
plt.figure(figsize=(10, 6))
for stock in tickers:
    plt.plot(cumulative_returns.index, cumulative_returns[stock], label=stock)
plt.title('Rendements Cumulés des 6 entreprises')
plt.xlabel('Date')
plt.ylabel('Rendement Cumulé')
plt.legend()
st.pyplot(plt)



# 3. Indicateurs clés de performance (KPI)
st.write("## 3. Indicateurs clés de performance (KPI)")
st.write("""
### Théorie :
Les **KPI** sont des indicateurs quantitatifs utilisés pour mesurer la performance d’un portefeuille. Le **rendement annualisé** mesure la performance moyenne sur une base annuelle.
""")
st.code("""
# Calcul des KPI : Rendement annualisé, volatilité, et ratio de Sharpe
annual_return = data['Daily Return'].mean() * 252
annual_volatility = data['Daily Return'].std() * np.sqrt(252)
risk_free_rate = 0.01  # Hypothèse d'un taux sans risque de 1%
sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
""", language='python')

annual_return = returns.mean() * 252
annual_volatility = returns.std() * np.sqrt(252)
risk_free_rate = 0.01
sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

st.write("### Tableau des KPI")
kpi_data = pd.DataFrame({
    'Rendement Annualisé (%)': annual_return * 100,
    'Volatilité Annualisée (%)': annual_volatility * 100,
    'Ratio de Sharpe': sharpe_ratio
})
st.write("### Tableau des KPI pour chaque entreprise")
st.table(kpi_data)


# 4. Modèle prédictif (LSTM)
st.write("## 4. Modèle prédictif (LSTM)")
st.write("""
### Théorie :
Les **réseaux de neurones LSTM** sont capables de modéliser les dépendances temporelles et d’anticiper les évolutions futures.
""")

# Définition de la fonction create_sequences
def create_sequences(data, sequence_length=60):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Code pour le LSTM
st.code("""
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Normalisation des données pour LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))

# Fonction pour créer les séquences
def create_sequences(data, sequence_length=60):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Modèle LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du modèle
model.fit(X, y, epochs=10, batch_size=32)
""", language='python')

# Normalisation des données et création des séquences
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))
X, y = create_sequences(scaled_data, sequence_length=60)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Création et entraînement du modèle LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

# Prédictions avec le modèle LSTM
st.write("### Prédictions des prix futurs")
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

# Visualisation des prédictions
plt.figure(figsize=(10, 6))
plt.plot(data.index[-len(predictions):], predictions, label='Prix prédit')
plt.plot(data.index[-len(predictions):], data['Adj Close'].values[-len(predictions):], label='Prix réel')
plt.title('Prédiction des prix futurs avec LSTM')
plt.xlabel('Date')
plt.ylabel('Prix')
plt.legend()
st.pyplot(plt)

# 5. Ajout des indicateurs techniques : RSI et MACD
st.write("## 5. Ajout des indicateurs techniques")
st.write("""
### Théorie :
Les **indicateurs techniques** sont des outils utilisés par les traders pour analyser les tendances du marché.
""")

# Définition de la fonction calculate_rsi
def calculate_rsi(data, window=14):
    delta = data['Adj Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Code pour le RSI et le MACD
st.code("""
# Calcul du RSI
def calculate_rsi(data, window=14):
    delta = data['Adj Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data)

# Calcul du MACD
data['EMA12'] = data['Adj Close'].ewm(span=12, adjust=False).mean()
data['EMA26'] = data['Adj Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA12'] - data['EMA26']
data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
""", language='python')

# Calcul des indicateurs techniques
data['RSI'] = calculate_rsi(data)
data['EMA12'] = data['Adj Close'].ewm(span=12, adjust=False).mean()
data['EMA26'] = data['Adj Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA12'] - data['EMA26']
data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Affichage des indicateurs techniques
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['RSI'], label='RSI')
plt.title('RSI (14 jours)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
st.pyplot(plt)

plt.figure(figsize=(10, 6))
plt.plot(data.index, data['MACD'], label='MACD')
plt.plot(data.index, data['Signal Line'], label='Signal Line')
plt.title('MACD et Ligne de Signal')
plt.xlabel('Date')
plt.ylabel('MACD')
plt.legend()
st.pyplot(plt)


# 6. Sélection d'une entreprise pour une analyse
st.write("## 6. Sélection d'une entreprise")
selected_stock = st.selectbox("Choisissez une entreprise", tickers)

# Visualiser le rendement cumulé de l'entreprise sélectionnée
st.write(f"### Visualisation du rendement cumulé pour {selected_stock}")
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns.index, cumulative_returns[selected_stock], label=selected_stock)
plt.title(f'Rendement cumulé de {selected_stock}')
plt.xlabel('Date')
plt.ylabel('Rendement Cumulé')
plt.legend()
st.pyplot(plt)

# Affichage des KPI pour l'entreprise sélectionnée
st.write(f"### KPI pour {selected_stock}")
selected_kpi = kpi_data.loc[selected_stock]
st.table(selected_kpi)

# 7. Comparaison des entreprises (jusqu'à toutes les entreprises)
st.write("## 7. Comparaison des entreprises")

# Ajouter une option pour sélectionner toutes les entreprises ou un sous-ensemble
selected_stocks = st.multiselect("Choisissez les entreprises à comparer", tickers, default=tickers)

# Comparaison des rendements cumulés entre les entreprises sélectionnées
if selected_stocks:
    st.write(f"### Comparaison des rendements cumulés : {' vs '.join(selected_stocks)}")
    plt.figure(figsize=(10, 6))
    for stock in selected_stocks:
        plt.plot(cumulative_returns.index, cumulative_returns[stock], label=stock)
    plt.title(f'Comparaison des rendements cumulés entre {" et ".join(selected_stocks)}')
    plt.xlabel('Date')
    plt.ylabel('Rendement Cumulé')
    plt.legend()
    st.pyplot(plt)

    # Comparaison des KPI entre les entreprises sélectionnées
    st.write(f"### Comparaison des KPI : {' vs '.join(selected_stocks)}")
    kpi_comparison = pd.DataFrame({stock: kpi_data.loc[stock] for stock in selected_stocks})
    st.table(kpi_comparison)
else:
    st.warning("Veuillez sélectionner au moins une entreprise pour la comparaison.")

# 8. Classement des entreprises par rendement
st.write("## 8. Classement des entreprises par Rendement Annualisé")
sorted_kpi_data = kpi_data.sort_values(by='Rendement Annualisé (%)', ascending=False)

# Afficher les entreprises triées par rendement
st.write("### Classement des entreprises par Rendement Annualisé")
st.table(sorted_kpi_data[['Rendement Annualisé (%)']])




# Ajouter les explications pour le RSI et MACD dans le tableau de bord
st.write("## Explications")

st.write("""
### Calcul du RSI :
- **Delta** : Différence des prix ajustés d'un jour à l'autre.
- **Gain et Perte** : Séparation des gains et des pertes pour calculer les moyennes.
- **Moyenne des Gains et Pertes** : Calculées sur une fenêtre de 14 jours.
- **RS** : Ratio des gains moyens sur les pertes moyennes.
- **RSI** : Indicateur de momentum basé sur le RS.

### Calcul du MACD :
- **EMA12 et EMA26** : Moyennes mobiles exponentielles sur 12 et 26 jours.
- **MACD** : Différence entre EMA12 et EMA26.
- **Signal Line** : Moyenne mobile exponentielle sur 9 jours de la MACD.

### Visualisation :
- **RSI** : Graphique montrant les niveaux de surachat et de survente.
- **MACD** : Graphique montrant la convergence et divergence des moyennes mobiles, aidant à identifier les retournements de tendance.
""")

# Ajouter la conclusion dans le tableau de bord
st.write("## Conclusion")

st.write("""
Ce tableau de bord interactif avec Streamlit te permet de :

- Télécharger et visualiser des données financières historiques.
- Calculer et afficher des rendements quotidiens et cumulés.
- Évaluer la performance du portefeuille à l'aide de KPI.
- Construire et entraîner un modèle LSTM pour prédire les prix futurs.
- Analyser les tendances du marché avec des indicateurs techniques (RSI et MACD).

### Points Clés :

- **Streamlit** facilite la création d'applications web interactives pour visualiser des analyses de données.
- **yFinance** permet de télécharger facilement des données financières historiques.
- **Pandas** et **NumPy** sont essentiels pour la manipulation et l'analyse des données.
- **Matplotlib** offre des capacités de visualisation puissantes pour représenter graphiquement les données.
- **TensorFlow/Keras** permettent de créer des modèles de machine learning avancés, comme les **LSTM**, pour les prévisions de séries temporelles.
- **Scikit-learn (MinMaxScaler)** est utilisé pour normaliser les données, améliorant ainsi la performance des modèles de machine learning.

Ce tableau de bord fournit une approche complète intégrant les théories financières, les concepts de gestion de portefeuille, les analyses quantitatives, les modèles prédictifs, et les visualisations interactives, offrant ainsi une ressource éducative précieuse pour les étudiants en finance et gestion de risque.

**ibrahima Gueye**

email : ibugueye@ngorweb.com
""")
