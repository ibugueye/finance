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
- [6. Sélection d'une entreprise](#6-sélection-dune-entreprise)
- [7. Comparaison des entreprises](#7-comparaison-des-entreprises)
- [8. Classement des entreprises par rendement](#8-classement-des-entreprises-par-rendement)
""")

# 1. Téléchargement des données historiques
st.write("## 1. Téléchargement des Données Historiques")
st.write("### Aperçu des données")
st.dataframe(data.head())

# 2. Calcul des rendements quotidiens et cumulés
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
annual_return = returns.mean() * 252
annual_volatility = returns.std() * np.sqrt(252)
risk_free_rate = 0.01
sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

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

# Normalisation des données et création des séquences
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['AAPL'].values.reshape(-1, 1))

# Définition de la fonction create_sequences
def create_sequences(data, sequence_length=60):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Préparation des séquences pour AAPL
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

# Prédictions avec le modèle LSTM
st.write("### Prédictions des prix futurs pour AAPL")
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

# Visualisation des prédictions
plt.figure(figsize=(10, 6))
plt.plot(data.index[-len(predictions):], predictions, label='Prix prédit')
plt.plot(data.index[-len(predictions):], data['AAPL'].values[-len(predictions):], label='Prix réel')
plt.title('Prédiction des prix futurs avec LSTM pour AAPL')
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

# Calcul du RSI pour AAPL
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data['AAPL'])

# Calcul du MACD pour AAPL
data['EMA12'] = data['AAPL'].ewm(span=12, adjust=False).mean()
data['EMA26'] = data['AAPL'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA12'] - data['EMA26']
data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Affichage des indicateurs techniques RSI et MACD pour AAPL
st.write("### Visualisation du RSI pour AAPL")
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['RSI'], label='RSI')
plt.title('RSI (14 jours) pour AAPL')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
st.pyplot(plt)

st.write("### Visualisation du MACD pour AAPL")
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['MACD'], label='MACD')
plt.plot(data.index, data['Signal Line'], label='Signal Line')
plt.title('MACD et Ligne de Signal pour AAPL')
plt.xlabel('Date')
plt.ylabel('MACD')
plt.legend()
st.pyplot(plt)

# 6. Sélection d'une entreprise pour une analyse
st.write("## 6. Sélection d'une entreprise")
selected_stock = st.selectbox("Choisissez une entreprise", tickers, key="selectbox_1")

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

# 7. Comparaison des entreprises
st.write("## 7. Comparaison des entreprises")

# Sélection des entreprises pour comparaison
selected_stocks = st.multiselect("Choisissez les entreprises à comparer", tickers, default=tickers, key="multiselect_1")

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
- **RS** : Ratio des gains moyens sur les pertes moyennes.
- **RSI** : Indicateur de momentum basé sur le RS.

### Calcul du MACD :
- **EMA12 et EMA26** : Moyennes mobiles exponentielles sur 12 et 26 jours.
- **MACD** : Différence entre EMA12 et EMA26.
- **Signal Line** : Moyenne mobile exponentielle sur 9 jours de la MACD.
""")

# Conclusion
st.write("## Conclusion")
st.write("""
Ce tableau de bord interactif avec Streamlit te permet de :

- Télécharger et visualiser des données financières historiques.
- Calculer et afficher des rendements quotidiens et cumulés.
- Évaluer la performance du portefeuille à l'aide de KPI.
- Construire et entraîner un modèle LSTM pour prédire les prix futurs.
- Analyser les tendances du marché avec des indicateurs techniques (RSI et MACD).

Points Clés :

- **Streamlit** facilite la création d'applications web interactives pour visualiser des analyses de données.
- **yFinance** permet de télécharger facilement des données financières historiques.
- **Pandas** et **NumPy** sont essentiels pour la manipulation et l'analyse des données.
- **TensorFlow/Keras** permettent de créer des modèles de machine learning avancés.
""")
