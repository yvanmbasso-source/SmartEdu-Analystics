import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ===================== CONFIG =====================
st.set_page_config(page_title="SmartEdu Analytics", layout="wide")
st.title("📊 SmartEdu Analytics - Analyse de données éducatives")

# ===================== UPLOAD =====================
st.sidebar.header("📁 Importation des données")
file = st.sidebar.file_uploader("Importer un fichier CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("📌 Aperçu des données")
    st.dataframe(df)

    # ===================== ANALYSE DESCRIPTIVE =====================
    st.subheader("📊 Analyse descriptive")
    st.write(df.describe())

    # ===================== CHOIX TARGET =====================
    target = st.sidebar.selectbox("🎯 Variable cible", df.columns)

    features = df.drop(columns=[target])

    # encodage simple
    X = pd.get_dummies(features, drop_first=True)
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ===================== GRAPHIQUES BASE =====================
    st.subheader("📈 Visualisations générales")

    fig, ax = plt.subplots()

    if y.dtype != "object":
        sns.histplot(y, kde=True, ax=ax)
        ax.set_title("Distribution de la variable cible")
    else:
        sns.countplot(x=y, ax=ax)
        ax.set_title("Répartition des classes")

    st.pyplot(fig)

    # ===================== BOX PLOT =====================
    st.subheader("📦 Boxplot des variables")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=pd.DataFrame(X), ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ===================== REGRESSION =====================
    st.header("1️⃣ Régression Linéaire")

    if y.dtype != "object" and y.nunique() > 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mse = mean_squared_error(y_test, pred)

        st.write("📉 MSE :", mse)

        # GRAPH REGRESSION
        fig, ax = plt.subplots()
        ax.scatter(y_test, pred)
        ax.set_xlabel("Valeurs réelles")
        ax.set_ylabel("Prédictions")
        ax.set_title("Régression Linéaire")
        st.pyplot(fig)

    # ===================== CLASSIFICATION =====================
    st.header("2️⃣ Classification supervisée")

    y_class = (y > y.mean()).astype(int) if y.dtype != "object" else pd.factorize(y)[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_class, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)

    st.write("🎯 Accuracy :", acc)

    # ===================== PCA =====================
    st.header("3️⃣ Réduction de dimension (PCA)")

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    st.write("Variance expliquée :", pca.explained_variance_ratio_)

    fig, ax = plt.subplots()
    ax.scatter(components[:, 0], components[:, 1], c=y_class, cmap="viridis")
    ax.set_title("Projection PCA")
    st.pyplot(fig)

    # ===================== KMEANS =====================
    st.header("4️⃣ Classification non supervisée (KMeans)")

    k = st.slider("Nombre de clusters", 2, 6, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df_cluster = df.copy()
    df_cluster["Cluster"] = clusters

    st.write(df_cluster)

    fig, ax = plt.subplots()
    ax.scatter(components[:, 0], components[:, 1], c=clusters, cmap="rainbow")
    ax.set_title("Clusters KMeans (PCA 2D)")
    st.pyplot(fig)

    # ===================== FIN =====================
    st.success("✔ Analyse terminée avec succès")

else:
    st.info("⬆️ Importez un fichier CSV pour commencer")
