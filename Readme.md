SmartEdu Analytics

Présentation du projet

SmartEdu Analytics est une application web développée en Python avec Streamlit permettant la collecte, l’analyse et la visualisation des données éducatives.

Elle est conçue dans le cadre du cours INF 232 EC2 - Analyse de données.

L’objectif principal est d’aider à analyser les performances des étudiants à travers des techniques statistiques et d’apprentissage automatique.

---

Objectifs

- Collecter des données éducatives (CSV)
- Effectuer une analyse descriptive automatique
- Appliquer la régression linéaire
- Réaliser une classification supervisée
- Effectuer une classification non supervisée (KMeans)
- Réduire la dimension des données (ACP / PCA)
- Visualiser les résultats avec des graphiques

---

Technologies utilisées

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

Fonctionnalités principales

1. Importation des données  
L’utilisateur charge un fichier CSV contenant les informations des étudiants.

2. Analyse descriptive  
- Moyenne  
- Écart-type  
- Minimum / Maximum  
- Distribution des données  

3. Régression linéaire  
Prédiction des performances des étudiants.

4. Classification supervisée  
Détection automatique des étudiants en réussite ou échec.

5. Classification non supervisée (KMeans)  
Regroupement des étudiants en clusters homogènes.

6. Réduction de dimension (PCA)  
Visualisation des données en 2D.

---

Graphiques inclus

L’application génère automatiquement :

- Histogrammes des notes  
- Diagrammes de distribution  
- Boxplots des variables  
- Courbe de régression  
- Projection PCA (2D)  
- Visualisation des clusters KMeans  

---

Lancer l'application en local

pip install -r requirements.txt  
streamlit run app.py  

---

Auteur

Projet réalisé dans le cadre du cours INF 232 EC2  
Analyse de données et Machine Learning appliqué à l’éducation.

Nom : Kenne Mbasso Yvan  
Matricule : 24F2736  

---

Améliorations futures

- Dashboard interactif avancé  
- Export PDF des résultats  
- Prédiction en temps réel  
- Connexion base de données  
