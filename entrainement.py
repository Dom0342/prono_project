import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Charger les données préparées (données déjà normalisées et prêtes pour l'entraînement)
df = pd.read_csv(r"C:\Users\domin\OneDrive\A_Projet_prono\atp_tennis_ready.csv")

# Liste des features utilisées pour l'entraînement du modèle
features = ['Rank_difference','winning_favourite','Gagnant_player_1','Indoor/Outdoor'
            ,'Surface_Carpet','Surface_Clay','Surface_Grass','Surface_Hard','Player_1_Form','Player_2_Form'
            ,'Pts_diff','Odd_ratio']

# Séparation des features (X) et de la cible (y)
X = df[features]  # Variables indépendantes
y = df['Gagnant_player_1']  # Variable dépendante (cible)


# Séparer les données en un ensemble d'entraînement (80%) et un ensemble de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index] 
X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]
# Initialisation du modèle de régression logistique
model = LogisticRegression()

# Entraînement du modèle avec les données d'entraînement
model.fit(X_train, y_train)


# Prédictions sur les jeux d'entraînement et de test
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Évaluer l'accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Accuracy sur les données d'entraînement : {train_accuracy:.4f}")
print(f"Accuracy sur les données de test : {test_accuracy:.4f}")

# Afficher la matrice de confusion
print("\nMatrice de confusion (test) :")
print(confusion_matrix(y_test, y_test_pred))

# Rapport détaillé (precision, recall, F1-score)
print("\nRapport de classification (test) :")
print(classification_report(y_test, y_test_pred))

# Sauvegarde du modèle entraîné pour une utilisation future
import joblib
joblib.dump(model, r"C:\Users\domin\OneDrive\A_Projet_prono\logistic_regression_model.pkl")

# Sauvegarde du scaler (si tu veux normaliser de nouvelles données de la même manière dans le futur)
joblib.dump(StandardScaler(), r"C:\Users\domin\OneDrive\A_Projet_prono\scaler.pkl")

print("Modèle et scaler sauvegardés avec succès.")
