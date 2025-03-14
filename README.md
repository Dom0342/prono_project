# Prono_project
# Ce projet est mon premier projet de Machine Learning visant a predire le resultat de match de  tennis en utilisant une regression logistique binaire
### Utilsation d'une base de donnees Kaggle regroupant les matchs de tennis ATP des 200 meilleurs au monde joueurs au monde depuis l'annee 2000.
# Preparation des donnees pour le modele de ML: 
### Suppression des colonnes avec valeurs manquantes,encodage one hot et frequentiel, calcul de correlation pour eviter de surchharger le modele, regularization pour reduire la variance du modele, encodage numeriques des noms des joueurs, visualisation des donnees avec pyplot et Tableau, detection et suppression des outliers, suppression des colonnes avec variance trop faible qui ne sont pas utiles pour l'entrainement du modele.
# Precision sur le modele
### Separation des donnees en donnees d'entrainement et de validation, utilisation d'un modele de regression logistique pour classification binaire, evaluation du modele avec accuracy_score et une matrice de confusion et obtention d'une precision de 78 pourcent.
## Ameliorations possibles
### Le projet est de continuer pour predire les cotes des matchs et les comparer a celles des bookmakers pour s'approcher cette fois d'une precision de 100% ce qui est impossible sur la prediction du gagnant du match, utilisation des face a face entre joueurs et de la formce actuelle du joueur pour ameliorer la precision du modele et utilisation d'un modele random forest de regression plus robuste et pouvant predire les cotes.
