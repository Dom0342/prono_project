import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filepath = r"C:\Users\Pc Domi\.cache\kagglehub\datasets\dissfya\atp-tennis-2000-2023daily-pull\versions\630\atp_tennis.csv"

df = pd.read_csv(filepath)

df['rank_difference'] = df['Rank_1'] - df['Rank_2']


# Créer le graphique
plt.figure(figsize=(10, 6))
sns.histplot(df['rank_difference'], kde=False, color='purple', bins=30)

# Ajouter des titres et des labels
plt.title('Distribution de la différence de classement entre les deux joueurs')
plt.xlabel('Différence de classement (Rank 1 - Rank 2)')
plt.ylabel('Fréquence')

# Afficher le graphique
plt.show()