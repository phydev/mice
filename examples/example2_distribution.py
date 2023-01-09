import seaborn as sns
import matplotlib.pyplot as plt

p = 3 # column to be plotted
custom_lines = [plt.Line2D([0], [0], color="red", lw=4),
                plt.Line2D([0], [0], color="grey", lw=4),
                plt.Line2D([0], [0], color="blue", lw=4)]

fig, ax = plt.subplots()

for m in range(len(imp)):
    sns.kdeplot(imp[m][:, p], label="Imputed", color="black", lw=0.2, ax=ax)
sns.kdeplot(X_amp[:,p], label="Missing", color="blue", ax=ax)
sns.kdeplot(df.to_numpy()[:, p], label="Complete", color="red",ax=ax)
plt.xlabel("Age (years)")
ax.legend(custom_lines, ['Complete', 'Imputed', 'Missing'], loc="upper left")
plt.savefig("qol_distribution_mice.png")