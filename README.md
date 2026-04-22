# DSLR - Hogwarts House Classification
## Documentation Complète et Détaillée

---

## Table des matières

1. [Vue d'ensemble du projet](#vue-densemble-du-projet)
2. [Structure du projet](#structure-du-projet)
3. [Les données](#les-données)
4. [Statistiques et describe](#statistiques-et-describe)
5. [Visualisation et graphiques](#visualisation-et-graphiques)
6. [Entraînement et prédiction](#entraînement-et-prédiction)
7. [Bonus et variantes](#bonus-et-variantes)
8. [Guide du Makefile](#guide-du-makefile)

---

## Vue d'ensemble du projet

### Objectif
Ce projet implémente un **classifieur multi-classe** pour prédire à quelle maison de Poudlard appartient un étudiant en fonction de ses notes académiques.

### Les 4 maisons
- **Ravenclaw**
- **Slytherin** 
- **Gryffindor**
- **Hufflepuff**
  
### Approche globale
- **Algorithme** : Régression logistique
- **Stratégie multi-classe** : One-vs-Rest (4 modèles binaires)
- **Optimisation** : Descente de gradient (3 variantes)
- **Dataset** : 1600 étudiants en entraînement, 400 en test

---

## Structure du projet

```
dslr/
├── Makefile                          # Orchestration
├── requirements.txt                  # Dépendances
│
├── datasets/
│   ├── dataset_train.csv            # 1600 étudiants + labels
│   └── dataset_test.csv             # 400 étudiants
│
├── src/
│   ├── logreg_train.py              # Entraînement
│   ├── logreg_predict.py            # Prédiction
│   ├── describe.py                  # Statistiques
│   ├── histogram.py                 # Histogrammes
│   ├── scatter_plot.py              # Corrélations
│   ├── pair_plot.py                 # Matrice de plots
│   ├── bonus.py                     # SGD, mini-batch
│   └── utils.py                     # Fonctions utiles
│
├── output/
│   ├── weights.csv                  # Poids entraînés (généré)
│   ├── houses.csv                   # Prédictions (généré)
│   └── *.png                        # Visualisations (généré)
│
└── README.md
```

---

## Les données

### Format et contenu

```
Total                   : 2000 étudiants (1600 train + 400 test)
Colonnes par étudiant   : 31
Colonnes utilisées      : 10
```

### Les 10 matières utilisées

1. Astronomy
2. Herbology
3. Defense Against the Dark Arts
4. Ancient Runes
5. Charms
6. Divination
7. Muggle Studies
8. History of Magic
9. Potions
10. Flying

### Distribution des maisons

Équilibrée : ~400 étudiants par maison dans l'entraînement.

---

## Statistiques et describe

### Concepts clés sur les statistiques

**Count** : Nombre d'étudiants avec une note valide

**Mean** : Performance moyenne

**Std (Écart-type)** : Variation entre étudiants
- Petit (< 5) → Performances homogènes (tous similaires)
- Grand (> 15) → Grandes différences (bons ET mauvais élèves)

**Min / Max** : Plage extrême

**Quartiles (25%, 50%, 75%)** : Découpe la distribution en 4 parts
```
25% des étudiants          50% des étudiants         75% des étudiants
ont une note < Q1          ont une note < Q2         ont une note < Q3
   (mauvais)                  (moyen)                    (bon)
```

**Exemple concret** :
```
Astronomy : 25%=6.78, 50%=12.45, 75%=18.90

Cela signifie :
- 25% des étudiants → notes < 6.78 (très faibles)
- 50% des étudiants → notes < 12.45 (moyen)
- 25% des étudiants → notes > 18.90 (bonnes)
```

**Bonus (--bonus)** :
- **Variance** : Écart-type² (mesure technique)
- **Skewness** : Asymétrie (-1 à +1)
  - Négatif → Plus de mauvaises notes
  - Positif → Plus de bonnes notes
- **Kurtosis** : Poids des extrêmes
  - Positif → Quelques "exceptions" (très bons/mauvais)
  - Négatif → Performances uniformes

---

## Visualisation et graphiques

### histogram.py - Histogrammes

Crée l'histogramme du cours **le plus homogène** entre les maisons.

**Score utilisé** : Écart entre les moyennes par maison / Moyenne générale

### scatter_plot.py - Corrélations

Crée le scatter montrant la paire de matière avec le **coefficient de Pearson** le plus fort.

**Pearson (r)** : Mesure si deux matières sont liées (-1 à 1)
- r = 1.0 → Corrélation positive (notes élevées dans les deux)
- r = -1.0 → Corrélation négative (fort dans une = faible dans l'autre)
- r = 0.0 → Pas de lien

### pair_plot.py - Matrice complète

**Diagonale** : Histogrammes de chaque matière
**Triangle** : Scatter plots entre paires

---

## Entraînement et prédiction

### Concepts clés

#### Régression logistique - Vue simple

C'est un algorithme qui prédit une **probabilité** (entre 0 et 1), pas une valeur continue.

**Analogie** : On veut prédire si un étudiant est à Ravenclaw ou non (oui/non).

##### Les étapes

1. **Calcul de la somme pondérée**
```
z = w₀ + w₁*x₁ + w₂*x₂ + ... + w₁₀*x₁₀
```
- `w₀` = biais (tendance de base)
- `w₁...w₁₀` = poids (coefficients appris)
- `x₁...x₁₀` = notes des 10 matières

2. **Transformation avec la sigmoïde** → convertit z en probabilité [0, 1]
```
σ(z) = 1 / (1 + e^(-z))

z très positif  → σ(z) ≈ 1.0 (99% Ravenclaw)
z = 0           → σ(z) = 0.5 (50% Ravenclaw)
z très négatif  → σ(z) ≈ 0.0 (1% Ravenclaw)
```

#### One-vs-Rest (OvR) - Pourquoi 4 modèles ?

La régression logistique est binaire, mais on a 4 maisons. **Solution** : créer 4 classifieurs binaires indépendants.

**Approche** :
```
Modèle 1 : Ravenclaw vs Rest (label 1 si Ravenclaw, 0 sinon)
Modèle 2 : Slytherin vs Rest (label 1 si Slytherin, 0 sinon)
Modèle 3 : Gryffindor vs Rest (label 1 si Gryffindor, 0 sinon)
Modèle 4 : Hufflepuff vs Rest (label 1 si Hufflepuff, 0 sinon)
```

**Prédiction finale** : Pour chaque étudiant, on calcule les 4 probabilités et on prend la maison avec le score le plus élevé.

#### Descente de gradient - Comment ça marche ?

C'est un algorithme qui **ajuste les poids** pour minimiser l'erreur.

**Formule de mise à jour (Batch GD)** :
```
Pour chaque époque :
  1. Prédire les 1600 étudiants avec les poids courants
  2. Calculer l'erreur pour chaque étudiant : (probabilité prédite - label réel)
  3. Calculer les gradients :
     - gradient_w0 = moyenne des erreurs
     - gradient_wj = moyenne(erreur * note_j)
  4. Mettre à jour : w = w - learning_rate * gradient
```

**Les 3 variantes** :

| Variante | Calcul | Mises à jour/époque | Fonction |
|----------|--------|-------------------|----------|
| **Batch GD** | Tous les 1600 étudiants | 1 | `GradientDescent()` |
| **SGD** | 1 étudiant à la fois | ~1600 | `stochastic_gradient_descent()` |
| **Mini-Batch** | Groupes de 32 | ~50 | `mini_batch_gradient_descent()` |

**Taux d'apprentissage (learning_rate)** :
- Valeur utilisée : `0.1`
- Trop petit (0.001) → converge très lentement
- Trop grand (1.0) → peut diverger

### Implémentation

#### logreg_train.py - Entraînement des modèles

**Ce que fait le script** :

1. **Charger et normaliser les données**
   - Lit `dataset_train.csv`
   - Extrait les 10 colonnes de notes choisies
   - Normalise chaque colonne en [0, 1] : `(x - min) / (max - min)`
   - Mémorise les min/max pour la dénormalisation

2. **Pour chaque maison** (Gryffindor, Hufflepuff, Ravenclaw, Slytherin) :
   - Créer les labels binaires : 1 si l'étudiant est de cette maison, 0 sinon
   - Initialiser 11 poids à 0 (1 biais + 10 features)
   - Appeler l'algorithme de descente de gradient (Batch, SGD ou Mini-Batch)
   - Dénormaliser les poids : `w_denorm = w / (max - min)`
   - Ajuster le biais : `w0 -= w * min / (max - min)`

3. **Sauvegarder les poids**
   - Créer un fichier `output/weights.csv` avec tous les poids

**Algorithmes disponibles** :

```bash
make train       # Batch GD (défaut)
make stochastic  # SGD
make mini-batch  # Mini-Batch
```

**Exemple de output - weights.csv** :
```csv
Hogwarts House,Biais,Astronomy,Herbology,Defense Against the Dark Arts,...
Ravenclaw,0.5234,-0.0123,0.0456,-0.0789,...
Slytherin,-0.2134,0.0234,-0.0567,0.0890,...
Gryffindor,0.1567,0.0123,0.0456,0.0789,...
Hufflepuff,0.2890,-0.0234,0.0567,0.0123,...
```

#### logreg_predict.py - Prédiction sur test

**Ce que fait le script** :

1. **Charger les poids** depuis `output/weights.csv`
2. **Lire les données de test** depuis `dataset_test.csv`
3. **Pour chaque étudiant du test** :
   - Calculer le score pour chaque maison : `score = w0 + w1*note1 + ... + w10*note10`
   - Transformer en probabilité avec sigmoid
   - Choisir la maison avec la probabilité maximale
4. **Sauvegarder les prédictions** dans `output/houses.csv`

**Utilisation** :
```bash
make predict
```

**Résultat - houses.csv** :
```csv
Index,Hogwarts House
0,Ravenclaw
1,Slytherin
2,Gryffindor
3,Hufflepuff
...
399,Ravenclaw
```

---

## Bonus et variantes

### bonus.py - SGD et mini-batch

Implémentations alternatives de la descente de gradient :

- `stochastic_gradient_descent()` : Mise à jour 1 étudiant par 1, ~1600 fois par époque
- `mini_batch_gradient_descent()` : Mise à jour sur des groupes de 32, ~50 fois par époque

Plus rapide que Batch GD pour les grands datasets.

---

## Guide du Makefile

### Setup

```bash
make              # Installation initiale (venv + dépendances)
make clean        # Supprime fichiers générés et le cache
make fclean       # Supprime tout (venv + output)
make re           # Réinitialise complètement
```

### Analyse des données

```bash
make describe     # Statistiques descriptives de base
make describe_all # + variance, skewness, kurtosis
```

### Visualisations

```bash
make histogram    # Meilleur histogramme (cours le plus discriminant)
make scatter      # Meilleur scatter (corrélation max)
make pair         # Matrice complète
make visu         # Tous les 3
```

### Entraînement et prédiction

```bash
make train        # Batch Gradient Descent (par défaut)
make stochastic   # Stochastic GD (SGD)
make mini-batch   # Mini-Batch GD
make predict      # Prédiction sur test.csv
```

