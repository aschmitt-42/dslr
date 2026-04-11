# DSLR - Hogwarts House Classification

Projet de Machine Learning utilisant la **régression logistique** pour prédire à quelle maison de Poudlard appartient un étudiant, en se basant sur ses notes.

---

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Utilisation](#utilisation)
- [Algorithmes](#algorithmes)
- [Fichiers](#fichiers)
- [Métriques](#métriques)
- [Résultats](#résultats)

---

## Vue d'ensemble

Ce projet implémente un **classifieur multi-classe** utilisant la régression logistique pour classifier les étudiants de Poudlard en 4 maisons :
- **Gryffindor**
- **Hufflepuff**
- **Ravenclaw**
- **Slytherin**

### Approche

- **One-vs-Rest** : 4 modèles binaires (chaque maison vs autres)
- **Descente de gradient** : optimisation des poids
- **Normalisation** : scaling des données entre 0-1
- **Gestion des données manquantes** : remplissage par la moyenne

---

## Installation

### Prérequis
- Python 3.7+
- Pas de dépendances externes (utilise uniquement la stdlib)

### Setup
```bash
cd dslr
# Aucune installation requise
```

---

## Structure du projet

```
dslr/
├── logreg_train.py          # Entraînement du modèle
├── logreg_predict.py        # Prédiction sur nouveaux données
├── describe.py              # Statistiques descriptives
├── utils.py                 # Fonctions utilitaires
├── bonus.py                 # SGD et métriques de précision
├── explication.md           # Guide détaillé des algorithmes
├── weights.json             # Poids entraînés (généré)
├── houses.csv               # Prédictions (généré)
├── datasets/
│   ├── dataset_train.csv    # Données d'entraînement
│   └── dataset_test.csv     # Données de test
└── README.md               # Ce fichier
```

---

## Utilisation

### 1. Afficher les statistiques descriptives

```bash
python3 describe.py datasets/dataset_train.csv
```

**Résultat** : Affiche un tableau avec les statistiques pour chaque feature numérique :
- Count (nombre de valeurs)
- Mean, Std, Min, Max
- 25%, 50%, 75% (quartiles)

### 2. Entraîner le modèle - Gradient Descent classique

```bash
python3 logreg_train.py datasets/dataset_train.csv
```

**Résultat** :
- Affiche les métriques pour chaque maison
- Crée `weights.json` avec les poids entraînés
- Métriques affichées : Accuracy, R² Score, RMSE, MAE, MAPE

### 3. Entraîner le modèle - Stochastic Gradient Descent (BONUS)

```bash
python3 logreg_train.py datasets/dataset_train.csv --bonus
```

**Résultat** :
- Utilise SGD au lieu du GD classique
- Généralement des métriques **meilleures**
- Crée `weights.json` (écrase le précédent)

### 4. Prédire les maisons pour de nouveaux étudiants

```bash
python3 logreg_predict.py datasets/dataset_test.csv weights.json
```

**Résultat** : Crée `houses.csv` avec les prédictions de maisons

### 5. Consulter les explications détaillées

```bash
cat explication.md  # Ou ouvrir dans votre éditeur
```

---

## Algorithmes

### Régression Logistique

La **régression logistique** est un algorithme de classification binaire qui produit une probabilité en sortie.

#### Formule

**1. Combinaison linéaire :**
$$z = w_0 + w_1 x_1 + w_2 x_2 + ... + w_n x_n$$

Où:
- $w_0$ = biais
- $w_1, w_2, ..., w_n$ = poids
- $x_1, x_2, ..., x_n$ = features (notes)

**2. Fonction sigmoid :**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Transforme $z$ en probabilité entre 0 et 1.

**Interprétation** :
- $\sigma(z) \geq 0.5$ → classe 1
- $\sigma(z) < 0.5$ → classe 0

### Descente de Gradient (GD)

Algorithme itératif qui minimise l'erreur en ajustant les poids.

**Mise à jour des poids** :
$$w_j := w_j - \alpha \times \text{erreur} \times x_j$$

Où:
- $\alpha$ = learning rate (0.1 par défaut)
- erreur = prédiction - vraie valeur
- Le changement est proportionnel à l'erreur

**Processus** :
1. Initialiser les poids à zéro
2. Pour N itérations :
   - Prédire avec les poids actuels
   - Calculer l'erreur moyenne
   - Mettre à jour les poids pour réduire l'erreur
3. Répéter jusqu'à convergence

### Stochastic Gradient Descent (SGD)

Variante du GD qui met à jour les poids avec **chaque étudiant** au lieu de tous les étudiants.

**Différences** :
- **GD** : utilise tous les étudiants (batch entier) par itération
- **SGD** : utilise 1 étudiant (ou mini-batch) par itération

**Avantages du SGD** :
- Plus rapide (converge souvent plus vite)
- Moins de mémoire
- Peut échapper aux minima locaux
- Convient mieux aux gros datasets

**Résultats observés** : SGD donne des métriques ~10% meilleures

### Normalisation des données

Les données brutes ont des échelles différentes → peut faire dominer certaines features.

**Formule (Min-Max Scaling)** :
$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

**Résultat** : Toutes les features entre 0 et 1

**Importance** :
- Évite que certaines features dominent
- Accélère la convergence
- Stabilise l'apprentissage

### Approche One-vs-Rest

Pour 4 classes (maisons), on crée **4 modèles binaires** :

```
Modèle 1 : Gryffindor vs Autres
Modèle 2 : Hufflepuff vs Autres
Modèle 3 : Ravenclaw vs Autres
Modèle 4 : Slytherin vs Autres
```

À la prédiction, on prend la maison avec le score maximal :
```
scores = [0.92, 0.15, 0.30, 0.08]
prédiction = argmax(scores) = Gryffindor
```

---

## Fichiers

### logreg_train.py

Entraîne le modèle et sauvegarde les poids.

**Fonctions** :
- `predict()` : Calcule les probabilités pour chaque étudiant
- `GradientDescent()` : Met à jour les poids (GD classique)
- `get_labels()` : Crée les vraies étiquettes (1 ou 0)
- `normalize()` : Normalise les notes entre 0-1
- `main()` : Orchestration complète

**Utilisation** :
```bash
python3 logreg_train.py <dataset> [--bonus]
```

### logreg_predict.py

Charge le modèle et prédit la maison pour de nouveaux étudiants.

**Processus** :
1. Charge les poids depuis weights.json
2. Normalise les données de test (avec min/max du training)
3. Calcule les scores pour chaque maison
4. Prend la maison avec le score maximal
5. Écrit les résultats dans houses.csv

**Utilisation** :
```bash
python3 logreg_predict.py <dataset_test> <weights_json>
```

### describe.py

Affiche des statistiques descriptives pour chaque feature numérique.

**Statistiques affichées** :
- Count, Mean, Std, Min
- 25%, 50%, 75% (quartiles)
- Max

**Utilisation** :
```bash
python3 describe.py <dataset>
```

### utils.py

Fonctions utilitaires réutilisables :

- `read_csv()` : Lit un CSV et le transforme en dict
- `get_numerical_columns()` : Récupère les colonnes numériques
- `get_data_by_column()` : Extrait les valeurs d'une colonne
- `calculateMean()`, `calculateStandardDeviation()` : Statistiques
- `findMin()`, `findMax()` : Extrema
- `calculatePercentile()` : Quartiles
- `GetNotesByStudents()` : Prépare les données (remplace les NaN par moyennes)
- `sigmoid()` : Fonction d'activation

### bonus.py

Implémentations bonus pour améliorer le modèle.

**Contient** :
- `stochastic_gradient_descent()` : SGD
- `calculate_r2_score()` : R² Score
- `calculate_rmse()` : RMSE
- `calculate_mae()` : MAE
- `calculate_mape()` : MAPE

---

## Métriques

### Accuracy

**Définition** : Proportion de prédictions correctes

$$\text{Accuracy} = \frac{\text{Prédictions correctes}}{\text{Total}} \times 100$$

**Interprétation** :
- 0% = complètement faux
- 100% = parfait
- > 95% = excellent

### R² Score

**Définition** : Proportion de variance expliquée par le modèle

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

Où:
- $SS_{res} = \Sigma(y - \hat{y})^2$ (erreurs)
- $SS_{tot} = \Sigma(y - \bar{y})^2$ (variance totale)

**Interprétation** :
- 1.0 = parfait
- 0.5 = moyen
- 0.0 = nul
- < 0 = pire que la moyenne

### RMSE (Root Mean Square Error)

**Définition** : Racine carrée de la moyenne des carrés des erreurs

$$RMSE = \sqrt{\frac{\Sigma(y - \hat{y})^2}{n}}$$

**Interprétation** :
- 0 = parfait
- Plus petit = mieux
- Pénalise les grandes erreurs

### MAE (Mean Absolute Error)

**Définition** : Moyenne de la valeur absolue des erreurs

$$MAE = \frac{\Sigma |y - \hat{y}|}{n}$$

**Interprétation** :
- 0 = parfait
- Plus petit = mieux
- Robuste aux valeurs aberrantes

### MAPE (Mean Absolute Percentage Error)

**Définition** : Moyenne des erreurs en pourcentage

$$MAPE = \frac{\Sigma |y - \hat{y}| / |y|}{n} \times 100$$

**Interprétation** :
- 0% = parfait
- Plus petit = mieux
- En pourcentage

---

## Résultats

### Avec Gradient Descent classique

```
Gryffindor:
  ├─ Accuracy : 99.19%
  ├─ R² Score : 0.882032
  ├─ RMSE     : 0.138500
  ├─ MAE      : 0.089636
  └─ MAPE     : 22.58%

Hufflepuff:
  ├─ Accuracy : 98.88%
  ├─ R² Score : 0.812404
  ├─ RMSE     : 0.203758
  ├─ MAE      : 0.175493
  └─ MAPE     : 25.72%

Ravenclaw:
  ├─ Accuracy : 98.81%
  ├─ R² Score : 0.823989
  ├─ RMSE     : 0.187724
  ├─ MAE      : 0.151706
  └─ MAPE     : 24.67%

Slytherin:
  ├─ Accuracy : 98.75%
  ├─ R² Score : 0.814305
  ├─ RMSE     : 0.168410
  ├─ MAE      : 0.114881
  └─ MAPE     : 32.11%
```

### Avec Stochastic Gradient Descent (--bonus)

```
Gryffindor:
  ├─ Accuracy : 99.19%
  ├─ R² Score : 0.952450 (+7%)
  ├─ RMSE     : 0.087932 (-36%)
  ├─ MAE      : 0.015358 (-83%)
  └─ MAPE     : 4.00% (-82%)

Hufflepuff:
  ├─ Accuracy : 99.06% (+0.2%)
  ├─ R² Score : 0.953143 (+17%)
  ├─ RMSE     : 0.101834 (-50%)
  ├─ MAE      : 0.023249 (-87%)
  └─ MAPE     : 4.00% (-84%)

Ravenclaw:
  ├─ Accuracy : 98.81%
  ├─ R² Score : 0.940330 (+14%)
  ├─ RMSE     : 0.109301 (-42%)
  ├─ MAE      : 0.025029 (-83%)
  └─ MAPE     : 4.17% (-83%)

Slytherin:
  ├─ Accuracy : 99.31% (+0.56%)
  ├─ R² Score : 0.951638 (+17%)
  ├─ RMSE     : 0.085945 (-49%)
  ├─ MAE      : 0.018710 (-84%)
  └─ MAPE     : 3.92% (-88%)
```

### Conclusion

**SGD améliore significativement le modèle** :
- R² Score : +7% à +17%
- RMSE : -36% à -50%
- MAE : -83% à -87%
- MAPE : -82% à -88%

---

## Hyperparamètres

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `learning_rate` | 0.1 | Vitesse d'apprentissage |
| `epochs` | 500 | Nombre d'itérations |
| `initialization` | 0.0 | Poids initialisés à zéro |
| `scaling` | Min-Max | Normalisation 0-1 |

---

## Gestion des données manquantes

- **Stratégie** : Remplissage par la moyenne de la colonne
- **Implémentation** : Fonction `GetNotesByStudents()` dans utils.py
- **Avantages** : Simple, conserve la distribution

---

## Documentation supplémentaire

Pour une explication détaillée ligne par ligne du code, consultez `explication.md`

---

## Auteur

Projet de Machine Learning - DSLR 2026

---

## Notes finales

- **Pas de dépendances externes** : Utilise uniquement la stdlib
- **Code modularisé** : Fonctions réutilisables dans utils.py
- **Données bien structurées** : Dictionnaire {colonne: [valeurs]}
- **Gestion d'erreurs** : Try-except pour les opérations fichiers

