# DSLR - Hogwarts House Classification
## Documentation Complète et Détaillée

---

## Table des matières

1. [Vue d'ensemble du projet](#vue-densemble-du-projet)
2. [Introduction à la régression logistique](#introduction-à-la-régression-logistique)
3. [Approche One-vs-All](#approche-one-vs-all)
4. [Structure du projet](#structure-du-projet)
5. [Guide du Makefile](#guide-du-makefile)
6. [Explications des algorithmes](#explications-des-algorithmes)
7. [Statistiques descriptives](#statistiques-descriptives)
8. [Fichiers Python - Guide complet](#fichiers-python---guide-complet)
9. [Description du dataset](#description-du-dataset)
10. [Métriques et interprétation](#métriques-et-interprétation)
11. [Comment interpréter les résultats](#comment-interpréter-les-résultats)

---

## Vue d'ensemble du projet

### Objectif
Ce projet implémente un **classifieur multi-classe** pour prédire à quelle maison de Poudlard appartient un étudiant en fonction de ses notes académiques.

### Les 4 maisons
- 🦅 **Ravenclaw** : Intelligence, sagesse
- 🐍 **Slytherin** : Ambition, ruse
- 🦁 **Gryffindor** : Courage, bravoure
- 🦡 **Hufflepuff** : Loyauté, travail
  
### Approche globale
- **Algorithme** : Régression logistique
- **Stratégie multi-classe** : One-vs-Rest (4 modèles binaires)
- **Optimisation** : Descente de gradient (3 variantes)
- **Dataset** : 1600 étudiants en entraînement, 400 en test
- **Features** : 10 matières académiques

---

## Introduction à la régression logistique

### Qu'est-ce que la régression logistique ?

La régression logistique est un **algorithme d'apprentissage supervisé** conçu pour la **classification binaire** (deux classes). Malgré son nom contenant "régression", c'est en réalité un **classifieur probabiliste**.

### Analogue simple
Imaginez que vous voulez prédire si un étudiant va être accepté à Ravenclaw ou non, basé sur sa note de Charms.

- Plus la note est élevée, plus la probabilité d'acceptation augmente
- Cette relation n'est pas linéaire mais suit une **courbe en S** (sigmoïde)
- La sortie est une **probabilité** entre 0 et 1

### Comment ça fonctionne ?

#### 1. Combinaison linéaire des features
```
z = w₀ + w₁*x₁ + w₂*x₂ + ... + w₁₀*x₁₀
```

Où :
- `z` = somme pondérée
- `w₀` = biais (constante)
- `w₁, w₂, ..., w₁₀` = poids (coefficients appris)
- `x₁, x₂, ..., x₁₀` = features (notes des 10 matières)

#### 2. Application de la fonction sigmoïde
```
σ(z) = 1 / (1 + e^(-z))
```

Cette fonction transforme `z` (qui peut être n'importe quel nombre) en une **probabilité** entre 0 et 1 :
- Quand z → +∞, σ(z) → 1 (probabilité d'appartenir à la classe)
- Quand z → -∞, σ(z) → 0 (probabilité de ne pas appartenir)
- Quand z = 0, σ(z) = 0.5 (50% de chance)

#### 3. Prise de décision
- Si probabilité ≥ 0.5 → prédiction = "Oui, cette classe"
- Si probabilité < 0.5 → prédiction = "Non, pas cette classe"

### Graphique de la sigmoïde

```
Probabilité (y)
    1.0 |                    ▄▄▄▄▄
        |                ▄▄▄█     █▄▄▄
        |            ▄▄█           █
    0.5 |        ▄▄█                 █▄▄
        |    ▄▄█                       █▄▄
        | ▄█                               █▄
    0.0 |_██_____________________________█__
        -5                0                5
               z (somme pondérée)
```

### Fonction de coût (Loss Function)

Pour mesurer l'erreur, on utilise l'**entropie croisée binaire (Binary Cross-Entropy)** :

```
J(w) = -(1/n) * Σ[yᵢ*log(ŷᵢ) + (1-yᵢ)*log(1-ŷᵢ)]
```

Où :
- `n` = nombre d'exemples
- `yᵢ` = label réel (0 ou 1)
- `ŷᵢ` = probabilité prédite
- Cette fonction pénalise fortement les mauvaises prédictions avec haute confiance

### Interprétation intuitive

**Exemple** : Prédire si un étudiant est en Ravenclaw
- Vraie classe : Ravenclaw (y = 1)
- Prédiction du modèle : 0.8 (80% de chance d'être Ravenclaw)
- Loss = -log(0.8) ≈ 0.22 (erreur faible)

vs.

- Vraie classe : Ravenclaw (y = 1)
- Prédiction du modèle : 0.1 (10% de chance d'être Ravenclaw)
- Loss = -log(0.1) ≈ 2.30 (grande erreur !)

---

## Approche One-vs-All

### Pourquoi avons-nous besoin de One-vs-All ?

La régression logistique est conçue pour la **classification binaire** (2 classes), mais nous avons **4 maisons**. La stratégie **One-vs-Rest (OvR)** ou **One-vs-All (OvA)** est une manière simple d'étendre la régression logistique aux problèmes multi-classes.

### Concept simple

Au lieu d'un seul modèle, on crée **4 modèles binaires** :

1. **Modèle 1** : "Ravenclaw" vs "Pas Ravenclaw" (autres maisons)
2. **Modèle 2** : "Slytherin" vs "Pas Slytherin"
3. **Modèle 3** : "Gryffindor" vs "Pas Gryffindor"
4. **Modèle 4** : "Hufflepuff" vs "Pas Hufflepuff"

### Processus d'entraînement

Pour chaque maison :
1. Créer des labels binaires : 1 si l'étudiant est dans cette maison, 0 sinon
2. Entraîner un modèle de régression logistique
3. Sauvegarder les poids du modèle

**Exemple pour Ravenclaw** :
```
Étudiant 1 : Ravenclaw   → label = 1
Étudiant 2 : Slytherin   → label = 0
Étudiant 3 : Ravenclaw   → label = 1
Étudiant 4 : Gryffindor  → label = 0
...
```

### Processus de prédiction

Pour un nouvel étudiant, on utilise les 4 modèles :

```
Étudiant inconnu avec notes : [60, 75, 80, ...]

Modèle 1 (Ravenclaw)  → Probabilité = 0.85 (85% Ravenclaw)
Modèle 2 (Slytherin)  → Probabilité = 0.20 (20% Slytherin)
Modèle 3 (Gryffindor) → Probabilité = 0.15 (15% Gryffindor)
Modèle 4 (Hufflepuff) → Probabilité = 0.10 (10% Hufflepuff)

Résultat final : Ravenclaw (probabilité la plus élevée = 0.85)
```

### Formule pour la prédiction multi-classe

```
Classe prédite = argmax(ŷ₁, ŷ₂, ŷ₃, ŷ₄)
                = classe avec la probabilité la plus élevée
```

### Avantages et limitations

**Avantages** :
- Simple à comprendre et implémenter
- Fonctionne bien en pratique
- Chaque modèle peut être entraîné indépendamment

**Limitations** :
- Pas d'optimisation directe pour le multi-classe
- Les 4 modèles ne "se parlent pas"
- Peut être sous-optimal comparé à des approches dédiées au multi-classe

---

## Structure du projet

```
dslr/
├── Makefile                          # Orchestration des tâches
├── requirements.txt                  # Dépendances Python
│
├── datasets/
│   ├── dataset_train.csv            # 1600 étudiants + labels
│   └── dataset_test.csv             # 400 étudiants (pas de labels)
│
├── src/
│   ├── logreg_train.py              # Entraînement du modèle
│   ├── logreg_predict.py            # Prédiction
│   ├── describe.py                  # Statistiques descriptives
│   ├── histogram.py                 # Visualisation histogrammes
│   ├── scatter_plot.py              # Visualisation corrélations
│   ├── pair_plot.py                 # Matrice de scatterplots
│   ├── bonus.py                     # SGD, mini-batch, métriques
│   └── utils.py                     # Fonctions utilitaires
│
├── docs/
│   └── README.md                    # Documentation initiale
│
├── output/
│   ├── weights.csv                  # Poids entraînés (généré)
│   ├── houses.csv                   # Prédictions (généré)
│   ├── histogram.png                # Meilleur histogramme (généré)
│   ├── histograms_all.png           # Tous les histogrammes (généré)
│   ├── scatter_best.png             # Meilleur scatter (généré)
│   ├── scatter_all.png              # Tous les scatters (généré)
│   └── pair_plot.png                # Matrice de plots (généré)
│
└── DETAILED_README.md               # Documentation complète (ce fichier)
```

---

## Guide du Makefile

Le `Makefile` automatise toutes les tâches du projet. Voici comment utiliser chaque règle :

### 1. **Setup du projet**

#### `make all` ou `make`
Initialise l'environnement virtuel et installe les dépendances.
```bash
make all
```
- Crée un dossier `venv/` avec Python isolé
- Installe les packages de `requirements.txt`
- **À exécuter une seule fois**

#### `make fclean`
Supprime tout (venv + output).
```bash
make fclean
```

#### `make re`
Réinitialise complètement le projet (fclean + all).
```bash
make re
```

#### `make clean`
Supprime uniquement les fichiers générés (__pycache__, output/).
```bash
make clean
```

### 2. **Analyse descriptive des données**

#### `make describe`
Affiche les statistiques descriptives de base.
```bash
make describe
```

**Affichage** :
```
          Astronomy    Herbology  Defense Against the Dark Arts
Count     1600.0       1600.0     1600.0
Mean      10.234       20.567     15.123
Std       5.123        8.234      6.456
Min       -100.0       -200.0     -150.0
25%       6.789        15.432     10.234
50%       10.123       20.567     14.987
75%       14.234       25.789     19.876
Max       150.0        250.0      200.0
```

#### `make describe_all`
Ajoute variance, skewness et kurtosis (statistiques bonus).
```bash
make describe_all
```

### 3. **Visualisation des données**

#### `make histogram`
Crée l'histogramme du cours **le plus homogène** par maison.
```bash
make histogram
```
- Sauvegarde dans `output/histogram.png`
- Affiche le score de pearson

#### `make histogram_all`
Crée 16 histogrammes (tous les cours).
```bash
make histogram_all
```
- Sauvegarde dans `output/histograms_all.png`

#### `make scatter`
Crée le scatter plot avec la meilleure corrélation entre deux matières.
```bash
make scatter
```
- Sauvegarde dans `output/scatter_best.png`

#### `make scatter_all`
Crée une grille de scatter plots pour toutes les paires de matières.
```bash
make scatter_all
```
- Sauvegarde dans `output/scatter_all.png`

#### `make pair`
Crée une matrice triangulaire de plots (diagonale = histogrammes).
```bash
make pair
```
- Sauvegarde dans `output/pair_plot.png`

#### `make visu`
Lance les trois visualisations (histogram + scatter + pair).
```bash
make visu
```
- Génère les trois fichiers PNG

### 4. **Entraînement du modèle**

#### `make train`
Entraîne le modèle avec **Batch Gradient Descent** (par défaut).
```bash
make train
```
- Utilise tous les 1600 étudiants à chaque mise à jour
- Sauvegarde les poids dans `output/weights.csv`
- Affiche les métriques d'entraînement

#### `make stochastic`
Entraîne avec **Stochastic Gradient Descent (SGD)**.
```bash
make stochastic
```
- Met à jour les poids avec 1 étudiant à la fois
- Plus de bruit, mais plus rapide
- Meilleur pour les grands datasets

#### `make mini-batch`
Entraîne avec **Mini-Batch Gradient Descent**.
```bash
make mini-batch
```
- Compromis entre Batch et SGD
- Met à jour tous les N étudiants (batch)
- Équilibre stabilité et vitesse

### 5. **Prédiction sur de nouvelles données**

#### `make predict`
Utilise les poids entraînés pour prédire sur le dataset de test.
```bash
make predict
```
- Charge les poids de `output/weights.csv`
- Prédit la maison pour les 400 étudiants de test
- Sauvegarde dans `output/houses.csv`

---

## Explications des algorithmes

### 1. Descente de gradient (Batch Gradient Descent)

#### Principe général

La descente de gradient est un **algorithme d'optimisation** qui ajuste les poids pour minimiser l'erreur.

#### Analogie : descente de montagne

Imaginez que vous êtes au sommet d'une montagne dans le brouillard :
- Vous ne voyez pas le sommet, mais vous voyez la pente juste devant vous
- Vous descendez dans la direction la plus raide
- Vous continuez jusqu'à trouver la vallée

Les poids fonctionnent de la même manière : on descend le gradient de la fonction de coût.

#### Formules mathématiques

1. **Calcul des prédictions**
```
Pour chaque étudiant i :
z_i = w₀ + Σ(w_j * x_{i,j})
ŷ_i = sigmoid(z_i) = 1 / (1 + e^(-z_i))
```

2. **Calcul des erreurs**
```
Pour chaque étudiant i :
error_i = ŷ_i - y_i
         = (prédiction) - (vraie valeur)
```

3. **Calcul des gradients**
```
Gradient pour le biais :
∇w₀ = (1/n) * Σ(error_i)

Gradient pour les poids :
∇w_j = (1/n) * Σ(error_i * x_{i,j})
```

Intuition : 
- Si on s'est trompé sur un étudiant, on pénalise les features qui ont contribué
- Si une feature était élevée et on s'est trompé, on réduit son poids

4. **Mise à jour des poids**
```
w₀_new = w₀_old - learning_rate * ∇w₀
w_j_new = w_j_old - learning_rate * ∇w_j
```

#### Pseudocode

```
Initialiser tous les poids à 0
Pour chaque époque (passage sur les données) :
    1. Prédire pour tous les 1600 étudiants
    2. Calculer l'erreur pour chaque étudiant
    3. Calculer les gradients moyens
    4. Mettre à jour tous les poids à la fois
    5. Répéter jusqu'à convergence
```

#### Avantages et inconvénients

**Avantages** :
- Convergence garantie vers un minimum
- Trajectoire stable et prédictible
- Fonctionne bien avec des datasets de taille moyenne

**Inconvénients** :
- Lent pour les très grands datasets
- Peut rester coincé dans les minima locaux
- Nécessite de charger toutes les données en mémoire

#### Taux d'apprentissage (Learning Rate)

C'est un **hyperparamètre** crucial :

```
learning_rate = 0.1  (par défaut)
```

- **Trop petit** (0.001) : convergence très lente
- **Trop grand** (1.0) : peut osciller sans converger
- **Juste** (0.1) : converge efficacement

Représentation visuelle :

```
Fonction de coût
      │
  100 │ ●
      │   ●
   50 │     ●  ← Learning rate trop grand (grande saute)
      │       ●
    0 │         ●●●●●●  ← Oscillations
      └────────────────── Epochs

Fonction de coût
      │
  100 │ ●●●●●●●●●●  ← Learning rate trop petit
      │ (progression très lente)
   50 │
      │
    0 │
      └────────────────── Epochs

Fonction de coût
      │
  100 │ ●
      │   ●
   50 │     ●  ← Learning rate idéal
      │       ●
    0 │         ●●●●●●  ← Convergence lisse
      └────────────────── Epochs
```

### 2. Stochastic Gradient Descent (SGD)

#### Différence avec Batch GD

**Batch GD** :
- Met à jour après avoir vu **tous les 1600** étudiants
- 1 mise à jour par époque

**SGD** :
- Met à jour après avoir vu **1 seul** étudiant
- 1600 mises à jour par époque

#### Formules

```
Pour chaque étudiant i (dans un ordre aléatoire) :
    z_i = w₀ + Σ(w_j * x_{i,j})
    ŷ_i = sigmoid(z_i)
    error_i = ŷ_i - y_i
    
    w₀_new = w₀_old - learning_rate * error_i
    w_j_new = w_j_old - learning_rate * error_i * x_{i,j}
```

Remarque : pas de moyenne sur n exemples, mise à jour immédiate.

#### Pseudocode

```
Initialiser tous les poids à 0
Pour chaque époque :
    Mélanger aléatoirement l'ordre des étudiants
    Pour chaque étudiant dans l'ordre aléatoire :
        1. Prédire pour cet étudiant
        2. Calculer l'erreur
        3. Mettre à jour les poids immédiatement
```

#### Visualisation

```
Coût lors de l'entraînement avec SGD vs Batch GD :

         Batch GD           SGD
Coût │    ●              ╱╲╱╲╱╲╱╲
     │     ●           ╱╲╱
     │      ●        ╱╲╱
     │       ●     ╱╲╱
     │        ●  ╱╲╱
     │         ●╱╱╱
     │         ●      ← Converge au même endroit
     └─────────────── Epochs
     
Batch : courbe lisse et monotone
SGD   : courbe bruyante mais moyenne décroissante
```

#### Avantages et inconvénients

**Avantages** :
- Beaucoup plus rapide que Batch GD
- Peut s'échapper des minima locaux (grâce au bruit)
- Idéal pour les grands datasets
- Peut traiter les données en streaming

**Inconvénients** :
- Trajectoire bruyante et imprévisible
- Peut diverger si learning_rate est trop grand
- Pas garantie de trouver le minimum global

#### Quand l'utiliser

- Datasets > 10,000 exemples
- Apprentissage en ligne (données arrivent en continu)
- Quand la rapidité est prioritaire

### 3. Mini-Batch Gradient Descent

#### Concept

C'est un **compromis** entre Batch et SGD :
- Traiter les données par **petits lots** (par exemple, 32 étudiants à la fois)
- Moins de mises à jour que SGD (plus stable)
- Plus rapide que Batch GD

#### Formules

```
Diviser les 1600 étudiants en batches (par exemple : 32 étudiants)
Nombre de batches = 1600 / 32 = 50 batches

Pour chaque époque :
    Mélanger aléatoirement l'ordre des batches
    Pour chaque batch de 32 étudiants :
        1. Prédire pour les 32 étudiants
        2. Calculer les 32 erreurs
        3. Calculer le gradient moyen sur les 32
        4. Mettre à jour les poids une fois
```

#### Pseudocode

```
Initialiser tous les poids à 0
batch_size = 32
Pour chaque époque :
    Mélanger les indices
    Pour chaque position i = 0, 32, 64, ..., 1600 :
        batch = étudiants[i : i+32]
        
        # Calcul des prédictions pour le batch
        gradients = []
        Pour chaque étudiant du batch :
            prédire
            calculer erreur
            ajouter à gradients
        
        # Mise à jour basée sur la moyenne du batch
        gradient_moyen = moyenne(gradients)
        mise à jour des poids
```

#### Comparaison visuelle

```
Nombre de mises à jour par époque :
- Batch GD      : 1 mise à jour
- Mini-Batch    : 50 mises à jour (1600/32)
- SGD           : 1600 mises à jour

Coût lors de l'entraînement :

Batch GD    : ●              (lisse, 1 point par époque)
              ●
              ●

SGD         : ● ● ● ● ● (bruit, 1600 points par époque)
              ● ● ● ●

Mini-Batch  : ●  ●  ●   (équilibre, 50 points par époque)
              ●  ●  ●
```

#### Avantages et inconvénients

**Avantages** :
- Équilibre entre stabilité et rapidité
- Moins d'impact du bruit qu'avec SGD
- Utilisation efficace du GPU/parallélisation
- Converge généralement plus vite que Batch
- Plus stable que SGD

**Inconvénients** :
- Choix du batch_size est important (hyperparamètre)
- Plus complexe que Batch GD

#### Quand l'utiliser

- **Cas idéal** : datasets de 1,000 à 1,000,000 exemples
- Quand on a ressources limitées mais qu'on veut la vitesse
- Dans la plupart des applications modernes

#### Choix du batch_size

```
Batch size = 1       → Stochastic GD
Batch size = n       → Batch GD
Batch size = 32, 64  → Mini-batch (optimal généralement)

Règle générale :
- Petit dataset (< 1000) : 32 - 64
- Dataset moyen (1000-10000) : 64 - 128
- Grand dataset (> 10000) : 128 - 256
```

#### Configuration du projet

```python
# Dans logreg_train.py
mini_batch_gradient_descent(
    notesByStudents,
    labels,
    weights,
    batch_size=32,          # Paire : 32 étudiants par batch
    learning_rate=0.1,
    epochs=500
)
```

### Comparaison synthétique des trois algorithmes

| Aspect | Batch GD | SGD | Mini-Batch |
|--------|----------|-----|-----------|
| **Mises à jour/époque** | 1 | n | n/batch_size |
| **Bruit** | Aucun | Très haut | Modéré |
| **Stabilité** | Excellente | Mauvaise | Bonne |
| **Vitesse** | Lente | Rapide | Très rapide |
| **Convergence** | Garantie | Non garantie | Garantie |
| **Mémoire** | Beaucoup | Peu | Peu |
| **Paramétrisation** | Simple | Simple | batch_size à tuner |
| **Idéal pour** | Petit dataset | Grand dataset | La plupart des cas |

---

## Statistiques descriptives

### Comprendre describe.py

Ce fichier calcule et affiche les **statistiques de base** de chaque cours académique.

### Les statistiques de base

#### 1. **Count** (Nombre de valeurs)

```
Définition : Nombre d'étudiants avec une note valide
Exemple    : 1550 (sur 1600 étudiants, 50 ont une note manquante)
Interprétation : 1550 données valides pour cette matière
```

#### 2. **Mean** (Moyenne)

```
Formule : μ = (1/n) * Σ(x_i)
Exemple : 15.34
Interprétation : 
  - Performance moyenne en cette matière
  - Si > 0 : forces globales
  - Si < 0 : faiblesses globales (notes négatives possibles!)
```

#### 3. **Std** (Écart-type)

```
Formule : σ = √[(1/n) * Σ(x_i - μ)²]
Exemple : 8.67
Interprétation :
  - Petite valeur (1-2) : notes très similaires, peu de variation
  - Grande valeur (> 10) : notes très différentes entre étudiants
  - Indique la "dispersion" des performances
```

Visualisation :

```
Petite écart-type :    Grande écart-type :
        ▁▃▄▅▆▅▄▃▁            ▁  ▂  ▃  ▄  ▆  ▇
     (concentré)     (dispersé)
```

#### 4. **Min et Max** (Minimum et Maximum)

```
Min : Note la plus basse
Max : Note la plus haute
Exemple : Min = -50.5, Max = 150.2
Interprétation : Étendue des notes possible
```

#### 5. **25%, 50%, 75%** (Quartiles/Percentiles)

```
Définition : Valeurs qui divisent les données en 4 parts égales

25% (1er quartile Q1) : 25% des étudiants ont une note ≤ Q1
50% (Médiane Q2)      : 50% des étudiants ont une note ≤ Q2
75% (3e quartile Q3)  : 75% des étudiants ont une note ≤ Q3

Exemple :
  25% : 8.2    (25% des étudiants < 8.2)
  50% : 15.1   (50% des étudiants < 15.1)
  75% : 22.5   (75% des étudiants < 22.5)

Interprétation : Forme et position de la distribution
```

Visualisation :

```
Distribution des notes avec quartiles

Count
  |
  |     ▁▂▃▄▅▆▇█
  |   ▁▃▄▅▆▇█
  | ▁▂▃▄▅▆▇██
  |▁▂▃▄▅▆▇██
  |___________________________
   -50  Q1  Q2  Q3  150
        (8.2)(15.1)(22.5)
```

### Les statistiques BONUS (--bonus)

#### 1. **Variance**

```
Formule : σ² = (1/n) * Σ(x_i - μ)²
Exemple : 75.17
Interprétation : Écart-type au carré, mesure la "dispersion"
Relation : Variance = Écart-type²
Note : Moins intuitive que l'écart-type, plus facile à calculer
```

#### 2. **Skewness** (Asymétrie)

```
Formule : γ₁ = (1/n) * Σ((x_i - μ) / σ)³
Gamme : De -∞ à +∞ (généralement -1 à 1)

Valeur négative (< -0.5) : Distribution asymétrique à gauche
  - Plus de notes élevées que basses
  - "Queue" vers les valeurs négatives
  - Exemple : Skewness = -0.8

   Fréquence
      |
      |    ▁▂▃
      |  ▁▃▅▇█   ← Queue vers la gauche
      |▂▅████
      |___________
       -50  0  50

Valeur proche de 0 : Distribution symétrique
  - Notes bien réparties de chaque côté
  - Exemple : Skewness = 0.05

   Fréquence
      |    ▂▃▂
      |  ▂▅▇█▅▂
      |▁▃▆██▆▃▁
      |___________

Valeur positive (> 0.5) : Distribution asymétrique à droite
  - Plus de notes basses que élevées
  - "Queue" vers les valeurs positives
  - Exemple : Skewness = 0.9

   Fréquence
      |        ▁▂▃
      |    ▂▅▇█▅▂▁
      |▇█████████
      |___________
```

Interprétation pour nos données :
- Skewness positif → Matière facile (beaucoup d'étudiants réussissent)
- Skewness négatif → Matière difficile (beaucoup d'étudiants échouent)

#### 3. **Kurtosis** (Aplatissement/Poids des queues)

```
Formule : γ₂ = (1/n) * Σ((x_i - μ) / σ)⁴ - 3
Gamme : De -∞ à +∞ (généralement -1 à 5)

Kurtosis < 0 (Platykurtique) : Distribution "aplatie"
  - Queues légères
  - Moins de valeurs extrêmes
  - Plus d'étudiants près de la moyenne
  
   Fréquence
      |
      |  ▂▂▂▂▂▂▂▂  ← Distribution plate
      |  ▂▂▂▂▂▂▂▂
      |__________

Kurtosis ≈ 0 (Mésokurtique) : Distribution "normale"
  - Queues modérées
  - Allure en cloche classique
  
   Fréquence
      |    ▁▃▅▇▅▃▁
      |  ▂▆███████▆▂
      |▁▄███████████▄▁
      |__________

Kurtosis > 0 (Leptokurtique) : Distribution "pointue"
  - Queues lourdes
  - Beaucoup de valeurs extrêmes
  - Pic au centre
  
   Fréquence
      |       ▅█▅
      |      ▅███▅
      |    ▂▅████▅▂  ← Valeurs extrêmes
      |▁▂▅████████▅▂▁
      |__________
```

Interprétation pour nos données :
- Kurtosis > 0 → Quelques étudiants "exceptionnels" (très bons ou très mauvais)
- Kurtosis < 0 → Performances plus uniformes

### Comment lire le tableau complet ?

```
          Astronomy  Herbology  Defense...
Count     1600.0     1550.0     1600.0       (1600 étudiants par matière)
Mean      12.34      15.67      -8.45        (Forces/faiblesses)
Std       8.45       9.12       10.23        (Dispersions)
Min       -100.0     -150.0     -200.0       (Pires notes)
25%       6.78       10.34      -15.23       (Quartiles)
50%       12.45      16.78      -9.12
75%       18.90      22.34      -2.56
Max       150.0      200.0      150.0        (Meilleures notes)
Variance  71.40      83.17      104.65       (BONUS)
Skewness  -0.23      0.15       0.45         (BONUS)
Kurtosis  0.67       -0.12      1.23         (BONUS)
```

---

## Fichiers Python - Guide complet

### 1. **utils.py** - Les utilitaires de base

Fichier contenant les fonctions réutilisables dans tout le projet.

#### Fonctions principales

##### `read_csv(filepath)`
```python
header, data = read_csv("datasets/dataset_train.csv")
```
- **Entrée** : Chemin vers un fichier CSV
- **Sortie** : 
  - `header` : Liste des noms de colonnes
  - `data` : Dictionnaire {nom_colonne: [valeurs]}
- **Utilité** : Lire tous les datasets du projet

##### `get_numerical_columns(header)`
```python
cols = get_numerical_columns(header)
# Résultat : ["Arithmetic", "Astronomy", ..., "Flying"]
```
- **Entrée** : Liste des colonnes
- **Sortie** : Liste des colonnes numériques (exclut Index, Hogwarts House, etc.)
- **Utilité** : Filtrer les données pour les algorithmes ML

##### `get_columns_for_gradient()`
```python
cols = get_columns_for_gradient()
# Résultat : ["Astronomy", "Herbology", "Defense Against the Dark Arts", ...]
```
- **Sortie** : Les 10 matières utilisées pour la régression logistique
- **Utilité** : Normaliser le choix des features dans tout le projet

##### `get_data_by_column(data, col_name)`
```python
notes = get_data_by_column(data, "Astronomy")
# Résultat : [15.3, 22.1, -5.2, 18.4, ...]
```
- **Entrée** : Le dictionnaire data et un nom de colonne
- **Sortie** : Liste de nombres (élimine les valeurs manquantes)
- **Utilité** : Extraire une colonne pour les statistiques

##### `get_data_by_column_with_none(data, col_name)`
```python
notes = get_data_by_column_with_none(data, "Astronomy")
# Résultat : [15.3, None, 22.1, None, -5.2, ...]
```
- **Sortie** : Même colonne mais avec None pour les valeurs manquantes
- **Utilité** : Conserver l'alignement des indices (pour la corrélation)

##### `calculateMean(values)`
```python
moyenne = calculateMean([10, 15, 20, 25])
# Résultat : 17.5
```
- **Formule** : Σ(x) / n
- **Utilité** : Calculer la moyenne (sans stdlib)

##### `calculateStandardDeviation(values, mean)`
```python
ecart_type = calculateStandardDeviation([10, 15, 20, 25], 17.5)
# Résultat : ≈ 6.45
```
- **Formule** : √[Σ(x - μ)² / n]
- **Utilité** : Mesurer la dispersion

##### `calculatePercentile(values, percentile)`
```python
q1 = calculatePercentile([10, 15, 20, 25, 30], 25)
# Résultat : 15
q2 = calculatePercentile([10, 15, 20, 25, 30], 50)
# Résultat : 20
```
- **Utilité** : Calculer quartiles et percentiles

##### `ListEachNotesByHouse(data, col_name)`
```python
notes_par_maison = ListEachNotesByHouse(data, "Astronomy")
# Résultat : {
#   "Ravenclaw": [15.3, 22.1, 18.4, ...],
#   "Slytherin": [8.2, 12.5, ...],
#   "Gryffindor": [20.1, 25.3, ...],
#   "Hufflepuff": [14.2, 16.5, ...]
# }
```
- **Utilité** : Grouper les notes par maison (pour visualisations)

##### `GetNotesByStudents(data, numerical_cols)`
```python
notes = GetNotesByStudents(data, get_columns_for_gradient())
# Résultat : [
#   [15.3, 22.1, 18.4, 12.5, ...],  # Étudiant 1
#   [8.2, 12.5, 20.1, 15.3, ...],   # Étudiant 2
#   ...
# ]
```
- **Utilité** : Format des données pour l'entraînement (une ligne par étudiant)

#### Fonctions mathématiques implémentées

**Remarque importante** : Le projet **n'utilise que la stdlib Python** (pas de NumPy/Pandas pour les calculs). Toutes les statistiques et opérations mathématiques sont implémentées "from scratch" :

- Moyenne : boucle avec somme
- Écart-type : boucle avec somme des écarts au carré
- Percentiles : tri et interpolation linéaire
- Etc.

Cela montre comment les mathématiques fonctionnent en détail.

---

### 2. **describe.py** - Statistiques descriptives

Affiche un tableau de statistiques pour chaque matière académique.

#### Fonctionnement

```python
python3 describe.py datasets/dataset_train.csv
```

**Étapes** :
1. Lire le CSV
2. Identifier les colonnes numériques
3. Pour chaque colonne :
   - Calculer count, mean, std, min, max, Q1, Q2, Q3
   - Afficher dans un tableau formaté

#### Fonction `describe_column(data, col_name, show_bonus=False)`

Calcule tous les statistiques pour une colonne.

```python
stats = describe_column(data, "Astronomy", show_bonus=True)
# Résultat : {
#   'Count': 1550,
#   'Mean': 15.34,
#   'Std': 8.67,
#   'Min': -100.0,
#   '25%': 8.2,
#   '50%': 15.1,
#   '75%': 22.5,
#   'Max': 150.0,
#   'Variance': 75.17,
#   'Skewness': -0.23,
#   'Kurtosis': 0.67
# }
```

#### Interprétation des résultats

Pour chaque matière, on peut répondre à des questions :

- **Matière facile ou difficile ?** Regarder la Mean
  - Mean élevée → Facile (étudiants réussissent)
  - Mean négative → Très difficile

- **Variation de performance ?** Regarder Std
  - Std élevée → Grand écart entre bons et mauvais élèves
  - Std basse → Performances similaires

- **Distribution équilibrée ?** Regarder Skewness
  - Proche de 0 → Équilibrée
  - Négative → Plus de mauvaises notes
  - Positive → Plus de bonnes notes

---

### 3. **histogram.py** - Visualisation par histogrammes

Crée des histogrammes pour identifier les matières qui **différencient les maisons**.

#### Concept clé : Score d'homogénéité

On cherche les matières où les maisons ont des **distributions différentes** (non-homogènes).

**Score d'homogénéité** = Écart-type de les moyennes par maison / Moyenne générale

```
Si 4 maisons ont des moyennes similaires → Score bas → Homogène
Si 4 maisons ont des moyennes très différentes → Score haut → Non-homogène
```

#### Fonctionnement

```python
make histogram
```

**Étapes** :
1. Pour chaque matière, calculer le score d'homogénéité
2. Tracer l'histogramme de la matière **la moins homogène** (la plus différenciante)
3. Sauvegarder dans `output/histogram.png`

**Exemple d'histogramme** :

```
Astronomy
Frequency
   │
50 │     ▁▃▅
   │  ▂▅▇█▇▅▂    Ravenclaw (bleu)
40 │ ▃▆████▅▃
   │▁▄▇███████▄▁
30 │▂▅██████████▅▂
   │
   │        ▁▃▅
20 │    ▂▅▇█▇▅▂    Slytherin (vert)
   │▁▃▆▇██████▇▆▃▁
   │
10 │           ▁▃▅  Gryffindor (rouge)
   │        ▂▅▇█▇▅▂
   │    ▁▄▇████▅▄▁
   │_________________
    -50  0  50 100 150
```

#### Options

```python
make histogram_all
```

Crée une grille 4×4 avec **tous les histogrammes** (16 matières).

---

### 4. **scatter_plot.py** - Visualisation de corrélations

Crée des diagrammes de dispersion (scatter plots) pour voir les **corrélations** entre deux matières.

#### Coefficient de Pearson (r)

Mesure la corrélation linéaire entre deux variables.

```
Formule : r = Σ[(x_i - μ_x)(y_i - μ_y)] / (n * σ_x * σ_y)

Gamme : De -1 à 1

r = 1.0    : Corrélation positive parfaite
  Notes en y augmentent quand notes en x augmentent

r = -1.0   : Corrélation négative parfaite
  Notes en y diminuent quand notes en x augmentent

r = 0.0    : Aucune corrélation
  Pas de lien linéaire entre les deux

Interprétation des valeurs :
|r| < 0.3  : Corrélation faible
0.3 < |r| < 0.7 : Corrélation modérée
|r| > 0.7  : Corrélation forte
```

#### Visualisations

```python
make scatter
```

Crée un scatter plot pour la paire de matières avec la **plus grande corrélation**.

```python
make scatter_all
```

Crée une grille avec **tous les scatters** (pairs de matières).

#### Exemple

```
Astronomy vs Potions (r = 0.82)

Potions
  150 │                        ●
      │                      ● ●
  100 │                    ● ● ●  ← Correlation positive forte
      │              ● ● ● ● ● ● ●
   50 │        ● ● ● ● ● ● ● ●
      │    ● ● ● ● ●
    0 │●●●●●
      │____________________________
     -100  -50  0  50  100  150
           Astronomy
```

---

### 5. **pair_plot.py** - Matrice de plots

Crée une **matrice triangulaire** où :
- **Diagonale** : Histogrammes (distribution de chaque matière)
- **Triangle inférieur** : Scatter plots (corrélations entre paires)

#### Visualisation

Exemple d'une matrice 3×3 (pour 3 matières seulement) :

```
            Astronomy    Potions    Charms
Astronomy   [Histo]      [Invisible] [Invisible]
Potions     [Scatter]    [Histo]    [Invisible]
Charms      [Scatter]    [Scatter]  [Histo]
```

Utilité : Voir d'un coup d'œil toutes les corrélations.

---

### 6. **logreg_train.py** - Entraînement du modèle

Fichier principal pour entraîner les 4 modèles de régression logistique.

#### Étapes principales

1. **Lecture des données**
```python
header, data = read_csv(sys.argv[1])
```

2. **Extraction des notes**
```python
notesByStudents = GetNotesByStudents(data, numerical_cols)
```
Format : 1600 × 10 (1600 étudiants, 10 matières)

3. **Normalisation (0-1)**
```python
normalizedNotes, mins, maxs = normalize(notesByStudents)
```
Chaque note est ramenée à l'intervalle [0, 1] pour améliorer la convergence.

4. **Entraînement de 4 modèles**
```python
for house in ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']:
    labels = get_labels(data, house)  # 1 si cette maison, 0 sinon
    weights = [0] * 11  # 1 biais + 10 features
    
    weights = GradientDescent(normalizedNotes, labels, weights, ...)
    weights_by_house[house] = denormalize_weights(weights, mins, maxs)
```

5. **Sauvegarde des poids**
```python
save_weights(weights_by_house, "output/weights.csv", numerical_cols)
```

#### Fonction `normalize(notesByStudents)`

```python
# Avant normalisation :
[[-487.88, 5.73, 4.78, ...],
 [-552.06, -5.99, 5.52, ...],
 ...]

# Après normalisation :
[[0.0, 0.5, 0.3, ...],    # Chaque valeur dans [0, 1]
 [0.1, 0.2, 0.6, ...],
 ...]

# Calcul : x_norm = (x - min) / (max - min)
```

Avantages :
- Toutes les features ont la même échelle
- La descente de gradient converge plus vite
- Facilite la comparaison des poids

#### Fonction `GradientDescent(notes, labels, weights, lr, epochs)`

Implémente l'algorithme Batch Gradient Descent.

```python
for epoch in range(epochs):
    # Prédictions
    probabilities = predict(notes, weights)
    
    # Erreurs
    errors = [probabilities[i] - labels[i] for i in range(len(notes))]
    
    # Gradients
    grad_w0 = mean(errors)
    grad_wj = mean(error_i * note_ij pour chaque feature j)
    
    # Mise à jour
    weights[0] -= lr * grad_w0
    weights[j] -= lr * grad_wj pour j = 1..10
```

#### Fonction `sigmoid(z)`

```python
sigmoid(z) = 1 / (1 + e^(-z))

Exemples :
sigmoid(-10) ≈ 0.0000453
sigmoid(0)   = 0.5
sigmoid(10)  ≈ 0.9999547
```

#### Sauvegarde des poids (weights.csv)

```
Hogwarts House,Biais,Astronomy,Herbology,Defense Against the Dark Arts,...
Ravenclaw,2.345,-0.123,0.456,-0.789,...
Slytherin,-1.234,0.234,-0.567,0.890,...
Gryffindor,0.567,0.123,0.456,0.789,...
Hufflepuff,1.890,-0.234,0.567,0.123,...
```

**Interprétation** :
- Biais positif : "Tendance" du modèle vers cette maison
- Poids positif : Si la note augmente, probabilité de cette maison ↑
- Poids négatif : Si la note augmente, probabilité de cette maison ↓
- Poids proche de 0 : Feature peu importante pour cette maison

---

### 7. **logreg_predict.py** - Prédiction

Utilise les poids entraînés pour classer les nouveaux étudiants.

#### Étapes

1. **Charger les poids**
```python
weights_by_house = load_weights("output/weights.csv")
```

2. **Lire les données de test**
```python
header, data = read_csv(dataset_test)
notesByStudents = GetNotesByStudents(data, numerical_cols)
```

3. **Pour chaque étudiant**
```python
for student in notesByStudents:
    scores = {}
    for house, weights in weights_by_house.items():
        # Calcul de la probabilité pour cette maison
        z = weights[0] + sum(w_j * note_j for j in range(10))
        prob = sigmoid(z)
        scores[house] = prob
    
    predicted_house = max(scores, key=scores.get)
    predicted_houses.append(predicted_house)
```

4. **Sauvegarder les prédictions**
```
Index,Hogwarts House
0,Ravenclaw
1,Slytherin
2,Gryffindor
...
```

---

### 8. **bonus.py** - SGD, mini-batch et métriques

Fichier avec les optimisations et métriques additionnelles.

#### Stochastic Gradient Descent

```python
def stochastic_gradient_descent(notesByStudents, labels, weights, lr, epochs):
    for epoch in range(epochs):
        indices = shuffle(list(range(len(notesByStudents))))
        
        for idx in indices:
            z = weights[0] + sum(weights[j+1] * notesByStudents[idx][j] ...)
            pred = sigmoid(z)
            error = pred - labels[idx]
            
            weights[0] -= lr * error
            weights[j+1] -= lr * error * notesByStudents[idx][j]
    
    return weights
```

Différence clé : Mise à jour après **chaque étudiant**, pas après tous.

#### Mini-Batch Gradient Descent

```python
def mini_batch_gradient_descent(notesByStudents, labels, weights, 
                               batch_size, lr, epochs):
    for epoch in range(epochs):
        indices = shuffle(list(range(len(notesByStudents))))
        
        for batch_start in range(0, len, batch_size):
            batch_indices = indices[batch_start:batch_start+batch_size]
            
            # Accumuler gradients sur le batch
            grad_w0 = 0
            grad_wj = [0] * 10
            
            for idx in batch_indices:
                # Calcul (similaire à SGD)
                error = sigmoid(...) - labels[idx]
                grad_w0 += error
                grad_wj[j] += error * notesByStudents[idx][j]
            
            # Mise à jour basée sur la moyenne du batch
            weights[0] -= lr * (grad_w0 / batch_size)
            weights[j] -= lr * (grad_wj[j] / batch_size)
    
    return weights
```

#### Métriques

##### R² Score

```python
def calculate_r2_score(y_true, y_pred):
    mean_y = mean(y_true)
    ss_res = sum((y_true[i] - y_pred[i])² for i)
    ss_tot = sum((y_true[i] - mean_y)² for i)
    return 1 - (ss_res / ss_tot)
```

**Interprétation** :
- R² = 1.0 : Prédictions parfaites
- R² = 0.5 : 50% de la variance expliquée
- R² = 0.0 : Aussi bon qu'une moyenne constante
- R² < 0.0 : Pire que la moyenne constante

##### RMSE (Root Mean Squared Error)

```python
def calculate_rmse(y_true, y_pred):
    mse = mean((y_true[i] - y_pred[i])² for i)
    return sqrt(mse)
```

**Interprétation** :
- Valeur en "unités" des données
- RMSE petit → Prédictions proches
- Pénalise les grandes erreurs (au carré)

##### MAE (Mean Absolute Error)

```python
def calculate_mae(y_true, y_pred):
    return mean(abs(y_true[i] - y_pred[i]) for i)
```

**Interprétation** :
- Valeur en "unités" des données
- Pas de pénalité quadratique
- Plus robuste aux valeurs extrêmes que RMSE

##### MAPE (Mean Absolute Percentage Error)

```python
def calculate_mape(y_true, y_pred):
    return 100 * mean(abs((y_true[i] - y_pred[i]) / y_true[i]) for i)
```

**Interprétation** :
- En pourcentage d'erreur relative
- MAPE = 5% → Erreur moyenne de 5% sur les prédictions

##### Variance, Skewness, Kurtosis

Voir section [Statistiques descriptives](#statistiques-descriptives).

---

## Description du dataset

### Caractéristiques générales

```
Total d'étudiants    : 2000 (1600 entraînement + 400 test)
Nombre de features   : 31 colonnes
Features utilisées   : 10 (matières académiques)
Features ignorées    : 21 (Index, noms, dates, etc.)
```

### Colonnes du dataset

#### Colonnes d'identification (non-utilisées)
- **Index** : Numéro unique par étudiant (0-1599 pour train)
- **Hogwarts House** : Maison de l'étudiant (label)
- **First Name** : Prénom
- **Last Name** : Nom de famille
- **Birthday** : Date de naissance (format AAAA-MM-JJ)
- **Best Hand** : Main dominante (Left/Right)

#### Colonnes de matières académiques (10 features utilisées)

Les 10 matières utilisées pour la régression logistique :

1. **Astronomy** : Études celestes
2. **Herbology** : Étude des plantes magiques
3. **Defense Against the Dark Arts** : Combat magique
4. **Ancient Runes** : Runes anciennes
5. **Charms** : Enchantements
6. **Divination** : Divination / Prédiction
7. **Muggle Studies** : Étude du monde non-magique
8. **History of Magic** : Histoire du monde magique
9. **Potions** : Préparation de potions
10. **Flying** : Équitation sur balai magique

#### Colonnes bonus (14 features ignorées)

Ces features existent dans le dataset mais ne sont pas utilisées pour la régression :
- Arithmancy
- Care of Magical Creatures
- Transfiguration
- (et 11 autres...)

**Raison** : Voir `get_columns_for_gradient()` qui détermine les 10 matières officielles.

### Distribution des maisons

```
Entraînement (1600 étudiants) :
Ravenclaw    : ~400 (25%)
Slytherin    : ~400 (25%)
Gryffindor   : ~400 (25%)
Hufflepuff   : ~400 (25%)

(Distribution équilibrée = bon pour l'entraînement)

Test (400 étudiants) :
Pas de labels fournis (c'est notre job de prédire !)
Probablement même distribution
```

### Format des notes

**Remarque importante** : Les notes ne sont **pas** sur 20 !

```
Plage observée : De -550 à +1100 environ
Valeurs négatives : Oui, possibles (pénalités ?)
Valeurs extrêmes : Oui, possibles (bonus ?)

Exemples de notes :
- Astronomy : -487.88, 697.74, -366.07, ...
- Potions : -232.79, -252.18, -227.34, ...
- Charms : 0.72, 0.09, -0.52, ...

Distribution :
- Beaucoup de matières ont une moyenne négative
- Suggestions : Notes avec pénalités ou notes normalisées originalement
```

### Valeurs manquantes

```
Quelques étudiants ont des données manquantes
Gestion dans le projet : Remplissage par la moyenne de la matière

Exemple :
Note manquante en Potions pour l'étudiant 5
→ Remplacer par la moyenne générale des notes de Potions
```

### Comment les données ont été générées

Basé sur le contexte du projet (Harry Potter) :
- Les 4 maisons ont probablement des profils académiques différents
- **Ravenclaw** : Fortes en tout (intelligence)
- **Gryffindor** : Bons en combat (DADA)
- **Slytherin** : Bons en magie sombre (Potions)
- **Hufflepuff** : Équilibrés

Les features sont les **notes académiques** :
- Réelles, variées, avec des manquants
- Normalisées et transformées (d'où les valeurs bizarres)
- Probablement générées synthétiquement pour ce cours

---

## Métriques et interprétation

### Métriques de classification binaire (par maison)

Pour chaque modèle (Ravenclaw vs autres, etc.), on reçoit :

#### 1. Accuracy (Précision)

```
Définition : Pourcentage de prédictions correctes

Accuracy = (Correct) / (Total) * 100

Gamme : 0% à 100%

Exemple :
    Prédictions sur 1600 étudiants
    1520 correctes
    80 incorrectes
    
    Accuracy = 1520 / 1600 * 100 = 95%

Interprétation :
- 95% → Excellent
- 80% → Bon
- 50% → Random guess (inutile)
- 25% → Pire que random (sur 4 classes)
```

**Limitation** : Si une classe est très rare, l'accuracy peut être trompeuse.

#### 2. R² Score

```
Définition : Proportion de variance expliquée

Formule : R² = 1 - (SS_res / SS_tot)

Gamme : -∞ à 1.0

Exemple :
- R² = 0.95 → Le modèle explique 95% de la variance
- R² = 0.50 → Le modèle explique 50% de la variance
- R² = 0.00 → Le modèle ne fait pas mieux qu'une moyenne
- R² = -0.5 → Le modèle est pire qu'une moyenne

Interprétation :
- Pour la régression logistique (classification), une valeur de 
  R² = 0.7-0.9 est généralement bonne
- R² < 0.5 : Modèle faible
- R² > 0.9 : Modèle excellent
```

#### 3. RMSE (Root Mean Squared Error)

```
Définition : Racine de l'erreur quadratique moyenne

Formule : RMSE = √(mean(erreur²))

Unité : Même que les données

Exemple :
Si les prédictions sont des probabilités (0-1) :
    RMSE = 0.05 → Erreur moyenne de 0.05 en probabilité
    
Interprétation :
- RMSE petit → Prédictions proches des vraies valeurs
- Pénalise les grandes erreurs (au carré)
- Utile pour identifier les erreurs graves
```

#### 4. MAE (Mean Absolute Error)

```
Définition : Erreur absolue moyenne

Formule : MAE = mean(|erreur|)

Unité : Même que les données

Exemple :
    Erreurs : [0.05, -0.03, 0.01, ...]
    MAE = mean(0.05, 0.03, 0.01, ...) = 0.02
    
Interprétation :
- MAE petit → Prédictions proches
- Sans pénalité quadratique (plus robuste aux outliers)
- Plus facile à interpréter que RMSE
```

#### 5. MAPE (Mean Absolute Percentage Error)

```
Définition : Erreur en pourcentage relatif

Formule : MAPE = 100 * mean(|erreur / vraie_valeur|)

Unité : Pourcentage

Exemple :
    Erreurs en pourcentage : [2%, 3%, 1%, ...]
    MAPE = 2.0%
    
Interprétation :
- MAPE = 5% → Erreur moyenne de 5% en relatif
- Utile pour comparer des prédictions à différentes échelles
- À prendre avec prudence si vraies_valeurs contiennent des 0

Note : Dans notre projet, c'est surtout un R² réinterprété
```

### Affichage des métriques

```bash
$ make train
Gryffindor: Accuracy = 91.2340%, R² Score = 0.7856, RMSE = 0.1234, MAE = 0.0987, MAPE = 9.87%
Hufflepuff: Accuracy = 89.5620%, R² Score = 0.7123, RMSE = 0.1456, MAE = 0.1123, MAPE = 11.23%
Ravenclaw: Accuracy = 93.4560%, R² Score = 0.8234, RMSE = 0.1012, MAE = 0.0856, MAPE = 8.56%
Slytherin: Accuracy = 90.8790%, R² Score = 0.7654, RMSE = 0.1289, MAE = 0.0945, MAPE = 9.45%
```

### Matrice de confusion (non-affichée, mais conceptuelle)

Pour évaluer un modèle binaire complet :

```
                 Prédiction
           Ravenclaw  Autre
Réel   Ravenclaw    TP      FN
       Autre        FP      TN

TP (True Positive)  : Bien classé comme Ravenclaw
FP (False Positive) : Mal classé comme Ravenclaw
FN (False Negative) : Mal classé comme autre
TN (True Negative)  : Bien classé comme autre

Métriques dérivées :
Précision = TP / (TP + FP)     (Quand on prédit Ravenclaw, on a raison combien de fois ?)
Rappel    = TP / (TP + FN)     (On détecte combien de vrais Ravenclaw ?)
F1        = 2 * (Precision * Recall) / (Precision + Recall)  (Moyenne harmonique)
```

---

## Comment interpréter les résultats

### Analyse des poids

Après l'entraînement, on a 4 modèles (1 par maison) dans `output/weights.csv` :

```csv
Hogwarts House,Biais,Astronomy,Herbology,Defense Against the Dark Arts,...
Ravenclaw,2.345,-0.123,0.456,-0.789,...
```

#### Interprétation

**Biais = 2.345** :
- Positive → Le modèle a une "tendance" à prédire Ravenclaw
- Negative → Le modèle a une tendance à prédire "Pas Ravenclaw"

**Astronomy = -0.123** :
- Négative → Plus la note d'Astronomy augmente, moins c'est Ravenclaw
- Positive → Plus la note d'Astronomy augmente, plus c'est Ravenclaw
- Proche de 0 → Astronomy n'est pas discriminant pour Ravenclaw

**Exemple d'interprétation complet** :
```
Pour Slytherin, supposons :
Biais = -1.5
Potions = 0.8
Astronomy = -0.2

Interprétation :
- Slytherin a un biais de -1.5 (tendance à prédire "Pas Slytherin")
- Si la note de Potions augmente → Probabilité Slytherin ↑ (+0.8)
- Si la note d'Astronomy augmente → Probabilité Slytherin ↓ (-0.2)

Signification : Les Slytherin sont bons en Potions, faibles en Astronomy
```

### Analyse de la performance

#### Comparaison entre algorithmes

```
              Batch    SGD    Mini-Batch
Gryffindor    91.2%   87.5%   90.8%
Hufflepuff    89.6%   85.2%   89.3%
Ravenclaw     93.5%   91.2%   93.1%
Slytherin     90.9%   88.3%   90.6%

Observations :
- Mini-Batch quasi égal à Batch (bon compromis)
- SGD un peu moins bon mais plus rapide
- Ravenclaw easiest (93.5%), Hufflepuff hardest (89.6%)
```

#### Comparaison entre matières

Regarder les poids absolus moyens par matière :

```
Matière                    Moyenne des |poids| par maison
Potions                    0.45  → Très discriminant
Defense Against the Dark   0.38  → Discriminant
Charms                     0.35  → Discriminant
Flying                     0.12  → Peu discriminant
Astronomy                  0.08  → Très peu discriminant
```

Interprétation :
- Potions : Les maisons ont des profils très différents (Slytherin?)
- Astronomy : Peu de différence entre les maisons

### Erreurs et problèmes courants

#### 1. **Accuracy basse (< 50%)**
```
Cause probable :
- Learning rate trop haut → ne converge pas
- Trop peu d'epochs → entraînement insuffisant
- Données mal normalisées

Solution :
- Réduire learning_rate
- Augmenter epochs
- Vérifier la normalisation
```

#### 2. **Accuracy stagne au 25%**
```
Interprétation : Le modèle prédit toujours la même classe
(sur 4 classes, prédire toujours la même = 25% de chance)

Cause probable :
- Données très déséquilibrées
- Weights initialisés mal
- One-vs-All n'a pas convergé correctement

Solution :
- Vérifier balance des classes
- Augmenter epochs
- Réduire learning_rate
```

#### 3. **Modèle fonctionne mal sur test**
```
Interprétation : Overfitting (surapprentissage)
- Fonctionne bien en entraînement
- Mal en test

Cause probable :
- Trop d'epochs (mémorisation)
- Les données d'entraînement et test trop différentes
- Un feature "leak" donnait des indices

Solution :
- Réduire epochs
- Validation croisée
- Régularisation (L1/L2) - non implémentée ici
```

#### 4. **Une maison toujours correcte, une toujours fausse**
```
Exemple : Ravenclaw 95%, Hufflepuff 60%

Cause probable :
- Hufflepuff a des patterns moins clairs
- Overlap avec d'autres maisons
- Features pas assez discriminantes

Analyse :
- Voir `make histogram` : Hufflepuff "mélangée" ?
- Voir `make scatter` : Corrélations insuffisantes ?
- Considérer d'autres features
```

### Évaluation globale du modèle

```
Performance globale = (Accuracy Gryffindor + ... + Accuracy Hufflepuff) / 4

Benchmark (approximate) :
- < 60% : Modèle faible
- 60-75% : Modèle acceptable
- 75-85% : Modèle bon
- 85-95% : Modèle excellent
- > 95% : Surapprentissage probable
```

### Résultats attendus

Basé sur une exécution standard :

```
$ make train
Gryffindor:  Accuracy ≈ 90-95%
Hufflepuff:  Accuracy ≈ 85-92%
Ravenclaw:   Accuracy ≈ 92-96%
Slytherin:   Accuracy ≈ 88-94%

Moyenne globale : ≈ 91% (excellent)
```

Cela indique que les features (matières académiques) sont **bien corrélées** avec les maisons.

---

## Conclusion

Ce projet démontre :

✅ **Concepts fondamentaux du ML** :
- Régression logistique
- Classification multi-classe (One-vs-All)
- Normalisation des données
- Optimisation par descente de gradient

✅ **Algorithmes d'optimisation** :
- Batch Gradient Descent (stable)
- Stochastic Gradient Descent (rapide)
- Mini-Batch Gradient Descent (équilibre)

✅ **Analyse et visualisation** :
- Statistiques descriptives
- Histogrammes et corrélations
- Interprétation des résultats

✅ **Implémentation from scratch** :
- Aucune lib ML (scikit-learn uniquement pour les métriques finales)
- Mathématiques implémentées explicitement
- Bonne compréhension des algorithmes

---

## Ressources supplémentaires

### Pour aller plus loin

**Sujets avancés** :
- Régularisation L1/L2 (prévenir l'overfitting)
- Cross-validation (meilleure évaluation)
- Softmax regression (alternative au One-vs-All)
- Réseau de neurones (généralisation de la logistic reg)

**Améliorations possibles** :
- Feature engineering (créer de nouvelles features)
- Sélection de features (garder les meilleures)
- Hyperparameter tuning (trouver le meilleur learning_rate)
- Ensemble learning (combiner plusieurs modèles)

---

**Projet réalisé sans dépendances ML externes** - Toutes les mathématiques sont transparentes ! 🎓
