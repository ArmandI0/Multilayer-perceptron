# Initialisation des poids dans les réseaux de neurones

Ce guide présente les 4 principales fonctions d'initialisation des poids utilisées dans les réseaux de neurones.

## 1. Initialisation Uniforme Simple

**Formule**: `W ~ U(-a, +a)`
- W : matrice des poids
- a : limite de l'intervalle (typiquement entre 0.01 et 0.05)

**Implémentation**:
```python
import numpy as np

def uniform_init(n_in, n_out, a=0.05):
    return np.random.uniform(-a, a, size=(n_in, n_out))
```

**Utilisation recommandée**:
- Pour des tests rapides
- Réseaux peu profonds
- Quand la simplicité est prioritaire

## 2. Initialisation Xavier/Glorot

**Formule**: 
- Uniforme: `W ~ U(-√(6/(n_in + n_out)), +√(6/(n_in + n_out)))`
- Normale: `W ~ N(0, √(2/(n_in + n_out)))`

**Implémentation**:
```python
def glorot_uniform(n_in, n_out):
    limit = np.sqrt(6/(n_in + n_out))
    return np.random.uniform(-limit, limit, size=(n_in, n_out))
```

**Utilisation recommandée**:
- Fonctions d'activation sigmoid ou tanh
- Réseaux profonds
- Quand le contrôle de la variance est important

## 3. Initialisation He

**Formule**:
- Normale: `W ~ N(0, √(2/n_in))`
- Uniforme: `W ~ U(-√(6/n_in), +√(6/n_in))`

**Implémentation**:
```python
def he_normal(n_in, n_out):
    return np.random.normal(0, np.sqrt(2/n_in), size=(n_in, n_out))
```

**Utilisation recommandée**:
- Fonction d'activation ReLU
- Réseaux très profonds
- Architecture moderne type ResNet

## 4. Initialisation LeCun

**Formule**:
- Normale: `W ~ N(0, √(1/n_in))`
- Uniforme: `W ~ U(-√(3/n_in), +√(3/n_in))`

**Implémentation**:
```python
def lecun_uniform(n_in, n_out):
    limit = np.sqrt(3/n_in)
    return np.random.uniform(-limit, limit, size=(n_in, n_out))
```

**Utilisation recommandée**:
- Fonction d'activation tanh
- Premier choix historique pour normalisation
- Quand seule la dimension d'entrée compte

## Points importants à retenir

1. **Choix de la méthode**:
   - Dépend de la fonction d'activation
   - Dépend de la profondeur du réseau
   - Impact sur la convergence

2. **Dimensionnement**:
   - Plus il y a de neurones, plus les poids sont petits
   - Évite l'explosion des valeurs
   - Maintient la variance du signal

3. **Framework support**:
```python
# PyTorch
torch.nn.init.xavier_uniform_(layer.weight)

# TensorFlow
tf.keras.initializers.GlorotUniform()
```

4. **Bonnes pratiques**:
   - Toujours initialiser les biais à 0 ou de petites valeurs
   - Vérifier la distribution des activations après initialisation
   - Adapter selon les résultats empiriques