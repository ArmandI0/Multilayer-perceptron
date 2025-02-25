# Backpropagation

## Pour la couche de sortie :

### 1-Calculer l'erreur : 
```
output_error = target - output
dE_dy = derivee partiel de la fonction de cout par rapport a yR
```
```math
∂y∂E​=∂y∂​E(y,t)
```

### 2-Calculer le delta : 
```
output_delta = output_error * dérivée_activation
dE_dz = dE_dy * deriveePartiel de la foncton d'activation pa rapport a Z
```
```math
∂z∂E​=∂y∂E​⋅∂z∂y​=∂y∂​E(y,t)⋅∂z∂​f(z)
```

### 3-Calcul du gradient :
Gradient = Transpose des entrees de la couche * delta

```math

∂W∂E​=AT⋅δ

```
### 4-Mise a jour des poids :

```
    Poids mis a jour = Poids - learning rate * gradient
```



```math
    W(t+1)=W(t)−η⋅∂W∂E​
```
### 5-Mise à jour des biais :
```
biais_mis_à_jour = biais_actuels - learning_rate * delta
```

$$b(t+1) = b(t) - \eta \cdot \delta$$

## Pour les couches cachees:

### 1-Calculer l'erreur:

dans le code on le calcul dans la couche directement et on le return c'est dE_dz
```
hidden_error = matrice_poids_couche_suivante.T · delta_couche_suivante
```

$\frac{\partial E}{\partial z^{(l-1)}} = \frac{\partial E}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial z^{(l-1)}} = (W^{(l)})^T \cdot \delta^{(l)}$

### 2-Calculer le delta :
```
hidden_delta = hidden_error * dérivée_activation_couche_cachée
```

$\frac{\partial E}{\partial z^{(l-1)}} = \frac{\partial E}{\partial a^{(l-1)}} \cdot \frac{\partial a^{(l-1)}}{\partial z^{(l-1)}} = (W^{(l)})^T \cdot \delta^{(l)} \odot \frac{\partial f(z^{(l-1)})}{\partial z^{(l-1)}}$

Où ⊙ représente le produit élément par élément (Hadamard) et f' est la dérivée de la fonction d'activation.

### 3-Calcul du gradient :
```
gradient_couche_cachée = activations_couche_précédente.T · hidden_delta
```

$\frac{\partial E}{\partial W^{(l-1)}} = \frac{\partial E}{\partial z^{(l-1)}} \cdot \frac{\partial z^{(l-1)}}{\partial W^{(l-1)}} = a^{(l-2)} \cdot (\frac{\partial E}{\partial z^{(l-1)}})^T$

### 4-Mise à jour des poids :
```
poids_mis_à_jour = poids_actuels - learning_rate * gradient_couche_cachée
```

$W^{(l-1)}(t+1) = W^{(l-1)}(t) - \eta \cdot \frac{\partial E}{\partial W^{(l-1)}}$

### 5-Mise à jour des biais :
```
biais_mis_à_jour = biais_actuels - learning_rate * hidden_delta
```

$b^{(l-1)}(t+1) = b^{(l-1)}(t) - \eta \cdot \frac{\partial E}{\partial b^{(l-1)}} = b^{(l-1)}(t) - \eta \cdot \frac{\partial E}{\partial z^{(l-1)}}$