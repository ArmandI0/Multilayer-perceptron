## Pour les couches cachees
1 : 
```
Delta_couche_cachée = Erreur_couche_cachée × Dérivée_fonction_activation(entrée_couche_cachée)

```

2 : 

```
Nouveau_poids = Ancien_poids + (Taux_apprentissage × Entrée_couche × Delta_couche_cachée)

```
Le processus se déroule ainsi :

Pour la couche de sortie :

Calculer l'erreur : output_error = target - output
Calculer le delta : output_delta = output_error * dérivée_activation(output_input)


Pour la couche cachée 3 :

Calculer l'erreur : hidden3_error = output_delta × weights_hidden3_to_output.T
Calculer le delta : hidden3_delta = hidden3_error * dérivée_activation(hidden3_input)
Mettre à jour les poids : weights_hidden3_to_output += hidden3 × output_delta × learning_rate


Pour la couche cachée 2 :

Calculer l'erreur : hidden2_error = hidden3_delta × weights_hidden2_to_hidden3.T
Calculer le delta : hidden2_delta = hidden2_error * dérivée_activation(hidden2_input)
Mettre à jour les poids : weights_hidden2_to_hidden3 += hidden2 × hidden3_delta × learning_rate


Pour la couche cachée 1 :

Calculer l'erreur : hidden1_error = hidden2_delta × weights_hidden1_to_hidden2.T
Calculer le delta : hidden1_delta = hidden1_error * dérivée_activation(hidden1_input)
Mettre à jour les poids : weights_hidden1_to_hidden2 += hidden1 × hidden2_delta × learning_rate
Mettre à jour les poids : weights_input_to_hidden1 += input × hidden1_delta × learning_rate
