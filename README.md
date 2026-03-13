# Self-Supervised-Temporal-Interpolation-for-Geospatial-Data-My-implementation

Interpolation Temporelle Vidéo pour les Données Géospatiales (STint)
Ce dépôt contient l'implémentation et l'évaluation du modèle STint (Self-supervised Temporal Interpolation), une méthode d'interpolation temporelle non supervisée spécialement conçue pour les données géospatiales. Ce projet a été réalisé dans le cadre du Master 2 IMA du Département d'Informatique de Sorbonne Université.

🌍 Contexte et Motivation
L'interpolation vidéo classique repose généralement sur des algorithmes de flux optique (optical flow) qui modélisent le mouvement des pixels entre des images consécutives. Or, les données géospatiales et climatiques (comme la température, les nuages, les courants marins) présentent des résolutions temporelles plus faibles et des déformations non rigides extrêmement complexes qui invalident les hypothèses de base du flux optique.

Ce projet implémente une solution alternative robuste : un apprentissage auto-supervisé indépendant du flux optique (flow-agnostic), permettant de générer des images intermédiaires cohérentes pour l'imagerie satellite et la modélisation climatique, sans nécessiter de vérité terrain ni de données de mouvement.

🧠 Méthodologie (Modèle STint)
L'approche repose sur un mécanisme de double cohérence cyclique (Dual Cycle Consistency). Le modèle apprend l'évolution temporelle des phénomènes physiques en tentant de reconstruire les images d'origine à travers une série de prédictions en deux étapes.

L'architecture sous-jacente utilise :

Un réseau U-Net pour la prédiction spatiale fine.

Des blocs Squeeze-and-Excitation (SE) intégrés pour modéliser efficacement les interdépendances entre les différentes caractéristiques convolutives.

📂 Structure du Dépôt
Le dépôt s'articule autour des 4 fichiers principaux suivants :

STint.pdf : L'article de recherche original ("STint: Self-supervised Temporal Interpolation for Geospatial Data") sur lequel s'appuie la méthode. Il décrit en profondeur le concept de la double cohérence cyclique.

notebook V2.ipynb : Un Jupyter Notebook complet contenant :

Le script de prétraitement des données climatiques (données CMEMS au format NetCDF4).

Les fonctions d'exploration et la gestion des masques (pour identifier les pixels correspondant à la terre ferme).

L'implémentation complète du réseau et le processus d'évaluation visuelle et quantitative en utilisant les métriques classiques (PSNR, SSIM, MSE).

training script V2.py : Le script Python dédié à l'entraînement du modèle PyTorch. Il contient la classe personnalisée DatasetNCFiles (pour charger les patchs extraits), applique les transformations de normalisation (transforms), et exécute la boucle d'entraînement tout en sauvegardant les poids du modèle et les métriques de loss (TensorBoard & Pickle).

Final report V2.pdf : Le rapport technique de fin de projet. Il détaille l'état de l'art, le problème technique abordé, l'architecture précise du réseau, le protocole expérimental et l'analyse fine des résultats obtenus.

⚙️ Installation et Prérequis
Le code est développé en Python et exploite le framework PyTorch. Pour exécuter les scripts, vous aurez besoin des bibliothèques suivantes :
