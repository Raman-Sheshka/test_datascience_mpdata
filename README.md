# Project

[![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
[![fastapi](https://img.shields.io/badge/FastAPI-009485?style=for-the-badge&logo=fastapi&logoColor=white)](https://img.shields.io/badge/FastAPI-3776AB?style=for-the-badge&logo=fastapi&logoColor=white)
[![django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=green)](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=green)

## Contexte et données

L’objectif est de fournir un classifier permettant de prédire qu’un joueur vaut le coup d’investir sur lui car il va durer plus de 5 ans en NBA en s’appuyant sur ses statistiques sportives. Le descriptif des paramètres du jeu de données est dans la table 1.

| Feature         | Description                          |
|-----------------|--------------------------------------|
| gp              | Matchs joués                         |
| min             | Minutes jouées                       |
| pts             | Points marqués                       |
| fgm             | Paniers réussis                      |
| fga             | Paniers tentés                       |
| fg%             | Pourcentage de réussite aux tirs     |
| 3p_made         | Paniers à trois points réussis       |
| 3pa             | Paniers à trois points tentés        |
| 3p%             | Pourcentage de réussite aux tirs à trois points |
| ftm             | Lancers francs réussis               |
| fta             | Lancers francs tentés                |
| ft%             | Pourcentage de réussite aux lancers francs |
| oreb            | Rebonds offensifs                    |
| dreb            | Rebonds défensifs                    |
| reb             | Rebonds totaux                       |
| ast             | Passes décisives                     |
| stl             | Interceptions                        |
| blk             | Contres                              |
| tov             | Balles perdues                       |
| target_5yrs     | Cible : Dure plus de 5 ans en NBA    |
| name            | Nom du joueur                        |

: Table 1 – description des features.

Le classifier entraîné est intégrer sous forme de requête unitaire dans un webservice API. Nous avons implementé deux possibilitées :

- Fast API
- Django

Le web service prends en entrée tous les paramètres pertinents et permet à un utilisateur faire une requête sur un seul joueur au modèle entraîné.

## install

'''bash
make reinstall package
'''

## Question 1 : Training

Vous trouverez ci-joint un fichier template (
test.py) à compléter qui lit et décode les données,
et propose une fonction de scoring.
Votre but sera de proposer, d’entraîner et de valider un classifier répondant le mieux possible
à l’objectif des investisseurs. Le fichier fournit également une fonction de scoring en recall,
que vous êtes libre de modifier tant que vous justifiez pourquoi. Plus que le résultat, c’est la
démarche analytique qui nous intéresse. Il n’y a pas de restrictions sur le format de remise, si
un jupyter notebook vous semble plus pertinent.

## Question 2 : Intégration

Une fois votre classifier entraîné, vous devrez l’intégrer sous forme de requête unitaire dans
un webservice. Vous êtes libre de choisir la librairie qui vous convient (flask, django ou autre)
Ce web service au format d’API REST devra prendre en entrée tous les paramètres pertinents
que vous aurez identifié comme s’il était mis à disposition d’un utilisateur voulant faire une
requête sur un seul joueur au modèle que vous aurez entraîné.

Call the API in your browser with this example: [http://localhost:8000/predict?gp=0.5&fgm=0.06&fg_pca=0.28&oreb=0.01&reb=0.04&pts=0.05&ftm=0.02&ft_pca=0.56&tov=0.14](http://localhost:8000/predict?gp=0.5&fgm=0.06&fg_pca=0.28&oreb=0.01&reb=0.04&pts=0.05&ftm=0.02&ft_pca=0.56&tov=0.14)
