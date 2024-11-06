
## Contexte et données

Vous trouverez ci-joint un dataset
nba_logreg.csv qui contient des statistiques sportives sur
les joueurs débutants de la NBA. L’objectif est de fournir un classifier permettant de prédire
qu’un joueur vaut le coup d’investir sur lui car il va durer plus de 5 ans en NBA en s’appuyant
sur ses statistiques sportives. Ce modèle vise a conseiller des investisseur cherchant a
capitaliser sur de futurs talents de la NBA.
Le descriptif des paramètres du jeu de données est le suivant :
Table 1 – description des features

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
