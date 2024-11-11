# Test MP DATA. API Webservice : Classificateur ML des joueurs de NBA

[![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
[![fastapi](https://img.shields.io/badge/FastAPI-009485?style=for-the-badge&logo=fastapi&logoColor=white)](https://img.shields.io/badge/FastAPI-3776AB?style=for-the-badge&logo=fastapi&logoColor=white)
[![django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=green)](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=green)
[![flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)

## Contexte et données

L’objectif est de fournir un classificateur permettant de prédire si un joueur vaut le coup d’investir sur lui car il va durer plus de 5 ans en NBA en s’appuyant sur ses statistiques sportives. Le descriptif des paramètres du jeu de données est dans la table 1.

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

: Table 1 – description des fonctionnalités.

Le classificateur entraîné est intégré sous forme de requête unitaire dans un webservice API. Nous avons implémenté trois possibilités :

- FastAPI : c'est mon outil de choix, très léger. Je n'ai pas fourni d'interface frontale, uniquement des requêtes et pas de sérialisation.
- Django : un serveur, uniquement des requêtes avec une sérialisation rudimentaire et une base de données basique.
- Flask : un petit outil avec une interface frontale rudimentaire pour l'utilisateur, pas de sérialisation.

Le web service prend en entrée tous les paramètres pertinents et permet à un utilisateur de faire une requête sur un seul joueur au modèle entraîné. Je n'ai pas prévu de déploiement, donc chaque webservice tournera uniquement en exécution locale.

## Installation

- Instanciez l'environnement virtuel avec votre outil de choix ; j'ai testé uniquement avec Python 3.10.6.
- Copiez l'archive dans un répertoire de votre choix et activez l'environnement virtuel.
- Dézipper l'archive.
- Ajoutez les fichiers ***.env*** et ***.envrc***, dans ***.envrc*** appellez ***dotenv*** si nécessaire.
- Dans le terminal, tapez la commande suivante pour installer les dépendances :

  ```bash
  make reinstall_package
  ```

L'archive contient les fichiers suivants :

```bash
.
├── Makefile
├── README.md
├── TODO.md
├── backend
│   └── server
│       ├── endpoints
│       │   ├── __init__.py
│       │   ├── admin.py
│       │   ├── apps.py
│       │   ├── migrations
│       │   │   ├── 0001_initial.py
│       │   │   ├── __init__.py
│       │   ├── models.py
│       │   ├── serializers.py
│       │   ├── tests.py
│       │   ├── urls.py
│       │   └── views.py
│       ├── manage.py
│       ├── ml
│       │   ├── __init__.py
│       │   ├── classifier.py
│       │   └── registry.py
│       └── server
│           ├── __init__.py
│           ├── asgi.py
│           ├── settings.py
│           ├── urls.py
│           └── wsgi.py
├── data
│   └── nba_logreg.csv
├── data_documents
│   ├── test.py
│   └── test_ds.pdf
├── flask
│   ├── app.py
│   ├── static
│   │   └── style.css
│   └── templates
│       ├── index.html
│       └── results.html
├── mpdatanba
│   ├── __init__.py
│   ├── api
│   │   ├── __init__.py
│   │   └── fast_api.py
│   ├── ml_logic
│   │   ├── __init__.py
│   │   ├── ml_workflow.py
│   │   └── preprocessor.py
│   └── utils.py
├── notebooks
│   ├── study_nba_players_prediction_ml.ipynb
│   └── summury_nba_players.ipynb
├── requirements.txt
├── save_models
│   ├── model_selected.pkl
│   └── scaler.pkl
└── setup.py
```

Le module principal est ***mpdatanba***, qui contient les modules nécessaires pour l'API FastAPI et les fonctions de prédiction et ***notebooks*** contient les notebooks Jupyter.

Le répertoire ***backend*** contient les applications Django, ***flask*** contient l'application Flask.

---

Le package fourni ne devrait pas être considéré comme un produit prêt à être déployé en production, il s'agit d'une démonstration (voir comme un ***"POC"***). Un livrable fini devrait notamment contenir au moins la ***"containerization"***, un container ***Docker*** par webservice implémenté, par exemple.

## Explications: Jupyter notebook

La démarche avec les commentaires (en anglais) se trouve dans un Jupyter notebook dans le répertoire ***notebooks***:

- [notebooks.study_nba_players_prediction_ml.ipynb](/notebooks/study_nba_players_prediction_ml.ipynb)

Dans le même notebook, vous trouverez des bouts de code permettant de tester les prédictions fournies par le modèle sauvegardé en utilisant les API développées.

## Fast API

Pour lancer le client d'API en développement local, exécutez la commande dans le terminal:

```bash
make run_fastapi
```

La page d'accueil de l'API FastAPI est accessible à l'adresse suivante:

  [http://127.0.0.1:8000/docs](http://http://127.0.0.1:8000/docs)

## Flask API

Pour lancer l'application Flask en développement local, exécutez la commande dans la terminal:

```bash
make run_flask
```

La page d'accueil de l'application Flask est accessible à l'adresse suivante:

  [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

une fois les paramètres soummis, le résultat de la prédiction s'affichera à l'adresse suivante:

  [http://127.0.0.1:5000/results](http://http://127.0.0.1:5000/results)

## Django

J'ai commencé à implémenter une application Django, mais je n'ai pas eu le temps de la finaliser : je n'ai pas implémenté la sérialisation des données,
donc la base de données ne serve pas pleinement içi.

Pour lancer l'application Django en développement local, exécutez la commande dans le terminal:

```bash
make run_django_server
```

Peut-être que vous devrez exécuter les commandes suivantes pour créer la base de données et les migrations:

```bash
python backend/server/manage.py migrate
```

L'algorithme ML peut être accessible à l'adresse suivante :

[http://127.0.0.1:8000/api/v1/income_classifier/predict](http://127.0.0.1:8000/api/v1/income_classifier/predict)
