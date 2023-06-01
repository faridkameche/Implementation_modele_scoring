# P7_scoring_credit

Dans le cadre de notre étude sur l’implémentation d’un modèle de scoring, nous avons pour but de déterminer pour chaque client la probabilité de faillite de remboursement d’un crédit. Nous présentons dans cette étude la manière dont cette probabilité a été déterminée pour chaque client. Pour cela, nous nous sommes basés sur une liste de clients dont nous savons si ce sont de bons clients (faible probabilité de remboursement d’un crédit) ou de mauvais clients (forte probabilité de remboursement d’un crédit)ainsi que diverses informations (professionnelles, personnelles, familiales…).

Nous avons effectué des modélisations et nous avons cherché à optimiser la justesse des prédictions mais aussi à minimiser le nombre de faux négatifs, c’est-à-dire le nombre de mauvais client que le modèle prédit comme étant de bons clients. 

Nous avons ensuite mis en place un dashboard interactif et une API. Nous avons séparé les dossiers selon leur utilité. Nous avons au sein du dossier GitHub un fichier contenant les données d’entrées (Datasets) qui ont été utilisées pour le prétraitement des données. Nous avons aussi un dossier où nous regroupons les fichiers et données issues de la modélisation notamment (dossier Outputs). Nous avons aussi un dossier pour la création de l’API et de son déploiement sur une le cloud (pythonanywhere). Nous avons aussi créé un dossier pour les tests unitaires. 

Nous avons placé à la racine du dossier le programme pour le dashboard ainsi que la liste packages utilisés et nécessaires pour l’API et le dashboard.

Les API et dashboard peuvent être trouvées aux adresses suivantes :

Dashboard : https://faridkameche-p7-scoring-credit.streamlit.app/

API : https://faridkam.pythonanywhere.com/

Dossier GitHub :
https://github.com/faridkameche/P7_scoring_credit

Dossier GitHub Actions : 
https://github.com/faridkameche/P7_scoring_credit/actions
