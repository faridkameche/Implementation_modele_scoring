#!/usr/bin/env python
# coding: utf-8

# # Importation des librairies et des données

# In[1]:


import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import math
# import itertools
# from sklearn import model_selection, preprocessing, pipeline
# from matplotlib.collections import LineCollection
# from matplotlib import colors
# from itertools import chain
# from matplotlib.lines import Line2D
# from sklearn.utils import resample
# from sklearn.metrics import fbeta_score, make_scorer, accuracy_score, confusion_matrix
import joblib
import random
from lightgbm import LGBMClassifier


# In[1]:


from flask import Flask, request, jsonify
# from flask_restful import Api, Resource, reqparse
# from fastapi.responses import HTMLResponse

pipeline_lgbm = joblib.load("Outputs/pipeline_lgbm.pkl")

app = Flask(__name__)    
    
@app.route("/")
def read_root():
    return {"Mode d'emploi API": "Ajouter /predict? à l'URL puis les variables suivantes. Sinon, utiliser /docs#", 
            "Proprietaire_voiture" : "0 : non; 1 : oui",
            "Proprietaire_maison" : "0 : non; 1 : oui",            
            "REGION_POPULATION_RELATIVE" : "ratio inférieur à 1", 
            "Pourcentage_prêt_retard" : "entier compris entre 0 et 100", 
            "Nombre_prêt_annulé_vendu" : "entier", 
            "Nombre_ancien_prêt_renouvelable" : "entier",            
            "Emploi_business" : "0 : non; 1 : oui",
            "Marié" : "0 : non; 1 : oui",
           }


@app.route("/predict", methods=['GET'])
def read_data():    
    Proprietaire_voiture = request.args.get('Proprietaire_voiture', type=int)
    Proprietaire_maison = request.args.get('Proprietaire_maison', type=int)
    REGION_POPULATION_RELATIVE = request.args.get('REGION_POPULATION_RELATIVE', type=float) 
    Pourcentage_prêt_retard = request.args.get('Pourcentage_prêt_retard', type=int)
    Nombre_prêt_annulé_vendu = request.args.get('Nombre_prêt_annulé_vendu', type=int)
    Nombre_ancien_prêt_renouvelable = request.args.get('Nombre_ancien_prêt_renouvelable', type=int)
    Emploi_business = request.args.get('Emploi_business', type=int) 
    Marié = request.args.get('Marié', type=int)

    Proprietaire_voiture = np.log10(Proprietaire_voiture+1)
    Proprietaire_maison = np.log10(Proprietaire_maison+1)
    REGION_POPULATION_RELATIVE = np.log10(REGION_POPULATION_RELATIVE+1)
    Pourcentage_prêt_retard = np.log10(Pourcentage_prêt_retard+1)
    Nombre_prêt_annulé_vendu = np.log10(Nombre_prêt_annulé_vendu+1)
    Nombre_ancien_prêt_renouvelable = np.log10(Nombre_ancien_prêt_renouvelable+1)
    Emploi_business = np.log10(Emploi_business+1)
    Marié = np.log10(Marié+1)

    val = [[Proprietaire_voiture, Proprietaire_maison, REGION_POPULATION_RELATIVE, Pourcentage_prêt_retard, 
            Nombre_prêt_annulé_vendu, Nombre_ancien_prêt_renouvelable, Emploi_business, Marié]]
    
    prediction = pipeline_lgbm.predict_proba(val).tolist()
    
#     results = []
#     for (valeur, prob) in zip(val, prediction.tolist()):
#         results.append({"probability": prob})    
    return jsonify({'probabilité faillite': prediction[0][1]})

if __name__ == "__main__":
    app.run()

