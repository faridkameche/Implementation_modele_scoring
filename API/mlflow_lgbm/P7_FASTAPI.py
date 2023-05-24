#!/usr/bin/env python
# coding: utf-8

# # Importation des librairies et des données

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import itertools
from sklearn import model_selection, preprocessing, pipeline
from matplotlib.collections import LineCollection
from matplotlib import colors
from itertools import chain
from matplotlib.lines import Line2D
from sklearn.utils import resample
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score, confusion_matrix
import joblib
import random
import streamlit as st
import mpld3
import streamlit.components.v1 as components
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, validation_curve, cross_val_score, learning_curve


# In[ ]:


import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

pipeline_lgbm = joblib.load("Outputs/pipeline_lgbm.pkl")
app = FastAPI()

@app.get("/")
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

@app.get("/predict")
def read_data(Proprietaire_voiture: int, Proprietaire_maison: int, REGION_POPULATION_RELATIVE: float, 
              Pourcentage_prêt_retard: int, Nombre_prêt_annulé_vendu: int, Nombre_ancien_prêt_renouvelable: int, 
              Emploi_business: int, Marié: int):

#     if(not(f0)):
#         raise HTTPException(status_code=400, 
#                             detail = "Please Provide a valid text message")
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
    
    prediction = pipeline_lgbm.predict_proba(val)
    
    results = []
    for (valeur, prob) in zip(val, prediction.tolist()):
        results.append({"probability": prob})    
    return {"Probabilité de faillite" : results[0]["probability"][1]}

def show_ui():
    st.title("API Demo")

if __name__ == "__main__":
    # Exécutez FastAPI en arrière-plan avec uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    # Exécutez Streamlit pour l'interface utilisateur
    show_ui()

