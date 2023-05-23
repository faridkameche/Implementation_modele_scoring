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
import streamlit.components.v1 as components
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, validation_curve, cross_val_score, learning_curve

# from P7_appli import f_entraînement


# In[2]:


import requests

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'inputs': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def main():
    MLFLOW_URI = 'https://faridkameche-p7-scoring-credit-apimlflow-lgbmp7-api-id7uq6.streamlit.app/invocations'

    st.title("Prédiction de la probabilité de faillite de remboursement d'un crédit")

    car_owner = st.number_input("Client propriétaire d'une voiture (Non : 0, Oui : 1)",
                                min_value=0, max_value=1)

    House_owner = st.number_input('Client propriétaire (Non : 0, Oui : 1)',
                                  min_value=0, max_value=1)

    REGION_POPULATION_RELATIVE = st.number_input('Population relative région (*1e6)',
                                                 min_value=0., max_value=1e6, step=1.)

    Ratio_retard = st.number_input('Pourcentage de prêt avec retard',
                                   min_value=0, value=100, step=1)

    Total_prêt_annulé_vendu = st.number_input("Nombre d'ancien prêt annulé/vendu",
                                              min_value=0, step=1)

    Prêt_renouvelable = st.number_input("Nombre d'ancien prêt renouvelable",
                                     min_value=0, step=1)

    business = st.number_input('Businessman(woman) (Non : 0, Oui : 1)', 
                               min_value=0, max_value=1)

    marie = st.number_input('Client marié(e) (Non : 0, Oui : 1)', 
                            min_value=0, max_value=1)
    
    dat = [[car_owner, House_owner, REGION_POPULATION_RELATIVE/1e6, Ratio_retard,
            Total_prêt_annulé_vendu, Prêt_renouvelable, business, marie]]

    predict_btn = st.button('Prédire')
    if predict_btn:        
        pred = None
        pred = request_prediction(MLFLOW_URI, dat)["predictions"][0][1]
        
        st.write(f"Le pourcentage de faillite est de {round(100*pred,5)} %")


if __name__ == '__main__':
    main()

