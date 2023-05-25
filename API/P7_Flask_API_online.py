#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os

os.environ["MKL_NUM_THREADS"] = "1"

os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)
app.config["DEBUG"] = True

pipeline_lgbm = joblib.load("/home/Faridkam/P7_scoring_credit/API/pipeline_lgbm.joblib")

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    Prop_voiture = np.log10(float(request.form.get("Propriétaire_voiture"))+1)
    Prop_maison = np.log10(float(request.form.get("Propriétaire_maison"))+1)
    Reg_pop_rel = np.log10((float(request.form.get("REGION_POPULATION_RELATIVE"))/1e6)+1)
    Pret_retard = np.log10(float(request.form.get("Pourcentage_prêt_retard"))+1)
    Pret_annul = np.log10(float(request.form.get("Nombre_prêt_annulé_vendu"))+1)
    Pret_renouvelable = np.log10(float(request.form.get("Nombre_ancien_prêt_renouvelable"))+1)
    Business = np.log10(float(request.form.get("Emploi_business"))+1)
    Married = np.log10(float(request.form.get("Marié"))+1)

    valeur_form = [Prop_voiture, Prop_maison, Reg_pop_rel, Pret_retard, Pret_annul, Pret_renouvelable, Business, Married]
    val = [np.array(valeur_form)]

    prediction_p = pipeline_lgbm.predict_proba(val)[0][1]

    return render_template('index.html', prediction=round(100*prediction_p,2))

