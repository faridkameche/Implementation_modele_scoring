#!/usr/bin/env python
# coding: utf-8

# In[138]:


import numpy as np
import joblib
import pandas as pd
import pytest

pipeline_lgbm = joblib.load("Outputs/pipeline_lgbm.joblib")

def predict():

    Prop_voiture = np.log10(np.random.randint(0, 2)+1)
    Prop_maison = np.log10(np.random.randint(0, 2)+1)
    Reg_pop_rel = np.log10((np.random.randint(0, 1e6)/1e6)+1)
    Pret_retard = np.log10(np.random.randint(0, 101)+1)
    Pret_annul = np.log10(np.random.randint(0, 33)+1)
    Pret_renouvelable = np.log10(np.random.randint(0, 33)+1)
    Business = np.log10(np.random.randint(0, 2)+1)
    Married = np.log10(np.random.randint(0, 2)+1)

    valeur_form = [Prop_voiture, Prop_maison, Reg_pop_rel, Pret_retard, Pret_annul, Pret_renouvelable, Business, Married]
    val = [np.array(valeur_form)]

    prediction_p = pipeline_lgbm.predict_proba(val)[0][1]

    return round(100*prediction_p,2)

def test_predict():
#     liste_feat = []
#     for i in range (100):
#         val_1 = np.log10(np.random.randint(0, 2)+1)
#         val_2 = np.log10(np.random.randint(0, 2)+1)
#         val_3 = np.log10((np.random.randint(0, 1e6)/1e6)+1)
#         val_4 = np.log10(np.random.randint(0, 101)+1)
#         val_5 = np.log10(np.random.randint(0, 33)+1)
#         val_6 = np.log10(np.random.randint(0, 33)+1)
#         val_7 = np.log10(np.random.randint(0, 2)+1)
#         val_8 = np.log10(np.random.randint(0, 2)+1)
#         liste_feat.append([val_1, val_2, val_3, val_4, val_5, val_6, val_7, val_8])        

#     data_to_test = pd.DataFrame(liste_feat)
#     y_0 = []
#     y_1 = []
#     diff_proba = []
#     for i in range (100):
#         y_0.append(pipeline_lgbm.predict_proba(data_to_test.values)[i][0])
#         y_1.append(pipeline_lgbm.predict_proba(data_to_test.values)[i][1])
#         diff_proba.append(y_0[i]+y_1[i])
    assert predict() <= 1

    
# def predict_str():
#     x_str = ["test", "string", "ne", "fonctionne", "pas", "avec", "le", "modèle"]
#     return pipeline_lgbm.predict_proba(x_str)

# def test_predict_str():
#     with pytest.raises(ValueError):
#         predict_str()


# In[ ]:




