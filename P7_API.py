#!/usr/bin/env python
# coding: utf-8

# # Importation des librairies et des données

# In[3]:


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

# from P7_appli import f_entraînement


# In[4]:


# def upsampling(df):
    
#     features = df.iloc[:, 1:].columns.tolist()
    
#     df_minority_upsampled = resample(df.loc[df["TARGET"]==1, features], 
#                                      replace=True,    
#                                      n_samples = df.loc[df["TARGET"]==0, "TARGET"].shape[0],    
#                                      random_state=65).reset_index(drop=True) 
#     dico = {}
#     for i in range(0,df_minority_upsampled.shape[0]):
#         dico[i] = i+df.shape[0]
        
#     df_minority_upsampled = df_minority_upsampled.rename(index=dico)
    
#     df_ups = pd.concat([df.loc[df["TARGET"]==0, features], df_minority_upsampled])
    
#     scaler = preprocessing.RobustScaler()
    
#     X_train, X_test, y_train, y_test = model_selection.train_test_split((scaler.fit_transform(np.log10((df_ups[features[1:]].values)+1))), 
#                                                                         df_ups["TARGET"].values.ravel(), 
#                                                                         train_size=0.75, 
#                                                                         random_state=65)

#     return X_train, X_test, y_train, y_test


# In[3]:


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
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'

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


# La fonction gauge nécessite une mémoire de 2.78 MB. 
# Les fichiers créés prennent 2.73 MB de mémoire.

# In[28]:


# def plot_critère(data_comparaison, data_to_show, variable, num_client):
    
#     plt.text(6, 8, f"Probabilité de remboursement : {100-int(round(data_comparaison.iloc[-1, num_client], 2))} %", 
#              ha="center", va="top", weight="bold", fontsize = 15)
#     k = 0
#     l = 0
#     for j in range(0, len(variable)):
#         plt.text(4*k, 6-l, f"{variable[j]}", weight="bold", ha="center", va="top", fontsize = 14)            
        
#         if (variable[j]=="Population\nrelative région"):
#             plt.text(4*k, 4-l, f"{round(data_to_show[j],6)}", ha="center", va="top", fontsize = 14)    
#         else:
#             plt.text(4*k, 4-l, f"{data_to_show[j]}", ha="center", va="top", fontsize = 14)            
        
#         k+=1
#         if (k==4):
#             l+=4
#             k=0 
            
#     plt.box(False)
#     plt.show()


# In[29]:


# def display_data_client(data_comparaison, num_client, liste_var):
    
#     plt.figure(figsize=(6,6))
    
#     dico1 = {"Propriétaire d'une voiture" : "Propriétaire\nvoiture", 
#              "Propriétaire d'un logement" : "Propriétaire\nlogement",
#              "Population relative région" : "Population\nrelative région",  
#              "Pourcentage de prêt avec retard" : "Pourcentage\nprêt avec retard", 
#              "Nombre de prêt annulé/vendu" : "Nombre prêt\nannulé/vendu", 
#              "Nombre d'ancien prêt renouvelable" : "Nombre\nd'ancien prêt\nrenouvelable", 
#              "Emploi business" : "Emploi business", 
#              "Marié(e)" : "Marié(e)", 
#              "Métier" : "Métier",
#              "Statut familial" : "Statut\nfamilial",
#              "Valeur crédit" : "Valeur\ncrédit",
#              "Revenus" : "Revenus",
#              "Nombre de prêt" : "Nombre prêt",
#              "Somme restante" : "Somme\nrestante",
#              "Credit rejeté" : "Credit\nrejeté",
#              "Nombre prêt actif" : "Nombre\nprêt actif", 
#              "Nombre prêt terminé" : "Nombre\nprêt terminé",
#              "Nombre prêt refusé" : "Nombre\nprêt refusé", 
#              "Nombre autres prêts" : "Nombre\nautres prêts",
#              "Probabilité remboursement" : "Probabilité\nremboursement"}
    
#     variable = []
#     for i in range (0, len(liste_var)):
#         variable.append(dico1[liste_var[i]])
    
#     plt.xlim(0,9)
#     plt.ylim(0,9)
    
#     plt.xticks([])
#     plt.yticks([])
#     col = data_comparaison.iloc[:,:].columns.tolist()
#     data_to_show = []
    
#     for var in liste_var:
#         data_to_show.append(data_comparaison.loc[data_comparaison["Critères"]==var, col[num_client]].item()) 
        
#     plot_critère(data_comparaison, data_to_show, variable, num_client)


# In[23]:


# st.set_page_config(
#     page_title="Dossier de demande crédit",
#     page_icon="✅",
#     layout="wide")

# st.title("Dossier de demande de crédit")

# df, score, seuil = f_entraînement()
# df = df.sort_values("SK_ID_CURR")

# col = df.iloc[:,2:10].columns
# liste_id = (df["SK_ID_CURR"].values.tolist())

# col1, col2 = st.columns([1, 3])

# placeholder = st.empty()

# with placeholder.container():
#     id_client = st.text_input("Entrer l'identifiant du client. Si inconnu taper 0")
    
#     if id_client=="0":
#         client_choisi = st.select_slider('Choisir un identifiant', options = (liste_id), 
#                                          label_visibility="hidden")
#         id_client = (client_choisi)

#     if id_client:
#         with col1:
#             client_proba = proba_faillite(id_client, df, score, seuil)
#             gauge(id_client, client_proba)
#             st.write(f'Le résultat obtenu a une précision de {round(100*score,1)} %.')
#             '''
#             Le seuil obtenu a été déterminé de façon à limiter le nombre de crédit donné à des clients 
#             n'allant pas au bout de leur prêt.
#             '''
#         with col2:  
#             shap_df, rang = plot_dist_shap(id_client, df, score, seuil)
        
#         l_rang = [rang, df["Rang"].min(), df["Rang"].max()]
        
#         col_1 = ['Client vehiculé (%)', 'Propriétaire (%)', 'Pop. relative région', 
#                  'Prêt avec retard  (%)', 'Ancien prêt annulé/vendu', 'Ancien prêt renouv.', 
#                  'Businessman(woman) (%)', 'Client marié(e) (%)']
        
#         data = pd.DataFrame(list(zip(df.loc[df["Rang"]==l_rang[0], col].mean().values.tolist(), 
#                                      df.loc[df["Rang"]==l_rang[1], col].mean().values.tolist(),
#                                      df.loc[df["Rang"]==l_rang[2], col].mean().values.tolist())), 
#                             columns = [f"Rang {rang}", "Meilleur rang", "Pire rang"], 
#                             index = col_1)
        
#         data.iloc[0:2,:] = data.iloc[0:2,:]*100
#         data.iloc[-2:,:] = data.iloc[-2:,:]*100
        
#         st.dataframe(data.T)
        
#         options = st.multiselect('Choisir 2 critères à afficher', col)
#         plot_client_shap(id_client, df, score, seuil, options)


# In[50]:


# st.set_page_config(
#     page_title="Dossier de demande crédit",
#     page_icon="✅",
#     layout="wide")

# st.title("Dossier de demande de crédit")

# df, score = f_entraînement()
# df = df.sort_values("SK_ID_CURR")

# liste_id = (df["SK_ID_CURR"].values.tolist())

# col1, col2 = st.columns([1, 2])

# id_client = st.text_input("Entrer l'identifiant du client. Si inconnu taper 0")

# if id_client=="0":
#     client_choisi = st.select_slider('Choisir un identifiant', options = (liste_id), 
#                                      label_visibility="hidden")
#     id_client = (client_choisi)

# if id_client:
#     if st.button("Aide"):
#         '''
#         L'application permet de comparer le client à 3 autres clients ayant des probabilités de remboursement. 
#         Si vous connaissez un ou des identifiants, veuillez les rentrer via le bouton ci-bas. 
#         Sinon, des clients seront sélectionnés de manière aléatoire avec comme condition :
#         - Client n°2 : client ayant un risque de faillite de remboursement de crédit similaire au client,
#         - Client n°3 : client ayant un risque de faillite de remboursement très faible et
#         - Client n°4 : client ayant un risque de faillite de remboursement très elevée.        
#         '''
#     col_id_1, col_id_2, col_id_3 = st.columns(3)
#     with col_id_1:
#         if st.button("Identifiant client n°2"):
#             client_2 = st.text_input("Entrer l'identifiant du client")
#         else:
#             client_2 = 0
#     with col_id_2:
#         if st.button("Identifiant client n°3"):
#             client_3 = st.text_input("Entrer l'identifiant du client")
#         else:
#             client_3 = 0
#     with col_id_3:
#         if st.button("Identifiant client n°4"):
#             client_4 = st.text_input("Entrer l'identifiant du client") 
#         else:
#             client_4 = 0
            
#     client_proba, data_comparaison = données_client(identifiant = id_client, 
#                                                     client_comp_2 = client_2, 
#                                                     client_comp_3 = client_3, 
#                                                     client_comp_4 = client_4)
    
#     col_data_comp = data_comparaison.iloc[:,1:].columns.tolist()
#     #     placeholder = st.empty()
    
#     with col1:
#         gauge(round(client_proba, 2), seuil = 0.60)
        
#         if st.button('Détails'):
#             st.write(f'Le résultat obtenu a une précision de {round(100*score,1)} %.')
#             '''
#             Il a été calculé en se basant sur les critères suivants : 
#             - Propriétaire d'une voiture,
#             - Propriétaire d'un logement,
#             - Population relative région,
#             - Pourcentage de prêt avec retard,
#             - Nombre de prêt annulé/vendu,
#             - Nombre d'ancien prêt renouvelable,
#             - Emploi business et
#             - Marié(e).
            
#             Le seuil obtenu a été déterminé de façon à limiter le nombre de crédit donné à des clients 
#             n'allant pas au bout de leur prêt.
#             '''
#         with col2:
#             options = st.multiselect('Critères',
#                                      df.iloc[:, 2:].columns.tolist())
#             if st.button('Valider'):
#                 tab1, tab2, tab3, tab4 = st.tabs([f"Client n°{id_client}", f"Client n°{col_data_comp[1]}", 
#                                                   f"Client n°{col_data_comp[2]}", f"Client n°{col_data_comp[3]}"])
                
#                 with tab1:
#                     tab1.subheader(f"Données sur le client n°{id_client}")
#                     tab1 = display_data_client(data_comparaison, 1, options)
                    
#                 with tab2:
#                     if (client_2==0):
#                         tab2.subheader("Client à risque similaire")
#                     else:
#                         tab2.subheader(f"Données sur le client n°{col_data_comp[1]}")
                        
#                     tab2 = display_data_client(data_comparaison, 2, options)
                        
#                 with tab3:
#                     if (client_3==0):
#                         tab3.subheader("Client à faible risque")
#                     else:
#                         tab3.subheader(f"Données sur le client n°{col_data_comp[2]}")
                    
#                     tab3 = display_data_client(data_comparaison, 3, options)
                    
#                 with tab4:
#                     if (client_4==0):
#                         tab4.subheader("Client à fort risque")
#                     else:
#                         tab4.subheader(f"Données sur le client n°{col_data_comp[3]}")
                        
#                     tab4 = display_data_client(data_comparaison, 4, options)


# In[ ]:


# plt.figure(figsize=(4, 4))
# plt.subplot(polar=True)

# label_loc = np.linspace(start = 0, stop = np.pi, num=100)

# liste_seuil = np.arange(0, 1.01, 0.01).round(2)
# proba_seuil = 0.60

# ind_seuil = 100-int(np.argwhere(liste_seuil==proba_seuil))

# gauge_up = np.ones(100)
# gauge_d = np.ones(100)-0.3

# plt.plot(label_loc[:ind_seuil], gauge_up[:ind_seuil], color = "green")
# plt.fill_between(label_loc[:ind_seuil], gauge_up[:ind_seuil], gauge_d[:ind_seuil], color = "green", alpha=0.9)

# plt.plot(label_loc[ind_seuil:], gauge_up[ind_seuil:], color = "red")
# plt.fill_between(label_loc[ind_seuil:], gauge_up[ind_seuil:], gauge_d[ind_seuil:], color = "red", alpha=0.9)

# val_client = 0.65
# val_faillite = 1-val_client

# ind_val_faillite = 100-int(np.argwhere(liste_seuil==val_faillite))

# if (val_client==0):
#     plt.annotate(f"{int(100*val_faillite)} %", xytext=(0,0), xy=(label_loc[ind_val_faillite-1], 0.9),
#                  arrowprops=dict(arrowstyle="wedge, tail_width=1", color="black", shrinkA=0),
#                  bbox=dict(boxstyle="circle", facecolor="black", edgecolor="black", linewidth=3), 
#                  fontsize=15, color="white", ha="center", weight = "bold")

# else:
#     plt.annotate(f"{int(100*val_faillite)} %", xytext=(0,0), xy=(label_loc[ind_val_faillite], 0.9),
#                  arrowprops=dict(arrowstyle="wedge, tail_width=1", color="black", shrinkA=0),
#                  bbox=dict(boxstyle="circle", facecolor="black", edgecolor="black", linewidth=3), 
#                  fontsize=15, color="white", ha="center", weight = "bold")

# # lines, labels = plt.thetagrids(np.arange(0, 181, (180/10)), [100,90,80,70,60,50,40,30,20,10,0], fontsize=12)

# plt.xlim(0, max(label_loc))

# x_ti_val = np.arange(max(label_loc), -0.01, -(max(label_loc)/10))
# x_ti_lab = np.arange(0,101,10)
# plt.xticks(x_ti_val[::2], x_ti_lab[::2], fontsize=13)
# plt.yticks([])

# plt.title("Probabilté de \nremboursement d'un crédit", ha="center", fontsize=15)
# plt.gca().spines["polar"].set_visible(False)
# # plt.gca().set_theta_direction("clockwise")
# # plt.gca().set_theta_offset(np.pi)
# plt.grid(visible=False)

# plt.box(False)

# plt.show()


# In[ ]:


# import plotly.graph_objects as go

# fig = go.Figure(go.Indicator(
#     domain = {'x': [0, 1], 'y': [0, 1]},
#     value = 69,
#     mode = "gauge+number",
#     title = {'text': "Probabilité"},
#     delta = {'reference': 380},
#     gauge = {'axis': {'range': [None, 100]},
#              'steps' : [
#                  {'range': [0, 250], 'color': "lightgray"},
#                  {'range': [250, 400], 'color': "gray"}],
#              'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 490}}))

# fig.show()


# In[ ]:


# import tracemalloc
# # starting the monitoring
# tracemalloc.start()
 
# # function call
# df = f_entraînement()
 
# # displaying the memory
# print(f"Le fichier créée prend {round(tracemalloc.get_traced_memory()[0]/1e6, 2)} MB de mémoire")
# print(f"La fonction nécessite au maximum {round(tracemalloc.get_traced_memory()[1]/1e6, 2)} MB de mémoire")

# # stopping the library
# tracemalloc.stop()


# In[33]:


# def f2_score(y_true, y_pred):
#     f2 = fbeta_score(y_true, y_pred, beta=2, pos_label=0)
#     return f2


# In[22]:


# from lightgbm import LGBMClassifier

# best_model = LGBMClassifier(random_state=rs, learning_rate=0.2, n_estimators=250, num_leaves=200)
# best_model.fit(X_train_ups, y_train_ups)

# grid_lgbm = joblib.load('grid_lgbm.pkl')

# y_train_bm = grid_lgbm.predict(X_train)
# y_test_bm = grid_lgbm.predict(X_test)

# F2_train_bm = f2_score(y_train, y_train_bm)
# print(f"Pour le modèle LGBM et les données d'entraînement," "\nle score F\u2082 vaut " f"{round(F2_train_bm, 5)}.")
# print("---------------------------------------------------------------------------------")
    
# F2_test_bm = f2_score(y_test, y_test_bm)
# print(f"Pour le modèle LGBM et les données de test," "\nle score F\u2082 vaut " f"{round(F2_test_bm, 5)}.")


# In[ ]:


# import timeit

# for i in range (10):
#     start = timeit.default_timer()
    
#     data_final = data()
    
#     stop = timeit.default_timer()
#     execution_time = round(stop - start, 2)
    
#     print(f"Program Executed in {execution_time} s")


# In[ ]:


# for i in range (3):
#     start = timeit.default_timer()
    
#     grid_lgbm = joblib.load('grid_lgbm.pkl')
#     y_train_bm = grid_lgbm.predict(X_train_ups)
    
# #     y_proba = best_model.predict_proba(données_client_scaler)
    
#     stop = timeit.default_timer()
#     execution_time = stop - start
    
#     print("Program Executed in " + str(execution_time))


# In[ ]:





# In[ ]:


#     if ((rang!=10) & (rang!=0)):
        
#         if (client_comp_2==0):
#             liste_same_client = data.loc[(data["Rang"]==rang), "SK_ID_CURR"].values            
#             client_2 = random.choice(liste_same_client)
            
#         if (client_comp_3==0):
#             liste_better_client = data.loc[(data["Rang"]>7) & (data["Rang"]!=rang), "SK_ID_CURR"].values            
#             client_3 = random.choice(liste_better_client)
            
#         if (client_comp_4==0):
#             liste_worse_client = data.loc[(data["Rang"]<7) & (data["Rang"]!=rang), "SK_ID_CURR"].values            
#             client_4 = random.choice(liste_worse_client)
        
#     elif (rang==10):
#         if (client_comp_2==0):
#             liste_same_client = data.loc[(data["Rang"]==10), "SK_ID_CURR"].values
#             liste_same_client = liste_same_client[liste_same_client!=client_1]                       
#             client_2 = random.choice(liste_same_client)
            
#         if (client_comp_3==0):
#             liste_better_client = data.loc[(data["Rang"]>7) & (data["Rang"]!=rang), "SK_ID_CURR"].values            
#             client_3 = random.choice(liste_better_client)
            
#         if (client_comp_4==0):    
#             liste_worse_client = data.loc[(data["Rang"]<7), "SK_ID_CURR"].values            
#             client_4 = random.choice(liste_worse_client)
            
#     elif (rang==0):
#         if (client_comp_2==0):
#             liste_same_client = data.loc[(data["Rang"]==0), "SK_ID_CURR"].values
#             liste_same_client = liste_same_client[liste_same_client!=client_1]                       
#             client_2 = random.choice(liste_same_client)
            
#         if (client_comp_3==0):
#             liste_better_client = data.loc[(data["Rang"]>7), "SK_ID_CURR"].values            
#             client_3 = random.choice(liste_better_client)
            
#         if (client_comp_4==0):    
#             liste_worse_client = data.loc[(data["Rang"]<7) & (data["Rang"]!=rang), "SK_ID_CURR"].values            
#             client_4 = random.choice(liste_worse_client)
        
#     if (client_comp_2!=0):
#         client_2 = client_comp_2    
        
#     if (client_comp_3!=0):
#         client_3 = client_comp_3
        
#     if (client_comp_4!=0):
#         client_4 = client_comp_4
        
#     data_client_1 = data.loc[data["SK_ID_CURR"]==client_1, col_fit]
#     data_client_2 = data.loc[data["SK_ID_CURR"]==client_2, col_fit]
#     data_client_3 = data.loc[data["SK_ID_CURR"]==client_3, col_fit]
#     data_client_4 = data.loc[data["SK_ID_CURR"]==client_4, col_fit]
        
#     data_comp = pd.concat([data_client_1, data_client_2, data_client_3, data_client_4]).reset_index(drop=True)
    
#     data_comp["Population relative région"] = data_comp["Population relative région"].round(6)
#     data_comp["Proba_faillite"] = round(100*data_comp["Proba_faillite"].round(2))
    
#     col_binaire = ["Propriétaire d'une voiture", "Propriétaire d'un logement", "Emploi business", "Marié(e)"]
        
#     for col in col_binaire:
#         data_comp.loc[data_comp[col]==1, col] = "Oui"
#         data_comp.loc[data_comp[col]==0, col] = "Non"
        
#     data_comparaison = data_comp.T.reset_index().rename(columns={"index" : "Critères", 
#                                                                  0 : f"{client_1}", 1 : f"{client_2}", 
#                                                                  2 : f"{client_3}", 3 : f"{client_4}"})

