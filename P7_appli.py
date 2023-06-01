#!/usr/bin/env python
# coding: utf-8

# # Importation des librairies et des données

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing, pipeline
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
from sklearn.preprocessing import FunctionTransformer


# ## Fusion des données des anciens crédits
# 
# Nous fusionnons les données sur les anciens crédits fait auprés de la banque Home Credit ("prev_credit_grouped") avec les données sur les anciens crédits fait auprés d'autres banques ("data_bureau_grouped")

# In[2]:


def fus_ancien_credit():
    
    data_bureau_grouped = pd.read_csv("Datasets/data_bureau_grouped_p7.csv", index_col=0)
    
    prev_credit_grouped = pd.read_csv("Datasets/prev_credit_grouped_p7.csv", index_col=0)
    
    data_merged = pd.merge(prev_credit_grouped, data_bureau_grouped, on="SK_ID_CURR", how="outer")
    
    data_merged["SK_ID_CURR"] = data_merged["SK_ID_CURR"].astype("int32")
    
    data_merged["Nombre_de_prêt"] = data_merged[["Total_credit", "Total_loans"]].sum(axis=1)
    
    data_merged["Total_prêt_annulé_vendu"] = data_merged[["Canceled", "Sold_Credit"]].sum(axis=1).astype("int8")
    
    data_merged["Prêt_renouvelable"] = data_merged["Revolving_loans"]
    
    data_merged["Prêt_renouvelable"] = data_merged["Prêt_renouvelable"].fillna(0).astype("int8")
    
    data_merged["Ratio_retard"] = np.ceil((data_merged[["Late_Issues", "Issues_Yes"]].sum(axis=1))*100/data_merged["Nombre_de_prêt"]).astype("int8")
        
    data_merged = data_merged[['SK_ID_CURR', 'Ratio_retard', 'Total_prêt_annulé_vendu', 'Prêt_renouvelable']]
    
    return data_merged


# In[3]:


def f_application_train(f_application = "Datasets/application_train_P7.csv"):
    
    dtype_application_train = {"SK_ID_CURR": np.int32, 
                               "TARGET": np.int32,
                               "FLAG_OWN_CAR": str, 
                               "FLAG_OWN_REALTY": str,
                               "NAME_FAMILY_STATUS": "category", 
                               "NAME_INCOME_TYPE": "category",
                               "REGION_POPULATION_RELATIVE": np.float64, 
                               "OBS_30_CNT_SOCIAL_CIRCLE":np.float16,
                               "EXT_SOURCE_1": np.float16, 
                               "EXT_SOURCE_2": np.float16,
                               "EXT_SOURCE_3": np.float16}
    
    col_application_train = list(dtype_application_train.keys())
    
    application = pd.read_csv(f_application, usecols=col_application_train, dtype=dtype_application_train)
    
    application = application.loc[application["NAME_FAMILY_STATUS"]!="Unknown",:]
    application = application.loc[application["OBS_30_CNT_SOCIAL_CIRCLE"].notnull(),:]
    
    application["mean_score"] = application[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    
    application["Source_revenus"] = np.nan
    
    application.loc[(application["NAME_INCOME_TYPE"]=="Pensioner") | 
                    (application["NAME_INCOME_TYPE"]=="Unemployed") | 
                    (application["NAME_INCOME_TYPE"]=="Student") |
                    (application["NAME_INCOME_TYPE"]=="Maternity leave") ,
                    "Source_revenus"] = "Retraité_Chômage_congé_mat"
    
    application.loc[(application["NAME_INCOME_TYPE"]=="Working") | 
                    (application["NAME_INCOME_TYPE"]=="Businessman") ,"Source_revenus"] = "Travail_business"
    
    application.loc[application["Source_revenus"].isna(),
                    "Source_revenus"] = application["NAME_INCOME_TYPE"]
    
    le = preprocessing.LabelEncoder()
   
    application["Car_owner"] = le.fit_transform(application["FLAG_OWN_CAR"])
    
    application["House_owner"] = le.fit_transform(application["FLAG_OWN_REALTY"])
    
    application = application[['SK_ID_CURR', 'TARGET', 'Car_owner', 'House_owner', 'Source_revenus', 
                               'NAME_FAMILY_STATUS', 'REGION_POPULATION_RELATIVE']]
    
    return application


# In[4]:


def data(fichier_client = "Outputs/client_restant.csv"):
    
    data_merged = fus_ancien_credit()

    application = f_application_train()
    
    data_train_test = pd.merge(application, data_merged, on="SK_ID_CURR", how="left")
    
    data_train_test = data_train_test.fillna(0)
            
    data_train_test.loc[data_train_test["Source_revenus"]=="Travail_business", "Travail_business"] = 1
    
    data_train_test.loc[data_train_test["Source_revenus"]!="Travail_business", "Travail_business"] = 0
    
    data_train_test.loc[data_train_test["NAME_FAMILY_STATUS"]=="Married", "Married"] = 1
    
    data_train_test.loc[data_train_test["NAME_FAMILY_STATUS"]!="Married", "Married"] = 0
    
    data_train_test = data_train_test.loc[data_train_test["Total_prêt_annulé_vendu"]<=33,:]
    data_train_test = data_train_test.loc[data_train_test["Prêt_renouvelable"]<25,:].reset_index(drop=True)
    
    data_train_test.drop(columns=["NAME_FAMILY_STATUS", "Source_revenus"], inplace=True)
    
#     data_train_test["REGION_POPULATION_RELATIVE"] = data_train_test["REGION_POPULATION_RELATIVE"].astype("float16")
    
#     col_int = ["TARGET", "Car_owner", "House_owner", "Ratio_retard", "Total_prêt_annulé_vendu", "Prêt_renouvelable", 
#                "Travail_business", "Married"]
    
#     data_train_test[col_int] = data_train_test[col_int].astype("int8")
    
    client_restant = pd.read_csv(fichier_client, usecols=["SK_ID_CURR"], dtype = {"SK_ID_CURR": np.int32})
    
    data_final = pd.merge(client_restant, data_train_test, on="SK_ID_CURR", how="left")
    
    data_final = data_final.rename(columns = {"Car_owner" : "Propriétaire d'une voiture", 
                                              "House_owner" : "Propriétaire d'un logement", 
                                              "REGION_POPULATION_RELATIVE" : "Population relative région", 
                                              "Ratio_retard" : "Pourcentage de prêt avec retard",
                                              "Total_prêt_annulé_vendu" : "Nombre de prêt annulé/vendu", 
                                              "Prêt_renouvelable" : "Nombre d'ancien prêt renouvelable", 
                                              "Travail_business" : "Emploi business", 
                                              "Married" : "Marié(e)"
                                             })
    
    return data_final


# In[5]:


def upsampling(df):
    
    features = df.iloc[:, 1:].columns.tolist()
    
    df_minority_upsampled = resample(df.loc[df["TARGET"]==1, features], 
                                     replace=True,    
                                     n_samples = df.loc[df["TARGET"]==0, "TARGET"].shape[0],    
                                     random_state=65).reset_index(drop=True) 
    
    dico = {}
    for i in range(0, df_minority_upsampled.shape[0]):
        dico[i] = i+df.shape[0]
        
    df_minority_upsampled = df_minority_upsampled.rename(index=dico)
    
    df_ups = pd.concat([df.loc[df["TARGET"]==0, features], df_minority_upsampled])
    
    scaler = preprocessing.RobustScaler()
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split((scaler.fit_transform(np.log10((df_ups[features[1:]].values)+1))), 
                                                                        df_ups["TARGET"].values.ravel(), 
                                                                        train_size=0.75, 
                                                                        random_state=65)

    return X_train, X_test, y_train, y_test


# ## Prédiction

# In[6]:


def f_entraînement():
    
    df = data()
    
    pipeline_lgbm = joblib.load('Outputs/pipeline_lgbm.pkl')
    
    score = accuracy_score(df["TARGET"].values, pipeline_lgbm.predict(np.log10(df.iloc[:, 2:].values + 1)))
        
    dtype_shap = {'SK_ID_CURR' : str, 
                  'SHAP_Propriétaire voiture' : np.float16, 
                  'SHAP_Propriétaire maison' : np.float16,
                  'SHAP_Emploi business' : np.float16, 
                  'SHAP_Marié' : np.float16, 
                  'SHAP_Population relative région' : np.float16,
                  'SHAP_Ancien prêt annulé/vendu' : np.float16, 
                  'SHAP_Ancien_prêt renouvelable' : np.float16,
                  'SHAP_Ratio prêt avec retard' : np.float16}
        
    shap_data = pd.read_csv("Outputs/shap_df_p7.csv", usecols=list(dtype_shap.keys()), dtype = dtype_shap)
    
    seuil = 40
    
    df["Proba_faillite"] = 100*pipeline_lgbm.predict_proba(np.log10(df.iloc[:,2:] + 1))[:, 1]
    
    k=0
    
    for i in range(0,101,5):   
        
        df.loc[(df["Proba_faillite"]<i+5) & (df["Proba_faillite"]>=i), "Rang"] = k
        k+=1
        
    df["Rang"] = df["Rang"].astype("int8")
    
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(str)
    
    df_final = pd.merge(df, shap_data, on="SK_ID_CURR", how="outer")
    
    return df_final.reset_index(drop=True), score, seuil


# In[13]:


def proba_faillite(identifiant, données, score, seuil):
    
    val_client = données.loc[données["SK_ID_CURR"]==identifiant,"SK_ID_CURR"].values.size
    col_fit = données.iloc[:,2:].columns.tolist()
    
    if (val_client==1):
        client_1 = données.loc[données["SK_ID_CURR"]==identifiant,"SK_ID_CURR"].item()
        client_proba = données.loc[données["SK_ID_CURR"]==identifiant, "Proba_faillite"].item()
    
    return client_proba


# In[16]:


def valeur_variable_categ(col, data):
    val_cat = []
    for elt in sorted(data["Rang"].unique()):
        val = data.loc[(data["Rang"]==elt) & (data[col]==1), 
                        col].shape[0]
        tot = data.loc[(data["Rang"]==elt), col].shape[0]
        
        val_cat.append(val*100/tot)
    
    return val_cat

def valeur_variable_num(col, data):
    val_num = []
    for elt in sorted(data["Rang"].unique()):
        val = data.loc[(data["Rang"]==elt), 
                        col].mean(axis=0)        
        val_num.append(val)
        
    return val_num


# In[17]:


def graph_variable(identifiant, shap_df, df, num_var, seuil):
    
    len_x = len(shap_df.iloc[num_var,2])
    rang_s = df.loc[df["Proba_faillite"]<seuil, "Rang"].max()
    
    rang = df.loc[df["SK_ID_CURR"]==identifiant, "Rang"].item()
    var_client = df.loc[df["SK_ID_CURR"]==identifiant, shap_df.iloc[num_var,0]].item()
    ind = (shap_df.index[shap_df["Variables"]==shap_df.iloc[num_var,0]].values).item() 

    plt.bar(np.arange(0, rang_s+1,1), shap_df.iloc[num_var, 2][0:rang_s+1], color="green", align="center")
    plt.bar(np.arange(rang_s+1,len_x,1), shap_df.iloc[num_var, 2][rang_s+1:], color="red", align="center")
        
    if (ind>3):
        plt.scatter(rang, var_client, color="black", label=f"Client n°{identifiant}")
    else:
        plt.vlines(x=rang, ymin=0, ymax=shap_df.iloc[num_var,-1], color="black", label=f"Client n°{identifiant}")
    
    plt.xlabel("Groupe de clients", fontsize = 7)
    plt.ylabel(f"{shap_df.iloc[num_var,-2]}", fontsize = 7, labelpad=10)
    
    plt.xticks(np.arange(0, len_x+2, 5))
    
    plt.xlim(-1, len_x+1)
    plt.ylim(0, (shap_df.iloc[num_var,-1]))
                   
    plt.legend(frameon=False, loc=1, fontsize=7)
    
    plt.title(f"{shap_df.iloc[num_var,-3]}""\npar groupes de clients", fontsize=10)


# In[18]:


def graph_shap(shap_df, identifiant):
    shap_df["Shap"] = shap_df["Shap"]*100/shap_df["Shap"].sum(axis=0)
    
    plt.barh(width = shap_df["Shap"], y = shap_df["Label"], color="black")
    
    plt.xlabel("Pourcentage d'impact sur la décision", labelpad = 10, fontsize=11)
    plt.ylabel("Critères", labelpad = 10, fontsize=11)
    
    plt.yticks(np.arange(0,shap_df.shape[0], 1), shap_df["Label"].values.tolist(), fontsize=10)
    
    plt.title("Classement du poids des critères\n"f"pour le client n°{identifiant}", fontsize=14, pad=20)


# In[19]:


def graph_distribution(df, client_proba, seuil):
    
    maxim = 1.1*df["Rang"].value_counts().max()
    
    plt.hist(df.loc[df["Proba_faillite"]<seuil, "Proba_faillite"], bins=np.arange(0,seuil+1,5), color = "green")
    plt.hist(df.loc[df["Proba_faillite"]>=seuil, "Proba_faillite"], bins=np.arange(seuil,101,5), color = "red")
    plt.vlines(x=client_proba, ymin=0, ymax=maxim, color="black")
    
    plt.ticklabel_format(axis='y', scilimits=[0,0])
    plt.xlim(0,100)
    plt.ylim(0, maxim)
    
    plt.xlabel("Probabilité de faillite (%)", fontsize=11, labelpad=10)
    plt.ylabel("Nombre de clients", fontsize = 11, labelpad=10)
    
    plt.title("Distribution de la probabilité de faillite\nde remboursement d'un crédit des clients", 
              fontsize = 14, pad=20)


# In[20]:


def plot_shap(données, identifiant, col_fit):
    
    val_voiture = valeur_variable_categ("Propriétaire d'une voiture", données)
    val_logement = valeur_variable_categ("Propriétaire d'un logement", données)
    val_emploi = valeur_variable_categ("Emploi business", données)
    val_marié = valeur_variable_categ("Marié(e)", données)
    
    val_pop_region = valeur_variable_num("Population relative région", données)
    val_pret_annulé = valeur_variable_num("Nombre de prêt annulé/vendu", données)
    val_pret_renouvelable = valeur_variable_num("Nombre d'ancien prêt renouvelable", données)
    val_pret_retard = valeur_variable_num("Pourcentage de prêt avec retard", données)
    
    shap_val = données.loc[données["SK_ID_CURR"]==identifiant, col_fit[-8:]].values.ravel().tolist()
    
    colonne = ["Propriétaire d'une voiture", "Propriétaire d'un logement", "Emploi business", 
               "Marié(e)", "Population relative région", "Nombre de prêt annulé/vendu", 
               "Nombre d'ancien prêt renouvelable", "Pourcentage de prêt avec retard"]
    
    col_val = [val_voiture, val_logement, val_emploi, val_marié, 
               val_pop_region, val_pret_annulé, val_pret_renouvelable, val_pret_retard]
    
    colonne_graph = ["Pourcentage de client vehiculé", "Pourcentage de propriétaire", 
                     "Pourcentage de businessman(woman)", 
                     "Pourcentage de client marié(e)", "Population relative région moyenne", 
                     "Ancien prêt annulé/vendu moyen", "Ancien prêt renouvelable moyen", 
                     "Pourcentage moyen de prêt avec retard"]
    
    colonne_label = ["Pourcentage\nclient vehiculé", "Pourcentage\n propriétaire", 
                     "Pourcentage\nbusinessman(woman)", 
                     "Pourcentage\nclient marié(e)", "Population relative\nrégion", 
                     "Ancien prêt\nannulé/vendu", "Ancien prêt\nrenouvelable", 
                     "Pourcentage\nprêt avec retard"]

    col_val = [val_voiture, val_logement, val_emploi, val_marié, 
               val_pop_region, val_pret_annulé, val_pret_renouvelable, val_pret_retard]

    col_max = [101, 101, 101, 101, 0.08, 32, 25, 101]
    
    shap_df = pd.DataFrame(list(zip(colonne, shap_val, col_val, colonne_graph, colonne_label, col_max)), 
                               columns = ["Variables", "Shap", "Valeur", 
                                          "Variable_g", "Label", "Max_g"]).sort_values("Shap").reset_index(drop=True)
    
    return shap_df


# In[21]:


def plot_dist_shap(identifiant, données, score, seuil):
    
    client_proba = proba_faillite(identifiant, données, score, seuil)
    
    rang = données.loc[données["SK_ID_CURR"]==identifiant, "Rang"].item()
    
    col_fit = données.iloc[:,2:].columns.tolist()
    
    shap_df = plot_shap(données, identifiant, col_fit)
    
    plt.figure(figsize=(15,4))
    
    plt.subplot(121)
    
    graph_distribution(données, client_proba, seuil)
    
    plt.subplot(122)
    
    graph_shap(shap_df, identifiant)
    
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.55, 
                        hspace=0.5)
    
    st.pyplot(plt.gcf())
#     plt.show()
    
    return shap_df, rang


# In[29]:


def plot_client_shap(identifiant, données, score, seuil, critère):    
    
    plt.figure(figsize=(11,4.2))
    if (len(critère)==0):
        plt.subplot(121)
        
        graph_variable(identifiant, shap_df, données, -1, seuil)
        
        plt.subplot(122)
        
        graph_variable(identifiant, shap_df, données, -2, seuil)
    
    elif (len(critère)==2):
        plt.subplot(121)
        ind_1 = (shap_df.index[shap_df["Variables"]==critère[0]].values).item() 
        graph_variable(identifiant, shap_df, données, ind_1, seuil)
        
        plt.subplot(122)
        ind_2 = (shap_df.index[shap_df["Variables"]==critère[1]].values).item() 
        graph_variable(identifiant, shap_df, données, ind_2, seuil)    
    
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.55, 
                        hspace=0.5)
    

#     plt.savefig("Outputs/données_client.jpg")
    
    fig_html = mpld3.fig_to_html(plt.gcf())
    components.html(fig_html, height = 500, scrolling=True)


# In[30]:


def gauge(identifiant, client_proba):
    
    plt.figure(figsize=(4, 4))
    plt.subplot(polar=True)
    
    label_loc = np.linspace(start = 0, stop = np.pi, num=100)
    
    liste_seuil = np.arange(0, 101, 1)
    
    ind_seuil = 100-int(np.argwhere(liste_seuil==seuil))
    
    gauge_up = np.ones(100)
    gauge_d = np.ones(100)-0.3
    
    plt.plot(label_loc[ind_seuil:], gauge_up[ind_seuil:], color = "green")
    plt.fill_between(label_loc[ind_seuil:], gauge_up[ind_seuil:], gauge_d[ind_seuil:], color = "green", alpha=0.9)
    
    plt.plot(label_loc[:ind_seuil], gauge_up[:ind_seuil], color = "red")
    plt.fill_between(label_loc[:ind_seuil], gauge_up[:ind_seuil], gauge_d[:ind_seuil], color = "red", alpha=0.9)
    
    val_faillite = round(client_proba,0)
    
    ind_val_faillite = 100-int(np.argwhere(liste_seuil==val_faillite))
    
    if (client_proba==0):
        plt.annotate(f"{int(client_proba)} %", xytext=(0,0), xy=(label_loc[ind_val_faillite-1], 0.9),
                     arrowprops=dict(arrowstyle="wedge, tail_width=1", color="black", shrinkA=0),
                     bbox=dict(boxstyle="circle", facecolor="black", edgecolor="black", linewidth=3), 
                     fontsize=15, color="white", ha="center", weight = "bold")
        
    else:
        plt.annotate(f"{int(client_proba)} %", xytext=(0,0), xy=(label_loc[ind_val_faillite], 0.9),
                     arrowprops=dict(arrowstyle="wedge, tail_width=1", color="black", shrinkA=0),
                     bbox=dict(boxstyle="circle", facecolor="black", edgecolor="black", linewidth=3), 
                     fontsize=15, color="white", ha="center", weight = "bold")
        
        
    plt.xlim(0, max(label_loc))
    
    x_ti_val = np.arange(max(label_loc), -0.01, -(max(label_loc)/10))
    x_ti_lab = np.arange(0,101,10)
    plt.xticks(x_ti_val[::2], x_ti_lab[::2], fontsize=13)
    plt.yticks([])
    
    plt.title("Probabilté de faillite de \nremboursement d'un crédit", ha="center", fontsize=15)
    plt.gca().spines["polar"].set_visible(False)
    
    plt.grid(visible=False)
    
    plt.box(False)
    
#     plt.savefig("Outputs/jauge_client.jpg")
#     fig_html = mpld3.fig_to_html(plt.gcf())
#     components.html(fig_html, height=200)
    st.pyplot(plt.gcf())
#     plt.show()


# In[147]:


st.set_page_config(
    page_title="Dossier de demande crédit",
    page_icon="✅",
    layout="wide")

st.title("Dossier de demande de crédit")

@st.experimental_memo  
def load_data():
    df, score, seuil = f_entraînement()
    return df, score, seuil

df, score, seuil = load_data()
df = df.sort_values("SK_ID_CURR")

col = df.iloc[:,2:10].columns
liste_id = (df["SK_ID_CURR"].values.tolist())

col1, col2 = st.columns([1, 3])

placeholder = st.empty()

with placeholder.container():
    id_client = st.text_input("Entrer l'identifiant du client. Si inconnu taper 0")
    
    if id_client=="0":
        client_choisi = st.select_slider('Choisir un identifiant', options = (liste_id), 
                                         label_visibility="hidden")
        id_client = (client_choisi)

    if id_client:
        with col1:
            client_proba = proba_faillite(id_client, df, score, seuil)
            gauge(id_client, client_proba)
            st.write(f'Le résultat obtenu a une précision de {round(100*score,1)} %.')
            '''
            Le seuil obtenu a été déterminé de façon à limiter le nombre de crédit donné à des clients 
            n'allant pas au bout de leur prêt.
            '''
        with col2:  
            shap_df, rang = plot_dist_shap(id_client, df, score, seuil)
        
        l_rang = [rang, df["Rang"].min(), df["Rang"].max()]
        
        col_1 = ['Client vehiculé (%)', 'Propriétaire (%)', 'Pop. relative région', 
                 'Prêt avec retard  (%)', 'Ancien prêt annulé/vendu', 'Ancien prêt renouv.', 
                 'Businessman(woman) (%)', 'Client marié(e) (%)']
        
        data = pd.DataFrame(list(zip(df.loc[df["Rang"]==l_rang[0], col].mean().values.tolist(), 
                                     df.loc[df["Rang"]==l_rang[1], col].mean().values.tolist(),
                                     df.loc[df["Rang"]==l_rang[2], col].mean().values.tolist())), 
                            columns = [f"Rang {rang}", "Meilleur rang", "Pire rang"], 
                            index = col_1)
        
        data.iloc[0:2,:] = data.iloc[0:2,:]*100
        data.iloc[-2:,:] = data.iloc[-2:,:]*100
        
        st.dataframe(data.T)
        
        options = st.multiselect('Choisir 2 critères à afficher', col)
        plot_client_shap(id_client, df, score, seuil, options)

