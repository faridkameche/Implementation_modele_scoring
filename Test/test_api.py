import numpy as np
import joblib
from P7_Flask_API_ import app
from flask import Flask, request, jsonify


def test_read_data():
    with app.test_request_context('/predict?Proprietaire_voiture=0&Proprietaire_maison=1&REGION_POPULATION_RELATIVE=0.008&Pourcentage_prêt_retard=0&Nombre_prêt_annulé_vendu=3&Nombre_ancien_prêt_renouvelable=2&Emploi_business=1&Marié=0'):
        assert request.path == '/predict'
        assert request.args['Proprietaire_voiture'] == '0'
        assert request.args['Proprietaire_maison'] == '1'
        assert request.args['REGION_POPULATION_RELATIVE'] == '0.008'
        assert request.args['Pourcentage_prêt_retard'] == '0'
        assert request.args['Nombre_prêt_annulé_vendu'] == '3'
        assert request.args['Nombre_ancien_prêt_renouvelable'] == '2'
        assert request.args['Emploi_business'] == '1'
        assert request.args['Marié'] == '0'
        
def test_response():
    with app.test_request_context('/predict?Proprietaire_voiture=0&Proprietaire_maison=1&REGION_POPULATION_RELATIVE=0.008&Pourcentage_prêt_retard=0&Nombre_prêt_annulé_vendu=3&Nombre_ancien_prêt_renouvelable=2&Emploi_business=1&Marié=0'):
        pipeline_lgbm = joblib.load("Outputs/pipeline_lgbm.pkl")
        Proprietaire_voiture = np.log10(float(request.args['Proprietaire_voiture'])+1)
        Proprietaire_maison = np.log10(float(request.args['Proprietaire_maison'])+1)
        REGION_POPULATION_RELATIVE = np.log10(float(request.args['REGION_POPULATION_RELATIVE'])+1)
        Pourcentage_prêt_retard = np.log10(float(request.args['Pourcentage_prêt_retard'])+1)
        Nombre_prêt_annulé_vendu = np.log10(float(request.args['Nombre_prêt_annulé_vendu'])+1)
        Nombre_ancien_prêt_renouvelable = np.log10(float(request.args['Nombre_ancien_prêt_renouvelable'])+1)
        Emploi_business = np.log10(float(request.args['Emploi_business'])+1)
        Marié = np.log10(float(request.args['Marié'])+1)
        
        val = [[Proprietaire_voiture, Proprietaire_maison, REGION_POPULATION_RELATIVE, Pourcentage_prêt_retard, 
                Nombre_prêt_annulé_vendu, Nombre_ancien_prêt_renouvelable, Emploi_business, Marié]]
        
        prediction = pipeline_lgbm.predict_proba(val).tolist()
        
        assert prediction[0][1] <= 1