#!/usr/bin/env python

import os
import json
import redis
from flask import Flask, request, jsonify
from flask_cors import CORS , cross_origin
import pandas as pd
from linkextractor import columnas
import numpy as np
from scipy.spatial.distance import cityblock


app = Flask(__name__)
CORS(app)

redis_conn = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

@app.route("/")
def index():
    return "Usage: http://<hostname>[:<prt>]/apix/<url>"

#----------------------------------------------------------------
total = {}
valoresfinal = {}
peliculasp = {}

'''@app.route('/api/csv', methods=['POST'])
def recibir_csv():
   
    if request.method == 'POST':
        data = request.get_json()  
        nombre = data.get('obj')  
        redis_conn.set('csv', json.dumps(nombre))
        return jsonify({"csv cargado correctamente a redis"})
    else:
        return jsonify({"mensaje": "Esta ruta solo acepta solicitudes POST"})'''
    


@app.route('/apix/valor', methods=['POST'])
def recibir_datos():
    global valoresfinal , peliculasp
    if request.method == 'POST':
        data = request.get_json()  

        csv_cached = redis_conn.get('csv')
        csv_data = json.loads(csv_cached)

        #nombre = data.get('obj')  
        

        col1 = data.get('col1')
        col2 = data.get('col2')
        col3 = data.get('col3')

        numero = data.get('numero')  
        numerox = int(numero)

        #peli = pd.DataFrame(nombre)
        peli = pd.DataFrame(csv_data)


        peli[col3] = pd.to_numeric(peli[col3], errors='coerce')
        #peli['movieId'] = pd.to_numeric(peli['movieId'], errors='coerce')
        peli[col1] = pd.to_numeric(peli[col1], errors='coerce')


        consolidated_dfmi = columnas(peli, col1, col2, col3)
        #consolidated_dfmi = pd.concat([consolidated_dfmi.iloc[:numerox], consolidated_dfmi.iloc[300:]])
        consolidated_dfmi = pd.concat([consolidated_dfmi.query(f'userId == {numerox}'), consolidated_dfmi.iloc[300:]])
        consolidated_dfmi = consolidated_dfmi.loc[~consolidated_dfmi.index.duplicated(keep='first')]
        consolidated_dfmi = consolidated_dfmi.fillna(0)


        def computeNearestNeighbor(dataframe, target_user, distance_metric=cityblock):
            distances = np.zeros(len(dataframe))
            target_row = dataframe.loc[target_user]  
            for i, (index, row) in enumerate(dataframe.iterrows()):
                if index == target_user:
                    continue  
                
                non_zero_values = (target_row != 0) & (row != 0)
                distance = distance_metric(target_row[non_zero_values].fillna(0), row[non_zero_values].fillna(0))
                distances[i] = distance
            
            sorted_indices = np.argsort(distances)
            sorted_distances = distances[sorted_indices]
            return list(zip(dataframe.index[sorted_indices], sorted_distances))
        

        target_user_id = numerox
        neighborsmi = computeNearestNeighbor(consolidated_dfmi, target_user_id)
        diccionario_resultante = dict(neighborsmi)
        valoresfinal = diccionario_resultante

        #pruebas
        cd2 = pd.DataFrame(neighborsmi)
        cd2.columns = ['Id_user', 'Distancias']

        primeros = cd2['Id_user'].unique().tolist()[:10]
        resul = peli.query('userId in @primeros')
        newx = resul.query('rating == 5.0')['movieId'].drop_duplicates()
        dictionary_final = dict(zip(newx.index, newx.values))
        peliculasp = dictionary_final

        '''peli = pd.DataFrame(nombre)
        data = peli['rating'].value_counts().sort_index(ascending=False)
        diccionario_resultante = data.to_dict()
        valoresfinal = diccionario_resultante'''

        cached_data = redis_conn.get('valoresfinal') #300 valores
        newcached_data = json.loads(cached_data)
        newcached_data.update(valoresfinal)
        
        

        



        #redis_conn.set('valoresfinal', json.dumps(valoresfinal))
        redis_conn.set('valoresfinal', json.dumps(newcached_data))
        redis_conn.set('peliculas', json.dumps(peliculasp))


        #return jsonify(valoresfinal)
        return jsonify(valoresfinal)
    else:
        return jsonify({"mensaje": "Esta ruta solo acepta solicitudes POST"})
#----------------------------------------------------------------



@app.route('/apix/valor', methods=['GET'])
def get_users():
    # Intenta recuperar datos desde Redis
    cached_data = redis_conn.get('valoresfinal') 
    if cached_data:
        return jsonify(json.loads(cached_data))
    else:
        return jsonify({"mensaje": "No hay valores finales almacenados en Redis para la instancia 2"})

@app.route('/apix/peliculas', methods=['GET'])
def get_peliculas():
    peliculas_cached = redis_conn.get('peliculas') 
    if peliculas_cached:
        peliculas = json.loads(peliculas_cached)
        return jsonify(peliculas)
    else:
        return jsonify({"mensaje": "No hay valores finales almacenados en Redis"})
    

@app.route('/apix/csv', methods=['GET'])
def get_csv():
    csv_cached = redis_conn.get('csv') 
    if csv_cached:
        csvx = json.loads(csv_cached)
        return jsonify(csvx)
    else:
        return jsonify({"mensaje": "No hay valores finales almacenados en Redis"})


app.run(host="0.0.0.0")
