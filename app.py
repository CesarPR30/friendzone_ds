# app.py
from flask import Flask, render_template, request, jsonify
import pickle  # Para cargar el modelo
import numpy as np

app = Flask(__name__)

# Cargar el modelo desde el archivo modelo.pkl
with open('modelo2.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Recibe datos del formulario
    grado_simpatico = int(request.form['Grado_Simpatico'])
    regalos_realizados = int(request.form['Regalos_Realizados'])
    te_habla_de_sus_citas = int(request.form['Te_habla_de_sus_citas'])
    tiempo_dedicado = int(request.form['Tiempo_dedicado'])
    amigos_presentados = int(request.form['Amigos_presentados'])
    veces_que_quedais_a_solas = int(request.form['Veces_que_quedais_a_solas'])
    consejos_dados = int(request.form['Consejos_dados'])
    print('*'*40,veces_que_quedais_a_solas)
    # Crear el array de características para la predicción
    features = np.array([[
        grado_simpatico,
        regalos_realizados,
        te_habla_de_sus_citas,
        tiempo_dedicado,
        amigos_presentados,
        veces_que_quedais_a_solas,
        consejos_dados
    ]])
    
    print(features)
    # Hacer la predicción usando el modelo cargado
    friendzone_score = ((model.predict_proba(features)[0])[1])*100# Suponiendo que el modelo devuelve un score
    print(friendzone_score)
    # Determina el resultado basado en el score
    if friendzone_score > 70:
        result = "Alta probabilidad de Friendzone 😢"
    elif friendzone_score > 40:
        result = "Riesgo moderado de Friendzone 🤔"
    else:
        result = "¡Bajas probabilidades! 🎉"

    return jsonify({"score": str(friendzone_score), "result": str(result)})

if __name__ == '__main__':
    app.run(debug=True)
"""
# app.py
from flask import Flask, render_template, request, jsonify
import random  # Simulación de predicción (puedes reemplazar esto con un modelo ML real)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Recibe datos del formulario
    grado_simpatico = int(request.form['Grado_Simpatico'])
    regalos_realizados = int(request.form['Regalos_Realizados'])
    te_habla_de_sus_citas = int(request.form['Te_habla_de_sus_citas'])
    tiempo_dedicado = int(request.form['Tiempo_dedicado'])
    amigos_presentados = int(request.form['Amigos_presentados'])
    veces_que_quedais_a_solas = int(request.form['Veces_que_quedais_a_solas'])
    consejos_dados = int(request.form['Consejos_dados'])

    # Simula una predicción (aquí puedes integrar tu modelo ML)
    friendzone_score = random.randint(0, 100)  # Score aleatorio entre 0 y 100

    # Determina el resultado basado en el score
    if friendzone_score > 70:
        result = "Alta probabilidad de Friendzone 😢"
    elif friendzone_score > 40:
        result = "Riesgo moderado de Friendzone 🤔"
    else:
        result = "¡Bajas probabilidades! 🎉"

    return jsonify({"score": friendzone_score, "result": result})

if __name__ == '__main__':
    app.run(debug=True)
"""