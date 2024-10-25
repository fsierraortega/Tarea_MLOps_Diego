from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os

# Cargar el modelo y el scaler desde los archivos .pkl
with open('modelo_arbol_decision.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

with open('scaler.pkl', 'rb') as archivo_scaler:
    scaler = pickle.load(archivo_scaler)

# Definir las características esperadas
columnas = ['Major_Axis_Length', 'Perimeter', 'Area', 'Convex_Area', 'Eccentricity']

# Definir el modelo de datos de entrada usando Pydantic
class CaracteristicasEntrada(BaseModel):
    Major_Axis_Length: float
    Perimeter: float
    Area: int
    Convex_Area: int
    Eccentricity: float

# Crear la aplicación FastAPI
app = FastAPI(title="Clasificación de Granos de Arroz")

@app.get("/")
async def root():
    return {"message": "Hola mundo"}

# Definir el endpoint para predicción
@app.post("/prediccion/")
async def model_predict(entrada: CaracteristicasEntrada):
    try:
        # Convertir la entrada en un DataFrame
        datos_entrada = pd.DataFrame([entrada.dict().values()], columns=columnas)
        
        # Escalar las características
        datos_entrada_scaled = scaler.transform(datos_entrada)
        
        # Realizar la predicción
        prediccion = modelo.predict(datos_entrada_scaled)
        probabilidad = modelo.predict_proba(datos_entrada_scaled)[:, 1]
        
        # Construir la respuesta
        resultado = {
            "Tipo de Arroz": 'Osmancik' if bool(prediccion[0]) else 'Cammeo',
            "Probabilidad Osmancik": float(probabilidad[0])
        }
        
        return resultado
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
