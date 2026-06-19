# Prophetia2

**Equipo de Desarrollo:**
* **César Becerra Valencia** y **José Luis Cortes Nava**: Cientificos de Computacion por la UNAM. Encargados de la arquitectura de software, la ingenieria de datos (Feature Engineering) y el entrenamiento del modelo predictivo (Machine Learning).
* **Dylan Eduardo Becerra Valencia**: Apostador profesional con solidos conocimientos en probabilidad. Encargado de auditar y validar los resultados matematicos del modelo, ofreciendo retroalimentacion experta para ajustar las probabilidades a escenarios reales de apuestas deportivas.

Prophetia2 es una arquitectura avanzada de Machine Learning diseñada para predecir los resultados de partidos de futbol utilizando datos tacticos detallados provenientes de StatsBomb. 

El nucleo predictivo del proyecto se basa en un modelo XGBoost Classifier, el cual evalua metricas tacticas complejas como Goles Esperados (xG), presiones, intercepciones y efectividad de pases. Para garantizar el rigor matematico y prevenir la fuga de datos (Data Leakage), el sistema emplea metodos de ingenieria de series temporales, calculando los promedios historicos (Rolling Averages) del estado de forma de los equipos antes del inicio de cada partido. En lugar de predecir resultados binarios, el modelo emite distribuciones probabilisticas (Victoria, Empate, Derrota) optimizadas mediante la metrica Log-Loss.

## Instalacion

Se recomienda utilizar un entorno virtual de Python. Para instalar las dependencias necesarias para la extraccion de datos, entrenamiento del modelo y visualizacion:

1. Clonar el repositorio.
2. Crear y activar el entorno virtual (opcional pero recomendado).
3. Instalar dependencias mediante pip:
   ```bash
   pip install -r requeriments.txt
   ```

## Flujo de Ejecucion

Para obtener predicciones y visualizar los resultados, se debe seguir un pipeline estandar de datos:

### 0. Descarga de Datos (Ingestion)
Para alimentar el modelo, primero debes descargar la base de datos abierta de StatsBomb (partidos y eventos tacticos). Ejecuta el script de ingestion:
```bash
python ingestion/statsbomb_ingestion.py
```
*Nota sobre el almacenamiento:* Los datos se guardan en formato `.parquet`, que es altamente comprimido. Descargar una buena cantidad de partidos (cientos o miles) ocupara aproximadamente entre **500 MB y 1.5 GB** de espacio en tu disco duro, dependiendo de las competiciones habilitadas en el script.

### 1. Procesamiento de Datos (Feature Engineering)
Antes de entrenar o predecir, es necesario transformar los eventos crudos en estadisticas consolidadas sin fuga de datos. Ejecute el siguiente script desde la raiz del proyecto:
```bash
python core/feature_engineering.py
```
Este script leera los archivos `.parquet` descargados de StatsBomb, calculara los promedios moviles para cada equipo, y generara el dataset de entrenamiento en la carpeta `data/processed/`.

### 2. Entrenamiento del Modelo de IA
Una vez procesados los datos, proceda a entrenar el modelo XGBoost:
```bash
python core/train.py
```
Este proceso dividira los datos, entrenara el algoritmo y evaluara el Log-Loss. Al finalizar, guardara el modelo matematico compilado (`prophetia_xgb_model.pkl`) en el directorio `core/save_models/`.

### 3. Visualizacion y Prediccion Cientifica (Jupyter Notebooks)
El analisis interactivo y las predicciones individuales se gestionan a traves de Jupyter Notebooks. Inicie su servidor de Jupyter:
```bash
jupyter notebook
```
Navegue a la carpeta `notebooks/` y utilice los siguientes archivos:

*   **01_data_exploration.ipynb**: Permite explorar la estructura cruda de los datos de StatsBomb.
*   **02_model_selection.ipynb**: Genera visualizaciones graficas sobre la correlacion tactica y revela que estadisticas (Feature Importance) considera XGBoost mas determinantes para ganar.
*   **03_live_dashboard.ipynb**: Contiene un panel interactivo (Dashboard). Ejecute todas las celdas para habilitar un selector desplegable de partidos. Al seleccionar un encuentro, el sistema utilizara el modelo entrenado para emitir las probabilidades exactas de Victoria, Empate o Derrota, comparando ademas la prediccion con el flujo tactico real del partido.
