# Prophetia2

**Equipo de Desarrollo:**
* **César Becerra Valencia** y **José Luis Cortes Nava**: Cientificos de Computacion por la UNAM. Encargados de la arquitectura de software, la ingenieria de datos (Feature Engineering) y el entrenamiento del modelo predictivo (Machine Learning).
* **Dylan Eduardo Becerra Valencia**: Apostador profesional con solidos conocimientos en probabilidad. Encargado de auditar y validar los resultados matematicos del modelo, ofreciendo retroalimentacion experta para ajustar las probabilidades a escenarios reales de apuestas deportivas.

Prophetia2 es una arquitectura avanzada de Machine Learning Cuantitativo (Quant) diseñada para predecir los resultados de partidos de fútbol y encontrar valor financiero (Value Bets) en el mercado de apuestas.

El núcleo predictivo del proyecto ha evolucionado hacia una avanzada **Arquitectura de Doble Stacking (Ensemble de Meta-Modelos)**. En lugar de predecir el resultado con un solo algoritmo monolítico, el sistema divide el trabajo en múltiples modelos especialistas y los ensambla en capas:
1. **Modelo Poisson (Fuerza Relativa):** Analiza exclusivamente ELO y xG para proyectar el flujo de goles (XGBoost).
2. **Modelo Contexto (Táctica y Fatiga):** Analiza rachas (Momentum), valor de plantilla (Transfermarkt) y métricas tácticas (XGBoost).
3. **Modelo Deep Learning (No-Linealidades):** Un perceptrón multicapa (*Multi-Layer Perceptron* en PyTorch) que captura relaciones matemáticas complejas ignoradas por los árboles de decisión.
4. **Modelo Caza-Empates (Draw-Catcher):** Un modelo binario enfocado exclusivamente en cazar la rentabilidad oculta de los empates.
5. **Modelo de Dinámica de Mercado:** Extrae información puramente financiera ("Steam" y "Vig") de las casas de apuestas más eficientes (Pinnacle/Asia).
6. **Meta-Modelo de Doble Stacking:** Consiste en dos niveles. Primero consolida un *Stacker Fundamental* con las probabilidades puramente deportivas y luego las cruza en un *Stacker Final* con las variables de mercado aplicando una fuerte penalidad L2 (Ridge) para evitar que el algoritmo se vuelva "perezoso" y asegurar un valor esperado (EV) genuino.

<!-- ## Instalacion

Se recomienda utilizar un entorno virtual de Python. Para instalar las dependencias necesarias para la extraccion de datos, entrenamiento del modelo y visualizacion:

1. Clonar el repositorio.
2. Crear y activar el entorno virtual (opcional pero recomendado).
3. Ejecutar el archivo dependencias.bat (Solo la primera vez).
4. Ejecutar los archivos iniciar_backend.bat y iniciar_frontend.bat -->

## Requisitos del Sistema (Hardware)

Prophetia2 es una arquitectura intensiva y está **optimizada nativamente para aceleración por hardware (NVIDIA CUDA)**, requiriendo procesar árboles de decisión (XGBoost GPU) y redes neuronales multicapa (PyTorch) sobre decenas de miles de filas con cientos de métricas.

*   **Mínimos (Para ejecutar inferencia y predecir sin re-entrenar):**
    *   CPU: Procesador moderno de 4 núcleos (ej. Intel Core i5 / Ryzen 5). Nota: Durante el entrenamiento, un procesador de 4-6 núcleos llegará al 100% de uso.
    *   RAM: 8 GB.
    *   GPU: No obligatoria, la inferencia puede correr en CPU.
*   **Recomendados (Para Entrenamiento Completo del Meta-Modelo):**
    *   CPU: 8 núcleos físicos o superior (ej. Ryzen 7 5700X o Intel i7). Con 8 núcleos, el uso de CPU rondará el 50%-60% durante el entrenamiento, evitando cuellos de botella.
    *   RAM: 16 GB - 32 GB (crucial para procesar el Dataset OOF en memoria).
    *   GPU: **Tarjeta gráfica NVIDIA con soporte CUDA (imprescindible para acelerar XGBoost y PyTorch)**. Se recomiendan mínimo 8GB de VRAM (ej. RTX 3060, RTX 4060 o superior). 
    *   *Nota de rendimiento:* Entrenar el pipeline completo con una RTX 4060 OC lleva a la GPU al 100% de uso constante y tarda aproximadamente **30 minutos** en converger, más 2-3 horas adicionales si se realiza la extracción web completa desde cero (Scraping de Transfermarkt).

## Arquitectura Matemática y Financiera

1. **Arquitectura Anti-Leakage y Single-Row:** El modelo filtra duplicados (Double-Row Betting) y calibra las probabilidades puramente fuera de muestra (OOS) garantizando métricas financieras honestas y robustas.
2. **Valoración de Plantillas (Proxy de Jugadores):** Mediante web scraping automatizado, Prophetia2 inyecta el valor de mercado contemporáneo (en millones de euros) de cada equipo desde Transfermarkt para sintetizar la calidad técnica de los jugadores sin sufrir la maldición de la dimensionalidad.
3. **Ratings Dinámicos (Ataque/Defensa) y ELO:** El sistema calcula ratings separados de ataque y defensa para cada equipo basándose en la métrica de Goles Esperados (xG), actualizándolos partido a partido sin contaminar el futuro.
2. **Modelo de Poisson Bivariado (Dixon-Coles):** Dado que el fútbol es un deporte de baja puntuación, los empates ocurren con mayor frecuencia de lo que sugiere la independencia estadística. Prophetia2 usa el ajuste de Dixon-Coles ($\rho \approx -0.15$) para inflar matemáticamente la probabilidad conjunta de resultados 0-0 y 1-1.
3. **Calibración Isotónica:** Las probabilidades crudas del XGBoost se ajustan de manera no-paramétrica para asegurar que una predicción del "60%" realmente se cumpla el 60% de las veces en la realidad.
4. **Simulación Financiera y Kelly Criterion:** Prophetia2 no solo clasifica; es un simulador de inversiones. El entrenamiento concluye con un backtest financiero riguroso. El algoritmo compara sus predicciones contra las **Cuotas de Cierre (Closing Odds)** (la línea más eficiente del mercado). Si encuentra Expectativa Matemática positiva (EV > 5%), el modelo aplica el **Criterio de Kelly** fraccionado para calcular exactamente qué porcentaje del Bankroll apostar (max 5%). Al final, reporta el ROI, Turnover y el Yield neto.

## Flujo de Ejecución

Para obtener predicciones y ejecutar las simulaciones financieras, sigue el pipeline estándar:

### 0. Descarga de Datos de Partidos (Ingestion)
Para alimentar el modelo, primero debes descargar los históricos de partidos y eventos tácticos (15 ligas principales y secundarias a lo largo de 12 años). Ejecuta el script:
```bash
python core/download_football_data.py
```

### 1. Descarga del Proxy de Jugadores (Transfermarkt)
Extrae el valor de mercado histórico de cada equipo para medir el nivel técnico de sus jugadores. Cuenta con guardado progresivo antibloqueos.
```bash
python ingestion/transfermarkt_scraper.py
```

### 2. Adaptacion de Datos (Data Adapter)
Para permitir que Prophetia2 consuma las diferentes fuentes de datos, primero debes estandarizar los eventos crudos en un DataFrame Intermedio Universal. Ejecuta el adaptador:
```bash
python core/data_adapter.py
```
Este script generara un dataset intermedio tabular unificado en `data/interim/intermediate_dataset.parquet`.

### 3. Extracción de Cuotas (Bookmaker Odds)
Descarga el historial de cuotas de casas de apuestas globales para entrenar el metamodelo y simular rentabilidad:
```bash
python ingestion/fetch_odds.py
```
Este script descarga cuotas de apertura (PSH) y cierre (PSCH) para las 15 ligas, convirtiéndolas a probabilidades implícitas (Vig-free).

### 4. Procesamiento Matemático (Feature Engineering)
El motor de Prophetia2 cruza las estadísticas tácticas avanzadas (Rolling Averages), el ELO Clásico, los ratings Glicko y fusiona los millones de euros de Transfermarkt con cada evento:
```bash
python core/feature_engineering.py
```
Se generará el dataset final de entrenamiento listo para la IA en `data/processed/matches_dataset.parquet`.

### 5. Entrenamiento de la Arquitectura de Stacking (Pipeline Principal)
Una vez procesados los más de 65,000 partidos históricos, ejecuta el orquestador principal que entrenará de manera secuencial todos los modelos especializados sin fuga de datos (Data Leakage):
```bash
python core/run_pipeline.py
```
Este script orquesta los siguientes pasos automáticamente:
1. **train_poisson.py**: Entrena el Modelo Poisson basado en xG y ELO, generando características Out-Of-Fold (OOF).
2. **train_context.py**: Entrena el Modelo de Contexto (XGBoost) con métricas tácticas y optimización `Optuna`.
3. **train_nn.py**: Entrena la Red Neuronal Profunda (`PyTorch`) para encontrar relaciones no-lineales.
4. **train_draws.py**: Entrena el modelo binario "Caza-Empates" especializado.
5. **train_market.py**: Entrena el modelo financiero que analiza el movimiento del "Smart Money".
6. **train_stacker.py**: Ejecuta el **Doble Stacking**, combinando predicciones fundamentales y financieras, aplicando Calibración Isotónica y emitiendo el `test_predictions.parquet` final.
7. **train_clv_model.py (Meta-Modelo Odds Drift):** Para proteger el bankroll, inyecta métricas de **Divergencia** para predecir hacia dónde se moverá la cuota de cierre y detectar trampas de valor.

### 7. Evaluación Financiera (Simulador de Bankroll)
Por último, evalúa el desempeño de la estrategia aplicando Kelly Criterion y Bayesian Blending:
```bash
python core/simulate_bankroll.py
```
Este script leerá las predicciones base, aplicará tus diccionarios de riesgo (**Kelly Fractions** y **EV Thresholds** por liga), simulará la rentabilidad de un Bankroll inicial de $1,000 y ejecutará pruebas de resistencia (Monte Carlo Bootstrapping) para medir tu Maximum Drawdown y Probabilidad de Ruina (PoR).

> **Benchmark Actual (Última Evaluación Financiera con Meta-Modelo Optuna):** Utilizando este flujo riguroso, el modelo ha demostrado métricas de clase institucional en *Out-of-Sample* (7,799 partidos validados), alcanzando un **WinRate del 55.7%**, un asombroso **Yield neto del 15.64%** y un **ROI sobre el bankroll del 160.78%**, con un **Maximum Drawdown muy controlado de 9.50%**. Aún más importante, el modelo logra un **CLV Positivo del +4.75%**, demostrando matemáticamente que vence el cierre del mercado. La **Probabilidad de Ruina (PoR) se mantiene en 0.00%** tras 10,000 simulaciones extremas de estrés (Bootstrap por bloques Monte Carlo).
> 
> *⚠️ **Nota Cuantitativa sobre la Varianza:** Debido a que el modelo opera bajo un régimen de riesgo extremadamente estricto, la selectividad fue de apenas un 11.4% (889 apuestas ejecutadas de 7,799 partidos analizados). Por el bajo volumen relativo de apuestas, se debe ser consciente de que la Ley de los Grandes Números aún no ha estabilizado estas métricas por completo frente a la varianza. Por ende, los porcentajes reportados de Yield, ROI y CLV deben tomarse con cautela como una excelente validación inicial del modelo, pero podrían sufrir regresión a la media al exponerse a tamaños de muestra más grandes.*

**¿Qué datos entran al modelo? (Inputs)**
- **Consenso del Mercado (Metamodelado):** Probabilidades de apertura del mercado (`open_prob_win`, `open_prob_draw`, `open_prob_loss`).
- **Calidad de Jugadores (Proxy):** Valor de mercado de la plantilla contemporánea extraído de Transfermarkt (`team_squad_value`, `squad_value_diff`).
- **Ratings Cuantitativos:** Fuerza Relativa de Ataque, ELO Clásico (`team_elo`, `elo_diff`), y Ratings Puros de Ataque/Defensa (`team_att_rating`, `team_def_rating`).
- **Tácticas Base (EMA-3/EMA-5):** Goles esperados creados y concedidos (`xg_created`, `xg_conceded`), xG-Chain, posesión, y acciones defensivas (presiones, intercepciones).
- **Contexto Avanzado:** Días de descanso, desgaste por competiciones europeas previas (`is_european_hangover`), e historial directo (H2H).

**¿Qué datos expulsa el modelo? (Outputs)**
En lugar de dar un resultado seco, el modelo expulsa **probabilidades calibradas de Valor** para las tres clases posibles:
- Probabilidad de Victoria Local ajustada (ej. 60.5%)
- Probabilidad de Empate ajustada por Dixon-Coles (ej. 24.5%)
- Expectativa Matemática de la Apuesta (EV) frente a la línea de cierre.

### 7. Visualizacion y Prediccion Cientifica (Jupyter Notebooks)
El analisis interactivo y las predicciones individuales se gestionan a traves de Jupyter Notebooks. Inicie su servidor de Jupyter:
```bash
jupyter notebook
```
Navegue a la carpeta `notebooks/` y utilice los siguientes archivos:

*   **01_data_exploration.ipynb**: Permite explorar la estructura cruda de los datos de StatsBomb.
*   **02_model_selection.ipynb**: Genera visualizaciones graficas sobre la correlacion tactica y revela que estadisticas (Feature Importance) considera XGBoost mas determinantes para ganar.
*   **03_live_dashboard.ipynb**: Contiene un panel interactivo (Dashboard). Ejecute todas las celdas para habilitar un selector desplegable de partidos. Al seleccionar un encuentro, el sistema utilizara el modelo entrenado para emitir las probabilidades exactas de Victoria, Empate o Derrota, comparando ademas la prediccion con el flujo tactico real del partido.

### 8. Predicción Financiera en Terminal (CLI)
Para ejecutar pronósticos en tiempo real basados en los últimos partidos procesados y calcular el riesgo (Bankroll Staking), utiliza nuestro menú interactivo de predicción cuantitativa:
```bash
cd core
python cli_predictor.py
```
El script te permitirá seleccionar primero la competición (Liga) y luego autocompletará los equipos participantes. Podrás introducir las cuotas del mercado actual y aplicar penalizaciones de fuerza estocástica si hay jugadores clave lesionados.

**Ejemplo de Salida:**
```text
+------------------------------------------------------+
| Prophetia2 - Quant Value Betting CLI                 |
| Initializing stochastic models and feature stores... |
+------------------------------------------------------+

=== ANÁLISIS CUANTITATIVO DEL PARTIDO ===
+-----------------------------------------------------------------------------+
|             |            |             |            |          EV |         |
|             | Odds       |       Prob. |      Prob. |   (Expected |         |
| Mercado     | (Bookie)   |      Bookie |     Modelo |      Value) | Kelly % |
|-------------+------------+-------------+------------+-------------+---------|
| 1 (Local -  | 2.10       |       47.6% |      34.7% |      -27.2% |   0.00% |
| Arsenal)    |            |             |            |             |         |
| X (Empate)  | 3.40       |       29.4% |      40.2% |      +36.7% |  15.30% |
| 2 (Visita - | 3.50       |       28.6% |      25.1% |      -12.1% |   0.00% |
| Chelsea)    |            |             |            |             |         |
+-----------------------------------------------------------------------------+

=== RECOMENDACIÓN DE STAKING ===
+---------------------------- SISTEMA DE STAKING -----------------------------+
| VALUE DETECTADO                                                             |
| Selección: Empate (X) @ 3.40                                                |
| Stake Recomendado: $30.61 (3.06% del bankroll)                              |
+-----------------------------------------------------------------------------+
```
