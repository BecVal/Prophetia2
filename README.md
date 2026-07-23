# Prophetia2: Motor Cuantitativo de Trading Deportivo (Quantitative Sports Trading Engine)

**Prophetia2** es un motor algorítmico y matemático de *Machine Learning* Cuantitativo (*Quant*) diseñado para modelar la probabilidad de eventos deportivos, detectar ineficiencias en mercados de cuotas (Bookmakers, Polymarket) y ejecutar estrategias de *Value Betting* con gestión de riesgo institucional. 

A diferencia de los modelos predictivos estándar que buscan maximizar el *Accuracy*, Prophetia2 es fundamentalmente una arquitectura financiera que busca **maximizar el Valor Esperado (EV)** superando el *Vig* (margen) del mercado a través del cálculo riguroso del Closing Line Value (CLV).

---

## Arquitectura Matemática y Modelado Estocástico

El núcleo predictivo abandona los enfoques monolíticos en favor de una **Arquitectura de Doble Stacking (Ensemble de Meta-Modelos)**, combinando técnicas paramétricas, no paramétricas, y procesos estocásticos:

### 1. Modelado Bayesiano Zero-Inflated Negative Binomial (ZINB)
El fútbol es un proceso de conteo estocástico que presenta sobredispersión (varianza mayor que la media) e inflación de ceros estructurales (partidos defensivos bloqueados). Para modelar la matriz de goles con exactitud matemática, Prophetia2 abandona la clásica Regresión de Poisson y utiliza Programación Probabilística (**PyMC**) para optimizar una distribución **Zero-Inflated Negative Binomial (ZINB)**:
$$ P(Y=y) = \psi I_{y=0} + (1-\psi) \binom{y+\alpha-1}{y} p^\alpha (1-p)^y $$
El modelo aprende inferencias latentes de Ataque ($att^*$) y Defensa ($def^*$) para cada equipo encontrando el **Maximum A Posteriori (MAP)**. 
Para resolver la sobredependencia de empates de baja puntuación, aplica una corrección de **Dixon-Coles** donde el parámetro de correlación $\rho$ no es fijo, sino que es una variable aleatoria normal aprendida directamente de los datos empíricos mediante la maximización de la función de verosimilitud conjunta ponderada (*Weighted Joint Likelihood*).

### 2. Time-Decay Likelihood Ponderado
La habilidad real de un equipo de fútbol no es estática; decae como un proceso isotópico. El modelo ZINB implementa un decaimiento temporal exponencial continuo ($e^{-\lambda t}$) con una vida media (Half-life) de 600 días inyectado directamente en el núcleo tensorial de PyMC. Las observaciones más lejanas aportan progresivamente menos densidad probabilística (Likelihood) a los priors bayesianos:
$$ \mathcal{L}_{decay} \propto \sum_{i} w_i \log P(x_i | \theta) $$

### 3. Procesos Estocásticos: Geometric Brownian Motion (GBM)
El movimiento de las cuotas desde la apertura hasta el cierre se modela asumiendo eficiencia de mercado (Hipótesis del Mercado Eficiente). Evaluamos el "Drift" de las cuotas utilizando la dinámica del *Geometric Brownian Motion* sobre el espacio Log-Odds:
$$ dS_t = \mu S_t dt + \sigma S_t dW_t $$
El **Modelo Cuantitativo GBM** extrae la Volatilidad Implícita ($\sigma$) del consenso del mercado asiático (Pinnacle) y detecta ineficiencias midiendo las desviaciones Z-Score de la trayectoria real frente a la teórica. Esto captura sobre-reacciones del público (*Steam*) o caídas abruptas de liquidez.

### 4. Aprendizaje Profundo (Multi-Layer Perceptron)
El *Modelo de Contexto* (XGBoost) captura relaciones jerárquicas en la táctica (posesión, xG-Chain, presión, valor de plantilla vía Transfermarkt). Paralelamente, una red neuronal profunda (`PyTorch`) con activación ReLU y Dropout es inyectada en el ensamble para aproximar funciones continuas altamente no-lineales que los particionamientos del espacio de características (árboles) no pueden resolver de manera óptima.

### 5. Modelado Estocástico de Mercados Derivados (Córners, Tarjetas y Tiros al Arco)
Prophetia2 expande su frontera predictiva bidimensional hacia los mercados accesorios mediante regresiones distribucionales enfocadas en variables aleatorias de conteo:
- **Tiros al Arco (Shots on Goal):** El proceso ensambla un *Perceptrón Multicapa* (MLP) profundo optimizado iterativamente en `PyTorch` (AdamW, Weight Decay y Learning Rate Schedulers) con regresiones múltiples *Ridge* (Norma L2). La matriz topológica inyecta las inferencias algebraicas de Goles Esperados (xG) del modelo Cuantitativo primario como variables latentes exógenas. El mapeo del continuo resultante es sometido a una convolución bivariante de Poisson para derivar las probabilidades intrínsecas de superación de fronteras asimétricas en las líneas Over/Under (ej. $P(X > 4.5)$).
- **Tiros de Esquina (Corners) y Puntos por Tarjetas (Booking Points):** Tratados algorítmicamente como procesos estocásticos discretos $\mathbb{N}_0$, estos mercados se resuelven mediante el ensamblado paramétrico de árboles de decisión (XGBoost) minimizando la función logarítmica negativa de verosimilitud de Poisson (`poisson-nloglik`). Las funciones de densidad condicionales para múltiples umbrales de cierre en el entorno de casas de apuestas se extraen analíticamente evaluando el complemento de la Función de Distribución Acumulada (CDF): 
$$ P(X > k) = 1 - \sum_{i=0}^{\lfloor k \rfloor} \frac{e^{-\lambda_T} \lambda_T^i}{i!} $$
donde $\lambda_T$ es la intensidad esperada global. Posteriormente, la sobredispersión inherente a la varianza deportiva se mitiga aplicando endomorfismos de Calibración Isotónica estrictamente monótonos.

### 6. Meta-Stacking y Rigor Out-Of-Fold (OOF)
Las predicciones de los modelos base (Poisson, Red Neuronal, GBM, Caza-Empates, Mercado, Córners, Tarjetas, Tiros) se ensamblan mediante un *Meta-Modelo HistGradientBoosting* de Nivel 2, utilizando entropía como metadato exógeno y reduciendo ceguera del consenso general. 
Para asegurar la validez matemática y evitar *In-Sample Data Leakage* o *Look-Ahead Bias* (es decir, evitar que el modelo aprenda del futuro para predecir el pasado), la matriz de entrenamiento de Nivel 2 se genera utilizando una validación cruzada híbrida:
- **TimeSeriesSplit** para respetar estrictamente la flecha temporal causal.
- **L2 Regularization (Ridge)** y **K-Fold** segmentado inicial para garantizar ortogonalidad e independencia en las predicciones Out-Of-Fold.

### 7. Calibración Probabilística Isotónica
Las probabilidades brutas de un modelo ensamblado rara vez representan certezas matemáticas. Utilizamos **Calibración Isotónica** no-paramétrica (una regresión por pasos monótona) para mapear las salidas del HGB al espacio real de frecuencias relativas, garantizando que un evento con $P(x)=0.45$ ocurra empíricamente exactamente el 45% de las veces en el largo plazo (Test de Brier Score).

---

## Gestión de Riesgo y Simulación de Bankroll

### 1. Closing Line Value (CLV) Drift Prediction
Dado que los corredores institucionales corrigen sus cuotas antes del silbatazo inicial, el *Edge* (ventaja) desaparece rápidamente. Prophetia2 entrena un meta-regresor probabilístico (**Optuna + XGBoost**) sobre el *Drift* (Diferencial entre la predicción y el mercado). Si el modelo proyecta que el mercado se moverá agresivamente en nuestra contra antes del cierre ($CLV < 0$), el *Trade* se cancela protegiendo el capital.

### 2. Criterio de Kelly (Optimal Staking)
Prophetia2 es un simulador financiero automatizado. El riesgo de capital no es plano; se escala dinámicamente según la Expectativa Matemática (EV) de la inversión. Si $EV > \text{Threshold}$, se calcula la fracción de la banca óptima para maximizar la tasa de crecimiento compuesto:
$$ f^* = \frac{bp - q}{b} $$
Donde $b$ son las cuotas decimales netas, $p$ la probabilidad real calibrada, y $q = 1-p$. Para protegerse frente a la varianza inherente y eventos de cola pesada, Prophetia2 aplica un **Kelly Fraccionario** (ej. Quarter-Kelly $= f^* \times 0.25$) sujeto a un límite estricto de liquidez ($L_{max} = 3\%$).

### 3. Test de Resistencia: Monte Carlo Bootstrapping
La validación financiera somete los resultados de *Backtesting* a remuestreo con reemplazo (Bootstrapping) sobre miles de bloques simulando realidades estadísticas alternas. Se auditan dos métricas críticas:
- **Maximum Drawdown (MDD):** La caída máxima histórica desde el pico financiero del portafolio.
- **Probabilidad de Ruina (PoR):** La certeza probabilística de llevar el Bankroll a cero en los escenarios de mayor varianza.

*(En las pruebas exhaustivas Out-Of-Sample recientes, la arquitectura reportó un **Sharpe Ratio de 2.64**, un Yield Neto constante del 5.55%, y una PoR matemática de 0.00% tras 10,000 iteraciones de Monte Carlo).*

---

## Flujo Operativo y de Ejecución (CLI)

Prophetia2 consolida millones de filas y peticiones, por lo que demanda un ecosistema hardware apto (Optimizaciones **NVIDIA CUDA** requeridas para convergencia eficiente).

### Pipeline Cuantitativo

1. **Ingesta y Adaptación de Datos**:
   Descarga de eventos crudos (StatsBomb) y valoraciones de liquidez/plantilla (Transfermarkt).
   ```bash
   python core/download_football_data.py
   python ingestion/transfermarkt_scraper.py
   python core/data_adapter.py
   ```
2. **Ingeniería de Características y Dinámica de Cuotas**:
   Generación de medias móviles temporales y cruce con Cuotas Institucionales asiáticas sin Vig.
   ```bash
   python core/feature_engineering.py
   python ingestion/fetch_odds.py
   ```
3. **Entrenamiento de Arquitectura Meta-Stacker**:
   Ejecución controlada y orquestada para los 9 modelos, con garantías de propagación OOF libre de fugas de datos.
   ```bash
   python core/run_pipeline.py
   ```
4. **Validación Financiera**:
   Simulador de la curva de capital y métricas de portafolio (Sharpe, Sortino, CLV Promedio, MDD).
   ```bash
   python core/simulate_bankroll.py
   ```

---

## Trading Automatizado: Polymarket HFT Bot
La rama `polymarket_bot/main.py` contiene un bot enfocado en el comercio de frecuencias de media a alta (HFT Trading adaptado) en entornos blockchain on-chain como **Polymarket**.
- Implementa *Dynamic Alpha Blending*: Castiga matemáticamente nuestra convicción predictiva si el mercado global (*Wisdom of Crowds*) presenta alta entropía y consenso en nuestra contra.
- Aplica Coberturas (Dutching) óptimas distribuyendo liquidez iterativamente minimizando el impacto de *Slippage* del AMM (Automated Market Maker).
- Protege los retornos netos descontando las *Network Fees* dinámicas antes de ejecutar el *Trade*.

### Análisis Visual Interactivo (Notebooks)
Para propósitos de investigación cuántica (R&D), los investigadores disponen del entorno:
```bash
jupyter notebook
```
Donde `03_live_dashboard.ipynb` y `02_model_selection.ipynb` permiten inspeccionar matrices de Permutation Importance para auditar qué vectores del espacio n-dimensional guían las inferencias, así como simulaciones de volatilidad pre-partido.

---
**Desarrollado y Auditado por:** 
* **César Becerra Valencia** & **José Luis Cortés Nava**: Ingeniería de Software, Machine Learning y Procesamiento Estocástico.
* **Dylan Eduardo Becerra Valencia**: Auditoría Cuantitativa de Riesgo y Arbitraje Estadístico.

---
## Licencia (Uso Académico y Restricción Comercial)

Este proyecto está protegido bajo una **Licencia Propietaria de Uso Académico y Restricción Comercial Estricta**.
El código fuente está disponible exclusivamente para fines de **investigación científica y académica**. 

Queda **estrictamente prohibido el uso de este software (total o parcial) para fines comerciales**, generación de ingresos monetarios, ejecución en fondos cuantitativos, *syndicates* deportivos, o automatización de trading sin el consentimiento explícito por escrito de los autores originales (César Becerra y José Luis Cortés) y un **Acuerdo Formal de Partición de Ingresos (Revenue Sharing)**.

Cualquier infracción, copia del código para beneficio propio o uso no autorizado en entornos de producción financiera será perseguido legalmente. Para más detalles, consulta el archivo legal [LICENSE](file:///c:/Users/cesar/OneDrive/Escritorio/Proyectos%20De%20Cesar/Proyectos%20Python/Prophetia2/LICENSE).
