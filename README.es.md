# Predicción de Salario Anual Ofrecido LCA

Proyecto final de Ciencia de Datos enfocado en la construcción de un modelo de Machine Learning capaz de estimar el **salario anual promedio ofrecido** en solicitudes laborales tipo **LCA**.

El proyecto utiliza datos históricos de certificaciones laborales, variables salariales, ubicación del empleo, clasificación ocupacional, sector económico y características administrativas de la solicitud para entrenar un modelo de regresión.

## Objetivo del proyecto

El objetivo principal es predecir la variable:

```text
offered_anual_avg_wage
```

Esta variable representa el salario anual promedio ofrecido. Al tratarse de una variable numérica continua, el problema se aborda como un caso de **regresión supervisada**.

## Dataset final

El dataset final utilizado para el modelo contiene aproximadamente:

```text
979.113 registros
20 columnas totales
19 variables predictoras
1 variable objetivo
```

La estructura general de tipos de datos es:

```text
category: 13 columnas
float64: 4 columnas
int64: 3 columnas
```

La variable objetivo es:

```text
offered_anual_avg_wage
```

Entre las variables predictoras se incluyen características como:

- clase de visa;
- posición full-time;
- estado del empleador;
- código NAICS;
- cantidad de trabajadores en el lugar de trabajo;
- ciudad y estado del lugar de trabajo;
- salario prevaleciente;
- nivel salarial;
- dependencia H-1B;
- historial de infractor intencional;
- grupo SOC de ocupación;
- año de recepción;
- duración del trámite;
- duración del empleo.

## Flujo de trabajo realizado

1. **Definición del problema**  
   Se planteó un problema de regresión para estimar salarios anuales ofrecidos en solicitudes laborales H-1B / LCA.

2. **Carga y filtrado de datos**  
   Se trabajó con un dataset histórico de solicitudes laborales, seleccionando variables relevantes para el modelo final.

3. **Limpieza y transformación**  
   Se normalizaron variables categóricas, se transformaron fechas y se generaron variables derivadas como duración del trámite y duración del empleo.

4. **Normalización de variables binarias**  
   Las columnas `FULL_TIME_POSITION`, `SECONDARY_ENTITY`, `WILLFUL_VIOLATOR` y `H_1B_DEPENDENT` fueron normalizadas a los valores `N` y `Y`.

5. **Entrenamiento del modelo**  
   Se entrenó un modelo de regresión utilizando librerías como `scikit-learn`, `xgboost` y `optuna`.

6. **Evaluación del modelo**  
   Se evaluó el rendimiento usando métricas de regresión como MAE, RMSE y R².

7. **Serialización del modelo**  
   El modelo final fue guardado con `pickle` y comprimido con `gzip` en formato `.pkl.gz`.

8. **Despliegue con Streamlit**  
   Se desarrolló una aplicación web en Streamlit para cargar el modelo y generar predicciones desde una interfaz interactiva.

## Instalación local

Clona el repositorio e instala las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución del notebook

El notebook principal de exploración y modelado se encuentra en:

```text
src/PROYECTO_FINAL_LCA_DISCLOSURE.ipynb
```

Desde ahí se realiza el análisis, preparación de datos, entrenamiento, evaluación y guardado del modelo.

## Ejecución de la aplicación Streamlit

Desde la raíz del proyecto, ejecuta:

```bash
streamlit run app.py
```

Si `app.py` está dentro de `src/`, usa:

```bash
streamlit run src/app.py
```

## Despliegue en Render

Para desplegar en Render se recomienda usar:

**Build Command**

```bash
pip install -r requirements_render.txt
```

**Start Command**

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Si el archivo está dentro de `src/`, el comando debe ser:

```bash
streamlit run src/app.py --server.port=$PORT --server.address=0.0.0.0
```

## Dependencias principales

El proyecto utiliza principalmente:

```text
pandas
numpy
plotly
xgboost
optuna
scikit-learn
streamlit
```

Las librerías `pickle`, `gzip`, `re` y `pathlib` no se agregan a `requirements.txt` porque forman parte de la librería estándar de Python.

## Archivos importantes

- `app.py`: aplicación Streamlit para predicción.
- `src/explore.ipynb`: notebook principal de análisis y modelado.
- `models/model.pkl.gz`: modelo final entrenado y comprimido.
- `data/processed/df_final.csv`: dataset final procesado.
- `data/app_options.json`: archivo opcional para cargar categorías reales en la app.
- `requirements_render.txt`: dependencias necesarias para ejecutar Render.
- `requirements.txt`: dependencias necesarias del proyecto.

## Nota sobre el modelo

Las predicciones entregadas por esta aplicación deben interpretarse como estimaciones aproximadas. El modelo fue entrenado con datos históricos y puede presentar errores ante valores atípicos, combinaciones poco frecuentes o escenarios distintos a los observados durante el entrenamiento.

Este proyecto tiene fines académicos y analíticos. No constituye asesoría financiera, legal, laboral ni migratoria.

## Autores

Proyecto desarrollado por **Oswaldo S.** y **Luis H.** como parte de la clase **4Geeks Academy Latam PT Data Science `latam-pt-ds-21`**.
