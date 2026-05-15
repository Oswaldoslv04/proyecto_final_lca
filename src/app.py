import streamlit as st
import pandas as pd
import pickle
import gzip
import os
import random


# ============================================================
# CONFIGURACIÓN GENERAL DE LA APP
# ============================================================

st.set_page_config(
    page_title="Predicción de Salario Anual H-1B",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# FUNCIÓN PARA CARGAR EL MODELO
# ============================================================

@st.cache_resource
def load_model():
    """
    Carga el modelo entrenado desde la carpeta models.
    El modelo fue guardado en formato .pkl.gz usando pickle + gzip.
    """

    model_path = "models/model.pkl.gz"

    if not os.path.exists(model_path):
        st.error(f"No se encontró el archivo del modelo en la ruta: {model_path}")
        st.stop()

    with gzip.open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


model = load_model()


# ============================================================
# OPCIONES BASADAS EN EL DATASET FINAL
# ============================================================

visa_class_options = [
    "H-1B",
    "E-3 Australian",
    "H-1B1 Chile",
    "H-1B1 Singapore"
]

# Columnas normalizadas previamente en el dataset:
# FULL_TIME_POSITION, SECONDARY_ENTITY, WILLFUL_VIOLATOR, H_1B_DEPENDENT
binary_options = ["N", "Y"]

state_options = [
    "CA", "TX", "NY", "NJ", "IL", "WA", "MA", "VA", "NC", "GA",
    "MI", "FL", "MD", "PA", "OH", "AZ", "MN", "TN", "CT", "MO",
    "CO", "IN", "OR", "WI", "DC", "UT", "SC", "KY", "LA", "AL",
    "IA", "KS", "NE", "OK", "NV", "AR", "DE", "RI", "NH", "ID",
    "NM", "MS", "ME", "HI", "WV", "ND", "SD", "MT", "AK", "WY",
    "VT", "PR", "GU", "VI", "MP", "Other"
]

pw_wage_level_options = ["Junior", "Intermedio", "Avanzado", "Experto"]

pw_oes_period_options = [
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025"
]

worksite_city_options = [
    "New York", "San Francisco", "Seattle", "Chicago", "Austin",
    "Houston", "San Jose", "Sunnyvale", "Mountain View", "Boston",
    "Dallas", "Atlanta", "Irving", "Plano", "Redmond",
    "San Diego", "Los Angeles", "Bellevue", "Santa Clara", "Jersey City",
    "Charlotte", "Phoenix", "Tampa", "Miami", "Philadelphia",
    "Arlington", "Cupertino", "Fremont", "Pittsburgh", "Detroit"
]

naics_code_options = [
    "541511", "541512", "611310", "54151", "541211",
    "622110", "541330", "523110", "45411", "518210",
    "5416", "541519", "54171", "611110", "541611",
    "511210", "3344", "51121", "334111", "454110"
]

soc_code_options = [
    "15-1132.00", "15-1252.00", "OTHER", "15-1133.00",
    "15-1121.00", "15-1299.08", "13-2051.00", "15-1199.02",
    "15-2031.00", "17-2141.00", "13-2011.00", "19-1042.00",
    "13-1111.00", "15-1211.00", "15-1199.01", "15-1199.08",
    "15-2051.00", "17-2072.00", "13-1161.00", "15-2041.00"
]


# ============================================================
# VALORES INICIALES DEL FORMULARIO
# ============================================================

default_values = {
    "visa_class": "H-1B",
    "full_time_position": "Y",
    "new_employment": 1,
    "employer_state": "CA",
    "naics_code": "541511",
    "worksite_workers": 1.0,
    "secondary_entity": "N",
    "worksite_city": "New York",
    "worksite_state": "CA",
    "prevailing_wage": 90000.0,
    "pw_wage_level": "II",
    "total_worksite_locations": 1.0,
    "h1b_dependent": "N",
    "willful_violator": "N",
    "pw_oes_period_group": "2023-2024",
    "soc_code_grouped": "15-1252.00",
    "received_year": 2023,
    "process_duration_days": 7,
    "employment_duration_days": 1095
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# FUNCIÓN PARA RELLENO ALEATORIO
# ============================================================

def fill_random_values():
    """
    Genera valores aleatorios respetando los rangos y categorías observadas
    en el dataset final utilizado para entrenar el modelo.
    """

    st.session_state.visa_class = random.choice(visa_class_options)

    # Variables binarias ya normalizadas a N/Y
    st.session_state.full_time_position = random.choice(binary_options)
    st.session_state.secondary_entity = random.choice(binary_options)
    st.session_state.h1b_dependent = random.choice(binary_options)
    st.session_state.willful_violator = random.choice(binary_options)

    st.session_state.new_employment = random.randint(1, 450)

    st.session_state.employer_state = random.choice(state_options)
    st.session_state.worksite_state = random.choice(state_options)
    st.session_state.worksite_city = random.choice(worksite_city_options)

    st.session_state.naics_code = random.choice(naics_code_options)
    st.session_state.soc_code_grouped = random.choice(soc_code_options)

    st.session_state.worksite_workers = float(random.randint(1, 450))
    st.session_state.total_worksite_locations = float(random.randint(1, 10))

    st.session_state.prevailing_wage = float(random.randint(15080, 300000))
    st.session_state.pw_wage_level = random.choice(pw_wage_level_options)
    st.session_state.pw_oes_period_group = random.choice(pw_oes_period_options)

    st.session_state.received_year = random.randint(2019, 2024)
    st.session_state.process_duration_days = random.randint(4, 152)
    st.session_state.employment_duration_days = random.randint(1, 1096)

# ============================================================
# FUNCIÓN PARA VACIAR / REINICIAR FORMULARIO
# ============================================================

def clear_form_values():
    """
    Reinicia el formulario a los valores por defecto.
    Esto permite limpiar rápidamente los parámetros ingresados o generados.
    """

    for key, value in default_values.items():
        st.session_state[key] = value

# ============================================================
# TÍTULO E INTRODUCCIÓN
# ============================================================

st.title("💼 Predicción de Salario Anual Ofrecido")
st.markdown("""
Esta aplicación utiliza un modelo de regresión entrenado con datos históricos de solicitudes laborales H-1B / LCA.

El objetivo es estimar el **salario anual promedio ofrecido** según las características principales de la solicitud.
""")


# ============================================================
# BOTONES DE RELLENO Y LIMPIEZA
# ============================================================

st.markdown("### 🎲 Rellenado automático")

col_random, col_clear = st.columns([4, 1])

with col_random:
    st.button(
        "Rellenar parámetros aleatoriamente",
        on_click=fill_random_values,
        use_container_width=True
    )

with col_clear:
    st.button(
        "Vaciar",
        on_click=clear_form_values,
        use_container_width=True
    )

st.info(
    "El botón de rellenado genera un caso de prueba usando categorías reales y rangos numéricos "
    "observados en el dataset final. El botón de vaciado reinicia el formulario a sus valores iniciales."
)

st.divider()

# ============================================================
# FORMULARIO DE ENTRADA
# ============================================================

st.markdown("### 📝 Parámetros de entrada")

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    # ------------------------------------------------------------
    # Columna 1: Información general
    # ------------------------------------------------------------
    with col1:
        st.markdown("#### Información general")

        visa_class = st.selectbox(
            "Clase de visa",
            visa_class_options,
            key="visa_class"
        )

        full_time_position = st.selectbox(
            "¿Es una posición de tiempo completo?",
            binary_options,
            key="full_time_position"
        )

        new_employment = st.number_input(
            "Cantidad de nuevos empleos solicitados",
            min_value=1,
            max_value=450,
            step=1,
            key="new_employment"
        )

        secondary_entity = st.selectbox(
            "¿Trabaja en una entidad secundaria?",
            binary_options,
            key="secondary_entity"
        )

        received_year = st.number_input(
            "Año de recepción",
            min_value=2019,
            max_value=2024,
            step=1,
            key="received_year"
        )

        process_duration_days = st.number_input(
            "Duración del trámite en días",
            min_value=4,
            max_value=152,
            step=1,
            key="process_duration_days"
        )

    # ------------------------------------------------------------
    # Columna 2: Ubicación y clasificación
    # ------------------------------------------------------------
    with col2:
        st.markdown("#### Ubicación y clasificación")

        employer_state = st.selectbox(
            "Estado del empleador",
            state_options,
            key="employer_state"
        )

        worksite_state = st.selectbox(
            "Estado del lugar de trabajo",
            state_options,
            key="worksite_state"
        )

        worksite_city = st.selectbox(
            "Ciudad del lugar de trabajo",
            worksite_city_options,
            key="worksite_city"
        )

        naics_code = st.selectbox(
            "Código NAICS",
            naics_code_options,
            key="naics_code"
        )

        soc_code_grouped = st.selectbox(
            "Grupo SOC de ocupación",
            soc_code_options,
            key="soc_code_grouped"
        )

        employment_duration_days = st.number_input(
            "Duración del empleo en días",
            min_value=1,
            max_value=1096,
            step=1,
            key="employment_duration_days"
        )

    # ------------------------------------------------------------
    # Columna 3: Salario, empresa y cumplimiento
    # ------------------------------------------------------------
    with col3:
        st.markdown("#### Salario y cumplimiento")

        prevailing_wage = st.number_input(
            "Salario prevaleciente anual",
            min_value=15080.0,
            max_value=300000.0,
            step=1000.0,
            key="prevailing_wage"
        )

        pw_wage_level = st.selectbox(
            "Nivel salarial PW",
            pw_wage_level_options,
            key="pw_wage_level"
        )

        pw_oes_period_group = st.selectbox(
            "Grupo del período OES",
            pw_oes_period_options,
            key="pw_oes_period_group"
        )

        total_worksite_locations = st.number_input(
            "Total de ubicaciones de trabajo",
            min_value=1.0,
            max_value=10.0,
            step=1.0,
            key="total_worksite_locations"
        )

        worksite_workers = st.number_input(
            "Cantidad de trabajadores en el lugar de trabajo",
            min_value=1.0,
            max_value=450.0,
            step=1.0,
            key="worksite_workers"
        )

        h1b_dependent = st.selectbox(
            "¿El empleador es H-1B dependent?",
            binary_options,
            key="h1b_dependent"
        )

        willful_violator = st.selectbox(
            "¿El empleador registra willful violator?",
            binary_options,
            key="willful_violator"
        )

    submitted = st.form_submit_button(
        "Generar predicción",
        use_container_width=True
    )


# ============================================================
# GENERACIÓN DE LA PREDICCIÓN
# ============================================================

if submitted:

    # Se construye un DataFrame con exactamente las mismas variables predictoras
    # utilizadas durante el entrenamiento del modelo.
    input_data = pd.DataFrame({
        "VISA_CLASS": [visa_class],
        "FULL_TIME_POSITION": [full_time_position],
        "NEW_EMPLOYMENT": [new_employment],
        "EMPLOYER_STATE": [employer_state],
        "NAICS_CODE": [naics_code],
        "WORKSITE_WORKERS": [worksite_workers],
        "SECONDARY_ENTITY": [secondary_entity],
        "WORKSITE_CITY": [worksite_city],
        "WORKSITE_STATE": [worksite_state],
        "PREVAILING_WAGE": [prevailing_wage],
        "PW_WAGE_LEVEL": [pw_wage_level],
        "TOTAL_WORKSITE_LOCATIONS": [total_worksite_locations],
        "H_1B_DEPENDENT": [h1b_dependent],
        "WILLFUL_VIOLATOR": [willful_violator],
        "pw_oes_period_group": [pw_oes_period_group],
        "soc_code_grouped": [soc_code_grouped],
        "received_year": [received_year],
        "process_duration_days": [process_duration_days],
        "employment_duration_days": [employment_duration_days]
    })

    # Columnas categóricas según la estructura final del dataset.
    categorical_cols = [
        "VISA_CLASS",
        "FULL_TIME_POSITION",
        "EMPLOYER_STATE",
        "NAICS_CODE",
        "SECONDARY_ENTITY",
        "WORKSITE_CITY",
        "WORKSITE_STATE",
        "PW_WAGE_LEVEL",
        "H_1B_DEPENDENT",
        "WILLFUL_VIOLATOR",
        "pw_oes_period_group",
        "soc_code_grouped",
        "received_year"
    ]

    # Columnas numéricas según la estructura final del dataset.
    numeric_cols = [
        "NEW_EMPLOYMENT",
        "WORKSITE_WORKERS",
        "PREVAILING_WAGE",
        "TOTAL_WORKSITE_LOCATIONS",
        "process_duration_days",
        "employment_duration_days"
    ]

    # Se respetan los tipos de datos usados en el dataset final.
    for col in categorical_cols:
        input_data[col] = input_data[col].astype("category")

    for col in numeric_cols:
        input_data[col] = pd.to_numeric(input_data[col], errors="coerce")

    try:
        prediction = model.predict(input_data)[0]

        st.success("✅ Predicción generada correctamente")

        st.markdown("### Resultado estimado")

        st.metric(
            label="Salario anual promedio ofrecido estimado",
            value=f"${prediction:,.2f} USD"
        )

        st.markdown("### Datos enviados al modelo")
        st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error("Ocurrió un error al generar la predicción.")
        st.write("Detalle del error:")
        st.code(str(e))


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.divider()

st.markdown("""
<div style='text-align: justify; color: #808080; font-size: 0.85em; padding: 14px; border-radius: 8px; background-color: rgba(128, 128, 128, 0.1);'>

<strong>Nota sobre el alcance del modelo predictivo:</strong><br><br>

Las estimaciones salariales presentadas por esta aplicación son generadas mediante un modelo de Machine Learning entrenado con datos históricos de solicitudes laborales.
Por lo tanto, el resultado debe interpretarse como una aproximación estadística y no como un valor definitivo o garantizado.

<br><br>

El modelo puede presentar márgenes de error, especialmente cuando recibe combinaciones de datos poco frecuentes, valores atípicos o escenarios distintos a los observados durante su entrenamiento.
Factores externos como cambios del mercado laboral, condiciones económicas, políticas migratorias, características específicas de la empresa contratante u otros elementos no incluidos en el dataset pueden influir en el salario real ofrecido.

<br><br>

Esta herramienta fue desarrollada con fines académicos y de apoyo analítico dentro de un proyecto de Ciencia de Datos. No constituye asesoramiento financiero, laboral ni legal.

</div>
""", unsafe_allow_html=True)