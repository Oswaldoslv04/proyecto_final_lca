import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import random
from pathlib import Path


# ============================================================
# CONFIGURACIÓN GENERAL DE LA APP
# ============================================================

st.set_page_config(
    page_title="Predicción de Salario Anual H-1B",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# ESTILOS DE INTERFAZ
# ============================================================

st.markdown(
    """
    <style>
        /* Contenedor general */
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
            max-width: 1280px;
        }

        /* Título principal */
        h1 {
            text-align: center;
            font-size: 2.05rem !important;
            margin-bottom: 0.35rem !important;
        }

        .app-subtitle {
            text-align: center;
            color: #a6a6a6;
            font-size: 0.92rem;
            margin-bottom: 0.9rem;
        }

        /* Títulos de sección centrados */
        .section-title {
            text-align: center;
            font-size: 1.55rem;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 0.8rem;
        }

        /* Compactar formulario */
        div[data-testid="stForm"] {
            max-width: 760px;
            margin: 0 auto;
            padding: 0.85rem 0.85rem 0.7rem 0.85rem;
            border-radius: 10px;
        }

        div[data-testid="stVerticalBlock"] {
            gap: 0.24rem;
        }

        div[data-testid="stNumberInput"],
        div[data-testid="stSelectbox"] {
            margin-bottom: -0.50rem;
        }

        label[data-testid="stWidgetLabel"] {
            font-size: 0.76rem !important;
            font-weight: 600;
            margin-bottom: -0.38rem !important;
        }

        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {
            min-height: 32px !important;
            font-size: 0.86rem !important;
        }

        h3 {
            margin-top: 0.25rem !important;
            margin-bottom: 0.35rem !important;
        }

        h4 {
            font-size: 1rem !important;
            margin-top: 0.10rem !important;
            margin-bottom: 0.35rem !important;
        }

        div[data-testid="stMarkdownContainer"] p {
            margin-bottom: 0.22rem;
        }

        .stButton > button {
            min-height: 34px;
        }

        /* Resultado centrado */
        .result-title {
            text-align: center;
            font-size: 1.55rem;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }

        div[data-testid="stMetric"] {
            text-align: center;
        }

        div[data-testid="stMetric"] label {
            justify-content: center;
        }

        /* Tabla de datos enviados más compacta */
        div[data-testid="stDataFrame"] {
            max-height: 430px;
        }

        .footer-note {
            max-width: 900px;
            margin: 1.3rem auto 0 auto;
            color: #9a9a9a;
            font-size: 0.86rem;
            line-height: 1.5;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FUNCIÓN PARA CARGAR EL PAQUETE DEL MODELO
# ============================================================

@st.cache_resource
def load_model_package():
    """
    Carga el paquete del modelo entrenado desde la carpeta models.
    El archivo .pkl.gz contiene un diccionario con modelo, columnas,
    variable objetivo, tipo de modelo y métricas.
    """

    model_path = Path("models") / "XGB_OPTUNA_50T.pkl.gz"

    if not model_path.exists():
        st.error(f"No se encontró el archivo del modelo en la ruta: {model_path}")
        st.stop()

    with gzip.open(model_path, "rb") as file:
        model_package = pickle.load(file)

    required_keys = [
        "model",
        "feature_names",
        "categorical_cols",
        "target",
        "model_type",
        "metrics"
    ]

    if not isinstance(model_package, dict):
        st.error("El archivo cargado no tiene el formato esperado. Se esperaba un diccionario.")
        st.stop()

    missing_keys = [key for key in required_keys if key not in model_package]

    if missing_keys:
        st.error("El paquete del modelo no contiene todas las claves necesarias.")
        st.write("Claves faltantes:", missing_keys)
        st.stop()

    if not hasattr(model_package["model"], "predict"):
        st.error("El objeto almacenado en la clave 'model' no tiene método predict().")
        st.stop()

    return model_package


model_package = load_model_package()

model = model_package["model"]
feature_names = model_package["feature_names"]
model_categorical_cols = model_package["categorical_cols"]
target = model_package["target"]
model_type = model_package["model_type"]
metrics = model_package["metrics"]


# ============================================================
# OPCIONES BASADAS EN EL DATASET FINAL
# ============================================================

visa_class_options = [
    "H-1B",
    "E-3 Australian",
    "H-1B1 Chile",
    "H-1B1 Singapore"
]

binary_options = ["N", "Y"]

state_options = [
    "CA", "TX", "NY", "NJ", "IL", "WA", "MA", "VA", "NC", "GA",
    "MI", "FL", "MD", "PA", "OH", "AZ", "MN", "TN", "CT", "MO",
    "CO", "IN", "OR", "WI", "DC", "UT", "SC", "KY", "LA", "AL",
    "IA", "KS", "NE", "OK", "NV", "AR", "DE", "RI", "NH", "ID",
    "NM", "MS", "ME", "HI", "WV", "ND", "SD", "MT", "AK", "WY",
    "VT", "PR", "GU", "VI", "MP", "OTHER"
]

pw_wage_level_options = ["Junior", "Intermedio", "Experto"]

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
    "Arlington", "Cupertino", "Fremont", "Pittsburgh", "Detroit",
    "OTHER"
]

naics_code_options = [
    "541511", "541512", "611310", "54151", "541211",
    "622110", "541330", "523110", "45411", "518210",
    "5416", "541519", "54171", "611110", "541611",
    "511210", "3344", "51121", "334111", "454110",
    "OTHER"
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
    "pw_wage_level": "Intermedio",
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

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "last_input_data" not in st.session_state:
    st.session_state.last_input_data = None


# ============================================================
# FUNCIONES DE CONTROL DEL FORMULARIO
# ============================================================

def fill_random_values():
    """
    Genera valores aleatorios respetando rangos y categorías del dataset final.
    """

    st.session_state.visa_class = random.choice(visa_class_options)

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

    st.session_state.prediction_result = None
    st.session_state.last_input_data = None


def reset_prediction():
    """
    Limpia la predicción generada y reinicia el formulario a valores base.
    """

    for key, value in default_values.items():
        st.session_state[key] = value

    st.session_state.prediction_result = None
    st.session_state.last_input_data = None


# ============================================================
# ENCABEZADO
# ============================================================

st.title("Predicción de Salario Anual Ofrecido")

st.markdown(
    """
    <div class="app-subtitle">
        Modelo de regresión para estimar el salario anual promedio ofrecido en solicitudes laborales H-1B / LCA.
    </div>
    """,
    unsafe_allow_html=True
)

_, model_info_col, _ = st.columns([1.2, 1.6, 1.2])
with model_info_col:
    with st.expander("Información del modelo"):
        st.write("Tipo de modelo:", model_type)
        st.write("Variable objetivo:", target)
        st.write("Cantidad de variables predictoras:", len(feature_names))
        st.write("Métricas guardadas:")
        st.json(metrics)


# ============================================================
# PARÁMETROS DE ENTRADA
# ============================================================

st.markdown(
    "<div class='section-title'>Parámetros de entrada</div>",
    unsafe_allow_html=True
)

with st.form("prediction_form"):
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")

    with col1:
        st.markdown("#### Información general")

        visa_class = st.selectbox(
            "Clase de visa",
            visa_class_options,
            key="visa_class"
        )

        full_time_position = st.selectbox(
            "Posición de tiempo completo",
            binary_options,
            key="full_time_position"
        )

        new_employment = st.number_input(
            "Nuevos empleos solicitados",
            min_value=1,
            max_value=450,
            step=1,
            key="new_employment"
        )

        secondary_entity = st.selectbox(
            "Entidad secundaria",
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
            "Trabajadores en el lugar de trabajo",
            min_value=1.0,
            max_value=450.0,
            step=1.0,
            key="worksite_workers"
        )

        h1b_dependent = st.selectbox(
            "Empleador H-1B dependent",
            binary_options,
            key="h1b_dependent"
        )

        willful_violator = st.selectbox(
            "Willful violator",
            binary_options,
            key="willful_violator"
        )

    st.markdown("---")

    _, random_col, _ = st.columns([1.3, 1, 1.3])
    with random_col:
        random_clicked = st.form_submit_button(
            "Rellenar parámetros aleatoriamente",
            on_click=fill_random_values,
            use_container_width=True
        )

    submitted = st.form_submit_button(
        "Generar predicción",
        use_container_width=True
    )



# ============================================================
# GENERACIÓN DE LA PREDICCIÓN
# ============================================================

if submitted:
    input_data = pd.DataFrame({
        "PREVAILING_WAGE": [prevailing_wage],
        "PW_WAGE_LEVEL": [pw_wage_level],
        "EMPLOYER_STATE": [employer_state],
        "WORKSITE_STATE": [worksite_state],
        "WORKSITE_CITY": [worksite_city],
        "VISA_CLASS": [visa_class],
        "NAICS_CODE": [naics_code],
        "FULL_TIME_POSITION": [full_time_position],
        "NEW_EMPLOYMENT": [new_employment],
        "TOTAL_WORKSITE_LOCATIONS": [total_worksite_locations],
        "WORKSITE_WORKERS": [worksite_workers],
        "SECONDARY_ENTITY": [secondary_entity],
        "H_1B_DEPENDENT": [h1b_dependent],
        "WILLFUL_VIOLATOR": [willful_violator],
        "pw_oes_period_group": [pw_oes_period_group],
        "soc_code_grouped": [soc_code_grouped],
        "received_year": [received_year],
        "process_duration_days": [process_duration_days],
        "employment_duration_days": [employment_duration_days]
    })

    numeric_cols = [
        "PREVAILING_WAGE",
        "NEW_EMPLOYMENT",
        "TOTAL_WORKSITE_LOCATIONS",
        "WORKSITE_WORKERS",
        "process_duration_days",
        "employment_duration_days"
    ]

    try:
        for col in model_categorical_cols:
            if col in input_data.columns:
                input_data[col] = input_data[col].astype("category")

        for col in numeric_cols:
            if col in input_data.columns:
                input_data[col] = pd.to_numeric(input_data[col], errors="coerce")

        input_data = input_data[feature_names]

        # El modelo predice el salario en escala logarítmica.
        # np.expm1 revierte la transformación log1p aplicada durante el entrenamiento.
        prediction_log = model.predict(input_data)[0]
        prediction = np.expm1(prediction_log)

        st.session_state.prediction_result = float(prediction)
        st.session_state.last_input_data = input_data.copy()

    except Exception as e:
        st.error("Ocurrió un error al generar la predicción.")
        st.write("Detalle del error:")
        st.code(str(e))


# ============================================================
# RESULTADOS
# ============================================================

if st.session_state.prediction_result is not None:
    st.success("Predicción generada correctamente")

    _, result_col, _ = st.columns([1.25, 1.5, 1.25])
    with result_col:
        st.markdown("<div class='result-title'>Resultado estimado</div>", unsafe_allow_html=True)

        st.metric(
            label="Salario anual promedio ofrecido estimado",
            value=f"${st.session_state.prediction_result:,.2f} USD"
        )

    input_display = (
        st.session_state.last_input_data
        .iloc[0]
        .astype(str)
        .reset_index()
    )
    input_display.columns = ["Variable", "Valor"]

    _, data_col, _ = st.columns([1.2, 1.6, 1.2])
    with data_col:
        st.markdown("<div class='result-title'>Datos enviados al modelo</div>", unsafe_allow_html=True)

        st.dataframe(
            input_display,
            use_container_width=True,
            hide_index=True,
            height=430
        )

        _, reset_col, _ = st.columns([1, 1.1, 1])
        with reset_col:
            st.button(
                "Generar nueva predicción",
                on_click=reset_prediction,
                use_container_width=True
            )


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer-note">
        Este modelo entrega una estimación basada en datos históricos de solicitudes laborales H-1B/LCA. El resultado no debe interpretarse como un valor definitivo, ya que puede variar por factores externos, condiciones del mercado laboral o características específicas no incluidas en el entrenamiento. Esta aplicación fue desarrollada con fines académicos y de apoyo analítico.
    </div>
    """,
    unsafe_allow_html=True
)