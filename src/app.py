from pathlib import Path
import pickle
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Predicción de salario H-1B / LCA",
    page_icon="💼",
    layout="centered"
)

TARGET = "offered_anual_avg_wage"

FEATURE_COLUMNS = [
    "RECEIVED_DATE",
    "VISA_CLASS",
    "FULL_TIME_POSITION",
    "BEGIN_DATE",
    "END_DATE",
    "NEW_EMPLOYMENT",
    "EMPLOYER_STATE",
    "NAICS_CODE",
    "WORKSITE_WORKERS",
    "SECONDARY_ENTITY",
    "WORKSITE_CITY",
    "WORKSITE_COUNTY",
    "WORKSITE_STATE",
    "PREVAILING_WAGE",
    "PW_WAGE_LEVEL",
    "TOTAL_WORKSITE_LOCATIONS",
    "H_1B_DEPENDENT",
    "WILLFUL_VIOLATOR",
    "pw_oes_period_group",
    "soc_code_grouped",
]

MODEL_DIR = Path("models")
MODEL_EXTENSION = "*.pkl"

# Rangos calculados desde df_final.csv
DATE_RANGES = {
    "RECEIVED_DATE": {"min": date(2019, 9, 25), "median": date(2022, 5, 16), "max": date(2024, 9, 23)},
    "BEGIN_DATE": {"min": date(2019, 9, 25), "median": date(2022, 10, 1), "max": date(2025, 3, 22)},
    "END_DATE": {"min": date(2019, 10, 30), "median": date(2025, 9, 30), "max": date(2028, 3, 21)},
}

NUMERIC_RANGES = {
    "NEW_EMPLOYMENT": {"min": 1, "q1": 1, "median": 1, "q3": 1, "max": 450},
    "NAICS_CODE": {"min": -271021, "q1": 336110, "median": 541511, "q3": 541512, "max": 928120},
    "WORKSITE_WORKERS": {"min": 1.0, "q1": 1.0, "median": 1.0, "q3": 1.0, "max": 450.0},
    "PREVAILING_WAGE": {"min": 15080.0, "q1": 67708.0, "median": 83658.0, "q3": 102669.0, "max": 300000.0},
    "TOTAL_WORKSITE_LOCATIONS": {"min": 1.0, "q1": 1.0, "median": 1.0, "q3": 2.0, "max": 10.0},
}

VISA_CLASSES = ["H-1B", "E-3 Australian", "H-1B1 Chile", "H-1B1 Singapore"]
YES_NO_VALUES = ["No", "Yes", "N", "Y"]
FULL_TIME_VALUES = ["Y", "N"]
PW_WAGE_LEVELS = ["I", "II", "III", "IV", "V"]
PW_OES_PERIODS = ["2024-2025", "2023-2024", "2022-2023", "2021-2022", "2020-2021", "2019-2020", "2018-2019"]

STATES = [
    "CA", "TX", "NJ", "NY", "IL", "WA", "MA", "VA", "NC", "GA", "MI", "FL", "MD", "PA", "OH",
    "AZ", "CO", "MN", "CT", "MO", "TN", "DC", "IN", "WI", "UT", "OR", "DE", "IA", "KS", "SC",
    "KY", "LA", "AL", "AR", "OK", "RI", "NE", "NV", "NH", "NM", "ID", "MS", "HI", "ME", "MT", "ND",
    "SD", "VT", "WV", "WY", "AK", "PR", "GU", "VI", "MP"
]

COMMON_WORKSITE_CITIES = [
    "New York", "San Francisco", "Seattle", "Chicago", "Austin", "Houston", "San Jose", "Sunnyvale",
    "Mountain View", "Boston", "Dallas", "Atlanta", "Irving", "Plano", "Redmond", "San Diego",
    "Los Angeles", "Bellevue", "Santa Clara", "Charlotte", "Alpharetta", "Cambridge", "Pittsburgh",
    "Philadelphia", "Phoenix", "Jersey City", "Hillsboro", "Columbus", "Washington"
]

COMMON_WORKSITE_COUNTIES = [
    "NEW YORK", "SANTA CLARA", "KING", "DALLAS", "COOK", "LOS ANGELES", "SAN FRANCISCO", "COLLIN",
    "FULTON", "HARRIS", "TRAVIS", "BOSTON CITY", "MIDDLESEX", "SAN MATEO", "MARICOPA",
    "ALAMEDA", "ORANGE", "SAN DIEGO", "MONTGOMERY", "FAIRFAX", "HUDSON", "OAKLAND", "WAKE",
    "WASHINGTON", "MECKLENBURG", "ALLEGHENY", "HENNEPIN", "WAYNE", "MERCER", "HILLSBOROUGH"
]

COMMON_SOC_CODES = [
    "15-1132.00", "15-1252.00", "OTHER", "15-1133.00", "15-1121.00", "15-1299.08", "13-2051.00",
    "15-1199.02", "15-2031.00", "17-2141.00", "13-2011.00", "19-1042.00", "13-1111.00",
    "15-1211.00", "15-1199.01", "15-1199.08", "15-2051.00", "17-2072.00", "13-1161.00",
    "15-2041.00", "17-2071.00", "15-1251.00", "15-1253.00", "15-1131.00", "15-1299.09",
    "15-1199.09", "19-1021.00", "17-2051.00", "17-2112.00", "15-2051.01"
]

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
@st.cache_resource
def load_model(model_path: str):
    """Carga un modelo .pkl desde la carpeta models usando pickle."""
    with open(model_path, "rb") as file:
        return pickle.load(file)


def find_pickle_models() -> list[str]:
    """Busca únicamente modelos con extensión .pkl dentro de la carpeta models."""
    if not MODEL_DIR.exists():
        return []
    return sorted(str(path) for path in MODEL_DIR.glob(MODEL_EXTENSION))


def format_currency(value: float) -> str:
    return f"${value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def random_date_between(start: date, end: date) -> date:
    days = (end - start).days
    return start + timedelta(days=random.randint(0, days))


def random_numeric_from_range(column: str):
    """Genera valores aleatorios realistas usando el rango intercuartílico del dataset."""
    info = NUMERIC_RANGES[column]

    if column in ["NEW_EMPLOYMENT", "WORKSITE_WORKERS"]:
        # En el dataset la gran mayoría de registros tiene valor 1; ocasionalmente generamos valores mayores.
        return random.choice([1, 1, 1, 1, random.randint(2, 10), random.randint(11, int(info["max"]))])

    if column == "TOTAL_WORKSITE_LOCATIONS":
        return random.choice([1, 1, 1, 2, random.randint(3, int(info["max"]))])

    if column == "NAICS_CODE":
        return random.choice([336110, 541511, 541512, 541519, 611310, 518210, random.randint(100000, 928120)])

    if column == "PREVAILING_WAGE":
        return round(random.uniform(info["q1"], info["q3"]), 2)

    return random.uniform(info["min"], info["max"])


def default_values() -> dict:
    return {
        "received_date": DATE_RANGES["RECEIVED_DATE"]["median"],
        "begin_date": DATE_RANGES["BEGIN_DATE"]["median"],
        "end_date": DATE_RANGES["END_DATE"]["median"],
        "visa_class": "H-1B",
        "full_time_position": "Y",
        "new_employment": int(NUMERIC_RANGES["NEW_EMPLOYMENT"]["median"]),
        "secondary_entity": "No",
        "h1b_dependent": "No",
        "willful_violator": "No",
        "employer_state": "CA",
        "worksite_state": "CA",
        "worksite_city": "New York",
        "worksite_county": "NEW YORK",
        "naics_code": int(NUMERIC_RANGES["NAICS_CODE"]["median"]),
        "soc_code_grouped": "15-1132.00",
        "pw_oes_period_group": "2023-2024",
        "pw_wage_level": "II",
        "prevailing_wage": float(NUMERIC_RANGES["PREVAILING_WAGE"]["median"]),
        "worksite_workers": float(NUMERIC_RANGES["WORKSITE_WORKERS"]["median"]),
        "total_worksite_locations": float(NUMERIC_RANGES["TOTAL_WORKSITE_LOCATIONS"]["median"]),
    }


def random_values() -> dict:
    received_date = random_date_between(DATE_RANGES["RECEIVED_DATE"]["min"], DATE_RANGES["RECEIVED_DATE"]["max"])
    begin_date = random_date_between(max(received_date, DATE_RANGES["BEGIN_DATE"]["min"]), DATE_RANGES["BEGIN_DATE"]["max"])
    end_date = random_date_between(max(begin_date + timedelta(days=30), DATE_RANGES["END_DATE"]["min"]), DATE_RANGES["END_DATE"]["max"])

    return {
        "received_date": received_date,
        "begin_date": begin_date,
        "end_date": end_date,
        "visa_class": random.choices(VISA_CLASSES, weights=[95, 2, 2, 1], k=1)[0],
        "full_time_position": random.choices(FULL_TIME_VALUES, weights=[98, 2], k=1)[0],
        "new_employment": int(random_numeric_from_range("NEW_EMPLOYMENT")),
        "secondary_entity": random.choices(YES_NO_VALUES, weights=[65, 17, 13, 5], k=1)[0],
        "h1b_dependent": random.choices(YES_NO_VALUES, weights=[60, 20, 15, 5], k=1)[0],
        "willful_violator": random.choices(YES_NO_VALUES, weights=[80, 1, 19, 1], k=1)[0],
        "employer_state": random.choice(STATES[:20]),
        "worksite_state": random.choice(STATES[:20]),
        "worksite_city": random.choice(COMMON_WORKSITE_CITIES),
        "worksite_county": random.choice(COMMON_WORKSITE_COUNTIES),
        "naics_code": int(random_numeric_from_range("NAICS_CODE")),
        "soc_code_grouped": random.choice(COMMON_SOC_CODES),
        "pw_oes_period_group": random.choices(PW_OES_PERIODS, weights=[2, 28, 17, 21, 21, 11, 1], k=1)[0],
        "pw_wage_level": random.choices(PW_WAGE_LEVELS, weights=[29, 49, 14, 8, 1], k=1)[0],
        "prevailing_wage": float(random_numeric_from_range("PREVAILING_WAGE")),
        "worksite_workers": float(random_numeric_from_range("WORKSITE_WORKERS")),
        "total_worksite_locations": float(random_numeric_from_range("TOTAL_WORKSITE_LOCATIONS")),
    }


def initialize_session_state():
    for key, value in default_values().items():
        st.session_state.setdefault(key, value)


def set_random_session_values():
    for key, value in random_values().items():
        st.session_state[key] = value


def build_input_dataframe() -> pd.DataFrame:
    input_data = pd.DataFrame([
        {
            "RECEIVED_DATE": str(st.session_state.received_date),
            "VISA_CLASS": st.session_state.visa_class,
            "FULL_TIME_POSITION": st.session_state.full_time_position,
            "BEGIN_DATE": str(st.session_state.begin_date),
            "END_DATE": str(st.session_state.end_date),
            "NEW_EMPLOYMENT": int(st.session_state.new_employment),
            "EMPLOYER_STATE": st.session_state.employer_state,
            "NAICS_CODE": int(st.session_state.naics_code),
            "WORKSITE_WORKERS": float(st.session_state.worksite_workers),
            "SECONDARY_ENTITY": st.session_state.secondary_entity,
            "WORKSITE_CITY": st.session_state.worksite_city.strip(),
            "WORKSITE_COUNTY": st.session_state.worksite_county.strip(),
            "WORKSITE_STATE": st.session_state.worksite_state,
            "PREVAILING_WAGE": float(st.session_state.prevailing_wage),
            "PW_WAGE_LEVEL": st.session_state.pw_wage_level,
            "TOTAL_WORKSITE_LOCATIONS": float(st.session_state.total_worksite_locations),
            "H_1B_DEPENDENT": st.session_state.h1b_dependent,
            "WILLFUL_VIOLATOR": st.session_state.willful_violator,
            "pw_oes_period_group": st.session_state.pw_oes_period_group,
            "soc_code_grouped": st.session_state.soc_code_grouped.strip(),
        }
    ])
    return input_data[FEATURE_COLUMNS]

# ============================================================
# INTERFAZ
# ============================================================
initialize_session_state()

st.title("💼 Predicción de salario ofrecido")
st.caption("Modelo de Machine Learning para estimar el salario anual promedio ofrecido en solicitudes LCA / H-1B.")

models_found = find_pickle_models()

if not models_found:
    st.error(
        "No se encontró ningún modelo `.pkl` dentro de la carpeta `models/`. "
        "Guarda el Pipeline o modelo final en esa carpeta con extensión `.pkl`."
    )
    st.stop()

selected_model_path = st.sidebar.selectbox(
    "Modelo a utilizar",
    models_found,
    format_func=lambda path: Path(path).name
)

model = load_model(selected_model_path)
st.sidebar.success(f"Modelo cargado: {Path(selected_model_path).name}")

st.info(
    "Los rangos del formulario fueron configurados con base en `df_final.csv`. "
    "Para el botón aleatorio se priorizan valores frecuentes y rangos realistas del dataset."
)

if st.button("🎲 Rellenar formulario aleatoriamente"):
    set_random_session_values()
    st.rerun()

with st.form("prediction_form"):
    st.subheader("📌 Datos de la solicitud")

    col1, col2 = st.columns(2)

    with col1:
        st.date_input(
            "Fecha de recepción",
            min_value=DATE_RANGES["RECEIVED_DATE"]["min"],
            max_value=DATE_RANGES["RECEIVED_DATE"]["max"],
            key="received_date"
        )
        st.date_input(
            "Fecha de inicio",
            min_value=DATE_RANGES["BEGIN_DATE"]["min"],
            max_value=DATE_RANGES["BEGIN_DATE"]["max"],
            key="begin_date"
        )
        st.date_input(
            "Fecha de término",
            min_value=DATE_RANGES["END_DATE"]["min"],
            max_value=DATE_RANGES["END_DATE"]["max"],
            key="end_date"
        )
        st.selectbox("Tipo de visa", VISA_CLASSES, key="visa_class")
        st.selectbox("Posición full time", FULL_TIME_VALUES, key="full_time_position")
        st.number_input(
            "Nuevos empleos",
            min_value=int(NUMERIC_RANGES["NEW_EMPLOYMENT"]["min"]),
            max_value=int(NUMERIC_RANGES["NEW_EMPLOYMENT"]["max"]),
            step=1,
            key="new_employment"
        )
        st.selectbox("Entidad secundaria", YES_NO_VALUES, key="secondary_entity")
        st.selectbox("H-1B dependent", YES_NO_VALUES, key="h1b_dependent")
        st.selectbox("Willful violator", YES_NO_VALUES, key="willful_violator")

    with col2:
        st.selectbox("Estado del empleador", STATES, key="employer_state")
        st.selectbox("Estado del lugar de trabajo", STATES, key="worksite_state")
        st.selectbox("Ciudad del lugar de trabajo", COMMON_WORKSITE_CITIES, key="worksite_city")
        st.selectbox("Condado del lugar de trabajo", COMMON_WORKSITE_COUNTIES, key="worksite_county")
        st.number_input(
            "NAICS code",
            min_value=int(NUMERIC_RANGES["NAICS_CODE"]["min"]),
            max_value=int(NUMERIC_RANGES["NAICS_CODE"]["max"]),
            step=1,
            key="naics_code",
            help="El mínimo negativo aparece porque ese valor existe en el dataset final procesado."
        )
        st.selectbox("SOC code agrupado", COMMON_SOC_CODES, key="soc_code_grouped")
        st.selectbox("Periodo OES / salario prevaleciente", PW_OES_PERIODS, key="pw_oes_period_group")
        st.selectbox("Nivel salarial prevaleciente", PW_WAGE_LEVELS, key="pw_wage_level")

    st.subheader("💰 Variables salariales y operativas")

    col3, col4 = st.columns(2)

    with col3:
        st.number_input(
            "Prevailing wage anual",
            min_value=float(NUMERIC_RANGES["PREVAILING_WAGE"]["min"]),
            max_value=float(NUMERIC_RANGES["PREVAILING_WAGE"]["max"]),
            step=1000.0,
            key="prevailing_wage"
        )
        st.number_input(
            "Trabajadores en el worksite",
            min_value=float(NUMERIC_RANGES["WORKSITE_WORKERS"]["min"]),
            max_value=float(NUMERIC_RANGES["WORKSITE_WORKERS"]["max"]),
            step=1.0,
            key="worksite_workers"
        )

    with col4:
        st.number_input(
            "Total worksite locations",
            min_value=float(NUMERIC_RANGES["TOTAL_WORKSITE_LOCATIONS"]["min"]),
            max_value=float(NUMERIC_RANGES["TOTAL_WORKSITE_LOCATIONS"]["max"]),
            step=1.0,
            key="total_worksite_locations"
        )

    submitted = st.form_submit_button("🔮 Predecir salario")

if submitted:
    input_data = build_input_dataframe()

    st.markdown("### 🧾 Registro enviado al modelo")
    st.dataframe(input_data, use_container_width=True)

    try:
        prediction = model.predict(input_data)
        predicted_wage = float(np.ravel(prediction)[0])

        st.success("Predicción generada correctamente ✅")
        st.metric(
            label="Salario anual promedio ofrecido estimado",
            value=format_currency(predicted_wage)
        )

    except Exception as error:
        st.error("No se pudo generar la predicción con las columnas actuales.")
        st.exception(error)

        expected_features = getattr(model, "feature_names_in_", None)
        if expected_features is not None:
            st.warning("El modelo parece esperar estas columnas:")
            st.write(list(expected_features))

        st.info(
            "Si el modelo fue entrenado con variables transformadas manualmente, guarda mejor un Pipeline completo "
            "que incluya imputación, codificación de variables categóricas y el modelo final."
        )

st.divider()
st.caption(
    "Aviso: esta predicción corresponde a una estimación generada por un modelo estadístico y puede presentar margen de error; "
    "no debe interpretarse como un valor definitivo."
)
