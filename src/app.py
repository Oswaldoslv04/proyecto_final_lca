import glob
from pathlib import Path

import joblib
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
MODEL_EXTENSIONS = ["*.pkl", "*.joblib", "*.sav"]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
@st.cache_resource
def load_model(model_path: str):
    """Carga el modelo entrenado desde la carpeta models."""
    return joblib.load(model_path)


def find_models() -> list[str]:
    """Busca modelos compatibles dentro de la carpeta models."""
    model_paths = []
    for extension in MODEL_EXTENSIONS:
        model_paths.extend(glob.glob(str(MODEL_DIR / extension)))
    return sorted(model_paths)


def normalize_yes_no(value: str) -> str:
    """Mantiene valores en formato parecido al dataset final."""
    if value in ["Yes", "Y"]:
        return value
    if value in ["No", "N"]:
        return value
    return value


def format_currency(value: float) -> str:
    return f"${value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ============================================================
# INTERFAZ
# ============================================================
st.title("💼 Predicción de salario ofrecido")
st.caption("Modelo de Machine Learning para estimar el salario anual promedio ofrecido en solicitudes LCA / H-1B.")

models_found = find_models()

if not models_found:
    st.error(
        "No se encontró ningún modelo dentro de la carpeta `models/`. "
        "Sube tu archivo `.pkl`, `.joblib` o `.sav` a esa carpeta."
    )
    st.stop()

selected_model_path = st.sidebar.selectbox(
    "Modelo a utilizar",
    models_found,
    format_func=lambda path: Path(path).name
)

model = load_model(selected_model_path)

st.sidebar.success(f"Modelo cargado: {Path(selected_model_path).name}")

with st.form("prediction_form"):
    st.subheader("📌 Datos de la solicitud")

    col1, col2 = st.columns(2)

    with col1:
        received_date = st.date_input("Fecha de recepción", value=pd.to_datetime("2024-05-01"))
        begin_date = st.date_input("Fecha de inicio", value=pd.to_datetime("2024-10-01"))
        end_date = st.date_input("Fecha de término", value=pd.to_datetime("2027-09-30"))
        visa_class = st.selectbox(
            "Tipo de visa",
            ["H-1B", "E-3 Australian", "H-1B1 Chile", "H-1B1 Singapore"]
        )
        full_time_position = st.selectbox("Posición full time", ["Y", "N"])
        new_employment = st.number_input("Nuevos empleos", min_value=1, max_value=450, value=1, step=1)
        secondary_entity = st.selectbox("Entidad secundaria", ["No", "Yes", "N", "Y"])
        h1b_dependent = st.selectbox("H-1B dependent", ["No", "Yes", "N", "Y"])
        willful_violator = st.selectbox("Willful violator", ["No", "Yes", "N", "Y"])

    with col2:
        employer_state = st.selectbox(
            "Estado del empleador",
            [
                "CA", "TX", "NJ", "NY", "IL", "WA", "MA", "VA", "NC", "GA", "MI", "FL", "MD", "PA", "OH",
                "AZ", "CO", "MN", "CT", "MO", "TN", "DC", "IN", "WI", "UT", "OR", "DE", "IA", "KS", "SC",
                "KY", "LA", "AL", "AR", "OK", "RI", "NE", "NV", "NH", "NM", "ID", "MS", "HI", "ME", "MT", "ND",
                "SD", "VT", "WV", "WY", "AK", "PR", "GU", "VI", "MP"
            ]
        )
        worksite_state = st.selectbox(
            "Estado del lugar de trabajo",
            [
                "CA", "TX", "NY", "NJ", "WA", "IL", "MA", "GA", "FL", "NC", "PA", "VA", "MI", "OH", "AZ",
                "MD", "CO", "MN", "CT", "MO", "TN", "IN", "DC", "WI", "UT", "OR", "IA", "DE", "KS", "SC",
                "KY", "LA", "AL", "AR", "OK", "RI", "NE", "NV", "NH", "NM", "ID", "MS", "HI", "ME", "MT", "ND",
                "SD", "VT", "WV", "WY", "AK", "PR", "GU", "VI", "MP"
            ]
        )
        worksite_city = st.text_input("Ciudad del lugar de trabajo", value="New York")
        worksite_county = st.text_input("Condado del lugar de trabajo", value="NEW YORK")
        naics_code = st.number_input("NAICS code", min_value=-999999, max_value=999999, value=541511, step=1)
        soc_code_grouped = st.text_input("SOC code agrupado", value="15-1132.00")
        pw_oes_period_group = st.selectbox(
            "Periodo OES / salario prevaleciente",
            ["2024-2025", "2023-2024", "2022-2023", "2021-2022", "2020-2021", "2019-2020", "2018-2019"]
        )
        pw_wage_level = st.selectbox("Nivel salarial prevaleciente", ["I", "II", "III", "IV", "V"])

    st.subheader("💰 Variables salariales y operativas")

    col3, col4 = st.columns(2)

    with col3:
        prevailing_wage = st.number_input(
            "Prevailing wage anual",
            min_value=15080.0,
            max_value=300000.0,
            value=83658.0,
            step=1000.0
        )
        worksite_workers = st.number_input(
            "Trabajadores en el worksite",
            min_value=1.0,
            max_value=450.0,
            value=1.0,
            step=1.0
        )

    with col4:
        total_worksite_locations = st.number_input(
            "Total worksite locations",
            min_value=1.0,
            max_value=10.0,
            value=1.0,
            step=1.0
        )

    submitted = st.form_submit_button("🔮 Predecir salario")

if submitted:
    input_data = pd.DataFrame([
        {
            "RECEIVED_DATE": str(received_date),
            "VISA_CLASS": visa_class,
            "FULL_TIME_POSITION": full_time_position,
            "BEGIN_DATE": str(begin_date),
            "END_DATE": str(end_date),
            "NEW_EMPLOYMENT": int(new_employment),
            "EMPLOYER_STATE": employer_state,
            "NAICS_CODE": int(naics_code),
            "WORKSITE_WORKERS": float(worksite_workers),
            "SECONDARY_ENTITY": normalize_yes_no(secondary_entity),
            "WORKSITE_CITY": worksite_city.strip(),
            "WORKSITE_COUNTY": worksite_county.strip(),
            "WORKSITE_STATE": worksite_state,
            "PREVAILING_WAGE": float(prevailing_wage),
            "PW_WAGE_LEVEL": pw_wage_level,
            "TOTAL_WORKSITE_LOCATIONS": float(total_worksite_locations),
            "H_1B_DEPENDENT": normalize_yes_no(h1b_dependent),
            "WILLFUL_VIOLATOR": normalize_yes_no(willful_violator),
            "pw_oes_period_group": pw_oes_period_group,
            "soc_code_grouped": soc_code_grouped.strip(),
        }
    ])

    input_data = input_data[FEATURE_COLUMNS]

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
            "Si entrenaste el modelo con variables ya codificadas manualmente, "
            "lo ideal es guardar y cargar un Pipeline completo que incluya el preprocesamiento "
            "y el modelo final."
        )

st.divider()
st.caption(
    "Nota: esta app espera que el modelo guardado acepte las columnas originales del dataset final. "
    "Para producción, se recomienda guardar un Pipeline de scikit-learn con imputación, encoding y modelo."
)