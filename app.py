import streamlit as st 
from image_processor import image_processor
import csv
from io import StringIO


def normalize_csv_text(raw_output: str) -> str:
    """Remove markdown fences and keep only CSV lines."""
    if not raw_output:
        return ""

    lines = []
    for line in raw_output.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        if not stripped:
            continue
        lines.append(line)

    return "\n".join(lines).strip()


def csv_to_table_rows(csv_text: str):
    """Parse CSV text into header + normalized data rows."""
    reader = csv.reader(StringIO(csv_text), skipinitialspace=True)
    parsed_rows = [row for row in reader if any(cell.strip() for cell in row)]

    if not parsed_rows:
        return [], []

    headers = [h.strip() for h in parsed_rows[0]]
    data_rows = parsed_rows[1:]

    normalized_rows = []
    total_cols = len(headers)
    for row in data_rows:
        current = [cell.strip() for cell in row]
        if len(current) < total_cols:
            current += [""] * (total_cols - len(current))
        elif len(current) > total_cols:
            current = current[:total_cols]
        normalized_rows.append(current)

    return headers, normalized_rows

st.set_page_config(
    page_title="Aplanador CSV",
    page_icon=":bar_chart:",
    layout="wide"
)


st.title("Herramienta de Aplanamiento de Tablas en Imágenes a CSV")
st.write("Convierte imágenes de tablas en muestras CSV aplanadas")

model_provider = st.radio("Seleccionar proveedor", options=["GPT", "Gemini"])

table_image = st.file_uploader("Cargar imagen de tabla", type=["jpg", "jpeg", "png"])

if table_image is not None: 
    
    st.image(table_image, caption="Tabla cargada")
    
    output = image_processor(table_image.getvalue())
    
    st.subheader("Tabla Aplanada")

    cleaned_csv = normalize_csv_text(output)
    headers, rows = csv_to_table_rows(cleaned_csv)

    if headers and rows:
        table_data = [dict(zip(headers, row)) for row in rows]
        st.dataframe(table_data, hide_index=True, use_container_width=True)
    elif headers:
        st.info("No hay filas de datos para mostrar.")
    else:
        st.error("No se pudo interpretar la salida como CSV válido.")
        st.code(output, language="text")
