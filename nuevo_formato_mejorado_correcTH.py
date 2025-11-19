import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, time, date, timedelta

# -------------------------------------------------
# CONFIG GENERAL
# -------------------------------------------------
st.set_page_config(
    page_title="Análisis de Velocidad - IMU Neumáticos",
    layout="wide"
)

st.title("📊 Análisis de Velocidad de Neumático (IMU)")

# -------------------------------------------------
# FUNCIONES AUXILIARES
# -------------------------------------------------
def _reset(fp):
    try:
        fp.seek(0)
    except Exception:
        pass

def _read_csv_any(path_or_buf, sep, encoding):
    # Soporta ruta (str/Path) o buffer (UploadedFile de Streamlit)
    if isinstance(path_or_buf, (str, os.PathLike)):
        return pd.read_csv(path_or_buf, sep=sep, engine="python", encoding=encoding)
    _reset(path_or_buf)
    return pd.read_csv(path_or_buf, sep=sep, engine="python", encoding=encoding)

def leer_archivo(path_or_buf):
    """
    Lector robusto:
      1) Prueba sep="\\t" (TSV) y luego autodetección (sep=None).
      2) Rota encodings: utf-8, utf-8-sig, cp1252, latin-1, utf-16(le/be).
      3) Último recurso: lee bytes -> decodifica cp1252 con 'replace' -> parsea.
    Devuelve un DataFrame con encabezados.
    """
    seps = ["\t", None]  # None = autodetecta separador
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "utf-16", "utf-16le", "utf-16be"]

    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = _read_csv_any(path_or_buf, sep=sep, encoding=enc)
                if df.shape[1] > 1:
                    return df
            except Exception as e:
                last_err = e
                continue

    # Fallback: bytes -> texto cp1252 (replace) -> autodetectar separador
    try:
        if isinstance(path_or_buf, (str, os.PathLike)):
            with open(path_or_buf, "rb") as f:
                raw = f.read()
        else:
            _reset(path_or_buf)
            raw = path_or_buf.read()

        if isinstance(raw, (bytes, bytearray)):
            text = raw.decode("cp1252", errors="replace")
        else:
            text = str(raw)

        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        if df.shape[1] > 1:
            return df
    except Exception as e:
        last_err = e

    raise last_err if last_err else UnicodeDecodeError("fallback", b"", 0, 1, "unknown encoding")

def reparar_ms_con_dos_puntos(serie: pd.Series) -> pd.Series:
    # Reemplaza el último ":" por "." para interpretar milisegundos
    return (
        serie.astype(str)
             .str.replace(r"(?<=\d):(\d{1,3})$", r".\1", regex=True)
             .str.strip()
    )

def encontrar_columna(df, candidatos):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidatos:
        key = cand.lower()
        if key in cols_lower:
            return cols_lower[key]
    # Búsqueda flexible por contiene (case-insensitive)
    for c in df.columns:
        cl = c.lower()
        if "angular" in cl and "x" in cl and "°/s" in cl:
            return c
        if cl in ["asx(°/s)", "gyrx(°/s)", "gyrox(°/s)", "wx(°/s)", "gx(°/s)"]:
            return c
    return None

def parse_hora(hora_str: str):
    try:
        h, m, s = hora_str.strip().split(":")
        return time(hour=int(h), minute=int(m), second=int(s))
    except Exception:
        return None

# -------------------------------------------------
# SIDEBAR: PARÁMETROS DE ENTRADA
# -------------------------------------------------
st.sidebar.header("⚙️ Parámetros de Entrada")

archivo = st.sidebar.file_uploader(
    "Sube tu archivo IMU (TXT/CSV con encabezados)",
    type=["txt", "csv"]
)

col_r, col_c = st.sidebar.columns(2)
radio_input = col_r.number_input(
    "Radio (m)",
    min_value=0.0,
    step=0.001,
    value=0.0
)
circ_input = col_c.number_input(
    "Circunferencia (mm)",
    min_value=0.0,
    step=1.0,
    value=4992.0
)

objetivo_kmh = st.sidebar.number_input(
    "Velocidad media objetivo (km/h)",
    min_value=0.0,
    step=0.1,
    value=20.0,
    help="Umbral de referencia para comparar en los gráficos."
)

# Cálculo de radio / circunferencia
if radio_input == 0 and circ_input > 0:
    radio_calc = circ_input / 1000.0 / (2 * np.pi)  # mm -> m
    circ_calc = circ_input
elif circ_input == 0 and radio_input > 0:
    radio_calc = radio_input
    circ_calc = 2 * np.pi * radio_input * 1000.0   # m -> mm
else:
    radio_calc = (
        radio_input
        if radio_input > 0
        else (circ_input / 1000.0 / (2 * np.pi) if circ_input > 0 else 0)
    )
    circ_calc = (
        circ_input
        if circ_input > 0
        else (2 * np.pi * radio_input * 1000.0 if radio_input > 0 else 0)
    )
    if radio_input > 0 and circ_input > 0:
        st.sidebar.info("Usando ambos valores tal cual fueron ingresados.")

st.sidebar.metric("Radio usado (m)", f"{radio_calc:.3f}")
st.sidebar.metric("Circunferencia (mm)", f"{circ_calc:.1f}")

if not archivo:
    st.info("📤 Sube un archivo IMU para comenzar el análisis.")
    st.stop()

# -------------------------------------------------
# CARGA DEL DATASET (robusta a separador y encoding)
# -------------------------------------------------
df = leer_archivo(archivo)
df.columns = df.columns.astype(str).str.strip()

# Guardamos las columnas originales para poder exportar
cols_original = list(df.columns)

# -------------------------------------------------
# NORMALIZACIÓN DE TIMESTAMP -> df["time"] (datetime)
# -------------------------------------------------
time_col_usada = None

cand_chip = [c for c in df.columns if c.lower().strip() in ["chip time()", "chip_time", "chiptime"]]
cand_time = [c for c in df.columns if c.lower().strip() in ["time", "timestamp"]]
cand_date = [c for c in df.columns if c.lower().strip() in ["date", "fecha"]]

if cand_chip:
    col_chip = cand_chip[0]
    s_chip = reparar_ms_con_dos_puntos(df[col_chip])
    dt_chip = pd.to_datetime(s_chip, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    df["time"] = dt_chip
    time_col_usada = col_chip

elif cand_time:
    col_time = cand_time[0]
    s_time = reparar_ms_con_dos_puntos(df[col_time])
    dt = pd.to_datetime(s_time, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(s_time, errors="coerce")
    df["time"] = dt
    time_col_usada = col_time

elif cand_date and cand_time:
    col_date, col_time = cand_date[0], cand_time[0]
    s_join = (df[col_date].astype(str).str.strip() + " " +
              df[col_time].astype(str).str.strip())
    s_join = reparar_ms_con_dos_puntos(s_join)
    df["time"] = pd.to_datetime(s_join, errors="coerce")
    time_col_usada = f"{col_date} + {col_time}"

else:
    st.error("No encontré un timestamp interpretable. Incluye 'Chip Time()', 'time' o 'Date'+'Time'.")
    st.stop()

# Limpiar y ordenar time
df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

# -------------------------------------------------
# CORRECCIÓN HORARIA DEL SENSOR
# -------------------------------------------------
st.sidebar.subheader("🕒 Corrección horaria del sensor")

aplicar_corr = st.sidebar.checkbox(
    "Aplicar corrección horaria",
    value=False,
    help="Útil cuando el reloj del sensor está adelantado o atrasado respecto a la hora real."
)

col_ch, col_cm = st.sidebar.columns(2)
corr_horas = col_ch.number_input("Horas", min_value=0, max_value=23, value=0, step=1)
corr_minutos = col_cm.number_input("Minutos", min_value=0, max_value=59, value=0, step=1)

corr_signo = st.sidebar.radio(
    "Tipo de corrección",
    ("Adelantar (+)", "Atrasar (−)"),
    index=1  # por defecto: atrasar
)

if aplicar_corr and (corr_horas > 0 or corr_minutos > 0):
    factor = 1 if corr_signo == "Adelantar (+)" else -1
    delta = timedelta(hours=factor * corr_horas, minutes=factor * corr_minutos)
    df["time"] = df["time"] + delta

    with st.sidebar.expander("Detalle de corrección aplicada"):
        sentido = "adelantadas" if factor == 1 else "atrasadas"
        st.write(
            f"Las marcas de tiempo del sensor se han **{sentido}** "
            f"en **{corr_horas} h {corr_minutos} min**."
        )

# -------------------------------------------------
# ELECCIÓN FLEXIBLE DE COLUMNA DE VELOCIDAD ANGULAR (°/s)
# -------------------------------------------------
cand_vel_cols = [
    "AsX(°/s)", "Angular velocity X(°/s)", "GyrX(°/s)", "GyroX(°/s)",
    "WX(°/s)", "W_x(°/s)", "Gx(°/s)", "OmegaX(°/s)"
]
col_vel = encontrar_columna(df, cand_vel_cols)
if not col_vel:
    st.error(
        "No encontré una columna de velocidad angular en °/s para el eje X.\n\n"
        "Candidatos esperados (insensible a mayúsculas): "
        + ", ".join(cand_vel_cols) +
        f"\n\nEncabezados detectados: {list(df.columns)}"
    )
    st.stop()

# Info de columna/timestamp usados
with st.sidebar.expander("ℹ️ Origen de datos usado", expanded=True):
    st.write(f"**Timestamp:** {time_col_usada}")
    st.write(f"**Velocidad angular (°/s):** {col_vel}")

# -------------------------------------------------
# CÁLCULO DE VELOCIDAD
# -------------------------------------------------
df[col_vel] = pd.to_numeric(df[col_vel], errors="coerce").replace([np.inf, -np.inf], np.nan)

# Velocidad con signo (km/h)
df["vel_signed_kmh"] = (df[col_vel] * np.pi / 180.0) * radio_calc * 3.6
# Velocidad absoluta (km/h)
df["vel_abs_kmh"] = df["vel_signed_kmh"].abs()

# dt, distancia por muestra y "distancia por hora" móvil (3600 s)
df["dt_s"] = df["time"].diff().dt.total_seconds().fillna(0).clip(lower=0)
vel_abs_m_s = df["vel_abs_kmh"] / 3.6
df["dist_m"] = vel_abs_m_s * df["dt_s"]

df_roll = df.set_index("time")
roll_dist_m = df_roll["dist_m"].rolling("3600s", min_periods=1).sum()
roll_dt_s   = df_roll["dt_s"].rolling("3600s", min_periods=1).sum()

# Distancia equivalente por hora (km/h)
df["eq_kmh_1h"] = np.where(
    roll_dt_s.values > 0,
    (roll_dist_m.values / (roll_dt_s.values / 3600.0)) / 1000.0,
    np.nan
)

# -------------------------------------------------
# SELECCIÓN DE RANGO POR TEXTO + RESET
# -------------------------------------------------
st.subheader("⏱ Selección de rango de tiempo")
st.caption(
    "Ingresa la hora de INICIO y FIN en formato HH:MM:SS. "
    "Los KPIs y el gráfico se calculan solo con ese tramo. "
    "Si quieres volver al rango completo del archivo, usa el botón Reset."
)

fecha_base = df["time"].dt.date.min()
hora_min = df["time"].min().time()
hora_max = df["time"].max().time()
hora_min_str = hora_min.strftime("%H:%M:%S")
hora_max_str = hora_max.strftime("%H:%M:%S")

if "start_text" not in st.session_state:
    st.session_state.start_text = hora_min_str
if "end_text" not in st.session_state:
    st.session_state.end_text = hora_max_str

if st.button("🔄 Reset al rango completo"):
    st.session_state.start_text = hora_min_str
    st.session_state.end_text = hora_max_str

col_start, col_end = st.columns(2)
start_text = col_start.text_input("Hora de inicio (HH:MM:SS)", value=st.session_state.start_text, key="start_text")
end_text   = col_end.text_input("Hora de fin (HH:MM:SS)", value=st.session_state.end_text, key="end_text")

start_hora = parse_hora(start_text)
end_hora   = parse_hora(end_text)

if start_hora is None or end_hora is None:
    st.error("Formato inválido. Usa HH:MM:SS (ej. 08:15:00).")
    st.stop()

start_dt = datetime.combine(fecha_base, start_hora)
end_dt   = datetime.combine(fecha_base, end_hora)

if end_dt < start_dt:
    # Interpretar que el tramo cruza medianoche: la hora final es del día siguiente
    st.warning("⚠️ La hora final es menor que la inicial. Asumo que el tramo cruza medianoche (fin = día +1).")
    end_dt = end_dt + timedelta(days=1)

# -------------------------------------------------
# FILTRAR EL TRAMO SELECCIONADO
# -------------------------------------------------
rango = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)].copy()
if rango.empty:
    st.warning("⚠️ No hay datos en el rango seleccionado. Ajusta las horas o usa Reset.")
    st.stop()

# -------------------------------------------------
# KPIs (valor absoluto)
# -------------------------------------------------
dt_local = rango["time"].diff().dt.total_seconds().fillna(0)
vel_abs_m_s_local = rango["vel_abs_kmh"] / 3.6
distancia_m = (vel_abs_m_s_local * dt_local).sum()

tiempo_total_seg = dt_local.sum()
tiempo_h = tiempo_total_seg / 3600.0
vel_media_kmh = (distancia_m / 1000.0) / tiempo_h if tiempo_h > 0 else 0.0
vel_max_kmh = float(rango["vel_abs_kmh"].max())
tiempo_min = tiempo_total_seg / 60.0

# -------------------------------------------------
# GRÁFICO DEL TRAMO SELECCIONADO
# -------------------------------------------------
st.subheader("📈 Velocidad vs Tiempo (tramo seleccionado)")
st.caption(
    "Azul = |velocidad| (km/h) → Velocidad en valor absoluto.\n"
    "Verde = velocidad con sentido de giro (km/h) → dirección (avance / retroceso).\n"
    "Morado = distancia equivalente por hora (km/h) calculada con ventana móvil de 3600 s.\n"
    f"Tramo analizado: {start_dt.strftime('%H:%M:%S')} → {end_dt.strftime('%H:%M:%S')}.\n"
    f"— Objetivo de velocidad media (línea fija): {objetivo_kmh:.1f} km/h."
)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=rango["time"], y=rango["vel_abs_kmh"],
        mode="lines", name="|Velocidad| (km/h)",
        line=dict(width=2, color="blue")
    )
)
fig.add_trace(
    go.Scatter(
        x=rango["time"], y=rango["vel_signed_kmh"],
        mode="lines", name="Velocidad con signo (km/h)",
        line=dict(width=1, color="green"), opacity=0.5
    )
)
fig.add_trace(
    go.Scatter(
        x=rango["time"], y=rango["eq_kmh_1h"],
        mode="lines", name="Distancia 1 h (km/h)",
        line=dict(width=2, color="purple")
    )
)
fig.add_hline(
    y=objetivo_kmh, line_width=2, line_dash="dash", line_color="red",
    annotation_text=f"Objetivo {objetivo_kmh:.1f} km/h", annotation_position="top left"
)
fig.update_layout(
    height=500, template="plotly_white",
    xaxis_title="Fecha / Hora", yaxis_title="km/h",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)
fig.update_xaxes(rangeslider_visible=True)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# KPIs
# -------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Velocidad Máxima (km/h)", f"{vel_max_kmh:.2f}")
c2.metric("Velocidad Media (km/h)", f"{vel_media_kmh:.2f}")
c3.metric("Distancia recorrida (m)", f"{distancia_m:.1f}")
c4.metric("Tiempo analizado (min)", f"{tiempo_min:.2f}")

# -------------------------------------------------
# EXPORTAR TRAMO SELECCIONADO IGUAL AL ORIGINAL
# -------------------------------------------------
st.subheader("💾 Exportar tramo seleccionado como archivo IMU")

st.caption(
    "El archivo exportado tendrá **las mismas columnas que el archivo original**, "
    "solo que recortado al rango de tiempo seleccionado. "
    "Puedes volver a subirlo a esta misma app sin problemas."
)

# Tomamos solo las columnas originales (sin columnas calculadas)
df_export = rango[cols_original].copy()

# Nombre de archivo
nombre_archivo = f"IMU_recorte_{start_dt.strftime('%Y%m%d_%H%M%S')}_{end_dt.strftime('%Y%m%d_%H%M%S')}.txt"

# Generamos TXT tipo IMU (tabulado)
txt_bytes = df_export.to_csv(index=False, sep="\t").encode("utf-8")

st.download_button(
    label="⬇️ Descargar archivo recortado (TXT, igual al original)",
    data=txt_bytes,
    file_name=nombre_archivo,
    mime="text/plain"
)

# -------------------------------------------------
# HISTOGRAMA eq_kmh_1h
# -------------------------------------------------
st.subheader("📊 Histograma de la velocidad media equivalente (ventana 1 h)")

data_hist = (
    rango["eq_kmh_1h"]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

if data_hist.size < 5 or data_hist.nunique() <= 1:
    st.info(
        "No hay suficientes valores válidos o variabilidad para mostrar el histograma "
        "de **Distancia 1 h (km/h)** en el tramo seleccionado. "
        "Ajusta el rango o verifica el archivo."
    )
else:
    with st.expander("⚙️ Ajustes del histograma", expanded=False):
        nbins = st.slider("Número de bins", min_value=10, max_value=120, value=40, step=5)
        show_kde = st.checkbox("Mostrar curva de densidad (aprox. KDE)", value=True)

    xmin, xmax = float(data_hist.min()), float(data_hist.max())
    if xmax == xmin:
        xmax = xmin + 1e-6

    edges = np.linspace(xmin, xmax, nbins + 1)
    counts, edges = np.histogram(data_hist, bins=edges)
    total = counts.sum()
    probs = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
    cum_probs = np.cumsum(probs)
    centers = (edges[:-1] + edges[1:]) / 2.0
    text_labels = [f"{p*100:.1f}%" for p in cum_probs]

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Bar(
            x=centers, y=probs, width=(edges[1:] - edges[:-1]),
            name="Distancia 1 h (km/h)", opacity=0.85,
            text=text_labels, textposition="outside",
            hovertemplate="<b>Centro:</b> %{x:.2f} km/h<br><b>Proporción:</b> %{y:.3f}<br><b>% acumulado:</b> %{text}<extra></extra>"
        )
    )

    if show_kde:
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(xmin, xmax, 400)
            kde = gaussian_kde(data_hist)
            dens = kde(xs)
            dens_scaled = dens / dens.max() * (probs.max() * 0.9 if probs.max() > 0 else 0.1)
            fig_hist.add_trace(
                go.Scatter(x=xs, y=dens_scaled, mode="lines", name="Densidad (KDE)", line=dict(width=2))
            )
        except Exception:
            pass

    if xmin <= objetivo_kmh <= xmax:
        fig_hist.add_vline(x=objetivo_kmh, line_width=2, line_dash="dash", line_color="red")

    fig_hist.update_layout(
        height=480, template="plotly_white",
        xaxis_title="Distancia 1 h (km/h) (equivale a velocidad media de la ventana)",
        yaxis_title="Proporción de observaciones",
        bargap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=50)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    pct_above = float((data_hist > objetivo_kmh).mean() * 100.0)
    st.caption(
        f"**Por encima del objetivo ({objetivo_kmh:.1f} km/h): {pct_above:.1f}%**  ·  "
        f"Nota: los textos sobre cada barra muestran el **% acumulado** hasta ese bin (1 decimal)."
    )
    mean_v = float(data_hist.mean()); median_v = float(data_hist.median()); p95_v = float(data_hist.quantile(0.95))
    st.caption(f"Resumen del tramo (eq_kmh_1h): Media={mean_v:.2f} · Mediana={median_v:.2f} · P95={p95_v:.2f} (km/h).")

# -------------------------------------------------
# NOTAS OPERATIVAS
# -------------------------------------------------
st.markdown(
    f"""
**Notas operativas:**

- Estás analizando desde **{start_dt.strftime("%H:%M:%S")}** hasta **{end_dt.strftime("%H:%M:%S")}** del día {df['time'].dt.date.min()} (ya considerando cualquier corrección horaria aplicada).
- El botón "🔄 Reset al rango completo" vuelve a poner las horas mínimas y máximas reales del archivo (ya corregidas si activaste la corrección).
- La línea **Distancia 1 h (km/h)** se calcula sumando la distancia de los **3600 s** previos y dividiéndola por el **tiempo realmente disponible** en esa ventana; si hay menos de 3600 s, se normaliza por ese tiempo (equivale a velocidad media de la ventana).
- El botón de **exportar tramo** genera un TXT con las mismas columnas que el archivo original, solo recortado en tiempo (los textos de las columnas originales no se modifican).
- En la leyenda puedes **apagar/encender** cualquier curva.
- Versión prueba Mauro Raffo ®Michelin.
"""
)

# streamlit run nuevo_formato_mejorado_correcTH.py
