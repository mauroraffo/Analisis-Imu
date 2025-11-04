import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, time

# -------------------------------------------------
# CONFIG GENERAL
# -------------------------------------------------
st.set_page_config(
    page_title="Análisis de Velocidad - IMU Neumáticos",
    layout="wide"
)

st.title("📊 Análisis de Velocidad de Neumático (IMU)")

# -------------------------------------------------
# SIDEBAR: PARÁMETROS DE ENTRADA
# -------------------------------------------------
st.sidebar.header("⚙️ Parámetros de Entrada")

archivo = st.sidebar.file_uploader(
    "Sube tu archivo IMU (Archivo en .txt)",
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
# CARGA Y LIMPIEZA DEL DATASET
# -------------------------------------------------
df = pd.read_csv(archivo, sep="\t", engine="python")
df.columns = df.columns.str.strip()

# Reparar timestamps tipo "2025-10-9 8:18:9:94" -> "2025-10-9 8:18:9.94"
df["time"] = (
    df["time"]
    .astype(str)
    .str.replace(r"(?<=\d):(\d{1,3})$", r".\1", regex=True)
)

df["time"] = pd.to_datetime(
    df["time"],
    format="%Y-%m-%d %H:%M:%S.%f",
    errors="coerce"
)

df = df.dropna(subset=["time"]).copy()
df = df.sort_values("time").reset_index(drop=True)

# -------------------------------------------------
# CÁLCULO DE VELOCIDAD
# -------------------------------------------------
# Nota: cambia "AsX(°/s)" si tu columna tiene otro nombre
df["AsX(°/s)"] = pd.to_numeric(df["AsX(°/s)"], errors="coerce")

# Velocidad con signo (km/h)
df["vel_signed_kmh"] = (df["AsX(°/s)"] * np.pi / 180.0) * radio_calc * 3.6

# Velocidad absoluta (km/h)
df["vel_abs_kmh"] = df["vel_signed_kmh"].abs()

# --- NUEVO: dt, distancia por muestra y "distancia por hora" móvil (3600 s)
df["dt_s"] = df["time"].diff().dt.total_seconds().fillna(0).clip(lower=0)
vel_abs_m_s = df["vel_abs_kmh"] / 3.6
df["dist_m"] = vel_abs_m_s * df["dt_s"]

df_roll = df.set_index("time")
roll_dist_m = df_roll["dist_m"].rolling("3600s", min_periods=1).sum()
roll_dt_s   = df_roll["dt_s"].rolling("3600s", min_periods=1).sum()

# Distancia equivalente por hora (km/h) = (distancia/tiempo)*1h, y a km
df["eq_kmh_1h"] = np.where(
    roll_dt_s.values > 0,
    (roll_dist_m.values / (roll_dt_s.values / 3600.0)) / 1000.0,
    np.nan
)

# -------------------------------------------------
# BLOQUE DE SELECCIÓN DE RANGO POR TEXTO + BOTÓN RESET
# -------------------------------------------------
st.subheader("⏱ Selección de rango de tiempo")

st.caption(
    "Ingresa la hora de INICIO y FIN en formato HH:MM:SS. "
    "Los KPIs y el gráfico se calculan solo con ese tramo. "
    "Si quieres volver al rango completo del archivo, usa el botón Reset."
)

# Tomamos la fecha base (asumimos 1 día continuo)
fecha_base = df["time"].dt.date.min()

# Extremos del dataset
hora_min = df["time"].min().time()
hora_max = df["time"].max().time()
hora_min_str = hora_min.strftime("%H:%M:%S")
hora_max_str = hora_max.strftime("%H:%M:%S")

# Inicializamos el estado solo la primera vez
if "start_text" not in st.session_state:
    st.session_state.start_text = hora_min_str
if "end_text" not in st.session_state:
    st.session_state.end_text = hora_max_str

# Botón para resetear a los límites reales del archivo
if st.button("🔄 Reset al rango completo"):
    st.session_state.start_text = hora_min_str
    st.session_state.end_text = hora_max_str

# Inputs manuales de hora (como texto, sin desplegable)
col_start, col_end = st.columns(2)

start_text = col_start.text_input(
    "Hora de inicio (HH:MM:SS)",
    value=st.session_state.start_text,
    key="start_text"
)

end_text = col_end.text_input(
    "Hora de fin (HH:MM:SS)",
    value=st.session_state.end_text,
    key="end_text"
)

def parse_hora(hora_str: str):
    """
    Convierte 'HH:MM:SS' en datetime.time.
    Si está mal, devuelve None.
    """
    try:
        h, m, s = hora_str.strip().split(":")
        return time(hour=int(h), minute=int(m), second=int(s))
    except Exception:
        return None

start_hora = parse_hora(start_text)
end_hora = parse_hora(end_text)

# Validación de formato
if start_hora is None or end_hora is None:
    st.error("Formato inválido. Usa HH:MM:SS (ej. 08:15:00).")
    st.stop()

# Armamos datetimes completos con la misma fecha base
start_dt = datetime.combine(fecha_base, start_hora)
end_dt   = datetime.combine(fecha_base, end_hora)

# Si el usuario los invierte, los corregimos
if end_dt < start_dt:
    st.warning("⚠️ La hora final es menor que la inicial. Intercambiando para calcular.")
    start_dt, end_dt = end_dt, start_dt

# -------------------------------------------------
# FILTRAR EL TRAMO SELECCIONADO
# -------------------------------------------------
rango = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)].copy()

if rango.empty:
    st.warning("⚠️ No hay datos en el rango seleccionado. Ajusta las horas o usa Reset.")
    st.stop()

# -------------------------------------------------
# KPIs BASADOS EN VALOR ABSOLUTO
# -------------------------------------------------
# Diferencia de tiempo entre muestras (s)
dt = rango["time"].diff().dt.total_seconds().fillna(0)

# Distancia recorrida (m) sumando el rodado real (velocidad absoluta)
vel_abs_m_s_local = rango["vel_abs_kmh"] / 3.6
distancia_m = (vel_abs_m_s_local * dt).sum()

# Tiempo total (h)
tiempo_total_seg = dt.sum()
tiempo_h = tiempo_total_seg / 3600.0

if tiempo_h > 0:
    vel_media_kmh = (distancia_m / 1000.0) / tiempo_h
else:
    vel_media_kmh = 0.0

# Velocidad máxima absoluta en ese tramo (km/h)
vel_max_kmh = rango["vel_abs_kmh"].max()

# Tiempo total analizado en minutos, para referencia
tiempo_min = tiempo_total_seg / 60.0

# -------------------------------------------------
# GRÁFICO DEL TRAMO SELECCIONADO
# -------------------------------------------------
st.subheader("📈 Velocidad vs Tiempo (tramo seleccionado)")

st.caption(
    "Azul = |velocidad| (km/h) → Velocidad en valor absoluto.\n"
    "Verde = velocidad con sentido de giro (km/h) → dirección (avance / retroceso).\n"
    "Morado = distancia equivalente por hora (km/h) calculada con ventana móvil de 3600 s.\n"
    f"Tramo analizado: {start_dt.strftime('%H:%M:%S')} → {end_dt.strftime('%H:%M:%S')}."
)

fig = go.Figure()

# Línea azul: magnitud de velocidad
fig.add_trace(
    go.Scatter(
        x=rango["time"],
        y=rango["vel_abs_kmh"],
        mode="lines",
        name="|Velocidad| (km/h)",
        line=dict(width=2, color="blue")
    )
)

# Línea verde: velocidad con signo (dirección de marcha)
fig.add_trace(
    go.Scatter(
        x=rango["time"],
        y=rango["vel_signed_kmh"],
        mode="lines",
        name="Velocidad con signo (km/h)",
        line=dict(width=1, color="green"),
        opacity=0.5
    )
)

# Línea morada: distancia equivalente por hora (km/h)
fig.add_trace(
    go.Scatter(
        x=rango["time"],
        y=rango["eq_kmh_1h"],
        mode="lines",
        name="Distancia 1 h (km/h)",
        line=dict(width=2, color="purple")
    )
)

fig.update_layout(
    height=500,
    template="plotly_white",
    xaxis_title="Fecha / Hora",
    yaxis_title="km/h",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

# Zoom visual extra dentro del tramo ya filtrado
fig.update_xaxes(rangeslider_visible=True)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# HISTOGRAMA: Distribución de "Distancia 1 h (km/h)" en el tramo
# -------------------------------------------------
st.subheader("📊 Histograma de la velocidad media equivalente (ventana 1 h)")

# Limpiar y validar datos
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

    # Resumen rápido
    mean_v = float(data_hist.mean())
    median_v = float(data_hist.median())
    p95_v = float(data_hist.quantile(0.95))

    # Construir histograma
    fig_hist = go.Figure()

    fig_hist.add_trace(
        go.Histogram(
            x=data_hist,
            nbinsx=nbins,
            name="Distancia 1 h (km/h)",
            histnorm="probability",
            opacity=0.85
        )
    )

    # Opcional: curva de densidad simple (KDE). Si no hay scipy, se omite.
    if show_kde:
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(data_hist.min(), data_hist.max(), 400)
            kde = gaussian_kde(data_hist)
            dens = kde(xs)
            # Escalar para que encaje visualmente con el histograma (probability)
            dens_scaled = dens / dens.max() * 0.12  # factor visual
            fig_hist.add_trace(
                go.Scatter(
                    x=xs, y=dens_scaled, mode="lines", name="Densidad (KDE)", line=dict(width=2)
                )
            )
        except Exception:
            pass  # si no está scipy, seguimos solo con el histograma

    # Marcas de referencia: media, mediana, p95
    for val, nombre, estilo in [
        (mean_v, "Media", "dash"),
        (median_v, "Mediana", "dot"),
        (p95_v, "P95", "dashdot"),
    ]:
        fig_hist.add_vline(x=val, line_width=2, line_dash=estilo)
        fig_hist.add_trace(
            go.Scatter(
                x=[val], y=[0],
                mode="markers",
                marker=dict(size=10),
                name=nombre
            )
        )

    fig_hist.update_layout(
        height=420,
        template="plotly_white",
        xaxis_title="Distancia 1 h (km/h) (equivale a velocidad media de la ventana)",
        yaxis_title="Proporción de observaciones",
        bargap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    st.caption(
        f"Resumen en el tramo: Media={mean_v:.2f} km/h · Mediana={median_v:.2f} km/h · P95={p95_v:.2f} km/h. "
        "La métrica **Distancia 1 h (km/h)** es la velocidad media en una ventana móvil de 3600 s."
    )

# -------------------------------------------------
# MOSTRAR KPIs
# -------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("Velocidad Máxima (km/h)", f"{vel_max_kmh:.2f}")
c2.metric("Velocidad Media (km/h)", f"{vel_media_kmh:.2f}")
c3.metric("Distancia recorrida (m)", f"{distancia_m:.1f}")
c4.metric("Tiempo analizado (min)", f"{tiempo_min:.2f}")

# -------------------------------------------------
# NOTAS OPERATIVAS
# -------------------------------------------------
st.markdown(
    f"""
**Notas operativas:**

- Estás analizando desde **{start_dt.strftime("%H:%M:%S")}** hasta **{end_dt.strftime("%H:%M:%S")}** del día {df['time'].dt.date.min()}.
- El botón "🔄 Reset al rango completo" vuelve a poner las horas mínimas y máximas reales del archivo.
- La línea **Distancia 1 h (km/h)** se calcula sumando la distancia de los **3600 s** previos y dividiéndola por el **tiempo realmente disponible** en esa ventana; si hay menos de 3600 s, se normaliza por ese tiempo (equivale a velocidad media de la ventana).
- En la leyenda puedes **apagar/encender** cualquier curva.
- Versión prueba Mauro Raffo ®Michelin.
"""
)

# streamlit run app_velocidad_mejorada2.py
