
# ============================================================
# Aplicativo Streamlit - Caudal ecológico / ambiental
# Versión simplificada sin parámetros editables
#
# Entrada esperada:
# - CSV/XLSX/XLS
# - Primera columna = fecha
# - Segunda columna = caudal
#
# Flujo:
# 1. Cargar información del local
# 2. Cargar archivo
# 3. Presionar "Correr análisis"
# 4. Ver resultados para: año completo, 3 épocas fijas y todos los meses
# ============================================================

import io
import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import gumbel_r, genextreme

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Caudal ecológico / ambiental",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONFIGURACIÓN FIJA DEL APLICATIVO
# ============================================================

PERMANENCIAS = [50, 60, 70, 75, 80, 85, 90, 95, 97, 99]
TEMPOS_RETORNO = [2, 5, 10, 20]
MIN_DIAS_ANO = 300

EPOCAS = {
    "Época 1 - Estiaje / mayor demanda": [12, 1, 2, 3],
    "Época 2 - Transición": [4, 5, 10, 11],
    "Época 3 - Aguas altas / menor demanda": [6, 7, 8, 9]
}

# ============================================================
# FUNCIONES DE INTERFAZ
# ============================================================

def mostrar_df(df: pd.DataFrame, titulo: Optional[str] = None, decimales: int = 4, height: Optional[int] = None):
    if titulo:
        st.subheader(titulo)
    if df is None or len(df) == 0:
        st.info("No hay datos disponibles para esta tabla.")
        return
    df_show = df.copy()
    num_cols = df_show.select_dtypes(include=[np.number]).columns
    df_show[num_cols] = df_show[num_cols].round(decimales)
    st.dataframe(df_show, use_container_width=True, height=height)


def fig_streamlit(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def download_excel_button(buffer: io.BytesIO, filename: str):
    st.download_button(
        label="⬇️ Descargar resultados en Excel",
        data=buffer,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ============================================================
# LECTURA Y PREPARACIÓN
# ============================================================

def ler_arquivo(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    uploaded_file.seek(0)
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, sep=None, engine="python")
        except Exception:
            uploaded_file.seek(0)
            try:
                return pd.read_csv(uploaded_file, sep=";")
            except Exception:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Formato no reconocido. Use CSV, XLSX o XLS.")


def preparar_serie(df_original: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, str, str]:
    if df_original.shape[1] < 2:
        raise ValueError("El archivo debe tener al menos dos columnas: fecha y caudal.")

    col_fecha = str(df_original.columns[0])
    col_q = str(df_original.columns[1])

    df = df_original.iloc[:, [0, 1]].copy()
    df.columns = ["data", "Q"]

    df["data"] = pd.to_datetime(df["data"], errors="coerce", dayfirst=True)
    df["Q"] = (
        df["Q"].astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["Q"] = pd.to_numeric(df["Q"], errors="coerce")

    df = df.dropna(subset=["data", "Q"])
    df = df[df["Q"] >= 0].copy()
    df = df.sort_values("data")

    if len(df) == 0:
        raise ValueError("No quedaron registros válidos después de convertir fecha y caudal.")

    df = df.set_index("data")
    # Solo se completa el eje diario; no se usa frecuencia mensual.
    df = df.resample("D").mean()
    df["ano"] = df.index.year
    df["mes"] = df.index.month
    df["epoca"] = df["mes"].apply(classificar_epoca)

    serie = df["Q"].dropna()
    if len(serie) == 0:
        raise ValueError("No quedaron datos válidos después de la limpieza.")

    return df, serie, col_fecha, col_q


def classificar_epoca(mes: int) -> str:
    for epoca, meses in EPOCAS.items():
        if int(mes) in meses:
            return epoca
    return "Sin clasificar"

# ============================================================
# MÉTODOS HIDROLÓGICOS
# ============================================================

def vazao_permanencia(q, permanencia: float) -> float:
    q = pd.Series(q).dropna()
    if len(q) == 0:
        return np.nan
    return float(np.percentile(q, 100 - permanencia))


def curva_permanencia(q) -> pd.DataFrame:
    q = pd.Series(q).dropna().sort_values(ascending=False).reset_index(drop=True)
    n = len(q)
    if n == 0:
        return pd.DataFrame(columns=["Permanencia (%)", "Q (m3/s)"])
    p = 100 * np.arange(1, n + 1) / (n + 1)
    return pd.DataFrame({"Permanencia (%)": p, "Q (m3/s)": q})


def tabela_permanencias(q, grupo: str, tipo: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Tipo": tipo,
        "Grupo": grupo,
        "Permanencia (%)": PERMANENCIAS,
        "Q (m3/s)": [vazao_permanencia(q, p) for p in PERMANENCIAS]
    })


def estatisticas_grupo(q, grupo: str, tipo: str) -> dict:
    s = pd.Series(q).dropna()
    if len(s) == 0:
        return {"Tipo": tipo, "Grupo": grupo}
    return {
        "Tipo": tipo,
        "Grupo": grupo,
        "n": int(s.count()),
        "Q mínimo": s.min(),
        "Q medio": s.mean(),
        "Q mediano": s.median(),
        "Q máximo": s.max(),
        "Desvío estándar": s.std(),
        "Coef. variación": s.std() / s.mean() if s.mean() != 0 else np.nan,
        "P25": s.quantile(0.25),
        "P75": s.quantile(0.75),
        "P95": s.quantile(0.95)
    }


def calcular_por_contextos(df: pd.DataFrame, serie: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calcula estadísticas y permanencias para año completo, tres épocas y todos los meses."""
    estat_rows = []
    perm_tables = []
    hoppe_rows = []
    tennant_rows = []

    contextos = []
    contextos.append(("Año completo", "Serie completa", serie))

    for epoca, g in df.groupby("epoca"):
        if epoca != "Sin clasificar":
            contextos.append(("Tres épocas", epoca, g["Q"]))

    for mes, g in df.groupby("mes"):
        contextos.append(("Mensual", f"Mes {int(mes)}", g["Q"]))

    for tipo, grupo, q in contextos:
        s = pd.Series(q).dropna()
        estat_rows.append(estatisticas_grupo(s, grupo, tipo))
        perm_tables.append(tabela_permanencias(s, grupo, tipo))

        hoppe_rows.extend([
            {"Tipo": tipo, "Grupo": grupo, "Método": "Hoppe Q80", "Función": "Actividades diarias", "Q (m3/s)": vazao_permanencia(s, 80)},
            {"Tipo": tipo, "Grupo": grupo, "Método": "Hoppe Q40", "Función": "Desove / reproducción", "Q (m3/s)": vazao_permanencia(s, 40)},
            {"Tipo": tipo, "Grupo": grupo, "Método": "Hoppe Q17", "Función": "Descarga / lavado del sustrato", "Q (m3/s)": vazao_permanencia(s, 17)}
        ])

        q_media = s.mean() if len(s) > 0 else np.nan
        for pct in [0.10, 0.30, 0.60]:
            tennant_rows.append({
                "Tipo": tipo,
                "Grupo": grupo,
                "Método": f"Tennant {int(pct*100)}% Qmedio",
                "Percentual": pct,
                "Q (m3/s)": q_media * pct if pd.notna(q_media) else np.nan
            })

    estatisticas_contextos = pd.DataFrame(estat_rows)
    permanencias_contextos = pd.concat(perm_tables, ignore_index=True) if perm_tables else pd.DataFrame()
    hoppe_contextos = pd.DataFrame(hoppe_rows)
    tennant_contextos = pd.DataFrame(tennant_rows)

    return estatisticas_contextos, permanencias_contextos, hoppe_contextos, tennant_contextos


def referencias_q60_q80_q90(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({
        "Tipo": "Año completo",
        "Grupo": "Serie completa",
        "Q60 (m3/s)": vazao_permanencia(df["Q"], 60),
        "Q80 (m3/s)": vazao_permanencia(df["Q"], 80),
        "Q90 (m3/s)": vazao_permanencia(df["Q"], 90)
    })
    for epoca, g in df.groupby("epoca"):
        if epoca == "Sin clasificar":
            continue
        rows.append({
            "Tipo": "Tres épocas",
            "Grupo": epoca,
            "Q60 (m3/s)": vazao_permanencia(g["Q"], 60),
            "Q80 (m3/s)": vazao_permanencia(g["Q"], 80),
            "Q90 (m3/s)": vazao_permanencia(g["Q"], 90)
        })
    for mes, g in df.groupby("mes"):
        rows.append({
            "Tipo": "Mensual",
            "Grupo": f"Mes {int(mes)}",
            "Q60 (m3/s)": vazao_permanencia(g["Q"], 60),
            "Q80 (m3/s)": vazao_permanencia(g["Q"], 80),
            "Q90 (m3/s)": vazao_permanencia(g["Q"], 90)
        })
    return pd.DataFrame(rows)


def metodo_ngprp(df: pd.DataFrame) -> pd.DataFrame:
    clim = df.dropna(subset=["Q"]).groupby("mes")["Q"].mean()
    if len(clim) == 0:
        return pd.DataFrame()
    meses_altos = clim.sort_values(ascending=False).head(3).index.tolist()
    rows = []
    for mes, g in df.groupby("mes"):
        q50 = vazao_permanencia(g["Q"], 50)
        q90 = vazao_permanencia(g["Q"], 90)
        if mes in meses_altos:
            criterio = "Q50 mensual"
            qeco = q50
        else:
            criterio = "Q90 mensual"
            qeco = q90
        rows.append({"Mes": int(mes), "Criterio": criterio, "Q eco (m3/s)": qeco, "Q50": q50, "Q90": q90})
    return pd.DataFrame(rows)


def metodo_abf(df: pd.DataFrame) -> Tuple[float, int, pd.DataFrame]:
    clim = df.dropna(subset=["Q"]).groupby("mes")["Q"].mean()
    if len(clim) == 0:
        return np.nan, np.nan, pd.DataFrame()
    mes_seco = int(clim.idxmin())
    q_abf = float(clim.min())
    clim_df = pd.DataFrame({"Mes": clim.index.astype(int), "Q media mensual": clim.values})
    return q_abf, mes_seco, clim_df


def medias_moveis_minimas_anuais(df: pd.DataFrame, janela: int, min_dias_ano: int = MIN_DIAS_ANO) -> pd.DataFrame:
    aux = df[["Q"]].copy()
    aux[f"Q{janela}"] = aux["Q"].rolling(janela, min_periods=janela).mean()
    aux["Ano"] = aux.index.year
    rows = []
    for ano, g in aux.groupby("Ano"):
        if g["Q"].count() >= min_dias_ano:
            val = g[f"Q{janela}"].min()
            if pd.notna(val):
                rows.append({"Ano": int(ano), f"Min_Q{janela}": float(val)})
    return pd.DataFrame(rows)


def calcular_qjt(df: pd.DataFrame, janela: int, distribuicao: str) -> pd.DataFrame:
    mins = medias_moveis_minimas_anuais(df, janela)
    if len(mins) < 10:
        return pd.DataFrame([{"Método": f"Q{janela},T", "Distribución": distribuicao, "Tiempo de retorno": np.nan, "Q (m3/s)": np.nan, "Observación": "Serie corta"}])
    x = mins[f"Min_Q{janela}"].dropna().values
    rows = []
    for T in TEMPOS_RETORNO:
        try:
            p = 1 / T
            if distribuicao == "gumbel":
                loc, scale = gumbel_r.fit(x)
                qjt = gumbel_r.ppf(p, loc=loc, scale=scale)
            else:
                c, loc, scale = genextreme.fit(x)
                qjt = genextreme.ppf(p, c, loc=loc, scale=scale)
            obs = "OK"
            qjt = max(float(qjt), 0)
        except Exception as exc:
            qjt = np.nan
            obs = f"Error: {exc}"
        rows.append({"Método": f"Q{janela},{T}", "Distribución": distribuicao, "Tiempo de retorno": T, "Q (m3/s)": qjt, "Observación": obs})
    return pd.DataFrame(rows)


def filtro_lyne_hollick(q, alpha: float = 0.925, passes: int = 3) -> pd.Series:
    q = pd.Series(q).dropna().copy()
    if len(q) < 10:
        return pd.Series(index=q.index, dtype=float)
    def one_pass(flow):
        quick = np.zeros(len(flow))
        base = np.zeros(len(flow))
        values = flow.values
        quick[0] = 0
        base[0] = values[0]
        for i in range(1, len(values)):
            quick[i] = alpha * quick[i - 1] + ((1 + alpha) / 2) * (values[i] - values[i - 1])
            quick[i] = max(quick[i], 0)
            base[i] = values[i] - quick[i]
            base[i] = max(base[i], 0)
            base[i] = min(base[i], values[i])
        return pd.Series(base, index=flow.index)
    base = q.copy()
    for _ in range(passes):
        base = one_pass(base)
        base = base[::-1]
        base = one_pass(base)
        base = base[::-1]
    return base.clip(lower=0, upper=q)


def calcular_bfi(df: pd.DataFrame, serie: pd.Series):
    base = filtro_lyne_hollick(serie)
    df_base = pd.DataFrame({"Q": serie, "Q_base": base}).dropna()
    if len(df_base) == 0 or df_base["Q"].sum() == 0:
        return df_base, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_base["ano"] = df_base.index.year
    df_base["mes"] = df_base.index.month
    df_base["epoca"] = df_base["mes"].apply(classificar_epoca)
    bfi_global = pd.DataFrame([{"Tipo": "Año completo", "Grupo": "Serie completa", "BFI": df_base["Q_base"].sum() / df_base["Q"].sum()}])
    bfi_mensal = df_base.groupby("mes").apply(lambda x: x["Q_base"].sum() / x["Q"].sum() if x["Q"].sum() != 0 else np.nan).reset_index(name="BFI")
    bfi_mensal["Tipo"] = "Mensual"
    bfi_mensal["Grupo"] = bfi_mensal["mes"].apply(lambda x: f"Mes {int(x)}")
    bfi_epoca = df_base.groupby("epoca").apply(lambda x: x["Q_base"].sum() / x["Q"].sum() if x["Q"].sum() != 0 else np.nan).reset_index(name="BFI")
    bfi_epoca = bfi_epoca[bfi_epoca["epoca"] != "Sin clasificar"].copy()
    bfi_epoca["Tipo"] = "Tres épocas"
    bfi_epoca["Grupo"] = bfi_epoca["epoca"]
    bfi_contextos = pd.concat([
        bfi_global[["Tipo", "Grupo", "BFI"]],
        bfi_epoca[["Tipo", "Grupo", "BFI"]],
        bfi_mensal[["Tipo", "Grupo", "BFI"]]
    ], ignore_index=True)
    bfi_anual = df_base.groupby("ano").apply(lambda x: x["Q_base"].sum() / x["Q"].sum() if x["Q"].sum() != 0 else np.nan).reset_index(name="BFI anual")
    return df_base, bfi_contextos, bfi_anual, bfi_mensal


def indicadores_iha_basicos(df: pd.DataFrame):
    q = df["Q"].copy()
    aux = pd.DataFrame({"Q": q})
    aux["Ano"] = aux.index.year
    aux["Mes"] = aux.index.month
    mensais = aux.groupby(["Ano", "Mes"])["Q"].agg(Media="mean", Mediana="median", Minimo="min", Maximo="max").reset_index()
    extremos = []
    for janela in [1, 3, 7, 30, 90]:
        mov = q.rolling(janela, min_periods=janela).mean()
        tmp = pd.DataFrame({"valor": mov, "Ano": mov.index.year}).dropna()
        for ano, g in tmp.groupby("Ano"):
            extremos.append({"Ano": int(ano), "Janela dias": janela, "Minimo": g["valor"].min(), "Maximo": g["valor"].max()})
    q25 = q.quantile(0.25)
    q75 = q.quantile(0.75)
    pulsos = []
    for ano, g in aux.groupby("Ano"):
        s = g["Q"].dropna()
        if len(s) < 30:
            continue
        alto = s > q75
        bajo = s < q25
        dif = s.diff().dropna()
        sinal = np.sign(dif)
        pulsos.append({
            "Ano": int(ano),
            "Limiar bajo Q25": q25,
            "Limiar alto Q75": q75,
            "Pulsos altos": int(((alto) & (~alto.shift(1).fillna(False))).sum()),
            "Pulsos bajos": int(((bajo) & (~bajo.shift(1).fillna(False))).sum()),
            "Tasa media subida": dif[dif > 0].mean(),
            "Tasa media descenso": dif[dif < 0].mean(),
            "Reversiones": int((sinal != sinal.shift(1)).sum())
        })
    return mensais, pd.DataFrame(extremos), pd.DataFrame(pulsos)


def teste_pettitt(x):
    x = pd.Series(x).dropna().values
    n = len(x)
    if n < 6:
        return None, np.nan, np.nan
    Kt = np.zeros(n)
    for t in range(n):
        s = 0
        for i in range(t + 1):
            for j in range(t + 1, n):
                s += np.sign(x[i] - x[j])
        Kt[t] = s
    K = float(np.max(np.abs(Kt)))
    idx = int(np.argmax(np.abs(Kt)))
    p = 2 * np.exp((-6 * K**2) / (n**3 + n**2))
    return idx, K, float(min(max(p, 0), 1))


def preparar_indicadores_pettitt(df: pd.DataFrame):
    rows = []
    for ano, g in df.groupby("ano"):
        s = g["Q"].dropna()
        if len(s) < MIN_DIAS_ANO:
            continue
        rows.append({
            "Ano": int(ano),
            "Q medio anual": s.mean(),
            "Q mediano anual": s.median(),
            "Q90 anual": vazao_permanencia(s, 90),
            "Q95 anual": vazao_permanencia(s, 95),
            "Q minimo anual": s.min(),
            "Q maximo anual": s.max(),
            "Q7 minimo anual": s.rolling(7, min_periods=7).mean().min(),
            "Q30 minimo anual": s.rolling(30, min_periods=30).mean().min()
        })
    return pd.DataFrame(rows)


def aplicar_pettitt(tab_ind: pd.DataFrame):
    rows = []
    if tab_ind is None or len(tab_ind) < 6:
        return pd.DataFrame()
    for indicador in tab_ind.columns:
        if indicador == "Ano":
            continue
        serie_ind = tab_ind[["Ano", indicador]].dropna()
        if len(serie_ind) < 6:
            continue
        idx, K, p = teste_pettitt(serie_ind[indicador])
        if idx is None:
            continue
        ano = int(serie_ind.iloc[idx]["Ano"])
        antes = serie_ind[serie_ind["Ano"] <= ano][indicador]
        depois = serie_ind[serie_ind["Ano"] > ano][indicador]
        ma = antes.mean()
        md = depois.mean()
        rows.append({
            "Indicador": indicador,
            "Ano cambio Pettitt": ano,
            "Estadistico K": K,
            "p-valor": p,
            "Significativo p<0.05": "Sí" if p < 0.05 else "No",
            "Media antes": ma,
            "Media después": md,
            "Cambio relativo (%)": ((md - ma) / ma) * 100 if ma != 0 and pd.notna(ma) else np.nan
        })
    return pd.DataFrame(rows)


def sugerir_ano_pettitt(tab_pettitt: pd.DataFrame):
    if tab_pettitt is None or len(tab_pettitt) == 0:
        return None, None, None, False
    qmedio = tab_pettitt[(tab_pettitt["Indicador"] == "Q medio anual") & (tab_pettitt["p-valor"] < 0.05)]
    if len(qmedio) > 0:
        row = qmedio.iloc[0]
    else:
        sig = tab_pettitt[tab_pettitt["p-valor"] < 0.05]
        row = sig.sort_values("p-valor").iloc[0] if len(sig) > 0 else tab_pettitt.sort_values("p-valor").iloc[0]
    return int(row["Ano cambio Pettitt"]), row["Indicador"], float(row["p-valor"]), bool(row["p-valor"] < 0.05)


def rva_simplificado(df: pd.DataFrame, ano_corte: int):
    rows = []
    for ano, g in df.groupby("ano"):
        s = g["Q"].dropna()
        if len(s) < MIN_DIAS_ANO:
            continue
        rows.append({
            "Ano": int(ano),
            "Periodo": "Referencia" if ano <= ano_corte else "Alterado",
            "Q medio": s.mean(),
            "Q mediano": s.median(),
            "Q minimo": s.min(),
            "Q maximo": s.max(),
            "Q7 minimo": s.rolling(7, min_periods=7).mean().min(),
            "Q30 minimo": s.rolling(30, min_periods=30).mean().min(),
            "Q90 anual": vazao_permanencia(s, 90),
            "Q10 anual": vazao_permanencia(s, 10)
        })
    ind = pd.DataFrame(rows)
    if len(ind) == 0 or ind["Periodo"].nunique() < 2:
        return pd.DataFrame()
    out = []
    for indicador in ["Q medio", "Q mediano", "Q minimo", "Q maximo", "Q7 minimo", "Q30 minimo", "Q90 anual", "Q10 anual"]:
        ref = ind[ind["Periodo"] == "Referencia"][indicador].dropna()
        alt = ind[ind["Periodo"] == "Alterado"][indicador].dropna()
        if len(ref) < 3 or len(alt) < 3:
            continue
        p25 = ref.quantile(0.25)
        p75 = ref.quantile(0.75)
        esp = ((ref >= p25) & (ref <= p75)).mean()
        obs = ((alt >= p25) & (alt <= p75)).mean()
        alt_h = np.nan if esp == 0 else ((obs - esp) / esp) * 100
        out.append({
            "Indicador": indicador,
            "Referencia P25": p25,
            "Referencia P75": p75,
            "Media referencia": ref.mean(),
            "Media alterado": alt.mean(),
            "Frecuencia esperada dentro RVA": esp,
            "Frecuencia observada dentro RVA": obs,
            "Alteración hidrológica (%)": alt_h
        })
    return pd.DataFrame(out)


def calcular_todo(df: pd.DataFrame, serie: pd.Series):
    estat_contextos, perm_contextos, hoppe_contextos, tennant_contextos = calcular_por_contextos(df, serie)
    q_refs = referencias_q60_q80_q90(df)
    ngprp = metodo_ngprp(df)
    q_abf, mes_abf, clim_mensal = metodo_abf(df)
    abf = pd.DataFrame([{"Método": "ABF simplificado", "Mes más seco": mes_abf, "Q ABF (m3/s)": q_abf}])
    qjt = pd.concat([calcular_qjt(df, 7, "gumbel"), calcular_qjt(df, 7, "gev"), calcular_qjt(df, 30, "gumbel"), calcular_qjt(df, 30, "gev")], ignore_index=True)
    df_base, bfi_contextos, bfi_anual, bfi_mensal = calcular_bfi(df, serie)
    iha_mensais, iha_extremos, iha_pulsos = indicadores_iha_basicos(df)
    ind_pettitt = preparar_indicadores_pettitt(df)
    pettitt = aplicar_pettitt(ind_pettitt)
    ano_pettitt, indicador_pettitt, p_pettitt, sig_pettitt = sugerir_ano_pettitt(pettitt)
    anos = sorted(df.dropna(subset=["Q"])["ano"].unique())
    ano_medio = int(np.median(anos)) if len(anos) else None
    fdc = curva_permanencia(serie)
    resumo = pd.DataFrame([
        {"Método": "Q90 global", "Q eco (m3/s)": vazao_permanencia(serie, 90), "Lectura": "Caudal igualado o excedido 90% del tiempo"},
        {"Método": "Q95 global", "Q eco (m3/s)": vazao_permanencia(serie, 95), "Lectura": "Condición de baja disponibilidad"},
        {"Método": "Q80 medio mensual", "Q eco (m3/s)": q_refs[q_refs["Tipo"] == "Mensual"]["Q80 (m3/s)"].mean(), "Lectura": "Promedio de referencias mensuales"},
        {"Método": "Tennant 30%", "Q eco (m3/s)": serie.mean() * 0.30, "Lectura": "Porcentaje de caudal medio"},
        {"Método": "ABF", "Q eco (m3/s)": q_abf, "Lectura": "Mes históricamente más seco"}
    ])
    return {
        "estat_contextos": estat_contextos,
        "perm_contextos": perm_contextos,
        "hoppe_contextos": hoppe_contextos,
        "tennant_contextos": tennant_contextos,
        "q_refs": q_refs,
        "ngprp": ngprp,
        "abf": abf,
        "q_abf": q_abf,
        "mes_abf": mes_abf,
        "clim_mensal": clim_mensal,
        "qjt": qjt,
        "df_base": df_base,
        "bfi_contextos": bfi_contextos,
        "bfi_anual": bfi_anual,
        "bfi_mensal": bfi_mensal,
        "iha_mensais": iha_mensais,
        "iha_extremos": iha_extremos,
        "iha_pulsos": iha_pulsos,
        "ind_pettitt": ind_pettitt,
        "pettitt": pettitt,
        "ano_pettitt": ano_pettitt,
        "indicador_pettitt": indicador_pettitt,
        "p_pettitt": p_pettitt,
        "sig_pettitt": sig_pettitt,
        "ano_medio": ano_medio,
        "fdc": fdc,
        "resumo": resumo
    }


def exportar_excel(info_local, info_serie, resultados, rva, criterio_rva):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        info_local.to_excel(writer, sheet_name="00_info_local", index=False)
        info_serie.to_excel(writer, sheet_name="01_info_serie", index=False)
        pd.DataFrame([{"Criterio RVA": criterio_rva}]).to_excel(writer, sheet_name="02_criterio_rva", index=False)
        resultados["estat_contextos"].to_excel(writer, sheet_name="03_estad_contextos", index=False)
        resultados["perm_contextos"].to_excel(writer, sheet_name="04_perm_contextos", index=False)
        resultados["q_refs"].to_excel(writer, sheet_name="05_q60_q80_q90", index=False)
        resultados["tennant_contextos"].to_excel(writer, sheet_name="06_tennant_contextos", index=False)
        resultados["hoppe_contextos"].to_excel(writer, sheet_name="07_hoppe_contextos", index=False)
        resultados["ngprp"].to_excel(writer, sheet_name="08_ngprp", index=False)
        resultados["abf"].to_excel(writer, sheet_name="09_abf", index=False)
        resultados["qjt"].to_excel(writer, sheet_name="10_qjt", index=False)
        resultados["bfi_contextos"].to_excel(writer, sheet_name="11_bfi_contextos", index=False)
        resultados["bfi_anual"].to_excel(writer, sheet_name="12_bfi_anual", index=False)
        resultados["iha_mensais"].to_excel(writer, sheet_name="13_iha_mensais", index=False)
        resultados["iha_extremos"].to_excel(writer, sheet_name="14_iha_extremos", index=False)
        resultados["iha_pulsos"].to_excel(writer, sheet_name="15_iha_pulsos", index=False)
        resultados["ind_pettitt"].to_excel(writer, sheet_name="16_ind_pettitt", index=False)
        resultados["pettitt"].to_excel(writer, sheet_name="17_pettitt", index=False)
        rva.to_excel(writer, sheet_name="18_rva", index=False)
        resultados["fdc"].to_excel(writer, sheet_name="19_curva_perm", index=False)
        resultados["resumo"].to_excel(writer, sheet_name="20_resumo", index=False)
    output.seek(0)
    return output

# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

st.title("💧 Caudal ecológico / ambiental")
st.caption("Aplicativo didáctico para análisis por año completo, tres épocas fijas y todos los meses.")

with st.expander("📘 Alcance del aplicativo", expanded=True):
    st.markdown("""
Este aplicativo calcula referencias hidrológicas de caudal ecológico/ambiental a partir de una serie diaria de caudales.

El análisis se ejecuta automáticamente para:

- **Año completo**;
- **Tres épocas fijas**: estiaje/mayor demanda, transición y aguas altas/menor demanda;
- **Todos los meses**.

Los resultados son comparativos y didácticos. No sustituyen una evaluación ecológica completa.
""")

st.sidebar.header("Información del local")
nombre_curso = st.sidebar.text_input("Curso de agua", "")
nombre_estacion = st.sidebar.text_input("Estación / punto de análisis", "")
departamento = st.sidebar.text_input("Departamento", "")
cuenca = st.sidebar.text_input("Cuenca hidrográfica", "")
tipo_analisis = st.sidebar.selectbox("Tipo de análisis", ["Didáctico", "Toma directa", "Embalse", "Ecológico", "Otro"])
uso_suelo = st.sidebar.text_area("Uso del suelo / observaciones", "")

info_local = pd.DataFrame([{
    "Curso de agua": nombre_curso,
    "Estación / punto": nombre_estacion,
    "Departamento": departamento,
    "Cuenca": cuenca,
    "Tipo de análisis": tipo_analisis,
    "Uso del suelo / observaciones": uso_suelo
}])

st.header("1. Carga de datos")
uploaded_file = st.file_uploader("Cargue un archivo CSV, XLSX o XLS. Primera columna = fecha; segunda columna = caudal.", type=["csv", "xlsx", "xls"])

if "analisis_ok" not in st.session_state:
    st.session_state["analisis_ok"] = False

correr = st.button("▶️ Correr análisis", type="primary", disabled=uploaded_file is None)

if uploaded_file is None:
    st.info("Cargue un archivo para habilitar el análisis.")
    st.stop()

if correr:
    try:
        df_original = ler_arquivo(uploaded_file)
        df, serie, col_fecha, col_q = preparar_serie(df_original)
        info_serie = pd.DataFrame([{
            "Fecha inicial": serie.index.min(),
            "Fecha final": serie.index.max(),
            "Datos válidos": serie.count(),
            "Años con datos": serie.index.year.nunique(),
            "Fallas en la serie diaria (%)": round(df["Q"].isna().mean() * 100, 2),
            "Caudal medio": serie.mean(),
            "Caudal mediano": serie.median(),
            "Caudal mínimo": serie.min(),
            "Caudal máximo": serie.max()
        }])
        with st.spinner("Calculando metodologías..."):
            resultados = calcular_todo(df, serie)
        st.session_state["analisis_ok"] = True
        st.session_state["df_original"] = df_original
        st.session_state["df"] = df
        st.session_state["serie"] = serie
        st.session_state["col_fecha"] = col_fecha
        st.session_state["col_q"] = col_q
        st.session_state["info_serie"] = info_serie
        st.session_state["resultados"] = resultados
        st.session_state["info_local"] = info_local
        st.success("Análisis ejecutado correctamente.")
    except Exception as exc:
        st.session_state["analisis_ok"] = False
        st.error(f"Error al ejecutar el análisis: {exc}")

if not st.session_state.get("analisis_ok", False):
    st.stop()

df_original = st.session_state["df_original"]
df = st.session_state["df"]
serie = st.session_state["serie"]
col_fecha = st.session_state["col_fecha"]
col_q = st.session_state["col_q"]
info_serie = st.session_state["info_serie"]
resultados = st.session_state["resultados"]
info_local = st.session_state["info_local"]

st.success(f"Archivo leído correctamente. Fecha = {col_fecha}; caudal = {col_q}.")

with st.expander("Vista previa y calidad de la serie", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        mostrar_df(df_original.head(20), "Vista previa del archivo")
    with c2:
        mostrar_df(info_serie, "Información básica de la serie")

st.header("2. Año de cambio y RVA")
mostrar_df(resultados["ind_pettitt"], "Indicadores anuales usados para Pettitt", height=260)
mostrar_df(resultados["pettitt"], "Resultados del test de Pettitt", height=260)

ano_pettitt = resultados["ano_pettitt"]
ano_medio = resultados["ano_medio"]
indicador_pettitt = resultados["indicador_pettitt"]
p_pettitt = resultados["p_pettitt"]
sig_pettitt = resultados["sig_pettitt"]
anos_validos = sorted(df.dropna(subset=["Q"])["ano"].unique())

if ano_pettitt is not None:
    st.info(f"Pettitt sugiere el año {ano_pettitt}, indicador {indicador_pettitt}, p-valor = {p_pettitt:.4f}, significativo: {'sí' if sig_pettitt else 'no'}.")

opciones = ["Usar año medio de la serie"]
if ano_pettitt is not None:
    opciones.insert(0, "Usar año sugerido por Pettitt")
opciones.append("Indicar otro año manualmente")
criterio = st.radio("Criterio para RVA", opciones)

if criterio == "Usar año sugerido por Pettitt" and ano_pettitt is not None:
    ano_corte = int(ano_pettitt)
    criterio_rva = f"Pettitt: {indicador_pettitt}"
elif criterio == "Indicar otro año manualmente":
    ano_corte = int(st.number_input("Año de corte manual", min_value=int(min(anos_validos)), max_value=int(max(anos_validos)), value=int(ano_medio), step=1))
    criterio_rva = "Manual"
else:
    ano_corte = int(ano_medio)
    criterio_rva = "Año medio de la serie"

st.markdown(f"**Año de corte usado:** {ano_corte}. Referencia: años ≤ {ano_corte}. Alterado: años > {ano_corte}.")
rva = rva_simplificado(df, ano_corte)
mostrar_df(rva, "RVA simplificado")

if len(resultados["ind_pettitt"]) > 0:
    indicador = st.selectbox("Indicador para visualizar el corte", [c for c in resultados["ind_pettitt"].columns if c != "Ano"])
    si = resultados["ind_pettitt"][["Ano", indicador]].dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(si["Ano"], si[indicador], marker="o", label=indicador)
    ax.axvline(ano_corte, linestyle="--", label=f"Año de corte: {ano_corte}")
    ax.set_xlabel("Año")
    ax.set_ylabel(indicador)
    ax.set_title("Indicador anual y año de corte")
    ax.grid(True)
    ax.legend()
    fig_streamlit(fig)

st.header("3. Resultados")
tabs = st.tabs(["Resumen", "Año/épocas/meses", "Métodos", "BFI/IHA", "Gráficos"])

with tabs[0]:
    mostrar_df(resultados["resumo"], "Resumen de métodos")

with tabs[1]:
    mostrar_df(resultados["estat_contextos"], "Estadísticas por año completo, tres épocas y meses", height=350)
    mostrar_df(resultados["perm_contextos"], "Permanencias por año completo, tres épocas y meses", height=450)
    mostrar_df(resultados["q_refs"], "Q60, Q80 y Q90 por año completo, tres épocas y meses")

with tabs[2]:
    mostrar_df(resultados["tennant_contextos"], "Tennant por año completo, tres épocas y meses", height=350)
    mostrar_df(resultados["hoppe_contextos"], "Hoppe por año completo, tres épocas y meses", height=350)
    mostrar_df(resultados["ngprp"], "NGPRP mensual")
    mostrar_df(resultados["abf"], "ABF")
    mostrar_df(resultados["qjt"], "Qj,T")

with tabs[3]:
    mostrar_df(resultados["bfi_contextos"], "BFI por año completo, tres épocas y meses")
    mostrar_df(resultados["bfi_anual"], "BFI anual")
    mostrar_df(resultados["iha_mensais"].head(150), "IHA mensual", height=350)
    mostrar_df(resultados["iha_extremos"].head(150), "IHA extremos", height=350)
    mostrar_df(resultados["iha_pulsos"].head(150), "IHA pulsos", height=350)

with tabs[4]:
    st.subheader("Serie temporal")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(serie.index, serie.values, linewidth=0.7)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Caudal (m3/s)")
    ax.grid(True)
    fig_streamlit(fig)

    st.subheader("Curva de permanencia")
    fdc = resultados["fdc"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(fdc["Permanencia (%)"], fdc["Q (m3/s)"], linewidth=1.5)
    for p in [60, 80, 90, 95]:
        ax.axvline(p, linestyle="--", label=f"Q{p}")
    ax.set_xlabel("Permanencia / excedencia (%)")
    ax.set_ylabel("Caudal (m3/s)")
    ax.grid(True)
    ax.legend()
    fig_streamlit(fig)

    st.subheader("Q60, Q80 y Q90 mensuales")
    qm = resultados["q_refs"][resultados["q_refs"]["Tipo"] == "Mensual"].copy()
    qm["Mes"] = qm["Grupo"].str.extract(r"(\d+)").astype(int)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(qm["Mes"], qm["Q60 (m3/s)"], marker="o", label="Q60")
    ax.plot(qm["Mes"], qm["Q80 (m3/s)"], marker="o", label="Q80")
    ax.plot(qm["Mes"], qm["Q90 (m3/s)"], marker="o", label="Q90")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Mes")
    ax.set_ylabel("Caudal (m3/s)")
    ax.grid(True)
    ax.legend()
    fig_streamlit(fig)

    st.subheader("Hidrograma medio mensual")
    clim = resultados["clim_mensal"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(clim["Mes"], clim["Q media mensual"], marker="o")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Mes")
    ax.set_ylabel("Caudal medio mensual")
    ax.grid(True)
    fig_streamlit(fig)

    st.subheader("Caudal observado y caudal base")
    db = resultados["df_base"]
    if len(db) > 0:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(db.index, db["Q"], linewidth=0.6, label="Caudal observado")
        ax.plot(db.index, db["Q_base"], linewidth=0.8, label="Caudal base")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Caudal")
        ax.grid(True)
        ax.legend()
        fig_streamlit(fig)

    st.subheader("RVA")
    if len(rva) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(rva["Indicador"], rva["Alteración hidrológica (%)"])
        ax.set_xlabel("Alteración hidrológica (%)")
        ax.grid(axis="x")
        fig_streamlit(fig)

st.header("4. Exportación")
excel = exportar_excel(info_local, info_serie, resultados, rva, criterio_rva)
st.download_button("⬇️ Descargar Excel", data=excel, file_name="resultados_caudal_ecologico.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
