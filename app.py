
# ============================================================
# Aplicativo Streamlit - Caudal ecológico / ambiental
# Entrada esperada: CSV/XLSX/XLS con primera columna = fecha y segunda columna = caudal
# ============================================================

import io
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import gumbel_r, genextreme

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Caudal ecológico",
    page_icon="💧",
    layout="wide"
)

# ============================================================
# CONFIGURACIÓN HIDROLÓGICA
# ============================================================

PERMANENCIAS = [50, 60, 70, 75, 80, 85, 90, 95, 97, 99]
TEMPOS_RETORNO = [2, 5, 10, 20]

# ============================================================
# FUNCIONES DE INTERFAZ
# ============================================================

def mostrar_df(df, titulo=None, decimales=4, height=None):
    if titulo:
        st.subheader(titulo)
    if df is None or len(df) == 0:
        st.info("No hay datos disponibles para esta tabla.")
    else:
        try:
            st.dataframe(df.round(decimales), use_container_width=True, height=height)
        except Exception:
            st.dataframe(df, use_container_width=True, height=height)

def download_excel_button(buffer, filename):
    st.download_button(
        label="⬇️ Descargar resultados en Excel",
        data=buffer,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ============================================================
# LECTURA Y PREPARACIÓN
# ============================================================

def ler_arquivo(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, sep=None, engine="python")
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Formato no reconocido. Use CSV, XLSX o XLS.")

def preparar_serie(df_original):
    if df_original.shape[1] < 2:
        raise ValueError("El archivo debe tener al menos dos columnas: fecha y caudal.")

    col_fecha = df_original.columns[0]
    col_q = df_original.columns[1]

    df = df_original.iloc[:, [0, 1]].copy()
    df.columns = ["data", "Q"]
    df["data"] = pd.to_datetime(df["data"], errors="coerce", dayfirst=True)
    df["Q"] = (
        df["Q"].astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df["Q"] = pd.to_numeric(df["Q"], errors="coerce")
    df = df.dropna(subset=["data", "Q"])
    df = df[df["Q"] >= 0].copy()
    df = df.sort_values("data")
    df = df.set_index("data")
    df = df.resample("D").mean()
    df["ano"] = df.index.year
    df["mes"] = df.index.month
    serie = df["Q"].dropna()
    if len(serie) == 0:
        raise ValueError("No quedaron datos válidos después de la limpieza.")
    return df, serie, col_fecha, col_q

def classificar_epoca(mes, epocas):
    for epoca, meses in epocas.items():
        if mes in meses:
            return epoca
    return "Sin clasificar"

# ============================================================
# MÉTODOS HIDROLÓGICOS
# ============================================================

def vazao_permanencia(q, permanencia):
    q = pd.Series(q).dropna()
    if len(q) == 0:
        return np.nan
    return np.percentile(q, 100 - permanencia)

def curva_permanencia(q):
    q = pd.Series(q).dropna().sort_values(ascending=False).reset_index(drop=True)
    n = len(q)
    if n == 0:
        return pd.DataFrame(columns=["Permanencia (%)", "Q (m3/s)"])
    p = 100 * np.arange(1, n + 1) / (n + 1)
    return pd.DataFrame({"Permanencia (%)": p, "Q (m3/s)": q})

def tabela_permanencias(q, grupo):
    return pd.DataFrame({
        "Grupo": grupo,
        "Permanencia (%)": PERMANENCIAS,
        "Q (m3/s)": [vazao_permanencia(q, p) for p in PERMANENCIAS]
    })

def medias_moveis_minimas_anuais(df_diario, janela, min_dias_ano=300):
    aux = df_diario[["Q"]].copy()
    aux[f"Q{janela}"] = aux["Q"].rolling(janela, min_periods=janela).mean()
    aux["Ano"] = aux.index.year
    resultados = []
    for ano, g in aux.groupby("Ano"):
        if g["Q"].count() >= min_dias_ano:
            resultados.append({"Ano": ano, f"Min_Q{janela}": g[f"Q{janela}"].min()})
    return pd.DataFrame(resultados).dropna()

def calcular_qjt(df_diario, janela=7, tempos_retorno=None, distribuicao="gumbel", min_dias_ano=300):
    if tempos_retorno is None:
        tempos_retorno = TEMPOS_RETORNO
    mins = medias_moveis_minimas_anuais(df_diario, janela, min_dias_ano)
    if len(mins) < 10:
        return pd.DataFrame([{
            "Método": f"Q{janela},T",
            "Distribución": distribuicao,
            "Tiempo de retorno": np.nan,
            "Q (m3/s)": np.nan,
            "Observación": "Serie corta para ajuste estadístico confiable"
        }]), mins
    x = mins[f"Min_Q{janela}"].dropna().values
    resultados = []
    for T in tempos_retorno:
        p = 1 / T
        if distribuicao == "gumbel":
            loc, scale = gumbel_r.fit(x)
            qjt = gumbel_r.ppf(p, loc=loc, scale=scale)
        elif distribuicao == "gev":
            c, loc, scale = genextreme.fit(x)
            qjt = genextreme.ppf(p, c, loc=loc, scale=scale)
        else:
            raise ValueError("Distribución no reconocida. Use gumbel o gev.")
        resultados.append({
            "Método": f"Q{janela},{T}",
            "Distribución": distribuicao,
            "Tiempo de retorno": T,
            "Q (m3/s)": max(qjt, 0),
            "Observación": "OK"
        })
    return pd.DataFrame(resultados), mins

def metodo_tennant(q_media):
    categorias = [
        ("Degradación elevada", 0.00, 0.10),
        ("Pobre / mínima", 0.10, 0.10),
        ("Débil / degradante", 0.30, 0.10),
        ("Buena", 0.40, 0.20),
        ("Muy buena", 0.50, 0.30),
        ("Excelente", 0.60, 0.40),
        ("Óptima - inferior", 0.60, 0.60),
        ("Óptima - superior", 1.00, 1.00),
        ("Lavado / máxima", 2.00, 2.00)
    ]
    return pd.DataFrame([{
        "Método": "Tennant/Montana",
        "Categoría": cat,
        "Percentual período 1": pct1,
        "Q período 1 (m3/s)": q_media * pct1,
        "Percentual período 2": pct2,
        "Q período 2 (m3/s)": q_media * pct2
    } for cat, pct1, pct2 in categorias])

def metodo_hoppe(df_diario):
    return pd.DataFrame([
        {"Método": "Hoppe simplificado", "Componente ecológico": "Actividades diarias", "Permanencia (%)": 80, "Q (m3/s)": vazao_permanencia(df_diario["Q"], 80)},
        {"Método": "Hoppe simplificado", "Componente ecológico": "Desove / reproducción", "Permanencia (%)": 40, "Q (m3/s)": vazao_permanencia(df_diario["Q"], 40)},
        {"Método": "Hoppe simplificado", "Componente ecológico": "Descarga / lavado del sustrato", "Permanencia (%)": 17, "Q (m3/s)": vazao_permanencia(df_diario["Q"], 17)}
    ])

def metodo_ngprp(df_diario):
    aux = df_diario.copy()
    aux["mes"] = aux.index.month
    mensal = df_diario["Q"].resample("M").mean().dropna()
    clim = mensal.groupby(mensal.index.month).mean()
    if len(clim) == 0:
        return pd.DataFrame()
    meses_altos = clim.sort_values(ascending=False).head(3).index.tolist()
    linhas = []
    for mes, g in aux.groupby("mes"):
        q50 = vazao_permanencia(g["Q"], 50)
        q90 = vazao_permanencia(g["Q"], 90)
        if mes in meses_altos:
            criterio = "Q50 mensual"
            qeco = q50
        else:
            criterio = "Q90 mensual"
            qeco = q90
        linhas.append({"Método": "NGPRP simplificado", "Mes": mes, "Criterio": criterio, "Q eco (m3/s)": qeco, "Q50": q50, "Q90": q90})
    return pd.DataFrame(linhas)

def metodo_abf(df_diario):
    mensal = df_diario["Q"].resample("M").mean().dropna()
    clim = mensal.groupby(mensal.index.month).mean()
    if len(clim) == 0:
        return np.nan, np.nan, pd.Series(dtype=float)
    mes_seco = clim.idxmin()
    q_abf = clim.min()
    return q_abf, mes_seco, clim

def filtro_lyne_hollick(q, alpha=0.925, passes=3):
    q = pd.Series(q).copy().dropna()
    if len(q) < 10:
        return pd.Series(index=q.index, dtype=float)
    def one_pass(flow):
        quick = np.zeros(len(flow))
        base = np.zeros(len(flow))
        values = flow.values
        quick[0] = 0
        base[0] = values[0]
        for i in range(1, len(values)):
            quick[i] = alpha * quick[i-1] + ((1 + alpha) / 2) * (values[i] - values[i-1])
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

def indicadores_iha_basicos(df_diario):
    q = df_diario["Q"].copy()
    aux = pd.DataFrame({"Q": q})
    aux["Ano"] = aux.index.year
    aux["Mes"] = aux.index.month
    mensais = aux.groupby(["Ano", "Mes"])["Q"].agg(Media="mean", Mediana="median", Minimo="min", Maximo="max").reset_index()
    extremos = []
    for janela in [1, 3, 7, 30, 90]:
        mov = q.rolling(janela, min_periods=janela).mean()
        tmp = pd.DataFrame({"valor": mov, "Ano": mov.index.year}).dropna()
        for ano, g in tmp.groupby("Ano"):
            extremos.append({"Ano": ano, "Janela dias": janela, "Minimo": g["valor"].min(), "Maximo": g["valor"].max()})
    extremos = pd.DataFrame(extremos)
    q25 = q.quantile(0.25)
    q75 = q.quantile(0.75)
    pulsos = []
    for ano, g in aux.groupby("Ano"):
        s = g["Q"].dropna()
        if len(s) < 30:
            continue
        alto = s > q75
        bajo = s < q25
        n_alto = ((alto) & (~alto.shift(1).fillna(False))).sum()
        n_bajo = ((bajo) & (~bajo.shift(1).fillna(False))).sum()
        dif = s.diff().dropna()
        subidas = dif[dif > 0]
        descidas = dif[dif < 0]
        sinal = np.sign(dif)
        reversoes = (sinal != sinal.shift(1)).sum()
        pulsos.append({
            "Ano": ano,
            "Limiar bajo Q25": q25,
            "Limiar alto Q75": q75,
            "Pulsos altos": n_alto,
            "Pulsos bajos": n_bajo,
            "Tasa media subida": subidas.mean() if len(subidas) > 0 else np.nan,
            "Tasa media descenso": descidas.mean() if len(descidas) > 0 else np.nan,
            "Reversiones": reversoes
        })
    return mensais, extremos, pd.DataFrame(pulsos)

# ============================================================
# PETTITT Y RVA
# ============================================================

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
    K = np.max(np.abs(Kt))
    indice_cambio = int(np.argmax(np.abs(Kt)))
    p_valor = 2 * np.exp((-6 * K**2) / (n**3 + n**2))
    return indice_cambio, K, min(max(p_valor, 0), 1)

def preparar_indicadores_anuais_para_pettitt(df_diario, min_dias_ano=300):
    dados = []
    for ano, g in df_diario.groupby(df_diario.index.year):
        s = g["Q"].dropna()
        if len(s) < min_dias_ano:
            continue
        dados.append({
            "Ano": ano,
            "Q medio anual": s.mean(),
            "Q mediano anual": s.median(),
            "Q90 anual": vazao_permanencia(s, 90),
            "Q95 anual": vazao_permanencia(s, 95),
            "Q minimo anual": s.min(),
            "Q maximo anual": s.max(),
            "Q7 minimo anual": s.rolling(7, min_periods=7).mean().min(),
            "Q30 minimo anual": s.rolling(30, min_periods=30).mean().min()
        })
    return pd.DataFrame(dados)

def aplicar_pettitt_indicadores(tab_indicadores):
    resultados = []
    if tab_indicadores is None or len(tab_indicadores) < 6:
        return pd.DataFrame()
    for indicador in tab_indicadores.columns:
        if indicador == "Ano":
            continue
        serie_ind = tab_indicadores[["Ano", indicador]].dropna()
        if len(serie_ind) < 6:
            continue
        indice, K, p = teste_pettitt(serie_ind[indicador])
        if indice is None:
            continue
        ano_cambio = int(serie_ind.iloc[indice]["Ano"])
        antes = serie_ind[serie_ind["Ano"] <= ano_cambio][indicador]
        depois = serie_ind[serie_ind["Ano"] > ano_cambio][indicador]
        media_antes = antes.mean()
        media_depois = depois.mean()
        cambio_relativo = ((media_depois - media_antes) / media_antes) * 100 if media_antes != 0 and not np.isnan(media_antes) else np.nan
        resultados.append({
            "Indicador": indicador,
            "Ano cambio Pettitt": ano_cambio,
            "Estadistico K": K,
            "p-valor": p,
            "Significativo p<0.05": "Sí" if p < 0.05 else "No",
            "Media antes": media_antes,
            "Media después": media_depois,
            "Cambio relativo (%)": cambio_relativo
        })
    return pd.DataFrame(resultados)

def sugerir_ano_pettitt(tab_pettitt):
    if tab_pettitt is None or len(tab_pettitt) == 0:
        return None, None, None, False
    candidato_qmedio = tab_pettitt[(tab_pettitt["Indicador"] == "Q medio anual") & (tab_pettitt["p-valor"] < 0.05)]
    if len(candidato_qmedio) > 0:
        linha = candidato_qmedio.iloc[0]
    else:
        significativos = tab_pettitt[tab_pettitt["p-valor"] < 0.05].copy()
        if len(significativos) > 0:
            linha = significativos.sort_values("p-valor").iloc[0]
        else:
            linha = tab_pettitt.sort_values("p-valor").iloc[0]
    return int(linha["Ano cambio Pettitt"]), linha["Indicador"], float(linha["p-valor"]), bool(float(linha["p-valor"]) < 0.05)

def rva_simplificado(df_diario, ano_corte, min_dias_ano=300):
    anos = sorted(df_diario.dropna(subset=["Q"]).index.year.unique())
    if len(anos) < 6:
        return pd.DataFrame(), ano_corte
    dados = []
    for ano, g in df_diario.groupby(df_diario.index.year):
        s = g["Q"].dropna()
        if len(s) < min_dias_ano:
            continue
        dados.append({
            "Ano": ano,
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
    ind = pd.DataFrame(dados)
    if len(ind) == 0 or ind["Periodo"].nunique() < 2:
        return pd.DataFrame(), ano_corte
    resultados = []
    for indicador in ["Q medio", "Q mediano", "Q minimo", "Q maximo", "Q7 minimo", "Q30 minimo", "Q90 anual", "Q10 anual"]:
        ref = ind[ind["Periodo"] == "Referencia"][indicador].dropna()
        alt = ind[ind["Periodo"] == "Alterado"][indicador].dropna()
        if len(ref) < 3 or len(alt) < 3:
            continue
        p25 = ref.quantile(0.25)
        p75 = ref.quantile(0.75)
        esperado = ((ref >= p25) & (ref <= p75)).mean()
        observado = ((alt >= p25) & (alt <= p75)).mean()
        alteracao = np.nan if esperado == 0 else ((observado - esperado) / esperado) * 100
        resultados.append({
            "Indicador": indicador,
            "Referencia P25": p25,
            "Referencia P75": p75,
            "Media referencia": ref.mean(),
            "Media alterado": alt.mean(),
            "Frecuencia esperada dentro RVA": esperado,
            "Frecuencia observada dentro RVA": observado,
            "Alteración hidrológica (%)": alteracao
        })
    return pd.DataFrame(resultados), ano_corte

# ============================================================
# CÁLCULOS PRINCIPALES
# ============================================================

def calcular_todo(df, serie, epocas, min_dias_ano):
    q_media = serie.mean()
    estatisticas = pd.DataFrame([{
        "n": serie.count(),
        "Q minimo": serie.min(),
        "Q medio": serie.mean(),
        "Q mediano": serie.median(),
        "Q maximo": serie.max(),
        "Desvio padrao": serie.std(),
        "Coeficiente variacion": serie.std() / serie.mean() if serie.mean() != 0 else np.nan,
        "P25": serie.quantile(0.25),
        "P75": serie.quantile(0.75),
        "P95": serie.quantile(0.95)
    }])
    tab_perm_global = tabela_permanencias(serie, "Serie completa")
    tab_perm_mensal = []
    for mes, g in df.groupby("mes"):
        temp = tabela_permanencias(g["Q"], f"Mes {mes}")
        temp["Mes"] = mes
        tab_perm_mensal.append(temp)
    tab_perm_mensal = pd.concat(tab_perm_mensal, ignore_index=True) if tab_perm_mensal else pd.DataFrame()
    tab_perm_epocas = []
    for epoca, g in df.groupby("Epoca"):
        if epoca == "Sin clasificar":
            continue
        temp = tabela_permanencias(g["Q"], epoca)
        temp["Epoca"] = epoca
        tab_perm_epocas.append(temp)
    tab_perm_epocas = pd.concat(tab_perm_epocas, ignore_index=True) if tab_perm_epocas else pd.DataFrame()
    tab_uruguay = []
    for mes, g in df.groupby("mes"):
        tab_uruguay.append({
            "Mes": mes,
            "Epoca": classificar_epoca(mes, epocas),
            "Q60 mensual (m3/s)": vazao_permanencia(g["Q"], 60),
            "Q80 mensual (m3/s)": vazao_permanencia(g["Q"], 80),
            "Q90 mensual (m3/s)": vazao_permanencia(g["Q"], 90)
        })
    tab_uruguay = pd.DataFrame(tab_uruguay)
    tab_tennant = metodo_tennant(q_media)
    tab_hoppe = metodo_hoppe(df)
    tab_ngprp = metodo_ngprp(df)
    q_abf, mes_abf, clim_mensal = metodo_abf(df)
    tab_abf = pd.DataFrame([{"Método": "ABF simplificado", "Mes más seco": mes_abf, "Q ABF (m3/s)": q_abf, "Observación": "Media histórica del mes más seco"}])
    tab_qjt_total = []
    for janela in [7, 30]:
        for dist in ["gumbel", "gev"]:
            tab_qjt, _ = calcular_qjt(df, janela=janela, tempos_retorno=TEMPOS_RETORNO, distribuicao=dist, min_dias_ano=min_dias_ano)
            tab_qjt_total.append(tab_qjt)
    tab_qjt_total = pd.concat(tab_qjt_total, ignore_index=True)
    base = filtro_lyne_hollick(serie, alpha=0.925, passes=3)
    df_base = pd.DataFrame({"Q": serie, "Q_base": base}).dropna()
    bfi_global = df_base["Q_base"].sum() / df_base["Q"].sum() if df_base["Q"].sum() != 0 else np.nan
    df_base["Ano"] = df_base.index.year
    df_base["Mes"] = df_base.index.month
    bfi_anual = df_base.groupby("Ano").apply(lambda x: x["Q_base"].sum() / x["Q"].sum() if x["Q"].sum() != 0 else np.nan).reset_index(name="BFI anual")
    bfi_mensal = df_base.groupby("Mes").apply(lambda x: x["Q_base"].sum() / x["Q"].sum() if x["Q"].sum() != 0 else np.nan).reset_index(name="BFI mensual")
    tab_bfi_global = pd.DataFrame([{"BFI global": bfi_global, "Interpretación": "Valores mayores sugieren mayor contribución relativa de caudal base"}])
    iha_mensais, iha_extremos, iha_pulsos = indicadores_iha_basicos(df)
    tab_indicadores_pettitt = preparar_indicadores_anuais_para_pettitt(df, min_dias_ano)
    tab_pettitt = aplicar_pettitt_indicadores(tab_indicadores_pettitt)
    ano_pettitt, indicador_pettitt, p_pettitt, sig_pettitt = sugerir_ano_pettitt(tab_pettitt)
    anos_disp = sorted(df.dropna(subset=["Q"]).index.year.unique())
    ano_medio = int(np.median(anos_disp)) if anos_disp else None
    fdc = curva_permanencia(serie)
    return {
        "q_media": q_media, "estatisticas": estatisticas,
        "tab_perm_global": tab_perm_global, "tab_perm_mensal": tab_perm_mensal, "tab_perm_epocas": tab_perm_epocas,
        "tab_uruguay": tab_uruguay, "tab_tennant": tab_tennant, "tab_hoppe": tab_hoppe, "tab_ngprp": tab_ngprp,
        "q_abf": q_abf, "mes_abf": mes_abf, "clim_mensal": clim_mensal, "tab_abf": tab_abf,
        "tab_qjt_total": tab_qjt_total, "df_base": df_base, "tab_bfi_global": tab_bfi_global,
        "bfi_anual": bfi_anual, "bfi_mensal": bfi_mensal,
        "iha_mensais": iha_mensais, "iha_extremos": iha_extremos, "iha_pulsos": iha_pulsos,
        "tab_indicadores_pettitt": tab_indicadores_pettitt, "tab_pettitt": tab_pettitt,
        "ano_pettitt": ano_pettitt, "indicador_pettitt": indicador_pettitt, "p_pettitt": p_pettitt, "sig_pettitt": sig_pettitt,
        "ano_medio": ano_medio, "fdc": fdc
    }

def construir_resumo(serie, q_media, tab_uruguay, q_abf, tab_qjt_total):
    resumo = pd.DataFrame([
        {"Grupo": "Permanencia global", "Método": "Q90 global", "Q eco (m3/s)": vazao_permanencia(serie, 90), "Lectura didáctica": "Caudal igualado o excedido 90% del tiempo"},
        {"Grupo": "Permanencia global", "Método": "Q95 global", "Q eco (m3/s)": vazao_permanencia(serie, 95), "Lectura didáctica": "Condición de baja disponibilidad más restrictiva"},
        {"Grupo": "Mensual", "Método": "Q80 mensual medio", "Q eco (m3/s)": tab_uruguay["Q80 mensual (m3/s)"].mean(), "Lectura didáctica": "Referencia mensual intermedia"},
        {"Grupo": "Mensual", "Método": "Q60 mensual medio", "Q eco (m3/s)": tab_uruguay["Q60 mensual (m3/s)"].mean(), "Lectura didáctica": "Referencia mensual menos restrictiva"},
        {"Grupo": "Tennant", "Método": "Tennant 10% Qmedio", "Q eco (m3/s)": q_media * 0.10, "Lectura didáctica": "Condición mínima o pobre"},
        {"Grupo": "Tennant", "Método": "Tennant 30% Qmedio", "Q eco (m3/s)": q_media * 0.30, "Lectura didáctica": "Condición intermedia"},
        {"Grupo": "Tennant", "Método": "Tennant 60% Qmedio", "Q eco (m3/s)": q_media * 0.60, "Lectura didáctica": "Condición más conservadora"},
        {"Grupo": "ABF", "Método": "ABF simplificado", "Q eco (m3/s)": q_abf, "Lectura didáctica": "Media histórica del mes más seco"},
        {"Grupo": "Hoppe", "Método": "Hoppe Q80", "Q eco (m3/s)": vazao_permanencia(serie, 80), "Lectura didáctica": "Actividades diarias de organismos"},
        {"Grupo": "Hoppe", "Método": "Hoppe Q40", "Q eco (m3/s)": vazao_permanencia(serie, 40), "Lectura didáctica": "Desove o reproducción"},
        {"Grupo": "Hoppe", "Método": "Hoppe Q17", "Q eco (m3/s)": vazao_permanencia(serie, 17), "Lectura didáctica": "Descarga o lavado de sustrato"}
    ])
    q710 = tab_qjt_total[(tab_qjt_total["Método"] == "Q7,10") & (tab_qjt_total["Distribución"] == "gumbel")]
    if len(q710) > 0 and not np.isnan(q710.iloc[0]["Q (m3/s)"]):
        resumo = pd.concat([resumo, pd.DataFrame([{"Grupo": "Mínimos móviles", "Método": "Q7,10 Gumbel", "Q eco (m3/s)": q710.iloc[0]["Q (m3/s)"], "Lectura didáctica": "Mínima media de 7 días con período de retorno de 10 años"}])], ignore_index=True)
    return resumo

def exportar_excel(info_local, info_serie, resultados, resumo, guia, tab_rva, criterio_ano_corte):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        info_local.to_excel(writer, sheet_name="00_info_local", index=False)
        info_serie.to_excel(writer, sheet_name="01_info_serie", index=False)
        pd.DataFrame([{"Criterio año corte RVA": criterio_ano_corte}]).to_excel(writer, sheet_name="01b_criterio_rva", index=False)
        resultados["estatisticas"].to_excel(writer, sheet_name="02_estadisticas", index=False)
        resultados["tab_perm_global"].to_excel(writer, sheet_name="03_perm_global", index=False)
        resultados["tab_perm_mensal"].to_excel(writer, sheet_name="04_perm_mensual", index=False)
        resultados["tab_perm_epocas"].to_excel(writer, sheet_name="05_perm_epocas", index=False)
        resultados["tab_uruguay"].to_excel(writer, sheet_name="06_q60_q80_q90", index=False)
        resultados["tab_tennant"].to_excel(writer, sheet_name="07_tennant", index=False)
        resultados["tab_hoppe"].to_excel(writer, sheet_name="08_hoppe", index=False)
        resultados["tab_ngprp"].to_excel(writer, sheet_name="09_ngprp", index=False)
        resultados["tab_abf"].to_excel(writer, sheet_name="10_abf", index=False)
        resultados["tab_qjt_total"].to_excel(writer, sheet_name="11_qjt", index=False)
        resultados["bfi_anual"].to_excel(writer, sheet_name="12_bfi_anual", index=False)
        resultados["bfi_mensal"].to_excel(writer, sheet_name="13_bfi_mensual", index=False)
        resultados["iha_mensais"].to_excel(writer, sheet_name="14_iha_mensual", index=False)
        resultados["iha_extremos"].to_excel(writer, sheet_name="15_iha_extremos", index=False)
        resultados["iha_pulsos"].to_excel(writer, sheet_name="16_iha_pulsos", index=False)
        resultados["tab_indicadores_pettitt"].to_excel(writer, sheet_name="17_ind_pettitt", index=False)
        resultados["tab_pettitt"].to_excel(writer, sheet_name="18_pettitt", index=False)
        tab_rva.to_excel(writer, sheet_name="19_rva", index=False)
        resumo.to_excel(writer, sheet_name="20_resumen", index=False)
        guia.to_excel(writer, sheet_name="21_guia", index=False)
        resultados["fdc"].to_excel(writer, sheet_name="22_curva_perm", index=False)
    output.seek(0)
    return output

# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

st.title("💧 Material didáctico: caudal ecológico / ambiental")
st.caption("Aplicativo para comparar metodologías hidrológicas a partir de una serie diaria de caudales.")

with st.expander("📘 Base teórica y alcance", expanded=True):
    st.markdown("""
El caudal ecológico no debe entenderse solo como un número fijo. Los ecosistemas acuáticos dependen de un
**régimen de caudales**, con variación mensual, épocas del año, pulsos, mínimos, máximos, caudal base y posibles
cambios en el tiempo.

Este aplicativo trabaja principalmente con **métodos hidrológicos**, porque el dato obligatorio es una serie de caudales.
Los resultados son referencias comparativas y didácticas; no sustituyen una evaluación ecológica completa.
""")

st.sidebar.header("1. Información del local")
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

st.sidebar.header("2. Parámetros")
min_dias_ano = st.sidebar.number_input("Mínimo de días válidos por año", min_value=100, max_value=366, value=300, step=10)

st.sidebar.markdown("### División en tres épocas")
epoca1 = st.sidebar.multiselect("Época 1 - Estiaje / mayor demanda", options=list(range(1, 13)), default=[12, 1, 2, 3])
epoca2 = st.sidebar.multiselect("Época 2 - Transición", options=list(range(1, 13)), default=[4, 5, 10, 11])
epoca3 = st.sidebar.multiselect("Época 3 - Aguas altas / menor demanda", options=list(range(1, 13)), default=[6, 7, 8, 9])

epocas = {
    "Época 1 - Estiaje / mayor demanda": epoca1,
    "Época 2 - Transición": epoca2,
    "Época 3 - Aguas altas / menor demanda": epoca3
}

uploaded_file = st.file_uploader("Cargue un archivo CSV, XLSX o XLS. Primera columna = fecha; segunda columna = caudal.", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("Cargue una serie diaria de caudales para iniciar el análisis.")
    st.stop()

try:
    df_original = ler_arquivo(uploaded_file)
    df, serie, col_fecha, col_q = preparar_serie(df_original)
except Exception as e:
    st.error(f"Error al leer o preparar el archivo: {e}")
    st.stop()

df["Epoca"] = df["mes"].apply(lambda m: classificar_epoca(m, epocas))

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

st.success(f"Archivo leído correctamente. Fecha = {col_fecha}; caudal = {col_q}.")

with st.expander("Vista previa y calidad de la serie", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        mostrar_df(df_original.head(20), "Vista previa del archivo")
    with c2:
        mostrar_df(info_serie, "Información básica de la serie")

with st.spinner("Calculando metodologías..."):
    resultados = calcular_todo(df, serie, epocas, min_dias_ano)

# ============================================================
# PETTITT + RVA
# ============================================================

st.header("Test de Pettitt y selección del año de corte para RVA")
st.markdown("""
El test de Pettitt se aplica sobre indicadores anuales para detectar un posible año de cambio.
El año sugerido puede usarse como corte para comparar un período de referencia con un período alterado.
""")

mostrar_df(resultados["tab_indicadores_pettitt"], "Indicadores anuales usados para Pettitt", height=300)
mostrar_df(resultados["tab_pettitt"], "Resultados del test de Pettitt", height=300)

ano_medio = resultados["ano_medio"]
ano_pettitt = resultados["ano_pettitt"]
indicador_pettitt = resultados["indicador_pettitt"]
p_pettitt = resultados["p_pettitt"]
sig_pettitt = resultados["sig_pettitt"]

if ano_pettitt is not None:
    st.info(
        f"Pettitt sugiere el año {ano_pettitt}, asociado al indicador {indicador_pettitt}. "
        f"p-valor = {p_pettitt:.4f}. Significativo al 5%: {'sí' if sig_pettitt else 'no'}."
    )
else:
    st.warning("No fue posible sugerir un año de cambio mediante Pettitt.")

anos_validos = sorted(df.dropna(subset=["Q"]).index.year.unique())
min_ano = int(min(anos_validos))
max_ano = int(max(anos_validos))

opciones_corte = ["Usar año medio de la serie"]
if ano_pettitt is not None:
    opciones_corte.insert(0, "Usar año sugerido por Pettitt")
opciones_corte.append("Indicar otro año manualmente")

opcion_corte = st.radio("Seleccione el criterio para el RVA", opciones_corte)

if opcion_corte == "Usar año sugerido por Pettitt" and ano_pettitt is not None:
    ano_corte = int(ano_pettitt)
    criterio_ano_corte = f"Pettitt: año sugerido por el indicador {indicador_pettitt}"
elif opcion_corte == "Indicar otro año manualmente":
    ano_corte = st.number_input("Año de corte manual", min_value=min_ano, max_value=max_ano, value=int(ano_medio), step=1)
    criterio_ano_corte = "Manual: año definido por el usuario"
else:
    ano_corte = int(ano_medio)
    criterio_ano_corte = "Automático: año medio de la serie"

st.markdown(f"""
**Año de corte usado para RVA:** {ano_corte}  
**Criterio de selección:** {criterio_ano_corte}

- Período de referencia: años ≤ {ano_corte}
- Período alterado: años > {ano_corte}
""")

tab_rva, _ = rva_simplificado(df, int(ano_corte), min_dias_ano=min_dias_ano)
mostrar_df(tab_rva, "Resultados del RVA simplificado")

if len(resultados["tab_indicadores_pettitt"]) > 0:
    indicador_grafico = st.selectbox("Indicador para visualizar el año de corte", [c for c in resultados["tab_indicadores_pettitt"].columns if c != "Ano"], index=0)
    serie_ind = resultados["tab_indicadores_pettitt"][["Ano", indicador_grafico]].dropna()
    media_antes = serie_ind[serie_ind["Ano"] <= ano_corte][indicador_grafico].mean()
    media_depois = serie_ind[serie_ind["Ano"] > ano_corte][indicador_grafico].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(serie_ind["Ano"], serie_ind[indicador_grafico], marker="o", label=indicador_grafico)
    ax.axvline(ano_corte, linestyle="--", label=f"Año de corte: {ano_corte}")
    ax.axhline(media_antes, linestyle=":", label="Media antes del corte")
    ax.axhline(media_depois, linestyle=":", label="Media después del corte")
    ax.set_xlabel("Año")
    ax.set_ylabel(indicador_grafico)
    ax.set_title(f"Indicador anual y año de corte - {indicador_grafico}")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# ============================================================
# RESULTADOS
# ============================================================

resumo = construir_resumo(serie, resultados["q_media"], resultados["tab_uruguay"], resultados["q_abf"], resultados["tab_qjt_total"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Resumen", "Permanencias", "Métodos clásicos", "BFI / IHA", "Gráficos"])

with tab1:
    st.header("Resumen comparativo general")
    st.markdown("""
Esta tabla reúne varios métodos en una misma escala. Los valores más bajos suelen representar criterios
más restrictivos o condiciones críticas; los valores más altos tienden a mantener más agua en el río.
""")
    mostrar_df(resultados["estatisticas"], "Estadísticas descriptivas")
    mostrar_df(resumo, "Resumen de métodos")

with tab2:
    st.header("Curva de permanencia y referencias mensuales")
    mostrar_df(resultados["tab_perm_global"], "Permanencias globales")
    mostrar_df(resultados["tab_perm_mensal"], "Permanencias mensuales", height=350)
    mostrar_df(resultados["tab_perm_epocas"], "Permanencias por tres épocas", height=350)
    mostrar_df(resultados["tab_uruguay"], "Q60, Q80 y Q90 mensual")

with tab3:
    st.header("Métodos clásicos")
    mostrar_df(resultados["tab_tennant"], "Tennant / Montana")
    mostrar_df(resultados["tab_hoppe"], "Hoppe simplificado")
    mostrar_df(resultados["tab_ngprp"], "NGPRP simplificado")
    mostrar_df(resultados["tab_abf"], "ABF simplificado")
    mostrar_df(resultados["tab_qjt_total"], "Qj,T")

with tab4:
    st.header("Caudal base, IHA y RVA")
    mostrar_df(resultados["tab_bfi_global"], "BFI global")
    mostrar_df(resultados["bfi_anual"], "BFI anual")
    mostrar_df(resultados["bfi_mensal"], "BFI mensual")
    mostrar_df(resultados["iha_mensais"].head(100), "IHA mensual", height=350)
    mostrar_df(resultados["iha_extremos"].head(100), "IHA extremos", height=350)
    mostrar_df(resultados["iha_pulsos"].head(100), "IHA pulsos", height=350)
    mostrar_df(tab_rva, "RVA simplificado")

with tab5:
    st.header("Gráficos didácticos")
    st.subheader("Serie temporal diaria")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(serie.index, serie.values, linewidth=0.7)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Caudal (m3/s)")
    ax.set_title("Serie temporal diaria de caudal")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Serie con referencias ecológicas")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(serie.index, serie.values, linewidth=0.6, label="Caudal observado")
    ax.axhline(vazao_permanencia(serie, 90), linestyle="--", label="Q90 global")
    ax.axhline(vazao_permanencia(serie, 95), linestyle="--", label="Q95 global")
    ax.axhline(resultados["q_media"] * 0.30, linestyle="--", label="Tennant 30%")
    ax.axhline(resultados["q_abf"], linestyle="--", label="ABF")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Caudal (m3/s)")
    ax.set_title("Serie de caudal con referencias de caudal ecológico")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Curva de permanencia")
    fdc = resultados["fdc"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(fdc["Permanencia (%)"], fdc["Q (m3/s)"], linewidth=1.5)
    ax.axvline(60, linestyle="--", label="Q60")
    ax.axvline(80, linestyle="--", label="Q80")
    ax.axvline(90, linestyle="--", label="Q90")
    ax.axvline(95, linestyle="--", label="Q95")
    ax.set_xlabel("Permanencia / excedencia (%)")
    ax.set_ylabel("Caudal (m3/s)")
    ax.set_title("Curva de permanencia global")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Comparación de metodologías")
    resumo_plot = resumo.dropna(subset=["Q eco (m3/s)"]).sort_values("Q eco (m3/s)")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(resumo_plot["Método"], resumo_plot["Q eco (m3/s)"])
    ax.set_xlabel("Caudal estimado (m3/s)")
    ax.set_ylabel("Método")
    ax.set_title("Comparación de metodologías de caudal ecológico")
    ax.grid(axis="x")
    st.pyplot(fig)

    st.subheader("Referencias mensuales Q60, Q80 y Q90")
    tab_u = resultados["tab_uruguay"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tab_u["Mes"], tab_u["Q60 mensual (m3/s)"], marker="o", label="Q60 mensual")
    ax.plot(tab_u["Mes"], tab_u["Q80 mensual (m3/s)"], marker="o", label="Q80 mensual")
    ax.plot(tab_u["Mes"], tab_u["Q90 mensual (m3/s)"], marker="o", label="Q90 mensual")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Caudal (m3/s)")
    ax.set_title("Referencias mensuales de caudal ambiental")
    ax.set_xticks(range(1, 13))
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Distribución mensual de caudales")
    df_box = df.dropna(subset=["Q"]).copy()
    df_box["Mes"] = df_box.index.month
    dados_box = [df_box[df_box["Mes"] == m]["Q"] for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(dados_box, labels=range(1, 13), showfliers=False)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Caudal (m3/s)")
    ax.set_title("Distribución mensual de caudales")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Hidrograma medio mensual")
    clim_mensal = resultados["clim_mensal"]
    clim_mensal_df = pd.DataFrame({"Mes": clim_mensal.index, "Q media mensual": clim_mensal.values})
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(clim_mensal_df["Mes"], clim_mensal_df["Q media mensual"], marker="o")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Caudal medio mensual (m3/s)")
    ax.set_title("Hidrograma medio mensual")
    ax.set_xticks(range(1, 13))
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Caudal base estimado")
    df_base = resultados["df_base"]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_base.index, df_base["Q"], linewidth=0.6, label="Caudal observado")
    ax.plot(df_base.index, df_base["Q_base"], linewidth=0.8, label="Caudal base estimado")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Caudal (m3/s)")
    ax.set_title("Separación aproximada de caudal base")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("BFI anual")
    bfi_anual = resultados["bfi_anual"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bfi_anual["Ano"], bfi_anual["BFI anual"], marker="o")
    ax.set_xlabel("Año")
    ax.set_ylabel("BFI")
    ax.set_title("Índice de caudal base anual")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("IHA: mínimos y máximos anuales de 7 días")
    ext7 = resultados["iha_extremos"][resultados["iha_extremos"]["Janela dias"] == 7].copy()
    if len(ext7) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ext7["Ano"], ext7["Minimo"], marker="o", label="Mínimo 7 días")
        ax.plot(ext7["Ano"], ext7["Maximo"], marker="o", label="Máximo 7 días")
        ax.set_xlabel("Año")
        ax.set_ylabel("Caudal (m3/s)")
        ax.set_title("IHA: mínimos y máximos anuales de 7 días")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    st.subheader("RVA simplificado")
    if len(tab_rva) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(tab_rva["Indicador"], tab_rva["Alteración hidrológica (%)"])
        ax.set_xlabel("Alteración hidrológica (%)")
        ax.set_ylabel("Indicador")
        ax.set_title("RVA simplificado")
        ax.grid(axis="x")
        st.pyplot(fig)
    else:
        st.info("No fue posible graficar RVA.")

# ============================================================
# EXPORTACIÓN
# ============================================================

st.header("Exportación")

guia = pd.DataFrame([
    {"Pregunta didáctica": "¿Qué métodos generan valores más bajos?", "Qué observar": "Q95, Q7,10, Q30,10", "Interpretación": "Representan condiciones críticas o de baja disponibilidad."},
    {"Pregunta didáctica": "¿Qué métodos generan valores intermedios?", "Qué observar": "Q80, Q90, Tennant 30%, ABF", "Interpretación": "Pueden funcionar como referencias operativas o preliminares."},
    {"Pregunta didáctica": "¿Qué métodos mantienen más agua en el río?", "Qué observar": "Tennant 40%, Tennant 60%, Hoppe Q40, Hoppe Q17", "Interpretación": "Son más conservadores desde el punto de vista ecológico, pero restringen más los usos."},
    {"Pregunta didáctica": "¿Qué aporta Pettitt?", "Qué observar": "Año de cambio y p-valor", "Interpretación": "Ayuda a identificar un posible punto de alteración hidrológica."},
    {"Pregunta didáctica": "¿Qué aportan IHA y RVA?", "Qué observar": "Pulsos, mínimos, máximos, tasas y alteración hidrológica", "Interpretación": "Permiten analizar el régimen hidrológico, no solamente un caudal mínimo."}
])

excel_buffer = exportar_excel(info_local, info_serie, resultados, resumo, guia, tab_rva, criterio_ano_corte)
download_excel_button(excel_buffer, "resultados_caudal_ecologico_streamlit.xlsx")
