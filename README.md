# Aplicativo Streamlit — Caudal ecológico / ambiental

Aplicativo didáctico para calcular y comparar metodologías hidrológicas de caudal ecológico/ambiental a partir de una serie diaria de caudales.

## Qué calcula

- Curva de permanencia global, mensual y por tres épocas del año.
- Q60, Q80 y Q90 mensual.
- Tennant / Montana.
- Hoppe simplificado.
- NGPRP simplificado.
- ABF simplificado.
- Q7,T y Q30,T con distribuciones Gumbel y GEV.
- Separación aproximada de caudal base y BFI.
- Indicadores IHA básicos.
- Test de Pettitt para sugerir año de cambio.
- RVA simplificado con año de corte por Pettitt, año manual o año medio.
- Exportación de resultados a Excel.

## Formato de los datos

El archivo debe ser `.csv`, `.xlsx` o `.xls`.

La primera fila debe tener los nombres de columnas. El nombre puede ser cualquiera.

El aplicativo asume automáticamente:

| Columna | Contenido |
|---|---|
| Primera columna | Fecha |
| Segunda columna | Caudal |

Ejemplo:

```csv
Fecha,Caudal
01/01/2000,12.5
02/01/2000,11.8
03/01/2000,10.9
```

## Ejecución local

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar:

```bash
streamlit run app.py
```

## Subir a GitHub y conectar con Streamlit Community Cloud

1. Crear un repositorio nuevo en GitHub.
2. Subir estos archivos:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `.streamlit/config.toml`
   - opcionalmente `data/ejemplo_caudal_diario.csv`
3. Entrar a Streamlit Community Cloud.
4. Crear una nueva app desde el repositorio.
5. Seleccionar `app.py` como archivo principal.

## Observación técnica

Los resultados son referencias hidrológicas comparativas y didácticas. No sustituyen una evaluación ecológica completa, especialmente en casos con obras hidráulicas, captaciones relevantes, ecosistemas sensibles o conflictos de uso.
