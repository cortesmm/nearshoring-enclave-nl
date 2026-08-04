import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def procesar_censos_inegi(rutas_archivos):
    """
    Lee, limpia y consolida múltiples archivos CSV del SAIC (INEGI),
    aplicando el marco teórico de Cadenas Globales de Valor y Spillovers.
    """
    lista_dfs = []
    
    # 1. LECTURA, LIMPIEZA INICIAL Y RENOMBRADO INDIVIDUAL
    # 1. LECTURA, LIMPIEZA INICIAL Y RENOMBRADO INDIVIDUAL
    for ruta in rutas_archivos:
        print(f"🔄 Leyendo y estandarizando archivo: {ruta}")
        df = pd.read_csv(ruta, skiprows=4, encoding='utf-8')
        
        # ELIMINAR espacios invisibles en los nombres de las columnas
        df.columns = df.columns.str.strip()
        
        # Diccionario de renombrado aplicado ANTES de juntar
        mapping = {}
        for col in df.columns:
            c = str(col).lower()
            if 'año' in c or 'anio' in c: mapping[col] = 'anio'
            elif 'entidad' in c: mapping[col] = 'entidad'
            elif 'municipio' in c: mapping[col] = 'municipio'
            elif 'estrato' in c: mapping[col] = 'estrato'
            elif 'actividad' in c: mapping[col] = 'actividad'
            elif 'ue ' in c or 'unidades' in c: mapping[col] = 'ue'
            elif 'h001a' in c: mapping[col] = 'po'
            elif 'h010a' in c: mapping[col] = 'po_rem'
            elif 'a131a' in c: mapping[col] = 'va'
            elif 'a111a' in c: mapping[col] = 'pb'
            elif 'm000a' in c: mapping[col] = 'ing'
            elif 'j000a' in c: mapping[col] = 'rem'
            elif 'a211a' in c: mapping[col] = 'inv'
            elif 'a221a' in c: mapping[col] = 'fbcf'
            elif 'q000a' in c: mapping[col] = 'act_fij'
            elif 'a121a' in c: mapping[col] = 'cons_int'

        df.rename(columns=mapping, inplace=True)
        
        # ==========================================
        # 🛡️ EL BLINDAJE ETL (NUEVO)
        # ==========================================
        # A) Eliminar columnas duplicadas (nos quedamos con la primera aparición)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # B) Filtrar y conservar SOLO las columnas de nuestro modelo
        columnas_teoricas = ['anio', 'entidad', 'municipio', 'estrato', 'actividad', 
                             'ue', 'po', 'po_rem', 'va', 'pb', 'ing', 'rem', 
                             'inv', 'fbcf', 'act_fij', 'cons_int']
        
        # Hacemos una intersección para evitar errores si alguna columna falta
        cols_a_conservar = [c for c in columnas_teoricas if c in df.columns]
        df = df[cols_a_conservar]
        # ==========================================

        lista_dfs.append(df)
        
    # AHORA SÍ: Pandas junta los archivos ya filtrados y libres de duplicados
    df_raw = pd.concat(lista_dfs, ignore_index=True)
    
    # 2. LIMPIEZA DE FILAS (Eliminar totales estatales y vacíos)
    # ... (EL CÓDIGO SIGUE IGUAL A PARTIR DE AQUÍ) ...
    
    # 2. LIMPIEZA DE FILAS (Eliminar totales estatales y vacíos)
    df_raw = df_raw.dropna(subset=['municipio']) # Aseguramos minúsculas
    df_raw = df_raw[df_raw['municipio'] != ""]
    df_raw = df_raw[~df_raw['actividad'].str.contains("Total|Todos", na=False, case=False)]
    
    # 4. EXTRACCIÓN DE CLAVES (Regex para cve_geo y cve_act)
    # Agregamos expand=False para que Pandas devuelva una Serie y no un DataFrame
    df_raw['cve_ent'] = df_raw['entidad'].str.extract(r'(\d+)', expand=False).astype(str).str.zfill(2)
    df_raw['cve_mun'] = df_raw['municipio'].str.extract(r'(\d+)', expand=False).astype(str).str.zfill(3)
    df_raw['cve_geo'] = df_raw['cve_ent'] + df_raw['cve_mun']
    
    # Limpiamos el nombre del municipio quitando los números
    df_raw['nom_mun'] = df_raw['municipio'].str.replace(r'\d+', '', regex=True).str.strip()
    
    # Lo mismo para la clave de actividad
    df_raw['cve_act'] = df_raw['actividad'].str.extract(r'(\d+)', expand=False).astype(str)
    
    # Asegurar tipos numéricos y rellenar nulos con 0
    vars_num = ['ue', 'po', 'po_rem', 'va', 'pb', 'ing', 'rem', 'inv', 'fbcf', 'act_fij', 'cons_int']
    for var in vars_num:
        df_raw[var] = pd.to_numeric(df_raw[var], errors='coerce').fillna(0)
        
    # 5. CLASIFICACIÓN TRACTORA vs PYME
    # Si el estrato contiene "251 y más", es tractora. Todo lo demás es PyME.
    df_raw['tipo_emp'] = np.where(df_raw['estrato'].str.contains('251'), 'tractora', 'pyme')
    
    # 6. AGRUPACIÓN MUNICIPAL POR SECTOR, TAMAÑO Y AÑO
    print("⚙️ Agrupando datos a nivel municipal y calculando suma de variables base...")
    df_agrupado = df_raw.groupby(['cve_geo', 'nom_mun', 'cve_act', 'tipo_emp', 'anio'])[vars_num].sum().reset_index()
    
    # 7. CONSTRUCCIÓN DE INDICADORES TEÓRICOS (Post-Agrupación para precisión matemática)
    print("🧠 Inyectando marco teórico (Dussel, Lascurain, Gereffi, Montaño)...")
    
    po_safe = df_agrupado['po'].replace(0, np.nan)
    pb_safe = df_agrupado['pb'].replace(0, np.nan)
    
    # Dussel: Ratio de Valor Agregado (Integración Local)
    df_agrupado['va_ratio'] = df_agrupado['va'] / pb_safe
    
    # Lascurain: Capacidad de Absorción (Intensidad de Capital)
    df_agrupado['ac_cap'] = df_agrupado['act_fij'] / po_safe
    
    # Gereffi: Ascenso Industrial (Productividad Marginal)
    df_agrupado['prod_trab'] = df_agrupado['va'] / po_safe
    
    # Montaño: Modernización Laboral (Tasa de Formalidad y Salario Medio)
    df_agrupado['tasa_formal'] = df_agrupado['po_rem'] / po_safe
    df_agrupado['sal_med'] = df_agrupado['rem'] / po_safe
    
    # Gereffi: Índice Complejo de Ascenso Industrial (UI = ln(Prod) + ln(Cap))
    # Protegemos logaritmos de números negativos o ceros
    prod_log = df_agrupado['prod_trab'].apply(lambda x: np.log(x) if x > 0 else 0)
    cap_log = df_agrupado['ac_cap'].apply(lambda x: np.log(x) if x > 0 else 0)
    df_agrupado['ui_index'] = prod_log + cap_log
    
    # Rellenar nans generados por divisiones
    df_agrupado.fillna(0, inplace=True)
    
    # 8. PIVOT: CREACIÓN DE LA BASE ANCHA
    print("🚀 Pivotando matriz (Base Ancha para Econometría Espacial)...")
    
    # Variables que queremos en nuestra tabla final
    vars_pivot = vars_num + ['va_ratio', 'ac_cap', 'prod_trab', 'tasa_formal', 'sal_med', 'ui_index']
    
    base_ancha = pd.pivot_table(
        df_agrupado,
        index=['cve_geo', 'nom_mun'],
        columns=['cve_act', 'tipo_emp', 'anio'],
        values=vars_pivot,
        fill_value=0
    )
    
    # Aplanar las columnas multinivel
    # El formato resultante será: variable_sector_tipo_año (Ej: po_336_tractora_2023)
    base_ancha.columns = [f'{col[0]}_{col[1]}_{col[2]}_{col[3]}' for col in base_ancha.columns]
    base_ancha = base_ancha.reset_index()
    
    print(f"✅ ¡Éxito! Base ancha final creada con {base_ancha.shape[0]} municipios y {base_ancha.shape[1]} columnas.")
    return base_ancha

# ==========================================
# BLOQUE DE EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    # Coloca aquí los nombres exactos de tus 3 descargas
    archivos_censales = [
        "SAIC_Exporta_2026730_213547665.csv", # 2013
        "SAIC_Exporta_2026730_212844129.csv", # 2018
        "SAIC_Exporta_2026730_212013903.csv"  # 2023
    ]
    
    # Ejecutar la magia
    df_final = procesar_censos_inegi(archivos_censales)
    
    # Exportar el resultado para consumirlo en el siguiente script (EDA)
    df_final.to_csv("Base_Ancha_NL_Nearshoring.csv", index=False, encoding='utf-8-sig')
    print("💾 Archivo 'Base_Ancha_NL_Nearshoring.csv' guardado correctamente.")
