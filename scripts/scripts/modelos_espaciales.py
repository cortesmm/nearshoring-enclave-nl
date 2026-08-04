import pandas as pd
import geopandas as gpd
import numpy as np
import libpysal
from spreg import OLS, ML_Lag, ML_Error
import warnings

warnings.filterwarnings('ignore')

class RegresionEspacialNearshoring:
    """
    Clase para la estimación de modelos de Regresión Espacial (OLS, SAR y SEM)
    que evalúa la causalidad del Nearshoring sobre el Upgrading de las PyMEs.
    """
    
    def __init__(self, ruta_csv, ruta_shp):
        self.ruta_csv = ruta_csv
        self.ruta_shp = ruta_shp
        self.gdf = self._cargar_y_unir_datos()
        self.w = self._crear_matriz_w()

    def _cargar_y_unir_datos(self):
        """Carga y une espacialmente los datos tabulares y cartográficos."""
        print("🌍 Cargando base ancha y mapa geográfico...")
        df = pd.read_csv(self.ruta_csv, dtype={'cve_geo': str})
        df['cve_geo'] = df['cve_geo'].str.zfill(5)
        
        mapa = gpd.read_file(self.ruta_shp)
        mapa['CVEGEO'] = mapa['CVEGEO'].astype(str).str.zfill(5)
        
        gdf = mapa.merge(df, left_on='CVEGEO', right_on='cve_geo', how='inner')
        print(f"✅ Unificación espacial completada: {gdf.shape[0]} municipios procesados.")
        return gdf

    def _crear_matriz_w(self):
        """Genera la Matriz de Pesos Espaciales normalizada por filas."""
        w = libpysal.weights.Queen.from_dataframe(self.gdf)
        w.transform = 'r'
        return w

    def estimar_modelos(self, var_dep, vars_indep):
        """
        Ejecuta MCO, evaluando pruebas LM, y posteriormente ajusta SAR y SEM.
        """
        print("\n" + "="*80)
        print(" 🎯 ESTIMACIÓN ECONOMÉTRICA ESPACIAL MULTIVARIADA")
        print("="*80)
        
        # Preparación de vectores N-dimensionales requeridos por spreg
        y = self.gdf[[var_dep]].values
        X = self.gdf[vars_indep].values
        
        nombres_x = vars_indep
        nombre_y = var_dep

        # ------------------------------------------------------------------
        # 1. MODELO MÍNIMOS CUADRADOS ORDINARIOS (MCO / OLS) + DIAGNÓSTICOS
        # ------------------------------------------------------------------
        print("\n🔹 [1/3] Estimando Modelo OLS (Base) y Diagnósticos de Lag/Error...")
        ols = OLS(y, X, w=self.w, name_y=nombre_y, name_x=nombres_x, spat_diag=True)
        print(ols.summary)

        # ------------------------------------------------------------------
        # 2. MODELO DE REZAGO ESPACIAL (SAR - Spatial Lag)
        # ------------------------------------------------------------------
        print("\n🔹 [2/3] Estimando Modelo SAR (Spatial Lag - ML_Lag)...")
        sar = ML_Lag(y, X, w=self.w, name_y=nombre_y, name_x=nombres_x)
        print(sar.summary)

        # ------------------------------------------------------------------
        # 3. MODELO DE ERROR ESPACIAL (SEM - Spatial Error)
        # ------------------------------------------------------------------
        print("\n🔹 [3/3] Estimando Modelo SEM (Spatial Error - ML_Error)...")
        sem = ML_Error(y, X, w=self.w, name_y=nombre_y, name_x=nombres_x)
        print(sem.summary)

# ==========================================
# BLOQUE DE EJECUCIÓN CON RUTAS QUEMADAS
# ==========================================
if __name__ == "__main__":
    # Rutas absolutas idénticas a los scripts anteriores
    ruta_csv = r"C:\Users\monge\OneDrive\Escritorio\Portafolio\Proyecto_1\Bases\Base_Ancha_NL_Nearshoring.csv"
    ruta_shp = r"C:\Users\monge\OneDrive\Escritorio\Portafolio\Proyecto_1\Bases\2025_1_19_MUN.shp"
    
    # Definición estricta de Variables del Marco Teórico
    VARIABLE_DEPENDIENTE = 'ui_index_541_pyme_2023'  # Upgrading de PyMEs
    
    VARIABLES_INDEPENDIENTES = [
        'act_fij_336_tractora_2023',   # Choque Nearshoring (Capital Fijo Automotriz)
        'ac_cap_541_pyme_2023',       # Capacidad de Absorción (Capital / Trabajo PyME - Lascurain)
        'tasa_formal_541_pyme_2023',  # Formalidad Laboral PyME (Montaño Hirose)
        'sal_med_541_pyme_2023'       # Calidad Salarial PyME (Montaño Hirose)
    ]
    
    try:
        modelo = RegresionEspacialNearshoring(ruta_csv, ruta_shp)
        modelo.estimar_modelos(VARIABLE_DEPENDIENTE, VARIABLES_INDEPENDIENTES)
    except Exception as e:
        print(f"❌ Error durante la estimación: {e}")
