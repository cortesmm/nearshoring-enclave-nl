import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import libpysal
from esda.moran import Moran, Moran_Local
from splot.esda import plot_moran, lisa_cluster
import warnings

warnings.filterwarnings('ignore')

class EconometriaEspacial:
    """
    Clase para el Análisis Exploratorio de Datos Espaciales (ESDA).
    Evalúa la existencia de Spillovers vs Enclaves a través de autocorrelación espacial.
    """
    
    def __init__(self, ruta_csv, ruta_shp):
        self.ruta_csv = ruta_csv
        self.ruta_shp = ruta_shp
        self.gdf = self._preparar_datos_espaciales()
        self.w = self._crear_matriz_vecindad()

    def _preparar_datos_espaciales(self):
        """Une la base ancha tabular con las geometrías del Shapefile."""
        print("🌍 Cargando y fusionando datos espaciales...")
        
        # 1. Cargar datos tabulares (asegurando el formato de 5 dígitos para la clave)
        df = pd.read_csv(self.ruta_csv, dtype={'cve_geo': str})
        df['cve_geo'] = df['cve_geo'].str.zfill(5)
        
        # 2. Cargar Shapefile (Mapa)
        mapa = gpd.read_file(self.ruta_shp)
        
        # Estandarizar la clave geográfica del mapa (el INEGI suele llamarla CVEGEO)
        if 'CVEGEO' in mapa.columns:
            mapa['CVEGEO'] = mapa['CVEGEO'].astype(str).str.zfill(5)
        else:
            raise ValueError("El shapefile no contiene la columna 'CVEGEO'. Verifica el nombre de la columna.")
            
        # 3. Fusión (Merge) Espacial
        gdf = mapa.merge(df, left_on='CVEGEO', right_on='cve_geo', how='inner')
        print(f"✅ Fusión exitosa: {gdf.shape[0]} polígonos listos para el análisis.")
        return gdf

    def _crear_matriz_vecindad(self):
        """Crea la Matriz W basada en contigüidad (Reina/Queen)."""
        print("🕸️ Construyendo la Matriz de Pesos Espaciales (W)...")
        # Usamos contigüidad tipo Queen (comparten frontera o un solo vértice)
        w = libpysal.weights.Queen.from_dataframe(self.gdf)
        # Estandarización por filas (para promediar el efecto de los vecinos)
        w.transform = 'r' 
        return w

    def moran_global(self, variable):
        """Calcula y grafica el Índice de Moran Global."""
        if variable not in self.gdf.columns:
            print(f"⚠️ La variable {variable} no existe en los datos.")
            return

        print(f"\n🌐 --- ÍNDICE DE MORAN GLOBAL PARA: {variable} ---")
        y = self.gdf[variable].values
        
        # Calcular Moran
        moran = Moran(y, self.w)
        
        # Imprimir resultados estadísticos
        print(f"Índice de Moran (I): {moran.I:.4f}")
        print(f"Valor-p (Significancia): {moran.p_sim:.4f}")
        
        if moran.p_sim < 0.05:
            if moran.I > 0:
                print("Conclusión: Existe Autocorrelación Espacial POSITIVA (Clústeres similares se agrupan).")
            else:
                print("Conclusión: Existe Autocorrelación Espacial NEGATIVA (Patrón de competencia).")
        else:
            print("Conclusión: El patrón es ALEATORIO (No hay evidencia de spillovers espaciales).")

        # Gráfico corregido: Dejamos que splot cree los ejes automáticamente
        fig, ax = plot_moran(moran, zstandard=True, figsize=(10, 4))
        # Usamos suptitle porque son múltiples gráficos
        plt.suptitle(f"Dispersión de Moran Global: {variable}", fontsize=12, y=1.05)
        plt.show()

    def moran_local_lisa(self, variable):
        """Calcula y mapea los Clústeres LISA (Local Indicators of Spatial Association)."""
        if variable not in self.gdf.columns:
            return

        print(f"\n📍 --- MAPA LISA PARA: {variable} ---")
        y = self.gdf[variable].values
        
        # Calcular Moran Local
        moran_loc = Moran_Local(y, self.w)
        
        # Graficar el Mapa de Clústeres
        fig, ax = plt.subplots(figsize=(10, 8))
        lisa_cluster(moran_loc, self.gdf, p=0.05, ax=ax)
        plt.title(f"Clústeres Espaciales LISA (p < 0.05): {variable}\n(Rojo=Derrame, Azul Oscuro=Rezago, Azul Claro/Rosa=Desplazamiento)")
        plt.tight_layout()
        plt.show()

# ==========================================
# BLOQUE DE EJECUCIÓN (MAIN)
# ==========================================
if __name__ == "__main__":
    # 1. Rutas exactas inyectadas usando 'r' (Raw Strings) para evitar errores de Windows
    ruta_csv = r"C:\Users\monge\OneDrive\Escritorio\Portafolio\Proyecto_1\Bases\Base_Ancha_NL_Nearshoring.csv"
    ruta_shp = r"C:\Users\monge\OneDrive\Escritorio\Portafolio\Proyecto_1\Bases\2025_1_19_MUN.shp"
    
    try:
        # Instanciar el modelo
        esda = EconometriaEspacial(ruta_csv, ruta_shp)
        
        # ---------------------------------------------------------
        # PRUEBA 1: EL CHOQUE EXÓGENO (El territorio del Nearshoring)
        # ---------------------------------------------------------
        # ¿La inversión extranjera se agrupa en clústeres?
        esda.moran_global('act_fij_336_tractora_2023')
        esda.moran_local_lisa('act_fij_336_tractora_2023')
        
        # ---------------------------------------------------------
        # PRUEBA 2: EL UPGRADING DE LAS PYMES (La variable a explicar)
        # ---------------------------------------------------------
        # ¿El ascenso industrial de las PyMEs es un fenómeno de contagio espacial?
        esda.moran_global('ui_index_541_pyme_2023')
        esda.moran_local_lisa('ui_index_541_pyme_2023')

    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
