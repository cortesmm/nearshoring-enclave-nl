import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mapclassify as mc
import warnings

warnings.filterwarnings('ignore')

class EdaEspacial:
    """
    Clase universal para Análisis Exploratorio de Datos (EDA) adaptada
    al análisis espacial de Spillovers y Cadenas Globales de Valor.
    """
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = self._load_data()
        
    def _load_data(self):
        """Carga los datos asegurando el formato estricto de la clave geográfica."""
        df = pd.read_csv(self.filepath, dtype={'cve_geo': str})
        
        # El seguro de vida espacial: rellenar con ceros a la izquierda (Ej. "19039")
        df['cve_geo'] = df['cve_geo'].astype(str).str.zfill(5)
        
        print(f"✅ Base cargada exitosamente: {df.shape[0]} municipios y {df.shape[1]} variables.")
        return df

    def estadistica_descriptiva(self, variables):
        """Genera un resumen estadístico para detectar dispersión."""
        print("\n📊 --- ESTADÍSTICA DESCRIPTIVA ---")
        # Filtramos solo las variables que existen en el dataframe
        vars_existentes = [v for v in variables if v in self.df.columns]
        desc = self.df[vars_existentes].describe().T
        # Formato numérico para facilitar lectura
        print(desc[['mean', 'std', 'min', '50%', 'max']].round(3))
        return desc

    def top_municipios(self, variable, n=5):
        """Revela los municipios con mayor concentración en una variable (Hotspots absolutos)."""
        if variable in self.df.columns:
            print(f"\n🏆 --- TOP {n} MUNICIPIOS EN: {variable} ---")
            top = self.df[['nom_mun', variable]].sort_values(by=variable, ascending=False).head(n)
            # Imprimir bonito
            for index, row in top.iterrows():
                print(f"   📍 {row['nom_mun']}: {row[variable]:,.2f}")
        else:
            print(f"⚠️ La variable {variable} no existe en la base.")

    def deteccion_outliers_cajas(self, variables):
        """Muestra la hiperconcentración visualmente mediante Boxplots."""
        vars_existentes = [v for v in variables if v in self.df.columns]
        if not vars_existentes: return
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df[vars_existentes], orient='h', palette="Set2")
        plt.title("Detección de Hiperconcentración Territorial (Outliers)")
        plt.xlabel("Valor")
        plt.tight_layout()
        plt.show()

    def histogramas_comparativos(self, var1, var2):
        """Compara la distribución de dos variables mediante un Scatter Plot."""
        if var1 in self.df.columns and var2 in self.df.columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=self.df, x=var1, y=var2, hue=var1, size=var1, sizes=(50, 400), palette="viridis", legend=False)
            
            # Anotar los nombres de los municipios más altos
            top_mun = self.df.sort_values(by=var1, ascending=False).head(3)
            for _, row in top_mun.iterrows():
                plt.text(row[var1], row[var2], f" {row['nom_mun']}", fontsize=9, weight='bold')

            plt.title(f"Relación Territorial: {var1} vs {var2}")
            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.show()

    def cortes_naturales(self, variable, k=5):
        """Usa Fisher-Jenks para encontrar los saltos reales en los datos para los mapas."""
        if variable in self.df.columns:
            # Filtramos los ceros para que no sesguen los cortes
            datos_limpios = self.df[self.df[variable] > 0][variable].dropna()
            if len(datos_limpios) > k:
                natural_breaks = mc.FisherJenks(datos_limpios, k=k)
                print(f"\n🗺️ --- CORTES NATURALES (Fisher-Jenks) PARA MAPA DE: {variable} ---")
                print(natural_breaks)
            else:
                print(f"\n⚠️ No hay suficientes municipios con datos > 0 para hacer cortes en {variable}.")

# ==========================================
# BLOQUE DE EJECUCIÓN (MAIN) - ENFOQUE: TRACTORAS
# ==========================================
if __name__ == "__main__":
    # 1. Ruta del archivo que acabamos de generar
    ruta_archivo = 'Base_Ancha_NL_Nearshoring.csv' 
    
    # 2. Instanciamos la clase
    eda = EdaEspacial(filepath=ruta_archivo)
    
    # ---------------------------------------------------------
    # EL CHOQUE EXÓGENO: INDUSTRIA AUTOMOTRIZ (SCIAN 336) 2023
    # ---------------------------------------------------------
    # Variables de interés: Unidades Económicas, Empleo, Acervo de Capital y Ratio de Valor Agregado
    vars_tractora_336 = [
        'ue_336_tractora_2023',      # ¿Cuántas macro plantas hay?
        'po_336_tractora_2023',      # ¿Cuánto empleo acaparan?
        'act_fij_336_tractora_2023', # El choque de Capital Puro (Inversión instalada)
        'va_ratio_336_tractora_2023' # El test de Dussel (Integración local)
    ]
    
    # A) Ver el resumen estadístico
    eda.estadistica_descriptiva(vars_tractora_336)
    
    # B) Descubrir a los "Reyes del Nearshoring" en Nuevo León
    eda.top_municipios('act_fij_336_tractora_2023', n=5)
    eda.top_municipios('po_336_tractora_2023', n=5)
    
    # C) Visualizar la hiperconcentración en Cajas (Boxplots)
    # Mostramos Empleo y Capital (las magnitudes más grandes)
    eda.deteccion_outliers_cajas(['po_336_tractora_2023', 'act_fij_336_tractora_2023'])
    
    # D) Probar la Teoría de Dussel: Scatter de Capital Instalado vs Ratio de Valor Agregado
    # ¿Los que tienen más inversión son los que menos valor agregan (ensambladoras)?
    eda.histogramas_comparativos('act_fij_336_tractora_2023', 'va_ratio_336_tractora_2023')
    
    # E) Calcular cortes para cuando pasemos al mapa en GeoDa o con splot
    eda.cortes_naturales('act_fij_336_tractora_2023')
