import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración global de estilo visual para Seaborn
sns.set_theme(style="whitegrid")

def tarea_2_inspeccion(ruta_csv):
    print("="*60)
    print("TAREA 2: CARGA E INSPECCIÓN DE DATOS")
    print("="*60)
    
    df = pd.read_csv(ruta_csv)
    
    print("\n1. Primeras 5 filas:")
    print(df.head())
    
    print("\n2. Forma del DataFrame (shape):")
    print(df.shape)
    
    print("\n3. Tipos de datos (info):")
    print(df.info())
    
    print("\n4. Cantidad de nulos por columna:")
    print(df.isnull().sum())
    
    print("\n5. Cantidad de filas duplicadas:")
    print(df.duplicated().sum())
    
    return df

def tarea_3_limpieza(df):
    print("\n" + "="*60)
    print("TAREA 3: CALIDAD Y LIMPIEZA DE DATOS")
    print("="*60)
    
    filas_iniciales = len(df)
    print(f"Número de filas antes de limpiar: {filas_iniciales}")
    
    # 1. Eliminar la fila de resumen final ('TOTAL')
    df = df[df['radicado'] != 'TOTAL'].copy()
    
    # 2. Unificar dependencias categóricas (espacios y mayúsculas)
    df['dependencia'] = df['dependencia'].astype(str).str.strip().str.title()
    
    # Asegurar conversión a tipos numéricos
    df['dias_respuesta'] = pd.to_numeric(df['dias_respuesta'], errors='coerce')
    df['reiteraciones'] = pd.to_numeric(df['reiteraciones'], errors='coerce')
    df['escalada'] = pd.to_numeric(df['escalada'], errors='coerce')
    
    # 3. Pregunta Clave: Corregir valor anómalo/imposible (-25 en dias_respuesta)
    media_antes = df['dias_respuesta'].mean()
    print(f"\n[Pregunta Clave] Media de 'dias_respuesta' ANTES de corregir el -25: {media_antes:.2f}")
    
    df.loc[df['dias_respuesta'] < 0, 'dias_respuesta'] = np.nan
    mediana_dias = df['dias_respuesta'].median()
    df['dias_respuesta'].fillna(mediana_dias, inplace=True)
    
    media_despues = df['dias_respuesta'].mean()
    print(f"[Pregunta Clave] Media de 'dias_respuesta' DESPUÉS de corregir: {media_despues:.2f}")
    
    # 4. Eliminar filas duplicadas
    duplicados_antes = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Filas duplicadas eliminadas: {duplicados_antes}")
    
    # Limpiar nulos residuales
    df = df.dropna(subset=['dias_respuesta', 'reiteraciones', 'escalada'])
    df['escalada'] = df['escalada'].astype(int)
    
    filas_finales = len(df)
    print(f"\nNúmero de filas después de la limpieza: {filas_finales}")
    print("Registros de cada clase (0 y 1) en la columna objetivo 'escalada':")
    print(df['escalada'].value_counts())
    
    # Guardar dataset limpio requerido por la rúbrica
    df.to_csv("../data/grupo_11_limpio.csv", index=False)
    print("\nDataset limpio guardado exitosamente en '../data/grupo_11_limpio.csv'")
    return df

def tarea_4_visualizacion(df):
    print("\n" + "="*60)
    print("TAREA 4: VISUALIZACIÓN CON SEABORN")
    print("="*60)
    
    # --- Gráfico 1: Dispersión de las dos variables numéricas coloreada por la columna objetivo ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, 
        x='dias_respuesta', 
        y='reiteraciones', 
        hue='escalada', 
        palette='Set1', 
        s=100
    )
    plt.title('Dispersión: Días de Respuesta vs Reiteraciones por Estado de Escalada')
    plt.xlabel('Días de Respuesta')
    plt.ylabel('Reiteraciones')
    plt.legend(title='Escalada (0=No, 1=Sí)')
    plt.tight_layout()
    plt.savefig('dispersion_variables.png', dpi=150)
    plt.close()
    print("Gráfico 1 guardado: dispersion_variables.png")
    
    # --- Gráfico 2: Boxplot que compara una variable numérica (días de respuesta) entre las clases 0 y 1 ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df, 
        x='escalada', 
        y='dias_respuesta', 
        palette='Set2'
    )
    plt.title('Distribución de Días de Respuesta según si la PQRS fue Escalada')
    plt.xlabel('Escalada (0 = No escalada, 1 = Escalada a control interno)')
    plt.ylabel('Días de Respuesta')
    plt.tight_layout()
    plt.savefig('boxplot_escalada.png', dpi=150)
    plt.close()
    print("Gráfico 2 guardado: boxplot_escalada.png")

def sigmoide(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def calcular_perdida(y_real, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))

def tarea_5_perceptron(df):
    print("\n" + "="*60)
    print("TAREA 5: PERCEPTRÓN DESDE CERO CON NUMPY")
    print("="*60)
    
    X = df[['dias_respuesta', 'reiteraciones']].values
    y = df['escalada'].values
    
    # Estandarización de X (media 0 y desviación 1 por columna)
    media_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    X_norm = (X - media_X) / std_X
    
    # Inicialización de pesos y bias
    np.random.seed(42)
    w = np.random.randn(2) * 0.01
    b = 0.0
    tasa_aprendizaje = 0.1
    epocas = 350
    
    perdidas = []
    exactitudes = []
    
    print(f"Pesos iniciales: w1={w[0]:.4f}, w2={w[1]:.4f}, b={b:.4f}")
    
    for epoca in range(epocas):
        z = np.dot(X_norm, w) + b
        y_pred = sigmoide(z)
        
        perdida = calcular_perdida(y, y_pred)
        perdidas.append(perdida)
        
        y_pred_bin = (y_pred >= 0.5).astype(int)
        exactitud = np.mean(y_pred_bin == y) * 100
        exactitudes.append(exactitud)
        
        # Cálculo de gradientes
        m = len(y)
        dw = np.dot(X_norm.T, (y_pred - y)) / m
        db = np.sum(y_pred - y) / m
        
        # Actualización de parámetros por descenso del gradiente
        w -= tasa_aprendizaje * dw
        b -= tasa_aprendizaje * db
        
    print(f"Pesos finales:   w1={w[0]:.4f}, w2={w[1]:.4f}, b={b:.4f}")
    print(f"Pérdida inicial: {perdidas[0]:.4f} | Pérdida final: {perdidas[-1]:.4f}")
    print(f"Exactitud final: {exactitudes[-1]:.2f}%")
    
    # Gráfico de la Tarea 5: Curva de aprendizaje
    plt.figure(figsize=(8, 5))
    plt.plot(range(epocas), perdidas, color='red')
    plt.title('Curva de Aprendizaje (Pérdida por Época)')
    plt.xlabel('Época')
    plt.ylabel('Pérdida (Cross-Entropy)')
    plt.savefig('curva_aprendizaje.png', dpi=150)
    plt.close()
    print("Gráfico 'curva_aprendizaje.png' guardado exitosamente.")
    
    # Gráfico de la Tarea 5: Frontera de decisión en unidades originales
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='dias_respuesta', y='reiteraciones', hue='escalada', palette='Set1', s=100)
    
    x1_min, x1_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    x1_vals = np.linspace(x1_min, x1_max, 100)
    x2_vals = -((w[0] * ((x1_vals - media_X[0]) / std_X[0]) + b) * std_X[1] / w[1]) + media_X[1]
    
    plt.plot(x1_vals, x2_vals, color='black', linewidth=2, label='Frontera de Decisión ($z=0$)')
    plt.title('Frontera de Decisión del Perceptrón')
    plt.xlabel('Días de Respuesta')
    plt.ylabel('Reiteraciones')
    plt.legend()
    plt.savefig('frontera_decision.png', dpi=150)
    plt.close()
    print("Gráfico 'frontera_decision.png' guardado exitosamente.")

if __name__ == "__main__":
    # Ejecución secuencial completa del parcial
    df_raw = tarea_2_inspeccion("../data/grupo_11.csv")
    df_clean = tarea_3_limpieza(df_raw)
    tarea_4_visualizacion(df_clean)
    tarea_5_perceptron(df_clean)