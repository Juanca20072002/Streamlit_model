import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import numpy as np
from pyod.utils.utility import standardizer
from pyod.models.combination import average
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import sklearn.metrics as metrics
from sklearn.metrics import auc, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from streamlit_option_menu import option_menu
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# Pestaña de la página
st.set_page_config(
    page_title="Sistema IDS IoT - Detección de Intrusiones",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar animaciones Lottie
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        st.error(f"Error al cargar la animación desde {url}. Estado HTTP: {r.status_code}")
        return None
    return r.json()

# Animaciones

lottie_loading = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_szlepvdh.json")
lottie_how_it_works = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_yd8fbnml.json")
lottie_network = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ggwq3ysg.json") 
lottie_success = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_ktwnwv5m.json")  
lottie_security = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_jcsfwbvi.json")  

# Verificar que las animaciones se han cargado correctamente
if lottie_loading is None or lottie_success is None:
    st.error("Error al cargar las animaciones Lottie. Por favor, verifica las URLs.")

# Inicializar el estado de la sesión para la navegación
if "page" not in st.session_state:
    st.session_state.page = "home"

#----------------------------------------------------  Estilos CSS --------------------------------------------------------------------------------------------------------------------------------
st.markdown("""
<style>
    body {
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
    }
        .main-header {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(90deg, #1E3D59 0%, #2E5077 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .success-animation {
        animation: fadeInOut 3s forwards;
    }
    @keyframes fadeInOut {
        0% { opacity: 0; }
        20% { opacity: 1; }
        80% { opacity: 1; }
        100% { opacity: 0; display: none; }
    }
    .stAlert {
        background-color: rgba(25, 25, 25, 0.5);
        color: white;
        border: none;
        padding: 1rem;
        border-radius: 10px;
    }
    .stButton > button {
        background: rgba(255, 255, 255, 0.1);
        border: none;
        color: #ffffff;
        padding: 10px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(0, 0, 0, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
    }
    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    .dashboard-form {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        max-width: 600px;
        margin: auto;
    }
    .notification {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        animation: fadeOut 2s forwards;
        animation-delay: 3.5s;
    }
    @keyframes fadeOut {
        from {opacity: 1;}
        to {opacity: 0; height: 0; padding: 0; margin: 0;}
    }
    .streamlit-expanderHeader, .stTextInput > div > div > input {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    .stDataFrame {
        color: #ffffff;
    }
    .card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .carousel {
        display: flex;
        overflow-x: auto;
        scroll-snap-type: x mandatory;
    }
    .carousel-item {
        flex: none;
        scroll-snap-align: start;
        margin-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------- INICIO PAGE----------------------------------------------------------------------------------------------------------------------------
# Navegación mejorada
with st.sidebar:
    st.image("https://img.icons8.com/color/48/000000/shield.png", width=50)
    selected = option_menu(
        "Navegación",
        ["Inicio", "Panel de Control"],
        icons=["house", "graph-up"],
        menu_icon="cast",
        default_index=0
    )
    st.session_state.page = "home" if selected == "Inicio" else "dashboard"

def show_home_page():
    st.title("🛡 Sistema de Detección de Intrusiones para IoT")
    if lottie_security:
        st_lottie(lottie_security, height=200)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("""
    Bienvenido al Sistema de Detección de Intrusiones en Redes IoT, 
    una solución avanzada diseñada para monitorear y detectar anomalías 
    en redes de IoT, brindando seguridad y confiabilidad sin precedentes. Nuestro sistema utiliza modelos 
    de vanguardia en detección de anomalías, como LOF (Local Outlier Factor), IForest (Isolation Forest) y KNN (K-Nearest Neighbors), 
    para analizar patrones complejos y alertar sobre posibles irregularidades en la red en tiempo real.
    """)

    col1, col2 = st.columns(2)
    
    with col1:  
        st.subheader("🎯 Objetivos del Sistema")
        st.write("""
        Nuestro sistema utiliza algoritmos avanzados de machine learning para:
        - 🔍 Detección en tiempo real de anomalías
        - 📊 Análisis predictivo de patrones
        - 🚫 Identificación de amenazas potenciales
        - 📈 Monitoreo continuo del rendimiento
        """)
        
    with col2:
        pass

    st.markdown("---")
    st.subheader("💬 Interpretación de Resultados")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 🟢 Métricas de Validación
        - Silhouette Score: Mide la calidad de los clusters (ideal > 0.5)
        - Calinski Score: Evalúa la separación entre clusters
        - Davies Score: Indica la similitud dentro de clusters
        """)
    
    with col2:
        st.markdown("""
        #### 🔰 Clasificación de Anomalías
        - Normal: Tráfico de red esperado
        - Anómalo: Patrones sospechosos
        - Métrica Combinada: Evaluación holística
        """)
    
    with col3:
        st.markdown("""
        #### 📈 Indicadores de Rendimiento
        - Precisión: Exactitud de detección
        - Recall: Cobertura de detección
        - F1-Score: Balance precisión-recall
        """)

    st.markdown("---")
    st.subheader("🔎 Modelos de Detección de Anomalías")
    with st.expander("LOF (Local Outlier Factor)"):
        st.write("""
        El modelo LOF (Local Outlier Factor) es un algoritmo de detección de anomalías que identifica puntos de datos que se encuentran en regiones de baja densidad en comparación con sus vecinos. Es útil para detectar comportamientos inusuales en la red IoT.
        """)
    with st.expander("IForest (Isolation Forest)"):
        st.write("""
        El modelo IForest (Isolation Forest) es un algoritmo de detección de anomalías que utiliza árboles de aislamiento para identificar puntos de datos anómalos. Es eficiente y efectivo para detectar anomalías en grandes conjuntos de datos.
        """)
    with st.expander("KNN (K-Nearest Neighbors)"):
        st.write("""
        El modelo KNN (K-Nearest Neighbors) es un algoritmo de detección de anomalías que clasifica puntos de datos en función de la distancia a sus vecinos más cercanos. Es útil para detectar anomalías en datos de red IoT.
        """)

    st.markdown("---")
    st.subheader("📖 Método de Uso del Panel")
    st.write("""
    Sigue estos pasos para utilizar el panel de detección de anomalías:
    """)
    st.markdown("""
    <div class="carousel">
        <div class="carousel-item">
            <h4>Paso 1: Cargar Datos</h4>
            <p>📤 Carga tus datos de red IoT en formato CSV.</p>
        </div>
        <div class="carousel-item">
            <h4>Paso 2: Seleccionar Modelo</h4>
            <p>📌 Selecciona el modelo de detección (LOF, IForest o KNN).</p>
        </div>
        <div class="carousel-item">
            <h4>Paso 3: Ejecutar Análisis</h4>
            <p>🖱 Ejecuta el análisis con un solo clic.</p>
        </div>
        <div class="carousel-item">
            <h4>Paso 4: Visualizar Resultados</h4>
            <p>📊 Visualiza los resultados en gráficos interactivos.</p>
        </div>
        <div class="carousel-item">
            <h4>Paso 5: Obtener Insights</h4>
            <p>📄 Obtén insights detallados sobre las anomalías detectadas.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Tipos de Resultados y Recomendaciones")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h4>✅ Resultados Normales</h4>
            <p>Los datos han sido clasificados como normales. No se han detectado anomalías significativas.</p>
            <p><strong>Recomendación:</strong> Continúa monitoreando la red regularmente.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <h4>⚠ Resultados Anómalos</h4>
            <p>Se han detectado datos anómalos en la red. Esto puede indicar posibles amenazas o comportamientos inusuales.</p>
            <p><strong>Recomendación:</strong> Investiga las anomalías detectadas y toma las medidas necesarias para mitigar posibles riesgos.</p>
        </div>
        """, unsafe_allow_html=True)

#---------------------------------------------------- DASHBOARD ----------------------------------------------------------------------------------------------------------------------------

def show_dashboard_page():
    st.title('❇ Panel de Control de Detección')
    st.markdown('</div>', unsafe_allow_html=True)

    # Rutas a los modelos LOF
    LOF_MODEL_PATHS = {
        "k_40": "model/lof_model_Bot-IoT_40.pkl",
        "k_50": "model/lof_model_Bot-IoT_50.pkl",
        "k_60": "model/lof_model_Bot-IoT_60.pkl",
        "k_70": "model/lof_model_Bot-IoT_70.pkl"
    }
    # Rutas a los modelos IFOREST
    IFOREST_MODEL_PATHS = {
        "200": "model/iforest_model_Bot_IoT_200.pkl",
        "300": "model/iforest_model_Bot_IoT_300.pkl",
        "400": "model/iforest_model_Bot_IoT_400.pkl",
        "500": "model/iforest_model_Bot_IoT_500.pkl"
    }
    # Rutas a los modelos KNN
    KNN_MODEL_PATHS = {
        "k_40": "model/knn_model_Bot-IoT_40.pkl",
        "k_50": "model/knn_model_Bot-IoT_50.pkl",
        "k_60": "model/knn_model_Bot-IoT_60.pkl",
        "k_70": "model/knn_model_Bot-IoT_70.pkl"
    }
    
    
    @st.cache_resource
    def load_models(model_paths):
        models = {}
        for key, path in model_paths.items():
            try:
                model = joblib.load(path)
                models[key] = model
            except Exception as e:
                st.error(f"Error al cargar el modelo {key}: {e}")
        return models
    
    def load_model():
        modelo_ocsvm = joblib.load('model/OCSVM_model2_Bot-IoT.pkl')  # Ruta al modelo guardado
        return modelo_ocsvm

    col1, col2 = st.columns([1, 2])
    
    # Interfas Uusuario Panel de Control
    with col1:
        st.markdown("### 🖥 Panel de Control")
        model_option = st.selectbox(
            "📌 Seleccione el modelo de detección",
            ('LOF', 'IForest', 'KNN','OCSVM'),
            help="Cada modelo utiliza diferentes técnicas para detectar anomalías"
        )
        metric_option = st.selectbox(
            "📌 Seleccione el tipo de metrica",
            ('Externas', 'Internas'),
            help="Cada modelo utiliza diferentes técnicas para detectar anomalías"
        )
        
        
        
        uploaded_file = st.file_uploader(
            "📂 Cargar archivo CSV",
            type="csv",
            help="Cargue sus datos de red en formato CSV"
        )
    

    with col2:
        if uploaded_file is not None:
            with st.spinner('⏳ Procesando datos...'):
                data = pd.read_csv(uploaded_file)
                st.markdown('<div class="notification" style="background-color: #4CAF50; color: white;">✅ Datos cargados exitosamente</div>', unsafe_allow_html=True)
                time.sleep(3)

                with st.expander("📊 Vista previa de datos"):
                    st.dataframe(data.head())
                    st.info(f"Dimensiones del dataset: {data.shape[0]} filas, {data.shape[1]} columnas")

    if uploaded_file is not None and st.button('🚀 Realizar Predicción', key='predict'):
        loading_placeholder = st.empty()
        cancel_button = st.empty()
        with loading_placeholder.container():
            if lottie_loading is not None:
                st_lottie(lottie_loading, height=200, key="loading")
                cancel_button.button("Cancelar Predicción", key='cancel')
        
        try:
            if model_option == 'OCSVM':
                # Cargar directamente el modelo OCSVM
                model_ocsvm = load_model()
                current_models = None  # No necesitamos current_models aquí
            else:
                # Seleccionar rutas para los otros modelos
                if model_option == 'LOF':
                    MODEL_PATHS = LOF_MODEL_PATHS
                elif model_option == 'IForest':
                    MODEL_PATHS = IFOREST_MODEL_PATHS
                elif model_option == 'KNN':
                    MODEL_PATHS = KNN_MODEL_PATHS
            # Cargar modelos desde las rutas
                current_models = load_models(MODEL_PATHS)
            
            
            if metric_option=='Externas':
                datay=data.iloc[:,-1]
                data=data.iloc[:, :-1]

            if model_option == 'OCSVM':
                data=data.drop('mean',axis=1)
                threshold=4000
                average_scores = model_ocsvm.decision_function(data.values)
                y_pred = np.where(average_scores <= threshold, 0, 1)
            else:
                # Procesamiento de datos para otros modelos
                test_scores = np.zeros((data.shape[0], len(current_models)))
                for i, (key, model) in enumerate(current_models.items()):
                    test_scores[:, i] = model.decision_function(data.values)

                # Normalización y predicción
                test_scores_norm = standardizer(test_scores)
                average_scores = average(test_scores_norm)

                # Umbrales según el modelo
                if model_option == 'LOF':
                    threshold = 0.19
                elif model_option == 'IForest':
                    threshold = 2
                elif model_option == 'KNN':
                    threshold = 0.4
                y_pred = np.where(average_scores <= threshold, 0, 1)

            # Limpiar animación de carga después de 3 segundos
            time.sleep(3)
            loading_placeholder.empty()

            # Mostrar animación de éxito
            with st.container():
                if lottie_success:
                    st_lottie(lottie_success, height=200, key="success", speed=1.5)
                    time.sleep(3)

            # Resultados y visualizaciones
             # Visualization section
            # Sección de visualización
            col1, col2 = st.columns(2)
            

            with col1:
                # Cálculo de las cantidades de normales y anómalos
                normal_count = np.sum(y_pred == 0)
                anomaly_count = np.sum(y_pred == 1)

                # Crear el gráfico de barras con Plotly
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Normal', 'Anómalo'],  # Etiquetas del eje X
                        y=[normal_count, anomaly_count],  # Valores del eje Y
                        marker_color=['#00C853', '#FF5252'],  # Colores personalizados
                        text=[f"{normal_count:,}", f"{anomaly_count:,}"],  # Texto en las barras
                        textposition='auto',  # Posición del texto
                        hoverinfo='y+text',  # Información al pasar el mouse
                        width=0.8  # Ancho de las barras
                    )
                ])

                # Configuración de diseño del gráfico
                fig.update_layout(
                    title="Distribución de Detecciones",  # Título del gráfico
                    title_x=0.5,  # Centrar el título
                    plot_bgcolor='rgba(0,0,0,0)',  # Fondo transparente del área de trazado
                    paper_bgcolor='rgba(0,0,0,0)',  # Fondo transparente del gráfico
                    font=dict(color='white'),  # Color del texto
                    showlegend=False,  # Ocultar leyenda
                    margin=dict(t=50, l=50, r=50, b=50),  # Márgenes del gráfico
                    width=800,  # Ancho total del gráfico
                    height=700,  # Altura total del gráfico
                    xaxis=dict(
                        title="Clasificación",  # Título del eje X
                        showgrid=False,  # Ocultar la cuadrícula del eje X
                        showline=True,  # Mostrar línea del eje X
                        linecolor='rgba(255,255,255,0.2)'  # Color de la línea del eje X
                    ),
                    yaxis=dict(
                        title="Cantidad",  # Título del eje Y
                        showgrid=True,  # Mostrar cuadrícula del eje Y
                        gridcolor='rgba(255,255,255,0.1)',  # Color de la cuadrícula del eje Y
                        zeroline=False  # Ocultar línea en Y=0
                    ),
                    bargap=0.2  # Espacio entre las barras
                )

                # Mostrar el gráfico en Streamlit
                st.plotly_chart(fig, use_container_width=True)  # Permitir que se ajuste al contenedor
                
                
            
                

            # Matriz de Confusión
            
            def confusion_matrix_threshold(actual,score, threshold):
                Actual_pred = pd.DataFrame({'Actual': actual, 'Pred': score})
                Actual_pred['Pred'] = np.where(Actual_pred['Pred']<=threshold,0,1)
                cm = pd.crosstab(Actual_pred['Actual'],Actual_pred['Pred'])
                return(cm)
            
            if(metric_option=='Externas'):
                with st.expander("📖 Explicación de las métricas Externas"):
                        
                        st.write("""
                        1. Precisión (Precision):
                        - Indica qué porcentaje de las predicciones positivas realizadas por el modelo son correctas.
                        - Fórmula: TP / (TP + FP)
                        - Ejemplo: Si el modelo predice 100 positivos y 98 son correctos, la precisión es 0.98.

                        2. Exactitud (Accuracy):
                        - Porcentaje de predicciones correctas sobre el total de predicciones realizadas.
                        - Fórmula: (TP + TN) / (TP + TN + FP + FN)
                        - Muestra qué tan bien el modelo clasifica en general.

                        3. F1-Score:
                        - Es la media armónica entre la Precisión y la Sensibilidad (Recall).
                        - Útil cuando las clases están desbalanceadas.
                        - Fórmula: 2 * (Precision * Recall) / (Precision + Recall)

                        4. Curva ROC y AUC:
                        - La curva ROC muestra la relación entre la Tasa de Verdaderos Positivos (TPR) y la Tasa de Falsos Positivos (FPR).
                        - El AUC (Área Bajo la Curva) mide qué tan bien el modelo separa las clases.
                        - Un valor cercano a 1 indica un excelente desempeño; 0.76 es aceptable.

                        5. Falsos Positivos (FP):
                        - Representan los casos negativos que el modelo clasificó erróneamente como positivos.
                        - Fórmula: FP / (FP + TN)
                        - Menos falsos positivos indican mejor rendimiento.

                        6. Falsos Negativos (FN):
                        - Representan los casos positivos reales que el modelo no detectó correctamente.
                        - Fórmula: FN / (FN + TP)
                        - Un valor bajo indica que el modelo es bueno para detectar positivos reales.

                        Nota:
                        - TP: Verdaderos Positivos
                        - TN: Verdaderos Negativos
                        - FP: Falsos Positivos
                        - FN: Falsos Negativos
                        """)    
                st.markdown("### 📊 Matriz de Confusión")
                data_y=datay.values
                matrix_confusion = confusion_matrix_threshold(datay, average_scores, threshold)
                st.table(matrix_confusion.style.format("{:,.0f}"))
                # Explicación interactiva
                with st.expander("📖 ¿Qué es la Matriz de Confusión?"):
                    st.markdown("""
                        La Matriz de Confusión es una herramienta para evaluar el desempeño de un modelo de clasificación. 
                        Muestra la cantidad de predicciones correctas e incorrectas de un modelo en una tabla con cuatro categorías:
                        - Verdaderos positivos (TP): Predicciones correctas de la clase positiva.
                        - Falsos negativos (FN): Predicciones incorrectas donde el modelo predijo la clase negativa.
                        - Verdaderos negativos (TN): Predicciones correctas de la clase negativa.
                        - Falsos positivos (FP): Predicciones incorrectas donde el modelo predijo la clase positiva.
                    """)
                
            if(metric_option=='Internas'):
                with st.expander("📖 Explicación de las métricas Internas"):
                        
                        st.write("""
                        1. Silhouette Score:
                        - El puntaje de silueta es una medida utilizada para evaluar la calidad de un agrupamiento (clustering) en un conjunto de datos.
                        - Se basa en la distancia entre los puntos de datos y su relación con otros grupos.
                        - Un valor cercano a 1 indica que el punto está bien agrupado.
                        - Un valor cercano a 0 indica que el punto está en la frontera entre dos grupos.
                        - Un valor negativo indica que el punto podría estar mal agrupado.

                        2. Calinski-Harabsz Score:
                        - Se enfoca en la separación entre los clusters y la cohesión dentro de ellos.
                        - Entre mas alto es la puntuación de esta métrica, mejor agrupación existe

                        3. Davies-bouldin Score:
                        - Se enfoca en la similitud entre pares de clusters, considerando tanto la compacidad como la separación.
                        - Cuanto más bajo sea el índice, más separados y compactos serán los clusters, lo que indica un agrupamiento de mayor calidad.
                        

                        """)    
            
            # New Traffic Pattern Visualization
            st.markdown("### 📈 Patrón de Tráfico y Anomalías")
            
            # Calculate traffic metrics
            traffic_data = data.mean(axis=1)  # Using mean of all features as traffic indicator
            
            # Calculate ceiling and floor
            window = 20  # Window size for rolling calculations
            ceiling = traffic_data.rolling(window=window).max()
            floor = traffic_data.rolling(window=window).min()
            
            # Create traffic pattern figure
            fig_traffic = go.Figure()

            # Add main traffic line
            fig_traffic.add_trace(go.Scatter(
                x=list(range(len(traffic_data))),
                y=traffic_data,
                mode='lines',
                name='Tráfico',
                line=dict(color='#4B9FE1', width=1)
            ))

            # Add ceiling line
            fig_traffic.add_trace(go.Scatter(
                x=list(range(len(ceiling))),
                y=ceiling,
                mode='lines',
                name='Techo',
                line=dict(color='#00C853', width=2, dash='dash')
            ))

            # Add floor line
            fig_traffic.add_trace(go.Scatter(
                x=list(range(len(floor))),
                y=floor,
                mode='lines',
                name='Suelo',
                line=dict(color='#FFA726', width=2, dash='dash')
            ))

            # Add anomaly points
            anomaly_indices = np.where(y_pred == 1)[0]
            fig_traffic.add_trace(go.Scatter(
                x=anomaly_indices,
                y=traffic_data[anomaly_indices],
                mode='markers',
                name='Anomalías',
                marker=dict(
                    color='#FF5252',
                    size=8,
                    symbol='circle'
                )
            ))

            # Update layout
            fig_traffic.update_layout(
                title="Patrón de Tráfico y Detección de Anomalías",
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(0,0,0,0)"
                ),
                margin=dict(t=50, l=50, r=50, b=50),
                xaxis=dict(
                    title="Muestras",
                    showgrid=False,
                    showline=True,
                    linecolor='rgba(255,255,255,0.2)'
                ),
                yaxis=dict(
                    title="Intensidad de Tráfico",
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    zeroline=False
                )
            )

            st.plotly_chart(fig_traffic, use_container_width=True)
                
            with col2:
                
                # Métricas de rendimiento
                X_test_norm = data.values
                if metric_option== 'Externas':
                    data_y=datay.values
                    #Matriz de confusion
                    def confusion_matrix_threshold(actual,score, threshold):
                        Actual_pred = pd.DataFrame({'Actual': actual, 'Pred': score})
                        Actual_pred['Pred'] = np.where(Actual_pred['Pred']<=threshold,0,1)
                        cm = pd.crosstab(Actual_pred['Actual'],Actual_pred['Pred'])
                        return(cm)
                    matrix_confusion=confusion_matrix_threshold(datay,average_scores,threshold)
                    
                    
                    #Metricas supervisadas
                    def Metricas_precision(Matriz):
                        # Cálculo de precisión
                        Precision = Matriz.iloc[0,0] / (Matriz.iloc[0,0] + Matriz.iloc[0,1])
                        
                        # Cálculo de exactitud
                        Exactitud = (Matriz.iloc[0,0] + Matriz.iloc[1,1]) / (Matriz.iloc[0,0] + Matriz.iloc[0,1] + Matriz.iloc[1,0] + Matriz.iloc[1,1])
                        
                        # Cálculo de especificidad
                        Especificidad = Matriz.iloc[1,1] / (Matriz.iloc[1,1] + Matriz.iloc[0,1])
                        
                        # Cálculo de la tasa de verdaderos positivos (TVP o Sensibilidad)
                        TVP = Matriz.iloc[0,0] / (Matriz.iloc[0,0] + Matriz.iloc[1,0])
                        
                        # Tasa de falsos negativos (TasaFN o FNR)
                        TasaFN = Matriz.iloc[1,0] / (Matriz.iloc[1,0] + Matriz.iloc[0,0])
                        
                        # Tasa de falsos positivos (TasaFP o FPR)
                        TasaFP = Matriz.iloc[0,1] / (Matriz.iloc[0,1] + Matriz.iloc[1,1])
                        
                        # Valor predictivo positivo (VPP o Precisión)
                        VPP = Matriz.iloc[0,0] / (Matriz.iloc[0,1] + Matriz.iloc[0,0])
                        
                        # Valor predictivo negativo (VPN)
                        VPN = Matriz.iloc[1,1] / (Matriz.iloc[1,1] + Matriz.iloc[1,0])
                        
                        # Cálculo de F1 Score
                        F1 = (2 * Precision * TVP) / (Precision + TVP)
                        return Precision, Exactitud, Especificidad, TVP, TasaFN, TasaFP, VPP, VPN, F1

                    Precision_n, Exactitud_n, Especificidad_n, TVP_n, TasaFN_n, TasaFP_n, VPP_n, VPN_n, F1_n=Metricas_precision(matrix_confusion)
                      
                    
                    Actual_predIF = pd.DataFrame({'Actual': datay, 'Pred':average_scores})
                    Actual_predIF['Pred'] = np.where(Actual_predIF['Pred']<=threshold,0,1)
                    actual=Actual_predIF['Actual']
                    pred=Actual_predIF['Pred']
                    
                    
                    with col2:
                        
                        
                    # Curva ROC
                        def plot_roc_curve(true_y, y_prob):
                            fpr, tpr, thresholds = roc_curve(true_y, y_prob)
                            roc_auc = auc(fpr, tpr)

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=fpr,
                                y=tpr,
                                mode='lines',
                                name=f'ROC curve = {roc_auc:.2f}',
                                line=dict(color='red', width=2)
                            ))
                            fig.add_trace(go.Scatter(
                                x=[0, 1],
                                y=[0, 1],
                                mode='lines',
                                name='Random Guess',
                                line=dict(color='blue', dash='dash', width=2)
                            ))

                            fig.update_layout(
                                title="Curva ROC",
                                title_x=0.5,
                                xaxis=dict(
                                    title="Tasa de Falsos Positivos (FPR)",
                                    range=[0, 1],
                                    gridcolor='rgba(255,255,255,0.1)',
                                    linecolor='rgba(255,255,255,0.2)',
                                    tickcolor='rgba(255,255,255,0.5)',
                                    showgrid=True,
                                    zeroline=False
                                ),
                                yaxis=dict(
                                    title="Tasa de Verdaderos Positivos (TPR)",
                                    range=[0, 1],
                                    gridcolor='rgba(255,255,255,0.1)',
                                    linecolor='rgba(255,255,255,0.2)',
                                    tickcolor='rgba(255,255,255,0.5)',
                                    showgrid=True,
                                    zeroline=False
                                ),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white', size=12),
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01,
                                    bgcolor="rgba(0,0,0,0)"
                                )
                            )

                            st.plotly_chart(fig, use_container_width=True)

                        plot_roc_curve(actual, pred)
                        curve_roc = roc_auc_score(actual, pred)
                    
                    
                    # Explicación interactiva para la curva ROC
                    with st.expander("📖 ¿Qué es la Curva ROC?"):
                        
                    
                        st.markdown("""
                    La Curva ROC (Receiver Operating Characteristic) muestra la relación entre la tasa de verdaderos positivos 
                    (TPR) y la tasa de falsos positivos (FPR) en diferentes umbrales de clasificación. 
                    - AUC (Área bajo la curva): Indica la capacidad del modelo para distinguir entre las clases.
                    - Cuanto mayor sea el área bajo la curva (AUC), mejor será el rendimiento del model""")        
                        
                    # Mostrar métricas en cards
                    st.markdown("### 📊 Métricas Externas")
                    cols = st.columns(6)
                    with cols[0]:
                        st.metric(
                            "Precision",
                            f"{Precision_n:.2f}",
                            delta="Bueno" if Precision_n > 0.75 else "Regular"
                        )
                    with cols[1]:
                        st.metric(
                            "Exactitud",
                            f"{Exactitud_n:.1f}",
                            delta="Bueno" if Exactitud_n > 0.75 else "Regular"
                        )
                    with cols[2]:
                        st.metric(
                            "F1-SCORES",
                            f"{F1_n:.2f}",
                            delta="Bueno" if F1_n > 0.75 else "Regular"
                        )
                    with cols[3]:
                        st.metric(
                            "Curve ROC",
                            f"{curve_roc:.2f}",
                            delta="Bueno" if curve_roc > 0.75 else "Regular"
                        )
                    with cols[4]:
                        st.metric(
                            "Falsos positivos",
                            f"{TasaFP_n:.2f}",
                            delta="Bueno" if TasaFP_n < 0.3 else "Regular"
                        )
                    with cols[5]:
                        st.metric(
                            "Falsos Negativos",
                            f"{TasaFN_n:.2f}",
                            delta="Bueno" if TasaFN_n < 0.3 else "Regular"
                        )
                    
                    
                    
                elif metric_option == 'Internas':
                    # Cálculo de métricas
                    silhouette = silhouette_score(data, y_pred)
                    calinski = calinski_harabasz_score(data, y_pred)
                    davies = davies_bouldin_score(data, y_pred)

                    # Mostrar métricas en cards
                    st.markdown("### 📊 Métricas Internas")
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric(
                            "Silhouette Score",
                            f"{silhouette:.3f}",
                            delta="Bueno" if silhouette > 0.5 else "Regular"
                        )
                    with cols[1]:
                        st.metric(
                            "Calinski Score",
                            f"{calinski:.1f}",
                            delta="Bueno" if calinski > 1000 else "Regular"
                        )
                    with cols[2]:
                        st.metric(
                            "Davies Score",
                            f"{davies:.3f}",
                            delta="Bueno" if davies < 0.5 else "Regular"
                        )
                        
                        
                

            # Tabla de resultados detallados
            with st.expander("📑 Detalles de las Predicciones"):
                results_df = pd.DataFrame({
                    'ID': data.index,
                    'Score de Anomalía': average_scores,
                    'Clasificación': ['0' if x == 0 else '1' for x in y_pred]
                })
                st.dataframe(results_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error en el procesamiento: {str(e)}")
            loading_placeholder.empty()

# Control de navegación
if st.session_state.page == "home":
    show_home_page()
else:
    show_dashboard_page()

# Pie de página mejorado
st.markdown("---")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🛡 Sistema de Detección de Anomalías IoT</p>
            <p>© 2024 Innovasic | UCC</p>
        </div>
        """,
        unsafe_allow_html=True)