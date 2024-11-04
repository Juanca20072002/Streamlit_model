import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pyod.utils.utility import standardizer  
from pyod.models.combination import average  


# Rutas a los modelos
MODEL_PATHS = {
    "k_40": "model/lof_model_k_40.pkl",
    "k_50": "model/lof_model_k_50.pkl",
    "k_60": "model/lof_model_k_60.pkl",
    "k_70": "model/lof_model_k_70.pkl"
}

# Función para cargar los modelos
def load_models():
    models = {}
    for key, path in MODEL_PATHS.items():
        try:
            model = joblib.load(path)
            models[key] = model
            st.success(f"Modelo LOF k={key} cargado exitosamente.")
        except FileNotFoundError:
            st.error(f"El archivo del modelo {key} no se encontró en la ruta: {path}")
        except Exception as e:
            st.error(f"Error al cargar el modelo {key}: {e}")
    return models

# Función para calcular la matriz de confusión
def confusion_matrix_threshold(actual, score, threshold):
    Actual_pred = pd.DataFrame({'Actual': actual, 'Pred': score})
    Actual_pred['Pred'] = np.where(Actual_pred['Pred'] <= threshold, 0, 1)
    cm = pd.crosstab(Actual_pred['Actual'], Actual_pred['Pred'], rownames=['Actual'], colnames=['Pred'])
    return cm

# Cargar los modelos
models = load_models()

# Interfaz de usuario de Streamlit
st.title('Innovasic Model V1')
st.write('Bienvenido a la aplicación de Innovasic Model V1. Aquí puedes predecir anomalías en tus datos.')

# Carga del archivo CSV
st.subheader("Subir archivo CSV para predicción")
uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

if uploaded_file is not None:
    # Leer el archivo CSV en un DataFrame de Pandas
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Datos cargados exitosamente:")
        st.dataframe(data)

        if st.button('Predecir'):
            if models:
                try:
                    # Inicializar un DataFrame para almacenar los scores de decision_function
                    test_scores = np.zeros((data.shape[0], len(models)))

                    # Calcular decision_function para cada modelo
                    for i, (key, model) in enumerate(models.items()):
                        test_scores[:, i] = model.decision_function(data.values)

                    # Normalizar los scores utilizando la función standardizer de pyod
                    test_scores_norm = standardizer(test_scores)

                    # Promediar los scores normalizados usando la función average
                    average_scores = average(test_scores_norm)

                    # Umbral de decisión (puedes ajustar este umbral según tu caso)
                    threshold = 0.000009  # Este es un ejemplo, ajusta según tu necesidad
                    y_pred = np.where(average_scores <= threshold, 0, 1)

                    # Crear un DataFrame con los resultados
                    results = pd.DataFrame({'index': data.index, 'Average Scores': average_scores, 'Predictions': y_pred})
                    st.write("Predicciones realizadas:")
                    st.dataframe(results)

                    # Si tienes etiquetas verdaderas para la evaluación
                    # y_test = ... # Deberías tener tus etiquetas verdaderas
                    # cm = confusion_matrix_threshold(y_test, average_scores, threshold)
                    # st.write("Matriz de confusión:")
                    # st.dataframe(cm)

                except Exception as e:
                    st.error(f"Error al realizar la predicción: {e}")
            else:
                st.error("Los modelos no están cargados correctamente.")
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")

# Agregar un pie de página
st.markdown("<hr>", unsafe_allow_html=True)
st.write("© 2024 Innovasic. Todos los derechos reservados.")
