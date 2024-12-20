# Proyecto de Clasificación de Puntaje de Crédito

Este proyecto analiza y predice puntajes de crédito utilizando una red neuronal secuencial de Keras/TensorFlow. Se asume que el conjunto de datos utilizado ha sido previamente limpiado y explorado.

## Análisis de Datos

Los datos se cargan desde un archivo CSV llamado `credit_score_cleaned.csv`. Se utilizan bibliotecas como `pandas`, `numpy`, `seaborn` y `matplotlib` para realizar el análisis exploratorio de datos (EDA).

Los pasos principales del análisis incluyen:

*   **Carga de datos:** Se utiliza `pd.read_csv()` para cargar el conjunto de datos.
*   **Visualización inicial:** Se usan `df.head()` para mostrar las primeras filas y `df.info()` para obtener información general sobre los tipos de datos y la presencia de valores nulos. `df.describe()` proporciona estadísticas descriptivas de las columnas numéricas.
*   **Análisis de correlación:** Se calcula la matriz de correlación entre las variables numéricas usando `df.corr()` y se visualiza con un mapa de calor (`sns.heatmap`) para identificar relaciones lineales entre las variables, incluyendo la correlación con el puntaje de crédito.
*   **Análisis de la distribución de las variables:** Se utilizan boxplots (`sns.boxplot`) para visualizar la distribución de cada variable numérica en relación con las diferentes categorías del puntaje de crédito, permitiendo identificar posibles diferencias en la distribución y la presencia de valores atípicos. Tambien se utilizan histogramas (`sns.histplot`) para visualizar la distribución de cada variable numerica
*   **Detección de valores atípicos (Outliers):** Se implementa una función (`detect_outliers_iqr`) que utiliza el rango intercuartílico (IQR) para identificar posibles valores atípicos en las variables numéricas. Se define un valor atípico como aquel que se encuentra fuera del rango definido por 3 veces el IQR sumado/restado al tercer/primer cuartil.
*   **Análisis del desbalanceo de clases:** Se analiza la distribución de las clases en la variable objetivo (`credit_score`) usando `df["credit_score"].value_counts(normalize = True)` para determinar si existe desbalanceo y la necesidad de aplicar técnicas de remuestreo.

## Preprocesamiento de Datos

El preprocesamiento de datos es crucial para preparar los datos para el entrenamiento del modelo. En este proyecto se realizan las siguientes transformaciones:

*   **Ingeniería de características (Feature Engineering):** Se crean nuevas columnas binarias a partir de la columna `type_of_loan`, representando la presencia o ausencia de cada tipo de préstamo.
*   **Eliminación de columnas:** Se eliminan columnas irrelevantes como `id`, `customer_id`, `name`, `ssn` y la columna original `type_of_loan` ya que la informacion se encuentra en las columnas creadas.
*   **Mapeo de variables categóricas ordinales:** Se mapean las columnas `payment_behaviour` y `credit_mix` a valores numéricos según un orden lógico.
*   **Codificación One-Hot:** Se aplica codificación one-hot a la columna `occupation` para convertirla en variables numéricas.
*   **Mapeo de la columna mes:** Se mapea la columna `month` a valores numericos.
*   **Manejo del desbalanceo de clases:** Se aplica la técnica de remuestreo SMOTE (`SMOTE()`) para balancear las clases en la variable objetivo. Esto ayuda a prevenir que el modelo esté sesgado hacia la clase mayoritaria.

## Modelo

Se utiliza una red neuronal secuencial de Keras/TensorFlow para la clasificación del puntaje de crédito. La arquitectura del modelo es la siguiente:

```python
model = Sequential()
model.add(Dense(units=128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001))) # Regularizacion L2
model.add(BatchNormalization()) # BatchNormalization
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.001))) # Regularizacion L2
model.add(BatchNormalization()) # BatchNormalization
model.add(Dropout(0.2))
model.add(Dense(units=3, activation='softmax')) # Softmax para clasificacion multiclase

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_cross
