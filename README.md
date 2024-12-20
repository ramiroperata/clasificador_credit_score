# Proyecto de Clasificación de Puntaje de Crédito

Este proyecto analiza y predice puntajes de crédito utilizando una red neuronal secuencial de Keras/TensorFlow. Se asume que el conjunto de datos utilizado ha sido previamente limpiado y explorado.

## Análisis de Datos

Los datos se cargan desde un archivo CSV llamado `credit_score_cleaned.csv`. Se utilizan bibliotecas como `pandas`, `numpy`, `seaborn` y `matplotlib` para realizar el análisis exploratorio de datos (EDA).

Los pasos principales del análisis incluyen:

*   **Carga de datos**
*   **Visualización inicial:** Se obervan las primeras filas, los tipos de los datos y valores nulos.
*   **Análisis de correlación:** Se calcula la matriz de correlación entre las variables numéricas usando `df.corr()` y se visualiza con un mapa de calor (`sns.heatmap`) para identificar relaciones lineales entre las variables, incluyendo la correlación con el puntaje de crédito.
*   **Análisis de la distribución de las variables:** Se utilizan boxplots (`sns.boxplot`) para visualizar la distribución de cada variable numérica en relación con las diferentes categorías del puntaje de crédito, permitiendo identificar posibles diferencias en la distribución y la presencia de valores atípicos. Tambien se utilizan histogramas (`sns.histplot`) para visualizar la distribución de cada variable numerica
*   **Detección de valores atípicos (Outliers):** Se implementa una función que utiliza el rango intercuartílico (IQR) para identificar posibles valores atípicos en las variables numéricas. Se define un valor atípico como aquel que se encuentra fuera del rango definido por 3 veces el IQR sumado/restado al tercer/primer cuartil.
*   **Análisis del desbalanceo de clases:** Se analiza la distribución de las clases en la variable objetivo para determinar si existe desbalanceo y la necesidad de aplicar técnicas de remuestreo.

## Preprocesamiento de Datos

El preprocesamiento de datos es crucial para preparar los datos para el entrenamiento del modelo. En este proyecto se realizan las siguientes transformaciones:

*   **Ingeniería de características (Feature Engineering):** Se crean nuevas columnas binarias a partir de la columna `type_of_loan`, representando la presencia o ausencia de cada tipo de préstamo.
*   **Eliminación de columnas:** Se eliminan columnas irrelevantes como `id`, `customer_id`, `name`, `ssn` y la columna original `type_of_loan` ya que la informacion se encuentra en las columnas creadas.
*   **Mapeo de variables categóricas ordinales:** Se mapean las columnas `payment_behaviour` y `credit_mix` a valores numéricos según un orden lógico.
*   **Codificación One-Hot:** Se aplica codificación one-hot a la columna `occupation` para convertirla en variables numéricas.
*   **Mapeo de la columna mes:** Se mapea la columna `month` a valores numericos.
*   **Manejo del desbalanceo de clases:** Se aplica la técnica de remuestreo SMOTE (`SMOTE()`) para balancear las clases en la variable objetivo. Esto ayuda a prevenir que el modelo esté sesgado hacia la clase mayoritaria.

## Modelo

Se utiliza una red neuronal secuencial de Keras/TensorFlow para la clasificación del credit score. La arquitectura del modelo es la siguiente:

```python
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(256, input_dim=X_train.shape[1], activation='relu'), # Activacion rectified linear para aprender relaciones no lineales
    BatchNormalization(), # Normalizacion por lote
    Dropout(0.25), # Dejamos fuera el 25% de los datos para evitar sobreajuste
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(3, activation='softmax') # Softmax para clasificacion multiclase
])
model.compile(optimizer = Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics = ['accuracy'])
```
Este modelo consta de:
* **Capas Densas (Dense)**: Capas totalmente conectadas que aprenden representaciones complejas de los datos. Se agrega regularizacion L2 a las capas densas para evitar el overfitting
* **Activación ReLU**: Se utiliza la función de activación ReLU para introducir no linealidad en el modelo.
* **Dropout**: Capas de Dropout para regularizar el modelo y evitar el sobreajuste.
* **Batch Normalization**: Normaliza las activaciones de la capa anterior para estabilizar y acelerar el entrenamiento.
* **Capa de salida con Softmax**: Se utiliza una capa densa con función de activación Softmax para la clasificación multiclase (tres categorías de puntaje de crédito).
* **Optimizador Adam**: Un optimizador eficiente para el entrenamiento de redes neuronales.
* **Función de pérdida sparse_categorical_crossentropy**: Adecuada para problemas de clasificación multiclase donde las etiquetas son números enteros.
* **Early Stopping**: Se implementa Early Stopping para detener el entrenamiento cuando la pérdida de validación deja de mejorar, evitando el sobreajuste y guardando los mejores pesos del modelo.


Es una arquitectura bastante simple, pero dio muy buenos resultados y servira bien bien como modelo base.

Ventajas de usar Redes Neuronales para la Clasificación de Puntaje de Crédito:

* **Capacidad para modelar relaciones no lineales**: Las redes neuronales pueden capturar relaciones complejas y no lineales entre las variables predictoras y el puntaje de crédito, lo cual es común en datos financieros.
* **Adaptabilidad**: Las redes neuronales pueden aprender y adaptarse a diferentes tipos de datos y patrones, lo que las hace versátiles para problemas de clasificación.
* **Manejo de alta dimensionalidad**: Pueden manejar eficientemente conjuntos de datos con una gran cantidad de características.
* **Extracción automática de características**: Las capas ocultas de la red neuronal aprenden representaciones de características de los datos, lo que puede ser útil cuando no se conocen las características más relevantes.

## Evaluación

El rendimiento del modelo se evalúa utilizando métricas como:

* **Exactitud (Accuracy)**: Proporción de predicciones correctas.
* **Precisión (Precision)**: Proporción de verdaderos positivos entre los predichos como positivos.
* **Recuperación (Recall)**: Proporción de verdaderos positivos identificados correctamente.
* **Puntaje F1 (F1-score)**: Media armónica de precisión y recuperación.
* **Matriz de confusión**: Visualización de la cantidad de instancias que se clasificaron correctamente y las que se clasificaron incorrectamente en cada clase.

## Resultados

El modelo consiguio una f1-score del 91%. Tambien tuvo un rendimiento muy bueno para identificar a los clientes de riesgo malo, con un recall del 95% para esta clase. Esto quiere decir que identifico correctamente al 95% de los clientes con un mal riesgo, permitiendonos evitarlos para mantener un buen perfil de riego. Con algunos cambios en la arquitectura sera muy facil consegui resultados aun mejores.

## Conclusion

Este proyecto demuestra la aplicación de aprendizaje automático para la clasificación de credit scores. Al analizar los datos, preprocesarlos adecuadamente y elegir un modelo adecuado, se puede desarrollar un sistema para predecir el riesgo. Los resultados obtenidos proporcionarán información sobre la eficacia del modelo y las posibles áreas de mejora.
