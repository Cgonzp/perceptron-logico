# Perceptrón Lógico

Implementación de un **perceptrón monocapa** desarrollado en Python para resolver un problema de clasificación binaria mediante aprendizaje supervisado. El proyecto incluye entrenamiento, predicción, visualización del proceso de aprendizaje y evaluación mediante métricas de clasificación.

## Descripción

Este proyecto implementa desde cero el algoritmo del perceptrón, utilizando NumPy para las operaciones matemáticas y Matplotlib para representar gráficamente su comportamiento durante el entrenamiento.

El modelo trabaja con entradas y salidas binarias bipolares (`-1` y `1`), empleando la función de activación `hardlims`.

La implementación permite:

- Entrenar un perceptrón mediante aprendizaje supervisado.
- Actualizar pesos y sesgo de acuerdo con el error de clasificación.
- Determinar la convergencia del modelo.
- Realizar predicciones sobre los patrones de entrada.
- Visualizar el error y los errores de clasificación por época.
- Generar una matriz de confusión y métricas de evaluación.
- Ejecutar pruebas automatizadas para verificar el funcionamiento.

## Fundamentos matemáticos

### 1. Modelo del perceptrón

El perceptrón calcula una combinación lineal de las entradas:

\[
net = \sum_{i=1}^{n} w_i x_i + b
\]

Donde:

- \(x_i\): valores de entrada.
- \(w_i\): pesos sinápticos.
- \(b\): sesgo (*bias*).
- \(n\): número de características.

### 2. Función de activación

Se utiliza la función `hardlims`, que produce una salida bipolar:

\[
f(net)=
\begin{cases}
1 & \text{si } net \geq 0\\
-1 & \text{si } net < 0
\end{cases}
\]

### 3. Regla de aprendizaje

Durante el entrenamiento, se calcula el error entre la salida esperada y la predicción:

\[
e = y-\hat{y}
\]

Los pesos y el sesgo se actualizan mediante:

\[
w_i^{nuevo}=w_i^{anterior}+\eta e x_i
\]

\[
b^{nuevo}=b^{anterior}+\eta e
\]

Donde \(\eta\) representa la tasa de aprendizaje.

### 4. Criterio de convergencia

El entrenamiento finaliza cuando todos los patrones del conjunto de entrenamiento son clasificados correctamente o cuando se alcanza el número máximo de épocas establecido.

## Estructura del proyecto

```text
PERCEPTRON_LOGICO/
│
├── original/                  # Implementaciones originales
│   ├── primer.py
│   ├── segundo.py
│   └── Base_de_datos.txt
│
├── src/                       # Código fuente modular
│   ├── __init__.py
│   ├── perceptron.py
│   └── visualization.py
│
├── tests/                     # Pruebas automatizadas
│   └── test_perceptron.py
│
├── results/                   # Resultados y evidencias
│
├── main.py                    # Punto de entrada
├── requirements.txt           # Dependencias
├── .gitignore
└── README.md
```

## Tecnologías utilizadas

| Tecnología | Aplicación |
|---|---|
| Python | Lenguaje de programación |
| NumPy | Operaciones matemáticas y procesamiento de datos |
| Matplotlib | Visualización del entrenamiento |
| Scikit-learn | Métricas de evaluación |
| Pytest | Pruebas automatizadas |

## Instalación y configuración

### 1. Clonar el repositorio

```bash
git clone URL_DEL_REPOSITORIO
cd PERCEPTRON_LOGICO
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
```

### 3. Activar el entorno virtual

En Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecución

Para ejecutar el entrenamiento y visualizar los resultados:

```bash
python main.py
```

El programa carga el conjunto de datos, inicializa el perceptrón, realiza el entrenamiento y presenta las predicciones y métricas de evaluación.

## Pruebas automatizadas

El proyecto incorpora pruebas mediante Pytest para verificar:

- Funcionamiento de la función de activación.
- Convergencia del entrenamiento.
- Generación de predicciones.
- Validación de etiquetas.
- Validación de dimensiones de entrada.
- Restricción de predicción antes del entrenamiento.
- Construcción de la matriz de confusión.
- Comportamiento del historial de entrenamiento sin ajustar el modelo.

Para ejecutar las pruebas:

```bash
python -m pytest -v
```

## Resultados experimentales

En la ejecución de referencia sobre el conjunto de datos incluido se obtuvieron los siguientes resultados:

| Indicador | Resultado |
|---|---:|
| Patrones de entrenamiento | 8 |
| Características de entrada | 3 |
| Clases | 2 |
| Épocas ejecutadas | 4 |
| Convergencia | Sí |
| Exactitud sobre entrenamiento | 100 % |

### Matriz de confusión

```text
                 Predicción
                 -1    1

Real       -1     1    0
            1     0    7
```

Las métricas obtenidas sobre los ocho patrones fueron:

- Precisión: 1.00.
- Recall: 1.00.
- F1-score: 1.00.

**Nota:** estos resultados corresponden al conjunto utilizado durante el entrenamiento. No representan una medición de generalización sobre datos independientes.

## Limitaciones

- El perceptrón monocapa está limitado a problemas linealmente separables.
- El entrenamiento depende del orden de presentación de los patrones.
- Los resultados pueden variar según la inicialización de los pesos, la tasa de aprendizaje y los datos utilizados.
- El conjunto de datos es pequeño y tiene fines académicos.
- Las métricas actuales no sustituyen una evaluación con datos de prueba independientes.

## Propósito académico

El proyecto tiene como finalidad estudiar e implementar los fundamentos de las redes neuronales artificiales, comprendiendo el proceso de aprendizaje supervisado, la actualización de pesos, la función de activación y la clasificación binaria.

---

**Desarrollado en Python | Proyecto académico de Inteligencia Artificial**