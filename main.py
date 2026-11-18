from sklearn.metrics import classification_report
import numpy as np
import time
inicio = time.perf_counter()

from src.perceptron import Perceptron
from src.visualization import (
    plot_training_history,
    plot_confusion_matrix
)


def main():

    # 1. Cargar dataset
    datos = np.loadtxt("original/Base_de_datos.txt")

    X = datos[:, :3]
    y = datos[:, 3]

    # 2. Crear modelo
    model = Perceptron(
        learning_rate=1.0,
        max_epochs=1000,
        random_state=0
    )

    # 3. Entrenar
    model.fit(X, y)

    # 4. Predecir
    predictions = model.predict(X)

    # 5. Mostrar resultados
    accuracy = np.mean(predictions == y) * 100

    print("\n========== PERCEPTRÓN LÓGICO ==========")
    print(f"Convergencia: {model.converged}")
    print(f"Épocas: {model.n_epochs}")
    print(f"Pesos: {model.weights}")
    print(f"Bias: {model.bias:.6f}")
    print(f"Exactitud: {accuracy:.2f}%")

    print("\n========== PREDICCIONES ==========")

    for entrada, real, prediccion in zip(X, y, predictions):
        print(
            f"Entrada: {entrada} | "
            f"Real: {int(real)} | "
            f"Predicción: {int(prediccion)}"
        )

    print("\n========== MÉTRICAS DE EVALUACIÓN ==========")

    print(
        classification_report(
            y,
            predictions,
            labels=[-1, 1],
            target_names=["Clase negativa", "Clase positiva"],
            zero_division=0
        )
    )

    print(f"\nTiempo antes de las gráficas: {time.perf_counter() - inicio:.4f} segundos")

    # 6. Visualizaciones
    plot_training_history(model)
    matrix = plot_confusion_matrix(y, predictions)

    print("\nMatriz de confusión:")
    print(matrix)

    print(f"Tiempo total: {time.perf_counter() - inicio:.4f} segundos")

if __name__ == "__main__":
    main()
