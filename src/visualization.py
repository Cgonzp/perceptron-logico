
import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(model, show=True):
    """
    Visualiza la evolución del MSE y los errores de clasificación
    durante el entrenamiento.
    """

    if not model.history:
        raise ValueError("El modelo no tiene historial de entrenamiento.")

    epochs = [item["epoch"] for item in model.history]
    mse = [item["mse"] for item in model.history]
    mistakes = [item["mistakes"] for item in model.history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Gráfica del error cuadrático medio
    axes[0].plot(epochs, mse, marker="o", linewidth=2)
    axes[0].set_title("Error cuadrático medio (MSE)")
    axes[0].set_xlabel("Épocas")
    axes[0].set_ylabel("MSE")
    axes[0].grid(alpha=0.3)

    # Gráfica de errores de clasificación
    axes[1].plot(
        epochs,
        mistakes,
        marker="o",
        color="red",
        linewidth=2
    )
    axes[1].set_title("Errores de clasificación")
    axes[1].set_xlabel("Épocas")
    axes[1].set_ylabel("Número de errores")
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Entrenamiento del Perceptrón - "
        f"{model.n_epochs} épocas",
        fontsize=14
    )

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, show=True):
    """
    Genera una matriz de confusión para clasificación bipolar.
    
    Clases:
        -1: Clase negativa
         1: Clase positiva
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Las etiquetas deben ser vectores.")

    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Las etiquetas deben tener igual longitud y no estar vacías.")

    if not np.all(np.isin(y_true, [-1, 1])):
        raise ValueError("Las etiquetas reales deben ser -1 o 1.")

    if not np.all(np.isin(y_pred, [-1, 1])):
        raise ValueError("Las predicciones deben ser -1 o 1.")

    classes = [-1, 1]

    matrix = np.zeros((2, 2), dtype=int)

    for real, predicted in zip(y_true, y_pred):
        row = classes.index(real)
        column = classes.index(predicted)
        matrix[row, column] += 1

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title("Matriz de confusión")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["-1", "1"])
    ax.set_yticklabels(["-1", "1"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=14
            )

    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return matrix
