
import numpy as np


def hardlims(value: float | np.ndarray) -> int | np.ndarray:
    """
    Función de activación bipolar Hard Limit Symmetric.

    Retorna:
        1  si value >= 0
        -1 si value < 0
    """
    return np.where(np.asarray(value) >= 0, 1, -1)


class Perceptron:
    """
    Implementación de un perceptrón simple para clasificación binaria.

    Utiliza:
        - Entradas bipolares.
        - Salidas {-1, 1}.
        - Activación hardlims.
        - Aprendizaje supervisado.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_epochs: int = 1000,
        random_state: int | None = 0
    ):
        if learning_rate <= 0:
            raise ValueError("La tasa de aprendizaje debe ser positiva.")

        if max_epochs <= 0:
            raise ValueError("El número máximo de épocas debe ser positivo.")

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state

        self.weights = None
        self.bias = None
        self.history = []
        self.converged = False
        self.n_epochs = 0

    def initialize(self, n_features: int) -> None:
        """Inicializa los pesos y el sesgo aleatoriamente."""

        rng = np.random.default_rng(self.random_state)

        self.weights = rng.random(n_features)
        self.bias = float(rng.random())

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Calcula la combinación lineal de entradas, pesos y sesgo."""

        self._check_fitted()

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != len(self.weights):
            raise ValueError("Las dimensiones de X no coinciden con el modelo.")

        return np.dot(X, self.weights) + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice las clases utilizando hardlims."""

        return hardlims(self.decision_function(X))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Entrena el perceptrón mediante actualización iterativa."""

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X debe ser una matriz no vacía.")

        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("Las dimensiones de X e y no coinciden.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contiene valores no finitos.")

        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("Las clases deben ser -1 o 1.")

        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("Las etiquetas deben ser numéricas.")

        y = y.astype(int)

        self.initialize(X.shape[1])
        self.history = []
        self.converged = False

        for epoch in range(1, self.max_epochs + 1):

            # Entrenamiento en línea: una muestra a la vez.
            for xi, target in zip(X, y):

                net_input = np.dot(self.weights, xi) + self.bias
                prediction = int(hardlims(net_input))

                error = target - prediction

                if error != 0:
                    self.weights += (
                        self.learning_rate * error * xi
                    )
                    self.bias += self.learning_rate * error

            # Evaluación después de actualizar todos los patrones.
            predictions = self.predict(X)

            errors = y - predictions
            mse = float(np.mean(np.square(errors)))

            mistakes = int(np.sum(y != predictions))

            self.history.append({
                "epoch": epoch,
                "mse": mse,
                "mistakes": mistakes
            })

            self.n_epochs = epoch

            # El entrenamiento termina cuando clasifica
            # correctamente todas las muestras.
            if mistakes == 0:
                self.converged = True
                break

        return self

    def _check_fitted(self) -> None:
        """Verifica que el modelo haya sido inicializado."""

        if self.weights is None or self.bias is None:
            raise RuntimeError(
                "El perceptrón todavía no ha sido entrenado."
            )
