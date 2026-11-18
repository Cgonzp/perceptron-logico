import numpy as np
import matplotlib.pyplot as plt

# Carga de datos desde un archivo de texto
datos = np.loadtxt('Base_de_datos.txt')

# Las tres primeras columnas son las entradas (x1, x2, x3), la última es la salida deseada T
X = datos[:, 0:3]
T = datos[:, 3]

# Inicialización fija de pesos y bias
#b = 4.544883182996896
#W = np.array([0.5488135, 0.71518937, 0.60276338])


# Inicialización de pesos y bias
np.random.seed(0)
b = 1 * np.random.rand()
W = 1 * np.random.rand(3)

ep = 0.0001  # Tolerancia
errorG = [1000]
epch = 0
Y = np.zeros(len(T))

# Función de activación tipo hardlims
def hardlims(a):
    return 1 if a >= 0 else -1

# Entrenamiento del perceptrón
while errorG[-1] >= ep:
    e = []
    for k in range(len(T)):
        a = np.dot(W, X[k]) + b
        Y[k] = hardlims(a)
        error = T[k] - Y[k]
        e.append(error)
        W += error * X[k]
        b += error
    epch += 1
    errorG.append(np.sum(np.square(e)) / 2)

    # Mostrar resultados por época
    print(f"Época {epch}, Error Global: {errorG[-1]:.4f}")
    print("  Pesos W:", W)
    print("  Bias b:", b)
    print("  Salida estimada Y:", Y)
    print("  Salida deseada T:", T)
    print("-" * 40)
    
    # Puedes descomentar la siguiente línea si quieres que se pause cada época
    # input("Presione Enter para continuar...")

# Resultados finales
print("Pesos Finales W:", W)
print("Bias Final b:", b)

# Gráfica del error global por época
plt.figure()
plt.semilogy(range(len(errorG)), errorG)
plt.title("Error Global por Época")
plt.xlabel("Época")
plt.ylabel("Error Global (log)")
plt.grid(True)
plt.show()