import numpy as np
import matplotlib.pyplot as plt

# Datos codificados directamente (8 combinaciones de entrada y su salida)
# Cada fila representa: fiebre, dolor de cabeza, fatiga, diagnóstico
datos = np.array([
    [-1, -1, -1, -1],
    [-1, -1,  1, -1],
    [-1,  1, -1, -1],
    [-1,  1,  1,  1],
    [ 1, -1, -1, -1],
    [ 1, -1,  1,  1],
    [ 1,  1, -1,  1],
    [ 1,  1,  1,  1]
])

# Separación de patrones de entrada y salida deseada
A = datos[:, 0]
B = datos[:, 1]
C = datos[:, 2]
X = np.column_stack((A, B, C))
T = datos[:, 3]

# Inicialización de pesos y bias
np.random.seed(0)  # Reproducibilidad
b = np.random.rand()
W = np.random.rand(3)

ep = 0.1  # Tolerancia
errorG = [1000]  # Error global inicial
epch = 0
Y = np.zeros(len(T), dtype=int)

# Función de activación tipo hardlims
def hardlims(a):
    return 1 if a >= 0 else -1

# Entrenamiento del perceptrón
while errorG[-1] >= ep:
    e = []
    print(f"\n==== Época {epch + 1} ====")
    for k in range(len(T)):
        entrada = X[k]
        salida_esperada = T[k]
        a = W[0] * A[k] + W[1] * B[k] + W[2] * C[k] + b
        Y[k] = hardlims(a)
        error = salida_esperada - Y[k]
        e.append(error)

        # Imprimir resultados parciales
        print(f"Entrada: {entrada}, Pesos: {W}, Sesgo: {b:.4f}, Salida predicha: {Y[k]}, Salida esperada: {salida_esperada}")

        if error != 0:
            # Actualización de pesos y bias
            W[0] += error * A[k]
            W[1] += error * B[k]
            W[2] += error * C[k]
            b += error
            print(f" -> Actualizando pesos a: {W} y sesgo a: {b:.4f}")

    epch += 1
    errorG.append(np.sum(np.square(e)) / 2)

# Resultados finales
print(f"\nEntrenamiento completo en {epch} épocas.")
print("=== Resultados finales ===")
print(f"Pesos finales: {W}")
print(f"Sesgo final: {b}")

# Verificación con los datos de entrada
print("=== Verificación con los datos de entrada ===")
for k in range(len(T)):
    entrada = X[k]
    salida_esperada = T[k]
    salida_predicha = hardlims(W[0] * A[k] + W[1] * B[k] + W[2] * C[k] + b)
    print(f"Entrada: {entrada}, Salida predicha: {salida_predicha}, Salida esperada: {salida_esperada}")

# Gráfica del error global por época
plt.figure()
plt.semilogy(range(len(errorG)), errorG)
plt.title("Error Global por Época")
plt.xlabel("Época")
plt.ylabel("Error Global (log)")
plt.grid(True)
plt.show()


# Prueba con un paciente ingresado por el usuario
print("\n=== Prueba interactiva con un nuevo paciente ===")
print("Por favor, ingrese los síntomas del paciente usando -1 (no) y 1 (sí)")

try:
    fiebre = int(input("¿Tiene fiebre? (-1 = No, 1 = Sí): "))
    dolor_cabeza = int(input("¿Tiene dolor de cabeza? (-1 = No, 1 = Sí): "))
    fatiga = int(input("¿Tiene fatiga? (-1 = No, 1 = Sí): "))

    # Validación básica
    if fiebre not in [-1, 1] or dolor_cabeza not in [-1, 1] or fatiga not in [-1, 1]:
        raise ValueError("Solo se permiten valores -1 o 1.")

    nuevo_paciente = np.array([fiebre, dolor_cabeza, fatiga])
    salida_predicha_nuevo = hardlims(np.dot(W, nuevo_paciente) + b)

    estado = "ENFERMO" if salida_predicha_nuevo == 1 else "SANO"
    print(f"\nDiagnóstico: El paciente está {estado} (salida = {int(salida_predicha_nuevo)})")

except ValueError as e:
    print(f"Error de entrada: {e}")
