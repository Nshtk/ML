import numpy as np
import matplotlib.pyplot as plt
import scipy


def grad_descent(learning_rate, num_iter=1000):
    x, y = 1.5, -1
    weight, bias=0.1, 0.01
    steps = []

    for iter_num in range(num_iter):
        steps.append([cur_x1, cur_x2, f(weight, bias)])

        # чтобы обновить значения cur_x1 и cur_x2, как мы помним с последнего занятия,
        # нужно найти производные (градиенты) функции f по этим переменным.
        x_gradient =learning_rate *
        y_gradient =

        weight -=f()
        bias -=
    return np.array(steps)

def main():
    import numpy as np

    # Используем векторную запись точки.

    x = np.array([1, 2])

    def f(x):

    # f(x) = 0.5x^2 + 0.2y^2

    return 0.5 * x[0] ** 2 + 0.2 * x[1] ** 2

    def grad(x):

    # grad(x) = [x, 0.4y]

    dx = x[0]

    dy = 0.4 * x[1]

    return np.array([dx, dy])

    print("Точка 0:", x)

    print("f(x) =", f(x))

    print()

    # Двигаем точку против градиента.

    x = x - grad(x)

    print("Точка 1:", x)

    print("f(x) =", f(x))



def F(x1, x2):
    return ((np.sin(x1 - 1) + x2 - 0.1) ** 2) + ((x1 - np.sin(x2 + 1) - 0.8) ** 2)


# Градиент
def Grad(x1, x2):
    return np.array([2 * x1 + 2 * (x2 + np.sin(x1 - 1) - 0.1) * np.cos(x1 - 1) - 2 * np.sin(x2 + 1) - 1.6,
                     2 * x2 - 2 * (x1 - np.sin(x2 + 1) - 0.8) * np.cos(x2 + 1) + 2 * np.sin(x1 - 1) - 0.2])

    # сам метод


def Gr_m(x1, x2, cnt=10000):
    alpha = 0.1  # Шаг сходимости

    eps = 0.000001  # точность

    X_prev = np.array([x1, x2])

    X = X_prev - alpha * Grad(X_prev[0], X_prev[1])

    t = 50  # Необходимый параметр, если выбрать  шаг №1
    k = 0

    l, s, p = 0.1, 1, 0.5  # Необходимые параметры для варианта шага №2

    while np.linalg.norm(X - X_prev) > eps and k < cnt:
        X_prev = X.copy()

        # alpha = 1/min(k+1,t)     # Шаг №1
        k = k + 1

        # alpha = l * ((s/(s+k))**p)   # Шаг №2

        alpha = F(X_prev[0] - alpha * Grad(X_prev[0], X_prev[1])[0],
                  X_prev[1] - alpha * Grad(X_prev[0], X_prev[1])[1])

        # Шаг №3. Не рабочий

        X = X_prev - alpha * Grad(X_prev[0], X_prev[1])  # Формула

    return X, k

if __name__ == '__main__':
    result = Gr_m(0, 0)

    print(result)

