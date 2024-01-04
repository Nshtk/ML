import math
import scipy
import numpy
from pylab import *
from sympy import *
import matplotlib.pyplot as plt

eps = 0.1
e = 10e-8

# def getGradientNumerical(func, vector):
#     vec_len = vector.shape[0]
#     gradient = np.zeros((vec_len, 1))
#     for i in range(vec_len):
#         base_vector =np.zeros((vec_len, 1))
#         base_vector[i, 0] = 1
#         func_plus = func(vector + (eps * base_vector))
#         func_minus = func(vector - (eps * base_vector))
#         a = ((func_plus - func_minus) / (2 * eps))
#     return gradient
def getGradientSymbolic(vector):
    x, y = Symbol('x'),  Symbol('y')
    function = eval(function_string)
    diff_y = str(function.diff(y)).replace('x', str(vector[0])).replace('y', str(vector[1]))
    diff_x = str(function.diff(x)).replace('x', str(vector[0])).replace('y', str(vector[1]))
    return numpy.array([eval(diff_x), eval(diff_y)])
def getGradient(vector: np.ndarray, dt: float = 0.00001) -> np.array:
    dxdt = (function(vector + np.array([dt, 0])) - function(vector)) / dt
    dydt = (function(vector + np.array([0, dt])) - function(vector)) / dt
    return numpy.array([dxdt, dydt])

def getGradientDescentPath(getGradient, vector: np.ndarray, minimum_global: np.ndarray, iterations_max: int = 64, learning_rate: float = 0.1) -> np.array:
    vector_copy = vector.copy()
    path = [np.array([vector_copy[0], vector_copy[1], function(vector_copy)])]

    i = 0
    while i < iterations_max and np.linalg.norm(path[-1] - minimum_global) > eps:
        vector_copy = vector_copy - learning_rate * getGradient(vector_copy)
        path.append(np.array([vector_copy[0], vector_copy[1], function(vector_copy)]))
        i += 1

    return np.array(path)
def getGradientDescentInertialPath(getGradient, vector: np.ndarray, minimum_global: np.ndarray, iterations_max: int = 64, learning_rate: float = 0.1, inertia_coef: float = 0.5) -> np.array:
    vector_copy = vector.copy()
    vector_copy_previous = vector.copy()
    path = [np.array([vector[0], vector[1], function(vector)])]

    i = 0
    while i < iterations_max and np.linalg.norm(path[-1] - minimum_global) > eps:
        vector_copy_tmp=vector_copy
        vector_copy = vector_copy - learning_rate * getGradient(vector_copy) + inertia_coef * (vector_copy - vector_copy_previous)
        vector_copy_previous = vector_copy_tmp
        path.append(np.array([vector_copy[0], vector_copy[1], function(vector_copy)]))
        i += 1

    return np.array(path)

def getGradientDescentAdaptivePath(getGradient, vector: np.ndarray, minimum_global: np.ndarray, iterations_max: int = 64, learning_rate: float = 0.1, adaptive_coef_1: float=0.5, adaptive_coef_2: float=0.9) -> np.array:
    vector_copy = vector.copy()
    path = [np.array([vector_copy[0], vector_copy[1], function(vector_copy)])]
    m_1 = np.array([0, 0])
    m_2 = np.array([0, 0])

    i = 0
    while i < iterations_max and np.linalg.norm(path[-1] - minimum_global) > eps:
        m_1 = adaptive_coef_1 * m_1 + (1 - adaptive_coef_1) * getGradient(vector)
        m_2 = adaptive_coef_2 * m_2 + (1 - adaptive_coef_2) * getGradient(vector) ** 2
        vector_copy = vector_copy - learning_rate * m_1 / (np.sqrt(m_2) + e)
        path.append(np.array([vector_copy[0], vector_copy[1], function(vector_copy)]))
        i += 1

    return np.array(path)

if __name__ == '__main__':
    function = None
    # function_string = input()  # Функция сферы 'xy[0] ** 2 + xy[1] ** 2' + мультифункция -xy[0] * np.sin(4 * np.pi * xy[0]) -xy[1] * np.sin(4 * np.pi * xy[1])
    # if function_string == '':
    function_string = 'xy[0] ** 2 + xy[1] ** 2'  # -20*math.exp(-0.2*math.sqrt(0.5*(xy[0]**2 + xy[1]**2)))-math.exp(0.5*math.cos(2*math.pi*xy[0])+math.cos(2*math.pi*xy[1]))+math.e+20
    minima_analytical=np.array([-0.54719, -1.54719, -1.9133])
    function_area=numpy.array([numpy.arange(-3, 3, 1), numpy.arange(-3, 3, 1)])
    vector_initial=np.array([2, 2.7])

    exec('function = lambda xy: ' + function_string)
    function_string = function_string.replace('xy[0]', 'x').replace('xy[1]', 'y')
    path = getGradientDescentPath(getGradient, vector_initial, minima_analytical, 1000)

    grid_x, grid_y = numpy.meshgrid(function_area[0], function_area[1])
    grid_z=function([grid_x, grid_y])
    best_z_position=numpy.unravel_index(grid_z.argmax(), grid_z.shape)

    print(f"Количество итераций: {len(path)}, минимум функции: {path[-1]}, погрешность: {abs(path[-1]-minima_analytical)}")
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(grid_x, grid_y, grid_z, cmap='hot', alpha=0.5)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], '-o', color='b', label='Градиентный спуск', alpha=0.7)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.legend(loc="upper left")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter3D([grid_x[best_z_position[0]][best_z_position[1]]], [grid_y[best_z_position[0]][best_z_position[1]]], [grid_z.max()], s=[100], c="g")

    plt.show()
