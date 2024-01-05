import math
import numpy
from sympy import *
from pylab import *
import optuna
import matplotlib.pyplot as plt

class Utility:
    eps = 0.001
    e = 10e-8

class TestFunction:
    def __init__(self, title: str, function_symbolic: str, area: np.ndarray, bounds: np.ndarray, minima_analytical: np.ndarray, vector_initial: np.ndarray) -> None:
        self._title = title
        self._function_symbolic = function_symbolic
        self._function = None
        exec('self._function = lambda xy: ' + function_symbolic)
        self._function_symbolic = self._function_symbolic.replace('xy[0]', 'x').replace('xy[1]', 'y')
        self._area = area
        self._bounds = bounds
        self._minima = minima_analytical
        self._vector_initial=vector_initial

    @property
    def title(self):
        return self._title
    @property
    def function(self):
        return self._function
    @property
    def function_symbolic(self):
        return self._function_symbolic
    @property
    def area(self):
        return self._area
    @property
    def minima(self):
        return self._minima
    @property
    def vector_initial(self):
        return self._vector_initial
    @vector_initial.setter
    def vector_initial(self, value):
        self._vector_initial=value

    def generateMeshes(self):
        grid_x, grid_y = numpy.meshgrid(self._area[0], self._area[1])
        grid_z = self._function([grid_x, grid_y])
        return grid_x, grid_y, grid_z
    def fitness(self, x: numpy.ndarray):
        return [self._function(x)]
    def get_bounds(self):
        return self._bounds
    # def getExtreams(func):
    #     optuna.logging.set_verbosity(optuna.logging.ERROR)
    #     study = optuna.create_study()
    #     study.optimize(trials, n_trials=200)
    #     found_params = [study.best_params["x[0]"], study.best_params["x[1]"]]


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
def getGradient(function: callable, vector: np.ndarray, dt: float = 0.00001) -> np.array:
    dxdt = (function(vector + np.array([dt, 0])) - function(vector)) / dt
    dydt = (function(vector + np.array([0, dt])) - function(vector)) / dt
    return numpy.array([dxdt, dydt])

def getGradientDescentPath(test_function: TestFunction, vector_initial: ndarray, getGradient, iterations_max: int = 64, learning_rate: float = 0.1) -> np.array:
    vector_copy = vector_initial.copy()
    path = [np.array([vector_copy[0], vector_copy[1], test_function.function(vector_copy)])]

    i = 0
    while i < iterations_max and np.linalg.norm(path[-1] - test_function.minima) > Utility.eps:
        vector_copy = vector_copy - learning_rate * getGradient(test_function.function, vector_copy)
        path.append(np.array([vector_copy[0], vector_copy[1], test_function.function(vector_copy)]))
        i += 1

    return np.array(path)
def getGradientDescentInertialPath(test_function: TestFunction, vector_initial: ndarray, getGradient, iterations_max: int = 64, learning_rate: float = 0.1, inertia_coef: float = 0.5) -> np.array:
    vector_copy = vector_initial.copy()
    vector_copy_previous = vector_initial.copy()
    path = [np.array([vector_initial[0], vector_initial[1], test_function.function(vector_initial)])]

    i = 0
    while i < iterations_max and np.linalg.norm(path[-1] - test_function.minima) > Utility.eps:
        vector_copy_tmp=vector_copy
        vector_copy = vector_copy - learning_rate * getGradient(test_function.function, vector_copy) + inertia_coef * (vector_copy - vector_copy_previous)
        vector_copy_previous = vector_copy_tmp
        path.append(np.array([vector_copy[0], vector_copy[1], test_function.function(vector_copy)]))
        i += 1

    return np.array(path)

def getGradientDescentAdaptivePath(test_function: TestFunction, vector_initial: ndarray, getGradient, iterations_max: int = 64, learning_rate: float = 0.1, adaptive_coef_1: float=0.5, adaptive_coef_2: float=0.9) -> np.array:
    vector_copy = vector_initial.copy()
    path = [np.array([vector_copy[0], vector_copy[1], test_function.function(vector_copy)])]
    m_1 = np.array([0, 0])
    m_2 = np.array([0, 0])

    i = 0
    while i < iterations_max and np.linalg.norm(path[-1] - test_function.minima) > Utility.eps:
        m_1 = adaptive_coef_1 * m_1 + (1 - adaptive_coef_1) * getGradient(test_function.function, vector_copy)
        m_2 = adaptive_coef_2 * m_2 + (1 - adaptive_coef_2) * getGradient(test_function.function, vector_copy) ** 2
        vector_copy = vector_copy - learning_rate * m_1 / (np.sqrt(m_2) + Utility.e)
        path.append(np.array([vector_copy[0], vector_copy[1], test_function.function(vector_copy)]))
        i += 1

    return np.array(path)

def plotTestFunction(test_function: TestFunction, higlight_maxima: bool = True, highlight_minima: bool = False):
    grid_x, grid_y, grid_z=test_function.generateMeshes()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(grid_x, grid_y, grid_z, cmap='hot', alpha=0.5)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], '-o', color='b', label='Градиентный спуск', alpha=0.7)
    ax.set_title(test_function.title, fontsize=12, fontweight="bold", loc="left")
    ax.legend(loc="upper left")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if higlight_maxima:
        position_z = numpy.unravel_index(grid_z.argmax(), grid_z.shape)
        ax.scatter3D([grid_x[position_z[0]][position_z[1]]], [grid_y[position_z[0]][position_z[1]]], [grid_z.max()], s=[100], c="g", label='Глобальный максимум')
    if highlight_minima:
        position_z = numpy.unravel_index(grid_z.argmin(), grid_z.shape)
        ax.scatter3D([grid_x[position_z[0]][position_z[1]]], [grid_y[position_z[0]][position_z[1]]], [grid_z.min()], s=[100], c="g", label='Глобальный минимум')
    plt.show()

global test_function_sphere
if __name__ == '__main__':
    # function_string = input()  # Функция сферы 'xy[0] ** 2 + xy[1] ** 2' + мультифункция -xy[0] * np.sin(4 * np.pi * xy[0]) -xy[1] * np.sin(4 * np.pi * xy[1])
    # if function_string == '':
    function_string = 'xy[0] ** 2 + xy[1] ** 2'  # -20*math.exp(-0.2*math.sqrt(0.5*(xy[0]**2 + xy[1]**2)))-math.exp(0.5*math.cos(2*math.pi*xy[0])+math.cos(2*math.pi*xy[1]))+math.e+20
    test_function_sphere=TestFunction("Функция сферы", function_string, numpy.array([numpy.arange(-3, 3, 1), numpy.arange(-3, 3, 1)]), numpy.array([-0.54719, -1.54719, -1.9133]), np.array([2, 2.7]))
    #function_multifunction=TestFunction("Мультифункция", function_string, numpy.array([numpy.arange(-3, 3, 1), numpy.arange(-3, 3, 1)]), numpy.array([-0.54719, -1.54719, -1.9133]), np.array([2, 2.7]))
    
    path = getGradientDescentAdaptivePath(test_function_sphere, test_function_sphere.vector_initial, getGradient, 1000)
    print(f"Количество итераций: {len(path)}, минимум функции: {path[-1]}, погрешность: {abs(path[-1]-test_function_sphere.minima)}")
    plotTestFunction(test_function_sphere)
