from sympy import *
import numpy
import pylab
import matplotlib.pyplot as plt
import celluloid

class Utility:
    eps = 0.1
    e = 10e-8

class TestFunction:
    def __init__(self, title: str, function_symbolic: str, area: numpy.ndarray, bounds: numpy.ndarray, minima_analytical: numpy.ndarray, vector_initial: numpy.ndarray) -> None:
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

def getGradientSymbolic(function: callable, vector):
    x, y = Symbol('x'), Symbol('y')
    function = eval(test_function.function_symbolic)
    diff_x = diff(function, x)
    diff_y=diff(function, y)
    return numpy.array([eval(str(diff_x).replace('x', str(vector[0])).replace('y', str(vector[1]))), eval(str(diff_y).replace('x', str(vector[0])).replace('y', str(vector[1])))])
def getGradient(function: callable, vector: numpy.ndarray, dt: float = 0.00001) -> numpy.array:
    dxdt = (function(vector + numpy.array([dt, 0])) - function(vector)) / dt
    dydt = (function(vector + numpy.array([0, dt])) - function(vector)) / dt
    return numpy.array([dxdt, dydt])

def getGradientDescentPath(test_function: TestFunction, vector_initial: numpy.ndarray, getGradient, iterations_max: int = 64, learning_rate: float = 0.1) -> numpy.array:
    vector_copy = vector_initial.copy()
    path = [numpy.array([vector_copy[0], vector_copy[1], test_function.function(vector_copy)])]

    i = 0
    while i < iterations_max and numpy.linalg.norm(path[-1] - test_function.minima) > Utility.eps:
        vector_copy = vector_copy - learning_rate * getGradient(test_function.function, vector_copy)
        path.append(numpy.array([vector_copy[0], vector_copy[1], test_function.function(vector_copy)]))
        i += 1

    return numpy.array(path)
def getGradientDescentInertialPath(test_function: TestFunction, vector_initial: numpy.ndarray, getGradient, iterations_max: int = 64, learning_rate: float = 0.1, inertia_coef: float = 0.5) -> numpy.array:
    vector_copy = vector_initial.copy()
    vector_copy_previous = vector_initial.copy()
    path = [numpy.array([vector_initial[0], vector_initial[1], test_function.function(vector_initial)])]

    i = 0
    while i < iterations_max and numpy.linalg.norm(path[-1] - test_function.minima) > Utility.eps:
        vector_copy_tmp=vector_copy
        vector_copy = vector_copy - learning_rate * getGradient(test_function.function, vector_copy) + inertia_coef * (vector_copy - vector_copy_previous)
        vector_copy_previous = vector_copy_tmp
        path.append(numpy.array([vector_copy[0], vector_copy[1], test_function.function(vector_copy)]))
        i += 1

    return numpy.array(path)

def getGradientDescentAdaptivePath(test_function: TestFunction, vector_initial: numpy.ndarray, getGradient, iterations_max: int = 64, learning_rate: float = 0.1, adaptive_coef_1: float=0.5, adaptive_coef_2: float=0.9) -> numpy.array:
    vector_copy = vector_initial.copy()
    path = [numpy.array([vector_copy[0], vector_copy[1], test_function.function(vector_copy)])]
    m_1 = numpy.array([0, 0])
    m_2 = numpy.array([0, 0])

    i = 0
    while i < iterations_max and numpy.linalg.norm(path[-1] - test_function.minima) > Utility.eps:
        m_1 = adaptive_coef_1 * m_1 + (1 - adaptive_coef_1) * getGradient(test_function.function, vector_copy)
        m_2 = adaptive_coef_2 * m_2 + (1 - adaptive_coef_2) * getGradient(test_function.function, vector_copy) ** 2
        vector_copy = vector_copy - learning_rate * m_1 / (numpy.sqrt(m_2) + Utility.e)
        path.append(numpy.array([vector_copy[0], vector_copy[1], test_function.function(vector_copy)]))
        i += 1

    return numpy.array(path)
def getGradientDescentEvolutionPath(test_function: TestFunction, vector_initial: numpy.ndarray, getGradient, iterations_max: int = 64, learning_rate: float = 0.1) -> numpy.array:
    vector_copy = vector_initial.copy()
    path = [numpy.array([vector_copy[0], vector_copy[1], test_function.function(vector_copy)])]
    z_best=float('inf')

    i = 0
    while i < iterations_max and numpy.linalg.norm(path[-1] - test_function.minima) > Utility.eps:
        vector_copy = vector_copy - learning_rate * getGradient(test_function.function, vector_copy)
        z=test_function.function(vector_copy)
        if z < z_best-Utility.eps*learning_rate:
            z_best=z
        else:
            learning_rate/=2
        path.append(numpy.array([vector_copy[0], vector_copy[1], test_function.function(vector_copy)]))
        i += 1

    return numpy.array(path)

def plotTestFunction(test_function: TestFunction, higlight_maxima: bool = True, highlight_minima: bool = False):
    grid_x, grid_y, grid_z=test_function.generateMeshes()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(test_function.title, fontsize=12, fontweight="bold", loc="left")
    ax.legend(loc="upper left")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    camera = celluloid.Camera(fig)

    for i in range(0, len(path), 1):
        if i < 2:
            ax.legend(loc='upper left')
        ax.plot_surface(grid_x, grid_y, grid_z, cmap='hot', alpha=0.5)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color='b', label='Градиентный спуск', alpha=0.7)
        ax.scatter(path[i, 0], path[i, 1], path[i, 2], '-o')
        camera.snap()
    if higlight_maxima:
        position_z = numpy.unravel_index(grid_z.argmax(), grid_z.shape)
        ax.scatter3D([grid_x[position_z[0]][position_z[1]]], [grid_y[position_z[0]][position_z[1]]], [grid_z.max()], s=[100], c="g", label='Глобальный максимум')
    if highlight_minima:
        position_z = numpy.unravel_index(grid_z.argmin(), grid_z.shape)
        ax.scatter3D([grid_x[position_z[0]][position_z[1]]], [grid_y[position_z[0]][position_z[1]]], [grid_z.min()], s=[100], c="g", label='Глобальный минимум')
    animation=camera.animate(interval=80, repeat=True)
    plt.show()

if __name__ == '__main__':
    # function_string = input()  # Функция сферы 'xy[0] ** 2 + xy[1] ** 2' + мультифункция -xy[0] * numpy.sin(4 * numpy.pi * xy[0]) -xy[1] * numpy.sin(4 * numpy.pi * xy[1])
    # if function_string == '':
    function_string = 'xy[0] ** 2 + xy[1] ** 2'
    test_function=TestFunction("Функция сферы", function_string, numpy.array([numpy.arange(-3, 3, 1), numpy.arange(-3, 3, 1)]), numpy.array([[-300., -300.], [300., 300.]]), numpy.array([0, 0, 0]), numpy.array([2, 2.7]))
    #test_function=TestFunction("Мультифункция", function_string, numpy.array([numpy.arange(-3, 3, 1), numpy.arange(-3, 3, 1)]), numpy.array([-0.54719, -1.54719, -1.9133]), numpy.array([2, 2.7]))
    path = getGradientDescentAdaptivePath(test_function, test_function.vector_initial, getGradient, 1000)
    print(f"Количество итераций: {len(path)}, минимум функции: {path[-1]}, погрешность: {abs(path[-1]-test_function.minima)}")
    plotTestFunction(test_function)
