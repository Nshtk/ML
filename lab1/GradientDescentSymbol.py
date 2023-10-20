import numpy
import math
from sympy import *
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d, Axes3D
def findMinimumAnalytically(z_str):
    x = Symbol('x')
    y = Symbol('y')
    func_z = eval(z_str)

    d1 = func_z.diff(x)
    d2 = d1.diff(x)  # = f.diff(x,2)
    extrema = real_roots(d1)
    for i in extrema:
        if d2.subs(x, i).is_positive:
            print('minimum', i)
        else:
            print('maxima', i)

def findGradient(z_str, vector):
    x = Symbol('x')
    y = Symbol('y')

    func_z = eval(z_str)
    dif_y=str(func_z.diff(y)).replace('y', str(vector[1])).replace('x', str(vector[0]))
    dif_x=str(func_z.diff(x)).replace('y', str(vector[1])).replace('x', str(vector[0]))

    return numpy.array([eval(dif_y), eval(dif_x)])

def minimize(func_z, func_z_str, vector):
    gradient=findGradient(func_z_str, vector)
    return vector - minimize_scalar(lambda l: func_z(vector - l * gradient)).x * gradient

if __name__ == '__main__':
    func_z_str = '(3 * args[0]**2 + args[1]**2 - args[0]*args[1] - 4*args[0])'
    func_z = lambda args: (3 * args[0] ** 2 + args[1] ** 2 - args[0] * args[1] - 4 * args[0])
    x=numpy.arange(-200, 200, 1.0)
    y=numpy.arange(-200, 200, 1.0)
    x_grid, y_grid=numpy.meshgrid(x, y)
    z_grid=func_z([x_grid, y_grid])
    eps = 0.0001

    func_z_str = func_z_str.replace('args[0]', 'x').replace('args[1]', 'y')
    vectors = [numpy.array([-150.0, 150.0])]
    vectors.append(minimize(func_z, func_z_str, vectors[0]))

    while True:
        vector_distance=vectors[-2]-vectors[-1]
        if(math.sqrt(vector_distance[0] ** 2 + vector_distance[1] ** 2)) <= eps:
            break
        vectors.append(minimize(func_z, func_z_str, vectors[-1]))

    ax = plt.axes(projection='3d')

    ax.plot_surface(x_grid, y_grid, z_grid)
    ax.plot([x[0] for x in vectors], [x[1] for x in vectors], [func_z(x) for x in vectors], color='b')

    plt.show()
