import numpy
import math
from pylab import *
from sympy import *
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

def findGradient(z_str, vector):
    func_z = eval(z_str)
    dif_y=str(func_z.diff(Symbol('y'))).replace('y', str(vector[1])).replace('x', str(vector[0]))
    dif_x=str(func_z.diff(Symbol('x'))).replace('y', str(vector[1])).replace('x', str(vector[0]))

    return numpy.array([eval(dif_y), eval(dif_x)])

def mininize(func_z, func_z_str, vector):
    gradient=findGradient(func_z_str, vector)
    return vector - minimize_scalar(lambda l: func_z(vector - l * gradient)).x * gradient

def main():
    func_z_str = '3 * a[0] ** 2 + a[1] ** 2 - a[0] * a[1] - 4 * a[0]'
    func_z = None
    exec('func_z = lambda a: ' + func_z_str)
    x=numpy.arange(-200, 200, 1.0)
    y=numpy.arange(-200, 200, 1.0)
    x_grid, y_grid=numpy.meshgrid(x, y)
    z_grid=func_z([x_grid, y_grid])
    eps = 0.0001

    func_z_str = func_z_str.replace('a[0]', 'x').replace('a[1]', 'y')
    vectors = [numpy.array([-150.0, 150.0])]
    vectors.append(mininize(func_z, vectors[0]))

    while True:
        vector_distance=vectors[-2]-vectors[-1]
        if(math.sqrt(vector_distance[0] ** 2 + vector_distance[1] ** 2)) <= eps:
            break
        vectors.append(mininize(func_z, vectors[-1]))

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(x_grid, y_grid, z_grid, cmap=hot)
    ax.plot([x[0] for x in vectors], [x[1] for x in vectors], [func_z(x) for x in vectors], color='b')

    plt.show()

main()