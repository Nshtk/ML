from lab1.GradientDescent import *
import pygmo

def plotOptimisation(test_function: TestFunction, algorithm, genes_count_max :int=100):
    population = pygmo.population(test_function_sphere, genes_count_max)
    fitnesses = []
    for i in range(genes_count_max):
        population = algorithm.evolve(population)
        fitnesses.append(population.get_f()[population.best_idx()])
    genes = numpy.linspace(0, genes_count_max, genes_count_max)

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(genes, numpy.array(fitnesses), '-', marker='.', label="Чемпион")
    ax.plot(genes, numpy.full(len(genes), test_function_sphere.minima[2]), '--', lw=2, label="Глобальный минимум")
    ax.set_title(test_function_sphere.title)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend()

if __name__=='__main__':
    test_function_sphere = TestFunction("Функция сферы", 'xy[0] ** 2 + xy[1] ** 2', numpy.array([numpy.arange(-3, 3, 1), numpy.arange(-3, 3, 1)]), numpy.array([[-300., -300.], [300., 300.]]), numpy.array([-0.54719, -1.54719, -1.9133]), np.array([2, 2.7]))
    plotOptimisation(test_function_sphere, pygmo.algorithm(pygmo.sga(1)))
    plotOptimisation(test_function_sphere, pygmo.algorithm(pygmo.de1220(1)))