import math

from numpy import exp, arange
from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show
import sympy

# fonksiyonun grafigini cizdirmek icin sorudan alakasiz

def func(x,y):
    return (5*x - y)**4 + (x - 2)**2 + x - 2*y + 12

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
def plotter(a,b,c):
    x = arange(2, 10, 0.1)
    y = arange((65/2 + 1/(2**(1/3)) - 4), 38, 0.1)
    X, Y = meshgrid(x, y)  # grid of point
    Z = func(X, Y)  # evaluation of the function on the grid

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                          cmap=cm.RdBu,linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()
    plt.plot(a, c)
    plt.show()
    plt.plot(b, c)
    plt.show()
    # for i in range(len(c)):
    #     ax.scatter(a[i]+2, b[i]+2, c[i]+2,
    #                linewidths=4, alpha=1,
    #                edgecolor='b',
    #                s = 455)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def func(x,y):
    return (5*x - y)**4 + (x - 2)**2 + x - 2*y + 12

def fprime(x, y):
    return [-(2*x + 20*(5*x - y)**3 - 3), -(-4*(5*x - y)**3 - 2)]

def alphaFunctionCreator(x_i, y_i, px, py):
    alpha = sympy.Symbol('alpha')
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    func = (5*x - y)**4 + (x - 2)**2 + x - 2*y + 12
    xx = sympy.solve(func.subs({'x': x_i + px * alpha, 'y': y_i + py * alpha}).diff(alpha))
    min = math.inf
    for i in xx:
        try:
            # print(fprime(x0+px*i, y0+py*i))
            # print(i,'sol',func.subs({'x': x0 + px * i, 'y': y0 + py * i}))
            if(i < min):
                min = i
        except TypeError:
            # type error verme sebebi bazi cozumlerin imaginary number icermesi onlari
            # compare edemeyip type error veriyor, direk ignore'ladim.
            pass
    return min
    # return fprime(x0 + px * min, y0 + py * min)

def GradientDescentSimple(func, fprime, x0, y0, tol=0.000000001, max_iter=100000000):
    # initialize x, f(x), and -f'(x)
    xk = x0
    yk = y0
    fk = func(xk, yk)
    prev_fk = 0
    px, py = fprime(xk, yk)
    # initialize number of steps, save x and f(x)
    num_iter = 0
    curve_x = [xk]
    curve_y = [yk]
    curve_z = [fk]
    print("x:", xk, "y:", yk, 'f(xk,yk):', fk)
    # take steps
    while (num_iter == 0 or prev_fk - fk > tol) and num_iter < max_iter:
        # print(num_iter, x0, y0)
        # calculate new x, f(x), and -f'(x)
        alpha = alphaFunctionCreator(xk, yk, px, py)
        xk = xk + alpha * px
        yk = yk + alpha * py
        prev_fk = fk
        fk = func(xk, yk)
        print("iteration:", num_iter, "x:", xk, "y:", yk, 'f(xk,yk):', fk,'alpha:',  alpha)
        px, py = fprime(xk, yk)
        num_iter += 1
        curve_x.append(xk)
        curve_y.append(yk)
        curve_z.append(fk)
    # print results
    if num_iter == max_iter:
        print('Gradient descent does not converge.')
    else:
        print(num_iter, x0, y0)
        print('Solution:\n  x = {:.8f}\n  y = {:.8f}\n  z = {:.8f}'.format(xk, yk, fk))

    return curve_x, curve_y, curve_z

# for i in range(6, 10000):
#     for j in range(60, 10000):
#         GradientDescentSimple(func, fprime, i, j)
x0 = 226255512512
y0 = 65/2 + 1/(2**(1/3)) - 456461
xs, ys, zs = GradientDescentSimple(func, fprime, x0, y0)
plotter(xs, ys, zs)
plt.show()