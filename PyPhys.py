#Author: Daniel Fassler

import numpy as np
import itertools

def summation(x, xerr) :
    """(tuple, tuple) -> (float, float)
    x - tuple containing the x_i's
    xerr - tuple containing uncertainty on the x_i's
    takes the sum of the measurement and propagate uncertainty
    If a value "y" is to be subtracted instead of added, write the pair as -y:yerr
    >>> Z, Zerr = summation((2,3,4),(.1,.2,.3))
    >>> print(Z)
    9
    >>> print(Zerr)
    0.37416573867739417
    """
    Z = 0
    Zerr = 0
    for value, error in itertools.zip_longest(x, xerr) :
        Z += value
        Zerr += error ** 2
    Zerr = np.sqrt(Zerr)
    return Z, Zerr

def power(x, xerr, power = 2) :
    """(float, float, int) -> (float, float)
    x - value of the measurement
    xerr - uncertainty on x
    power - the power for x to be raised (can be negative), the default value is 2, as squaring a value is extremely common
    Takes x ** power and propagate uncertainty
    >>> Z, Zerr = power(2, .1)
    >>> Z
    4
    >>> Zerr
    0.4
    >>> Z, Zerr = power(2, .1, 3)
    >>> Z
    8
    >>> Zerr
    1.2000000000000002
    """
    Z = x ** power
    Zerr = abs(power * x ** (power - 1)) * xerr
    return Z, Zerr

def logarithm(x, xerr) :
    """(float, float) -> (float, float)
    x - value of the measurement
    xerr - uncertainty on x
    Takes ln(x) and propagates uncertainty
    >>> Z, Zerr = logarithm(5, .1)
    >>> Z
    1.6094379124341003
    >>> Zerr
    0.02
    """
    Z = np.log(x)
    Zerr = xerr / x
    return Z, Zerr

def sine(x, xerr) :
    """(float, float) -> (float, float)
    x - value of the measurement
    xerr - uncertainty on x
    Takes sin(x) and propagates uncertainty (in radians)
    >>> Z, Zerr = sine(np.pi, .1)
    >>> round(Z) 
    0
    >>> Zerr
    0.1
    """
    Z = np.sin(x)
    Zerr = abs(np.cos(x)) * xerr
    return Z, Zerr

def cosine(x, xerr) :
    """(float, float) -> (float, float)
    x - value of the measurement
    xerr - uncertainty on x
    Takes cos(x) and propagates uncertainty (in radians)
    >>> Z, Zerr = cosine(np.pi / 2, .1)
    >>> round(Z) 
    0
    >>> Zerr
    0.1
    """
    Z = np.cos(x)
    Zerr = abs(np.sin(x)) * xerr
    return Z, Zerr

def tangent(x, xerr) :
    """(float, float) -> (float, float)
    x - value of the measurement
    xerr - uncertainty on x
    Takes tan(x) and propagates uncertainty (in radians)
    >>> Z, Zerr = tangent(np.pi, .1)
    >>> round(Z) 
    0
    >>> Zerr
    0.1
    """
    Z = np.tan(x)
    Zerr = (1 + Z ** 2) * xerr
    return Z, Zerr

def product(x, xerr) :
    """(tuple, tuple) -> (float, float)
    x - value of the measurement
    xerr - uncertainty on x
    takes the product of the measurements and propagate uncertainty
    if some of the value 'y' is to be divided instead of multiplied, use power(y, yerr, -1) where yerr is the uncertainty on y

    >>> Z, Zerr = product((2,3,4),(.1,.2,.3))
    >>> print(Z)
    24
    >>> print(Zerr)
    0.11211353372561426
    """
    Z = 1
    Zerr = 0
    for value, error in itertools.zip_longest(x, xerr):
        Z *= value
        Zerr += (error/value) ** 2
    Zerr = np.sqrt(Zerr)

    return Z, Zerr

def division(x, xerr, y, yerr, m = 1, n = 1, k = 1) :
    """(float, float, float, float, int, int, int) -> (float, float)
    - x : measurement x
    - xerr : uncertainty on x
    - y : measurement y
    - yerr : ucnertainty on y
    - m : power on x
    - n : power on y (if you want to multiply instead of divie, n can be negative)
    - k : coefficient for the multiplication
    >>> Z, Zerr = division(2, .1, 4, .2)
    >>> Z
    0.5
    >>> Zerr
    0.03535533905932738
    >>> Z, Zerr = division(2, .1, 4, .2, 3, 4, 5)
    >>> Z
    0.15625
    >>> Zerr
    0.0390625
    """
    Z = k * (x ** m / y ** n)
    Zerr = Z * np.sqrt((m * xerr / x)**2 + (n * yerr / y)**2)
    return Z, Zerr

def least_squares(x, y) :
    """(tuple, tuple) -> (float, float, float, float)
    - x : tuple containing x-coordinates of the data points
    - y : tuple containing y-coordinates of the data points
    Find m, c, merr, cerr using the method of least squares, that is the
    coefficients m and c such that the line mx + c is the best fit for the data points
    merr is the error on m,
    cerr is the error on c
    """
    N = len(x)
    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_xy = 0
    for i in range(N) :
        sum_x += x[i]
        sum_y += y[i]
        sum_x2 += x[i] ** 2
        sum_xy += x[i] * y[i]
    delta = N * sum_x2 - sum_x ** 2

    m = (N * sum_xy - (sum_x * sum_y)) / delta
    c = (sum_x2 * sum_y - sum_x * sum_xy) / delta
    alpha_CU = 0
    for i in range(N) :
        alpha_CU += (y[i] - m * x[i] - c) ** 2
    alpha_CU /= (N - 2)
    alpha_CU = np.sqrt(alpha_CU)

    merr = alpha_CU * np.sqrt(N/ delta)
    cerr = alpha_CU * np.sqrt(sum_x2 / delta)

    return m, merr, c, cerr

def chi_square(observed, expected) :
    """(tuple, tuple) -> float
    - observed : the measured value of the y-coordinates of the data points
    - expected : the expected value of the y-coordinates of the data points from a fit
    Compute a chi square value for a fit
    """
    chi_square = 0
    for o, e in itertools.zip_longest(observed, expected) :
        chi_square += (o - e) ** 2 / e
    return chi_square

def residuals(observed, expected) : 
    """(tuple, tuple) -> list
    - observed : the measured value of the y-coordinates of the data points
    - expected : the expected value of the y-coordinates of the data points from a fit
    Compute the residuals for a fit
    """
    res = []
    for o,e in itertools.zip_longest(observed, expected) :
        res.append(e-o)
    return res

def weighted_least_squares(x, y, yerr) :
    """(tuple, tuple, tuple) -> (float, float, float, float)
    - x : the x-coordinates of the data points
    - y : the y-coordinates of the data points
    - yerr : the error on the y-coordinate
    Find the coefficients for a line of best fit (least squares) when uncertainty isn't uniform
    """
    w = []
    for a in yerr :
        w.append(1/a**2)
    sum_w = 0
    sum_wx = 0
    sum_wy = 0
    sum_wxy = 0
    sum_wx2 = 0

    for i in range(len(x)) :
        sum_w += w[i]
        sum_wx += w[i] * x[i]
        sum_wy += w[i] * x[i]
        sum_wxy += w[i] * x[i] * y[i]
        sum_wx2 += w[i] * x[i] ** 2 
    
    delta = sum_w * sum_wx2 - sum_wx ** 2

    m = (sum_w * sum_wxy - sum_wx * sum_wy) / delta
    c = (sum_wx2 * sum_wy - sum_wx * sum_wxy) / delta
    merr = np.sqrt(sum_w / delta)
    cerr = np.sqrt(sum_wx2 / delta)

    return m, merr, c, cerr

if __name__ == '__main__' :
    import doctest

    doctest.testmod()

    import matplotlib.pyplot as plt
    x = (1, 2, 3, 4)
    xerr = (.1, .1, .1, .1)
    y = (1, 2, 3, 4)
    yerr = (.2, .3, .4, .5)

    plt.subplot(2, 1, 1)
    plt.errorbar(x, y, xerr = xerr, yerr = yerr, fmt='o')
    m, merr, c, cerr = weighted_least_squares(x, y, yerr)
    y_best_fit = []
    for i in x :
        y_best_fit.append(m * i + c)
    plt.subplot(2, 1, 1)
    plt.plot(x, y_best_fit)
    print(f'm: {m} +/- {merr}')
    print(f'c: {c} +/- {cerr}')
    print(chi_square(y, y_best_fit))

    plt.subplot(2, 1, 2)
    plt.scatter(x, residuals(y, y_best_fit))
    plt.axhline(0, color = 'r')
    plt.show()
    
    
