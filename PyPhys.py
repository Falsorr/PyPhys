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
    >>> Z 
    0
    >>> Zerr
    .1
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
    >>> Z 
    0
    >>> Zerr
    .1
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
    >>> Z 
    0
    >>> Zerr
    .1
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
    """
    Z = k * (x ** m / y ** n)
    Zerr = Z * np.sqrt((m * xerr / x)**2 + (n * yerr / y)**2)
    return Z, Zerr

if __name__ == '__main__' :
    Z, Zerr = division(2, .1, 4, .2, 3, 4, 5)
    print(Z, Zerr)