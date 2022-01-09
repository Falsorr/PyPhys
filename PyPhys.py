import numpy as np
import doctest

def summation(x) :
    """(dict) -> float, float
    x - dictionnary where the key/value pair represent measurement/uncertainty
    takes the sum of the measurement and propagate uncertainty

    >>> Z, Zerr = summation({2:.1, 3:.2, 4:.3})
    >>> print(Z)
    9
    >>> print(Zerr)
    0.37416573867739417
    """
    Z = 0
    Zerr = 0
    for value, error in x.items() :
        Z += value
        Zerr += error ** 2
    Zerr = np.sqrt(Zerr)
    return Z, Zerr


if __name__ == '__main__' :
    doctest.testmod()
