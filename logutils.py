__author__ = 'arenduchintala'

from math import exp, log
from math import pi


def lognormpdf(x, mean, sd):
    var = float(sd) ** 2
    denom = (2 * pi * var) ** .5
    num = exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return log(num / denom)


def logadd(x, y):
    """
    trick to add probabilities in logspace
    without underflow
    """
    # Todo: handle special case when x,y=0
    if x == 0.0 and y == 0.0:
        return log(exp(x) + exp(y))
    elif x == float('-inf') and y == float('-inf'):
        return float('-inf')
    elif x >= y:
        return x + log(1 + exp(y - x))
    else:
        return y + log(1 + exp(x - y))


def logadd_of_list(a_list):
    """
    uses logadd trick to sum a list of log probabilities
    """
    logsum = a_list[0]
    for i in a_list[1:]:
        logsum = logadd(logsum, i)
    return logsum


def normpdf(x, mean, sd):
    var = float(sd) ** 2
    pi = 3.1415926
    denom = (2 * pi * var) ** .5
    num = exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom