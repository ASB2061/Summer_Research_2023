import numpy as np
import numexpr as ne
import math
from math import cos, sin, pow, log, sqrt, tan, e

V = [1, 2, 3, 4]

inputFunc = 'V[0] + V[1]'

print(ne.evaluate(inputFunc))