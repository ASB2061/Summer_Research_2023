#!/adiel the line for running this in the interpreter? ill ask jeova about htis

"""This program is meant to investigate the viability of using the golden ratio
as a method of locating minimums for convex, multivariable functions. This program
'divides and conquers' the equations by minimizing at one variable for each iteration
in the main loop for minimizing. Each variable is minimized until the margin
between the upper and lower bounds for its search space is 0.001. We also compare
the difference between the sum of all older guesses and newer guesses and stop once
that margin is one thousandth. Then we return an array with the minimized values."""

# import statements: the primary libraries used are numpy and matplotlib
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

__author__ = "Adiel Benisty and Jeova Farias Sales Rocha Neto"
__copyright__ = "None"  # not sure
__credits__ = ["Adiel Benisty", "Jeova Farias Sales Rocha Neto"]
__license__ = "GPL"  # not sure
__version__ = "1.8"  # also not sure what to make this
__maintainer__ = "Adiel Benisty"
__email__ = "abenisty@haverford.edu"
__status__ = "Prototype"

# global variables
gldRto = (sqrt(5) + 1) / 2

"""This function finds the output of a convex function with any number of variables,
since it uses a and b matrices to represent the coefficients of each term in the 
convex function. Then they are multiplied out by transposing the variable matrix
and taking the dot products of the a and b matrices with the variable matrix."""


def fMatrixDotProduct(aMatrix: np.array, bMatrix: np.array, vMatrix: np.array) -> float or str:
    if np.shape(aMatrix)[1] == np.shape(bMatrix)[1] == np.shape(vMatrix)[0]:
        return vMatrix.T.dot(aMatrix.dot(vMatrix)) + bMatrix.dot(vMatrix)
    else:
        return 'Issue with matrix dimensions'


"""This builds two matrices that are used to construct a convex, multivariable 
function. Either it will receive a number of dimensions wanted by the user or it
 will choose a number of dimensions between 10 and 1000. The next question is
 determining how we can create positive semi definite matrices exclusively when
 using this function."""


def matrixBuilder(d: int = None) -> tuple:
    if d is None:  # if there are no dimensions
        d = random.randint(10, 1000)

    aMatrix = np.random.rand(d, d)
    aMatrix = np.dot(aMatrix, aMatrix.transpose()) + 0.001 * np.eye(d)  # this
    # aMatrix = np.array([[1, 2], [2, 1]])
    # creates a positive semi-definite matrix plus a scalar that makes it a positive definite matrix
    # aMatrix = np.eye(d) * np.random.rand(d, d)
    bMatrix = np.random.rand(1, d)
    # bMatrix = np.ones((1, d))
    # print(np.linalg.inv(aMatrix))
    minX = - 0.5 * np.linalg.inv(aMatrix).dot(bMatrix.T)
    return aMatrix, bMatrix, minX


"""This creates random bounds for each matrix. The margin is quite wide with 
anything between -100 to 100. Some changes will be to determine how to modify
the bounds if we have enough information to guess that the minimum is not in the 
search space that was created."""


def boundBuilder(numOfVars: int) -> tuple or list:
    upperBoundList = []
    lowerBoundList = []
    lowerBound = None
    upperBound = None
    for i in range(numOfVars):
        while True:
            try:
                lowerBound = random.uniform(-100, 100)
                upperBound = random.uniform(lowerBound, 101)
                break
            except lowerBound == upperBound:
                pass
        lowerBoundList.append(lowerBound)
        upperBoundList.append(upperBound)
    return lowerBoundList, upperBoundList


"""The main portion of the program. This actually minimizes each variable using 
the golden ratio. We save time by only having to recalculate one of the guesses
after the first iteration, since the others become part of the new search space.
This function recurs until the difference between the upper and lower bounds is 
one thousandth."""


def goldenSearchMatrix(lower: float, upper: float, index: int, aMatrix: np.array, bMatrix: np.array, guesses: np.array,
                       var1=None, var2=None, f1=None, f2=None) -> float:
    if var1 is None and var2 is None:  # If there are no initial guesses
        var1 = upper - ((upper - lower) / gldRto)
        guesses[index] = var1
        f1 = matrixSlicer(index, aMatrix, bMatrix, guesses)
        var2 = lower + ((upper - lower) / gldRto)
        guesses[index] = var2
        f2 = matrixSlicer(index, aMatrix, bMatrix, guesses)
        if abs(lower - upper) <= 0.00001:
            return lower
        elif f1 >= f2:
            return goldenSearchMatrix(var1, upper, index, aMatrix, bMatrix, guesses, var2, None, f2, None)
        elif f1 < f2:
            return goldenSearchMatrix(lower, var2, index, aMatrix, bMatrix, guesses, None, var1, None, f1)
    elif var2 is None:
        var2 = lower + ((upper - lower) / gldRto)
        guesses[index] = var2
        f2 = matrixSlicer(index, aMatrix, bMatrix, guesses)
        if abs(lower - upper) <= 0.00001:
            return lower
        elif f1 >= f2:
            return goldenSearchMatrix(var1, upper, index, aMatrix, bMatrix, guesses, var2, None, f2, None)
        elif f1 < f2:
            return goldenSearchMatrix(lower, var2, index, aMatrix, bMatrix, guesses, None, var1, None, f1)
    elif var1 is None:
        var1 = upper - ((upper - lower) / gldRto)
        guesses[index] = var1
        f1 = matrixSlicer(index, aMatrix, bMatrix, guesses)
        if abs(lower - upper) <= 0.00001:
            return lower
        elif f1 >= f2:
            return goldenSearchMatrix(var1, upper, index, aMatrix, bMatrix, guesses, var2, None, f2, None)
        elif f1 < f2:
            return goldenSearchMatrix(lower, var2, index, aMatrix, bMatrix, guesses, None, var1, None, f1)


"""This function is crucial in the golden search function, since it allows us to 
minimize for one variable without changing the others. This only evaluates """


def matrixSlicer(indexation: int, aMatrix: np.array, bMatrix: np.array, vMatrix: np.array) -> float:
    return fMatrixDotProduct(aMatrix, bMatrix, vMatrix)

    columnA = aMatrix[:, indexation]
    rowA = aMatrix[indexation, :]
    vIndexed = vMatrix[indexation]
    return bMatrix[0, indexation] * vIndexed + vIndexed * (
                vMatrix.dot(rowA) + vMatrix.T.dot(columnA) - vIndexed * aMatrix[indexation, indexation])


def multiVarMinimizerMatrix(aMatrix: np.array, bMatrix: np.array, boundedValues: list or tuple) -> float or str:
    lowerBounds = boundedValues[0]
    upperBounds = boundedValues[1]
    previousMatrixGuesses = np.zeros(bMatrix.size)
    newMatrixGuesses = np.zeros(bMatrix.size)
    guessDifference = 1
    if upperBounds is list or tuple:
        for iterable in range(len(upperBounds)):
            previousMatrixGuesses[iterable] = random.uniform(lowerBounds[iterable], upperBounds[iterable])
    else:
        previousMatrixGuesses[0] = random.uniform(lowerBounds, upperBounds)
    while abs(guessDifference) >= 0.000000001:
        previousMatrixGuesses = newMatrixGuesses
        if lowerBounds is list or tuple:
            for iteration in range(len(lowerBounds)):
                newMatrixGuesses[iteration] = goldenSearchMatrix(lowerBounds[iteration], upperBounds[iteration],
                                                                 iteration, aMatrix, bMatrix, previousMatrixGuesses)
                previousMatrixGuesses[iteration] = newMatrixGuesses[iteration]
            guessDifference = abs(sum(previousMatrixGuesses) - sum(newMatrixGuesses))
        else:
            return goldenSearchMatrix(lowerBounds, upperBounds, 0, aMatrix, bMatrix, previousMatrixGuesses)
    return previousMatrixGuesses


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    timesElapsed = []
    dimensions = []
    aMatrixDimensions = []

    d = 2
    testMatrixList = matrixBuilder(d)
    A = testMatrixList[0]
    b = testMatrixList[1]
    x = testMatrixList[2]
    bounds = ((np.squeeze(x) - 50), (np.squeeze(x) + 50))
    print(f"bounds: \t\t{bounds}")
    if np.all(np.linalg.eigvals(A)) > 0:
        # print(f'{b} \n {bounds}')

        # A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # b = np.array([[0, 0, 0]])
        # bounds = ([-1, -1, -1], [1, 1, 1])
        programAnswer = multiVarMinimizerMatrix(A, b, bounds)
        print(f'Optimization Solution:  {programAnswer}')
        print(f'Optimizer value:  {fMatrixDotProduct(A, b, programAnswer)}')
        val = fMatrixDotProduct(A, b, programAnswer)
        # for i in range(10):
        #     print(f'Optimizer noise:  {fMatrixDotProduct(A, b, programAnswer +  np.squeeze(np.random.rand(d, 1))) > val}')

        # print(fMatrixDotProduct(A, b, programAnswer))
        print(f'True Solution: \t\t {np.squeeze(x)}')
        x_val = fMatrixDotProduct(A, b, x)

        print(f'True value: \t\t {np.squeeze(x_val)}')
        # print(f'True Solution: \t\t {x_val < val}')
        # print(fMatrixDotProduct(A, b, x))
    else:
        print('Not a positive definite matrix.')