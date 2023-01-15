import doctest
import random
import numpy as np
import numexpr as ne
import math
from math import cos, sin, pow, log, sqrt, tan, e
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# For convex optimization, we first assume that the inputted function is continuous, or we cannot really perform
# anything on the inputted function. We then need to verify that it is convex

# global variables
global gldRto
gldRto = (sqrt(5) + 1) / 2


def f(V: list) -> float or str:
    return 5 * V[0] ** 2 + 3 * V[1] ** 2 + 5 * V[2] ** 2 + 3 * V[3] ** 2 + 5 * V[4] ** 2 + 3 * V[5] ** 2 + 3 * V[6] ** 2


def fMatrixDotProduct(aMatrix: np.array, bMatrix: np.array, vMatrix: np.array) -> float or str:
    # print(f'This is the AMatrix {aMatrix} \n ')
    # print(f'This is the bMatrix {bMatrix} \n')
    # print(f'This is the vMatrix {vMatrix} \n')
    # print('The final answer: \n')
    if np.shape(aMatrix)[1] == np.shape(bMatrix)[1] == np.shape(vMatrix)[0]:
        return vMatrix.T.dot(aMatrix.dot(vMatrix)) + bMatrix.dot(vMatrix)
    else:
        return 'Issue with matrix dimensions'


def matrixBuilder() -> list:
    vMatrix = np.random.rand(random.randint(10, 10))
    aMatrix = np.random.rand(np.shape(vMatrix)[0], np.shape(vMatrix)[0])
    bMatrix = np.random.rand(1, np.shape(vMatrix)[0])
    return [aMatrix, bMatrix, vMatrix]


def boundBuilder(numOfVars: int) -> list:
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
    return [lowerBoundList, upperBoundList]


# V[2] + abs(V[3] - 3) + V[4] ** 2 - 3 * V[5] + V[6] ** 2 - 3


# def fOfXY(x: float, y: float) -> float or str:
#     func = s
#     if 'x' in func or 'y' in func:
#         outputOfFunction = ne.evaluate(func)
#         return outputOfFunction
#     else:
#         return 'Error'


# def bisectionForOneVar(lower: float, upper: float, ) -> float or str:
#     c = (lower + upper) / 2
#     xOne = (lower + c) / 2
#     xTwo = (upper + c) / 2
#
#     if lower < upper and xOne < xTwo:
#         if abs(lower - upper) <= 0.00001:
#             return round(c)
#         elif f(xOne) > f(xTwo):
#             return bisectionForOneVar(xOne, upper)
#         elif f(xOne) < f(xTwo):
#             return bisectionForOneVar(lower, xTwo)
#     else:
#         return "Error, lower and upper values are not in proper form or solution cannot be guaranteed."


def goldenSearchForOneVar(lower: float, upper: float, index: int, guesses: list, var1=None, var2=None, f1=None,
                          f2=None) -> float or str:
    # We make optional parameters to allow us to send either the x1 and f1 or the x2 and f2

    # The advantage is that we can reuse these test points as they become the new bounds as we get to smaller intervals

    if var1 is None and var2 is None:
        var1 = upper - ((upper - lower) / gldRto)
        guesses[index] = var1
        f1 = f(guesses)
        var2 = lower + ((upper - lower) / gldRto)
        guesses[index] = var2
        f2 = f(guesses)
        if abs(lower - upper) <= 0.0001:
            return round(lower)
        elif f1 >= f2:
            return goldenSearchForOneVar(var1, upper, index, guesses, var2, None, f2, None)
        elif f1 < f2:
            return goldenSearchForOneVar(lower, var2, index, guesses, None, var1, None, f1)
    elif var2 is None:
        var2 = lower + ((upper - lower) / gldRto)
        guesses[index] = var2
        f2 = f(guesses)
        if abs(lower - upper) <= 0.0001:
            return round(lower)
        elif f1 >= f2:
            return goldenSearchForOneVar(var1, upper, index, guesses, var2, None, f2, None)
        elif f1 < f2:
            return goldenSearchForOneVar(lower, var2, index, guesses, None, var1, None, f1)
    elif var1 is None:
        var1 = upper - ((upper - lower) / gldRto)
        guesses[index] = var1
        f1 = f(guesses)
        if abs(lower - upper) <= 0.0001:
            return round(lower)
        elif f1 >= f2:
            return goldenSearchForOneVar(var1, upper, index, guesses, var2, None, f2, None)
        elif f1 < f2:
            return goldenSearchForOneVar(lower, var2, index, guesses, None, var1, None, f1)


def multivariateFunctionMinimization(varInput: list) -> list or str:
    # we expect input for the function when it is first called to be in the format where each pair in this tuple is a
    # set of bounds for that variable. So there will be half the number of bounds for the number of variables.
    numOfVariables = int(len(varInput) / 2)
    lowerBounds = []
    upperBounds = []
    global previousGuesses
    previousGuesses = []
    global newGuesses
    newGuesses = [0] * numOfVariables
    differenceOfGuesses = 0.0
    for i in range(len(varInput)):  # This sets up the lower and upper bound lists
        if i % 2 == 0:
            lowerBounds.append(varInput[i])
        else:
            upperBounds.append(varInput[i])

    for i in range(numOfVariables):  # We set up the first list of initial guesses
        if lowerBounds[i] is None or upperBounds[i] is None:
            raise Exception("Bound is missing or some input error has occurred.")
        else:
            previousGuesses.append(random.uniform(lowerBounds[i], upperBounds[i]))

    while abs(sum(previousGuesses) - sum(newGuesses)) >= 0.0000001:
        previousGuesses = newGuesses
        for i in range(numOfVariables):
            newGuesses[i] = goldenSearchForOneVar(lowerBounds[i], upperBounds[i], i, previousGuesses)
            previousGuesses[i] = newGuesses[i]

    return f"Minimums: {newGuesses}"


def goldenSearchMatrix(lower: float, upper: float, index: int, aMatrix: np.array, bMatrix: np.array, guesses: np.array,
                       var1=None, var2=None, f1=None, f2=None) -> int:
    if var1 is None and var2 is None:
        var1 = upper - ((upper - lower) / gldRto)
        guesses[index] = var1
        f1 = matrixSlicer(index, aMatrix, bMatrix, guesses)
        guesses[index] = lower + ((upper - lower) / gldRto)
        f2 = matrixSlicer(index, aMatrix, bMatrix, guesses)
        if abs(lower - upper) <= 0.0001:
            return round(lower)
        elif f1 >= f2:
            return goldenSearchForOneVar(var1, upper, index, guesses, guesses[index], None, f2, None)
        elif f1 < f2:
            return goldenSearchForOneVar(lower, guesses[index], index, guesses, None, var1, None, f1)
    elif var2 is None:
        var2 = lower + ((upper - lower) / gldRto)
        guesses[index] = var2
        f2 = matrixSlicer(index, aMatrix, bMatrix, guesses)
        if abs(lower - upper) <= 0.0001:
            return round(lower)
        elif f1 >= f2:
            return goldenSearchForOneVar(var1, upper, index, guesses, var2, None, f2, None)
        elif f1 < f2:
            return goldenSearchForOneVar(lower, var2, index, guesses, None, var1, None, f1)
    elif var1 is None:
        guesses[index] = upper - ((upper - lower) / gldRto)
        f1 = matrixSlicer(index, aMatrix, bMatrix, guesses)
        if abs(lower - upper) <= 0.0001:
            return round(lower)
        elif f1 >= f2:
            return goldenSearchForOneVar(guesses[index], upper, index, guesses, var2, None, f2, None)
        elif f1 < f2:
            return goldenSearchForOneVar(lower, var2, index, guesses, None, guesses[index], None, f1)


def matrixSlicer(i: int, aMatrix: np.array, bMatrix: np.array, vMatrix: np.array) -> float:
    # vMatrix = np.squeeze(vMatrix)
    # print(vMatrix)
    columnA = aMatrix[:, i]
    rowA = aMatrix[i, :]
    vIndexed = vMatrix[i]
    return bMatrix[0, i] * vIndexed + vIndexed * (vMatrix.dot(rowA) + vMatrix.T.dot(columnA) - vIndexed * aMatrix[i, i])
    # return bMatrix[0, i] * vMatrix[i, 0] + (
    #             np.squeeze(vMatrix.dot(aMatrix[i, :])) + np.squeeze(vMatrix.T.dot(aMatrix[:, i])))


def multiVarMinimizerMatrix(aMatrix: np.array, bMatrix: np.array, vMatrix: np.array, bounds: list) -> float or str:
    lowerBounds = bounds[0]
    upperBounds = bounds[1]
    previousMatrixGuesses = np.array
    newMatrixGuesses = np.ones(len(lowerBounds)) * 0

    for i in range(len(upperBounds)):
        previousMatrixGuesses = np.append(previousMatrixGuesses, random.uniform(lowerBounds[i], upperBounds[i]))
    print(previousMatrixGuesses)
    while abs(np.sum(previousMatrixGuesses) - np.sum(newMatrixGuesses)) >= 0.000001:
        previousMatrixGuesses = newMatrixGuesses
        for i in range(len(lowerBounds)):
            newMatrixGuesses[i] = goldenSearchMatrix(lowerBounds[i], upperBounds[i], i, aMatrix, bMatrix,
                                                     previousMatrixGuesses)
            previousMatrixGuesses[i] = newMatrixGuesses[i]
    return f'{previousMatrixGuesses}'


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # s = input("Please provide a function with the variable x: ")
    # a = int(input("Please provide a lower limit for your search space: "))
    # b = int(input("Please provide an upper limit for your search space: "))
    # print(bisectionForOneVar(a, b))

    # print(multivariateFunctionMinimization([-5, 7, -9, 3, -2, 4, 0, 5, -7, 1, -10, 2, -3, 5])) ////Important Line\\\\
    matrixList = matrixBuilder()
    aAMatrix = matrixList[0]
    bBMatrix = matrixList[1]
    varMatrix = matrixList[2]

    print(multiVarMinimizerMatrix(aAMatrix, bBMatrix, varMatrix, boundBuilder(int(varMatrix.size))))

    # bounds = boundBuilder(10)
    # lowerBounds = bounds[0]
    # upperBounds = bounds[1]
    # previousMatrixGuesses = np.array
    # newMatrixGuesses = np.ones(len(lowerBounds)) * 0
    # for i in range(len(upperBounds)):
    #     previousMatrixGuesses = np.append(previousMatrixGuesses, random.uniform(lowerBounds[i], upperBounds[i]))
    #
    # print(previousMatrixGuesses)
    # print(newMatrixGuesses)

    # print(multiVarMinimizerMatrix(aAMatrix, bBMatrix, varMatrix, boundBuilder(int(varMatrix.size))))
    # print(matrixSlicer(3, matrixList[0], matrixList[1], matrixList[2]))
    # print(fMatrixDotProduct(matrixList[0], matrixList[1], matrixList[2]))
    # print(matrixSlicer(1, matrixList[0], matrixList[1], matrixList[2]))

    # previousMatrixGuesses = np.array([[1, 1], [1, 1]])
    # previousMatrixGuesses = np.append(previousMatrixGuesses, np.array([[2], [1]]), 1)
    # print(previousMatrixGuesses)
    # print(boundBuilder(5))
    # listTool = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # var1 = 5
    # index = 3

    # goldenSearchForOneVar()

    # s = "x ** 2 - 9x + y ** 2 - 20y + x * y"
    # x, y = sp.symbols("x y")
    # expr = 9 * x * y
    # print(expr)
    # x = 4
    # expr = 9 * x * y
    # print(expr)
    #
    # b = np.arange(0.2, 3.2, 0.2)
    # d = np.arange(0.1, 1.0, 0.1)
    #
    # B, D = np.meshgrid(b, d)
    # nu = np.sqrt(1 + (2 * D * B) ** 2) / np.sqrt((1 - B ** 2) ** 2 + (2 * D * B) ** 2)
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(B, D, nu)
    # plt.xlabel('b')
    # plt.ylabel('d')
    # plt.show()

    # x1 = upper - ((upper - lower) / gldRto)
    # x2 = lower + ((upper - lower) / gldRto)
    # f1 = fOfX(x1)
    # f2 = fOfX(x2)
    # print(f1, f2)

    # figure out to be more efficient with f1 and f2 and x1 and x2
    # when f1 is greater than f2, we make x1 our new lower bound and x2 the new x1. The only number that needs to be
    # recalculated is the new x2
    # if f2 is greater than f1, we make x2 the new upper bound and reuse x1 as the new x2 and the lower is also reused.
