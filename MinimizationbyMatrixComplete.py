#!/adiel the line for running this in the interpreter? ill ask jeova about htis

"""Description of Code"""

import random
import random
import time
from math import sqrt
import matplotlib.pyplot as plt

import numpy as np

__author__ = "Adiel Benisty and Jeova Farias Sales Rocha Neto"
__copyright__ = "None"  # not sure
__credits__ = ["Adiel Benisty", "Jeova Farias Sales Rocha Neto"]
__license__ = "GPL"  # not sure
__version__ = "1.7"  # also not sure what to make this
__maintainer__ = "Adiel Benisty"
__email__ = "abenisty@haverford.edu"
__status__ = "Prototype"

# For convex optimization, we first assume that the inputted function is continuous, or we cannot really perform
# anything on the inputted function. We then need to verify that it is convex

# global variables
gldRto = (sqrt(5) + 1) / 2


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
    vMatrix = np.random.rand(random.randint(10, 2500))
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


def goldenSearchMatrix(lower: float, upper: float, index: int, aMatrix: np.array, bMatrix: np.array, guesses: np.array,
                       var1=None, var2=None, f1=None, f2=None) -> float:
    # print(guesses[index])
    if var1 is None and var2 is None:
        var1 = upper - ((upper - lower) / gldRto)
        guesses[index] = var1
        f1 = matrixSlicer(index, aMatrix, bMatrix, guesses)
        var2 = lower + ((upper - lower) / gldRto)
        guesses[index] = var2
        f2 = matrixSlicer(index, aMatrix, bMatrix, guesses)
        if abs(lower - upper) <= 0.00001:
            return round(lower)
        elif f1 >= f2:
            return goldenSearchMatrix(var1, upper, index, aMatrix, bMatrix, guesses, var2, None, f2, None)
        elif f1 < f2:
            return goldenSearchMatrix(lower, var2, index, aMatrix, bMatrix, guesses, None, var1, None, f1)
    elif var2 is None:
        var2 = lower + ((upper - lower) / gldRto)
        guesses[index] = var2
        f2 = matrixSlicer(index, aMatrix, bMatrix, guesses)
        if abs(lower - upper) <= 0.00001:
            return round(lower)
        elif f1 >= f2:
            return goldenSearchMatrix(var1, upper, index, aMatrix, bMatrix, guesses, var2, None, f2, None)
        elif f1 < f2:
            return goldenSearchMatrix(lower, var2, index, aMatrix, bMatrix, guesses, None, var1, None, f1)
    elif var1 is None:
        var1 = upper - ((upper - lower) / gldRto)
        guesses[index] = var1
        f1 = matrixSlicer(index, aMatrix, bMatrix, guesses)
        if abs(lower - upper) <= 0.00001:
            return round(lower)
        elif f1 >= f2:
            return goldenSearchMatrix(var1, upper, index, aMatrix, bMatrix, guesses, var2, None, f2, None)
        elif f1 < f2:
            return goldenSearchMatrix(lower, var2, index, aMatrix, bMatrix, guesses, None, var1, None, f1)


def matrixSlicer(indexation: int, aMatrix: np.array, bMatrix: np.array, vMatrix: np.array) -> float:
    # vMatrix = np.squeeze(vMatrix)
    # print(vMatrix)
    columnA = aMatrix[:, indexation]
    rowA = aMatrix[indexation, :]
    vIndexed = vMatrix[indexation]
    return bMatrix[0, indexation] * vIndexed + vIndexed * (
            vMatrix.dot(rowA) + vMatrix.T.dot(columnA) - vIndexed * aMatrix[indexation, indexation])
    # return bMatrix[0, i] * vMatrix[i, 0] + (
    #             np.squeeze(vMatrix.dot(aMatrix[i, :])) + np.squeeze(vMatrix.T.dot(aMatrix[:, i])))


def multiVarMinimizerMatrix(aMatrix: np.array, bMatrix: np.array, vMatrix: np.array, bounds: list) -> float or str:
    lowerBounds = bounds[0]
    upperBounds = bounds[1]
    previousMatrixGuesses = np.zeros(vMatrix.size)
    newMatrixGuesses = np.zeros(vMatrix.size)
    guessDifference = 1

    for iterable in range(len(upperBounds)):
        previousMatrixGuesses[iterable] = random.uniform(lowerBounds[iterable], upperBounds[iterable])
    # previousMatrixGuesses = np.append(previousMatrixGuesses, random.uniform(lowerBounds[i], upperBounds[i]))
    # print(previousMatrixGuesses)

    while guessDifference >= 0.00001:
        previousMatrixGuesses = newMatrixGuesses
        for iteration in range(len(lowerBounds)):
            newMatrixGuesses[iteration] = goldenSearchMatrix(lowerBounds[iteration], upperBounds[iteration], iteration,
                                                             aMatrix, bMatrix, previousMatrixGuesses)
            previousMatrixGuesses[iteration] = newMatrixGuesses[iteration]
        guessDifference = abs(sum(previousMatrixGuesses) - sum(newMatrixGuesses))

    return previousMatrixGuesses


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # s = input("Please provide a function with the variable x: ")
    # a = int(input("Please provide a lower limit for your search space: "))
    # b = int(input("Please provide an upper limit for your search space: "))
    # print(bisectionForOneVar(a, b))

    # print(multivariateFunctionMinimization([-5, 7, -9, 3, -2, 4, 0, 5, -7, 1, -10, 2, -3, 5])) ////Important Line\\\\

    # fig, _ = plt.subplots()
    # type(fig)
    # one_tick = fig.axes[0].yaxis.get_major_ticks()[0]
    # type(one_tick)
    #
    # rng = np.arange(50)
    # rnd = np.random.randint(0, 10, (3, rng.size))
    # yrs = 1950 + rng
    #
    # fig, ax = plt.subplots(5)
    # ax.stackplot(yrs, rng + rnd, ['Eastasia', 'Eurasia', 'Oceania'])
    # ax.set_title('Combined debt growth over time')
    # ax.legend('upper left')
    # ax.set_ylabel('Total debt')
    # ax.set_xlim(yrs[0], yrs[-1])
    # fig.tight_layout()

    timesElapsed = []
    dimensions = []
    aMatrixDimensions = []

    for i in range(100):
        timeStart = time.time()
        matrixList = matrixBuilder()
        aAMatrix = matrixList[0]
        bBMatrix = matrixList[1]
        varMatrix = matrixList[2]
        minimizedMatrix = multiVarMinimizerMatrix(aAMatrix, bBMatrix, varMatrix, boundBuilder(int(varMatrix.size)))
        finalTime = time.time() - timeStart
        timesElapsed.append(finalTime)
        dimensions.append(minimizedMatrix.size)
        aMatrixDimensions.append(aAMatrix.shape)

    # coef = np.polyfit(dimensions,timesElapsed,10)
    # poly1d_fn = np.poly1d(coef)

    # dimensionsSorted = []

    plt.style.use('seaborn-whitegrid')
    plt.title('The Effect of Dimensions on Runtime of Minimization Algorithm')
    plt.scatter(dimensions, timesElapsed, color='red')
    # plt.plot(dimensions, timesElapsed, 'ro', poly1d_fn(sorted(dimensions)), '--k')
    plt.xlabel('dimensions')
    plt.ylabel('time (seconds)')
    plt.show()

    #
    # # make the data
    # np.random.seed(3)
    # x = 4 + np.random.normal(0, 2, 24)
    # y = 4 + np.random.normal(0, 2, len(x))
    # # size and color:
    # sizes = np.random.uniform(15, 80, len(x))
    # colors = np.random.uniform(15, 80, len(x))
    #
    # # plot
    # fig, ax = plt.subplots()
    #
    # ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)
    #
    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
    #        ylim=(0, 8), yticks=np.arange(1, 8))
    #
    # plt.show()

    # print(timesElapsed)
    # print(dimensions)
    # print(aMatrixDimensions)
    print(f'Total time: {sum(timesElapsed)}')
