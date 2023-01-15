# def f(V: list) -> float or str: return 5 * V[0] ** 2 + 3 * V[1] ** 2 + 5 * V[2] ** 2 + 3 * V[3] ** 2 + 5 * V[4] **
# 2 + 3 * V[5] ** 2 + 3 * V[6] ** 2


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


# def goldenSearchForOneVar(lower: float, upper: float, index: int, guesses: list, var1=None, var2=None, f1=None,
#                           f2=None) -> float or str:
#     # We make optional parameters to allow us to send either the x1 and f1 or the x2 and f2
#
# The advantage is that we can reuse these test points as they become the new bounds as we get to smaller intervals
#
#     if var1 is None and var2 is None:
#         var1 = upper - ((upper - lower) / gldRto)
#         guesses[index] = var1
#         f1 = f(guesses)
#         var2 = lower + ((upper - lower) / gldRto)
#         guesses[index] = var2
#         f2 = f(guesses)
#         if abs(lower - upper) <= 0.0001:
#             return round(lower)
#         elif f1 >= f2:
#             return goldenSearchForOneVar(var1, upper, index, guesses, var2, None, f2, None)
#         elif f1 < f2:
#             return goldenSearchForOneVar(lower, var2, index, guesses, None, var1, None, f1)
#     elif var2 is None:
#         var2 = lower + ((upper - lower) / gldRto)
#         guesses[index] = var2
#         f2 = f(guesses)
#         if abs(lower - upper) <= 0.0001:
#             return round(lower)
#         elif f1 >= f2:
#             return goldenSearchForOneVar(var1, upper, index, guesses, var2, None, f2, None)
#         elif f1 < f2:
#             return goldenSearchForOneVar(lower, var2, index, guesses, None, var1, None, f1)
#     elif var1 is None:
#         var1 = upper - ((upper - lower) / gldRto)
#         guesses[index] = var1
#         f1 = f(guesses)
#         if abs(lower - upper) <= 0.0001:
#             return round(lower)
#         elif f1 >= f2:
#             return goldenSearchForOneVar(var1, upper, index, guesses, var2, None, f2, None)
#         elif f1 < f2:
#             return goldenSearchForOneVar(lower, var2, index, guesses, None, var1, None, f1)


# def multivariateFunctionMinimization(varInput: list) -> list or str:
#     # we expect input for the function when it is first called to be in the format where each pair in this tuple is a
#     # set of bounds for that variable. So there will be half the number of bounds for the number of variables.
#     numOfVariables = int(len(varInput) / 2)
#     lowerBounds = []
#     upperBounds = []
#     global previousGuesses
#     previousGuesses = []
#     global newGuesses
#     newGuesses = [0] * numOfVariables
#     differenceOfGuesses = 0.0
#     for i in range(len(varInput)):  # This sets up the lower and upper bound lists
#         if i % 2 == 0:
#             lowerBounds.append(varInput[i])
#         else:
#             upperBounds.append(varInput[i])
#
#     for i in range(numOfVariables):  # We set up the first list of initial guesses
#         if lowerBounds[i] is None or upperBounds[i] is None:
#             raise Exception("Bound is missing or some input error has occurred.")
#         else:
#             previousGuesses.append(random.uniform(lowerBounds[i], upperBounds[i]))
#
#     while abs(sum(previousGuesses) - sum(newGuesses)) >= 0.0000001:
#         previousGuesses = newGuesses
#         for i in range(numOfVariables):
#             newGuesses[i] = goldenSearchForOneVar(lowerBounds[i], upperBounds[i], i, previousGuesses)
#             previousGuesses[i] = newGuesses[i]
#
#     return f"Minimums: {newGuesses}"


# print(minimizedMatrix)
# print(np.shape(minimizedMatrix))
# print(f"Time elapsed: {time.time() - timeStart} seconds")

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
