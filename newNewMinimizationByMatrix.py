# import statements: the primary libraries used are numpy and matplotlib
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import pandas as pd

golden_ratio = (1 + sqrt(5)) / 2


def matrix_dot_product(a_Matrix: np.array, b_Matrix: np.array, v_Matrix: np.array):
    return v_Matrix.T.dot(a_Matrix.dot(v_Matrix)) + b_Matrix.dot(v_Matrix)


def matrices_builder(dimensions: int = None):
    if dimensions is None:
        dimensions = random.randint(4, 25)
    a_Matrix = np.random.rand(dimensions, dimensions)
    a_Matrix = np.dot(a_Matrix, a_Matrix.T) + 0.001 * np.eye(dimensions)
    b_Matrix = np.random.rand(1, dimensions)
    true_solution = - 0.5 * np.linalg.inv(a_Matrix).dot(b_Matrix.T)
    return a_Matrix, b_Matrix, true_solution


def bounds_builder(true_answer: np.array, dimensions: int = None):
    lower_Bounds = np.zeros(dimensions)
    upper_Bounds = np.zeros(dimensions)
    lower_bound = None
    upper_bound = None
    for i in range(dimensions):
        while True:
            try:
                lower_bound = -20 + true_answer[i]
                upper_bound = 20 + true_answer[i]
                break
            except lower_bound == upper_bound:
                pass
        lower_Bounds[i] = lower_bound
        upper_Bounds[i] = upper_bound
    return lower_Bounds, upper_Bounds


def randomboundBuilder(numOfVars: int) -> list:
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


def minimizing_algorithm(lower_bounds_array: np.array, upper_bounds_array: np.array, a_matrix: np.array,
                         b_matrix: np.array):
    previous_guesses, current_guesses = np.zeros(b_matrix.size), np.zeros(b_matrix.size)

    for iteration in range(previous_guesses.size):
        previous_guesses[iteration] = random.uniform(lower_bounds_array[iteration], upper_bounds_array[iteration])
    # We first check if the bounds have the local minimum in them by looking at the endpoints and the golden points.
    for var in range(previous_guesses.size):
        repeat = True
        while repeat:  # this loop will verify that the local minimum is within the endpoints given to the function.
            randomGuess = previous_guesses[var]
            a = lower_bounds_array[var]
            previous_guesses[var] = a
            f_a = matrix_dot_product(a_matrix, b_matrix, previous_guesses)
            x_1 = upper_bounds_array[var] - (upper_bounds_array[var] - lower_bounds_array[var]) / golden_ratio
            previous_guesses[var] = x_1
            f_x_1 = matrix_dot_product(a_matrix, b_matrix, previous_guesses)
            x_2 = lower_bounds_array[var] + (upper_bounds_array[var] - lower_bounds_array[var]) / golden_ratio
            previous_guesses[var] = x_2
            f_x_2 = matrix_dot_product(a_matrix, b_matrix, previous_guesses)
            b = upper_bounds_array[var]
            previous_guesses[var] = b
            f_b = matrix_dot_product(a_matrix, b_matrix, previous_guesses)
            if f_a == min(f_a, f_x_1, f_x_2, f_b):
                lower_bounds_array[var] = (golden_ratio * a - b) / (golden_ratio - 1)
            elif f_b == min(f_a, f_x_1, f_x_2, f_b):
                upper_bounds_array[var] = (golden_ratio * b - a) / (golden_ratio - 1)
            else:
                repeat = False
    # guesses[index] = x_1
    # f_1 = matrix_dot_product(a_Matrix, b_Matrix, guesses)
    # guesses[index] = x_2
    # f_2 = matrix_dot_product(a_Matrix, b_Matrix, guesses)

    tolerance = accuracy(upper_bounds_array, lower_bounds_array)
    while tolerance > 0.000001:
        for i in range(b_matrix.size):
            current_guesses[i] = golden_search_algorithm(lower_bounds_array[i], upper_bounds_array[i], i, a_matrix,
                                                         b_matrix, previous_guesses)
            previous_guesses[i] = current_guesses[i]
            # accuracy = abs((t_solution - matrix_dot_product(a, b, previous_guesses)) / t_solution)
            # if accuracy < tolerance:
            #     break
            if i % 10 == 0:
                print(current_guesses)
            tolerance = accuracy(upper_bounds_array, lower_bounds_array)
    return current_guesses


def accuracy(upper_limits: np.array, lower_limits: np.array):
    toleration = []
    for upper, lower in zip(upper_limits, lower_limits):
        toleration.append(abs(upper - lower))
    return max(toleration)


def golden_search_algorithm(lower_limit: float, upper_limit: float, index: int, a_Matrix: np.array, b_Matrix: np.array,
                            guesses: np.array, x_1: float = None, f_1: float = None, x_2: float = None,
                            f_2: float = None):
    # if t[index] < lower_limit or t[index] > upper_limit:
    #     raise Exception('The solution has escaped the bounds.')
    if abs(upper_limit - lower_limit) > 10 ** -8:
        if x_1 is None and x_2 is None:
            x_1 = upper_limit - (upper_limit - lower_limit) / golden_ratio
            x_2 = lower_limit + (upper_limit - lower_limit) / golden_ratio
            guesses[index] = x_1
            f_1 = matrix_dot_product(a_Matrix, b_Matrix, guesses)
            guesses[index] = x_2
            f_2 = matrix_dot_product(a_Matrix, b_Matrix, guesses)
            guesses[index] = sum([x_1, x_2]) / 2
            if f_1 > f_2:
                return golden_search_algorithm(x_1, upper_limit, index, a_Matrix, b_Matrix, guesses, x_2, f_2,
                                               None, None)
            elif f_1 < f_2:
                return golden_search_algorithm(lower_limit, x_2, index, a_Matrix, b_Matrix, guesses, None, None,
                                               x_1, f_1)
            else:
                return guesses[index]
        elif x_1 is None:
            x_1 = upper_limit - (upper_limit - lower_limit) / golden_ratio
            guesses[index] = x_1
            f_1 = matrix_dot_product(a_Matrix, b_Matrix, guesses)
            guesses[index] = sum([x_1, x_2]) / 2
            if f_1 > f_2:
                return golden_search_algorithm(x_1, upper_limit, index, a_Matrix, b_Matrix, guesses, x_2, f_2,
                                               None, None)
            elif f_1 < f_2:
                return golden_search_algorithm(lower_limit, x_2, index, a_Matrix, b_Matrix, guesses, None, None,
                                               x_1, f_1)
            else:
                return guesses[index]
        elif x_2 is None:
            x_2 = lower_limit + (upper_limit - lower_limit) / golden_ratio
            guesses[index] = x_2
            f_2 = matrix_dot_product(a_Matrix, b_Matrix, guesses)
            guesses[index] = sum([x_1, x_2]) / 2
            if f_1 > f_2:
                return golden_search_algorithm(x_1, upper_limit, index, a_Matrix, b_Matrix, guesses, x_2, f_2,
                                               None, None)
            elif f_1 < f_2:
                return golden_search_algorithm(lower_limit, x_2, index, a_Matrix, b_Matrix, guesses, None, None,
                                               x_1, f_1)
            else:
                return guesses[index]
    else:
        return sum([upper_limit, lower_limit]) / 2


# def bound_expander_checker(solution: np.array, lower__bounds: np.array, upper__bounds: np.array):
#     # x_1 = upper_limit - (upper_limit - lower_limit) / golden_ratio
#     # x_2 = lower_limit + (upper_limit - lower_limit) / golden_ratio
#     # | . . |
#     # |     .     .    |
#     # is the idea here. or the opposite for the other direction.
#     bounds_changed = [0] * lower__bounds.size
#     for i in range(lower__bounds.size):
#         x_1 = upper__bounds[i] - (upper__bounds[i] - lower__bounds[i]) / golden_ratio
#
#         x_2 = lower__bounds[i] + (upper__bounds[i] - lower__bounds[i]) / golden_ratio
#         if solution[i] == lower__bounds[i] and bounds_changed[i] != 1:
#             new_gold_point_two = lower__bounds[i]
#             new_lower = (golden_ratio * new_gold_point_two - upper__bounds[i]) / (golden_ratio - 1)
#             lower__bounds[i] = new_lower
#             bounds_changed[i] = 1
#         elif solution[i] == upper__bounds[i] and bounds_changed[i] != 1:
#             new_gold_point_one = upper__bounds[i]
#             new_upper = (golden_ratio * new_gold_point_one - lower__bounds[i]) / (golden_ratio - 1)
#             upper__bounds[i] = new_upper
#             bounds_changed[i] = 1
#     return lower__bounds, upper__bounds, bounds_changed
#
#
# def new_minimizing_algorithm(lower_bounds_array: np.array, upper_bounds_array: np.array, a_matrix: np.array,
#                              b_matrix: np.array, old_solution: np.array, bounds_checker: list = None):
#     for i in range()
#     previous_guesses, current_guesses = old_solution, old_solution
#     summation = 0
#     for check in bounds_checker:
#         if check == 0:
#             summation += 1
#         elif check == 1:
#             break
#     if summation == len(bounds_checker):
#         return old_solution
#     for iteration in range(previous_guesses.size):
#         if bounds_checker[iteration] == 0:
#             continue
#         else:
#             previous_guesses[iteration] = random.uniform(lower_bounds_array[iteration], upper_bounds_array[iteration])
#     # accuracy =
#     # while accuracy > tolerance:
#     for _ in range(1, 101):
#         for i in range(b_matrix.size):
#             if bounds_checker[i] == 1: # skips when the minimum for that variable has already been found.
#                 current_guesses[i] = golden_search_algorithm(lower_bounds_array[i], upper_bounds_array[i], i, a_matrix,
#                                                              b_matrix, previous_guesses)
#                 previous_guesses[i] = current_guesses[i]
#                 # accuracy = abs((t_solution - matrix_dot_product(a, b, previous_guesses)) / t_solution)
#                 # if accuracy < tolerance:
#                 #     break
#             else:
#                 continue
#         if _ % 10 == 0:
#             print(current_guesses)
#     return current_guesses


def plot_curve(A, b, p, string):
    range_x = [-1, 1]
    range_y = [-1, 1]
    x, y = np.meshgrid(np.linspace(range_x[0], range_x[1], 50), np.linspace(range_y[0], range_y[1], 50))

    f = A[0, 0] * x ** 2 + A[1, 1] * y ** 2 + A[0, 1] * x * y + A[1, 0] * y * x \
        + b[1] * y + b[0] * x

    _, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.scatter(p[0], p[1])
    plt.title(string)
    plt.contour(x, y, f)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    d = 2
    the_matrices = matrices_builder(d)
    # lower_bounds, upper_bounds = bounds_builder(the_matrices[2], d)[0], bounds_builder(the_matrices[2], d)[1]
    # print(lower_bounds)
    # print(upper_bounds)
    lower_bounds, upper_bounds = randomboundBuilder(2)

    #a, b, t = the_matrices[0], the_matrices[1], the_matrices[2]
    # a = np.eye(2) * np.random.rand(2)
    a = np.array([[3, -2], [-2, 3]])
    b = np.squeeze(np.random.rand(1, d))
    # b = np.zeros(d)
    t = np.squeeze(- 0.5 * np.linalg.solve(a, b))

    t_solution = matrix_dot_product(a, b, t)

    limitOfSolution = 0.000001
    optimized_solution = minimizing_algorithm(lower_bounds, upper_bounds, a, b)

    plot_curve(a, b, optimized_solution, "OPT")
    plot_curve(a, b, t, "TRUE")

    # print(abs(np.sum(np.zeros(5)) - np.sum(np.ones(5))))
    # print(main_guesses)
    print(f'The A matrix: \n {a}')
    print(f'The b matrix: \n {b}')
    print(f'The true solution: \n {t}')
    print(f'The true value: \t {t_solution}')
    print(
        f'Optimized Solution: \n {optimized_solution}')
    print(f'Optimized Value: \n {matrix_dot_product(a, b, optimized_solution)}')
