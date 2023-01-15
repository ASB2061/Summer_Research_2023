import random
import time as t
# import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

__author__ = "Adiel Benisty and Jeova Farias Sales Rocha Neto"
__copyright__ = "N/A"
__credits__ = ["Adiel Benisty", "Jeova Farias Sales Rocha Neto"]
__license__ = "N/A"
__version__ = "N/A"
__maintainer__ = "Adiel Benisty"
__email__ = "abenisty@haverford.edu"
__status__ = "Prototype"

golden_ratio = (1 + sqrt(5)) / 2
threshold = 0.00001

"""This function returns the matrix dot product. This is a way of computing a two degree, convex function using matrices
instead of representing the function as a list."""


def matrix_dot_product(a_Matrix: np.array, b_Matrix: np.array, v_Matrix: np.array):
    return v_Matrix.T.dot(a_Matrix.dot(v_Matrix)) + b_Matrix.dot(v_Matrix)


"""Matrices_Builder constructs random convex equations of two degrees. If there is no dimension count provided, the 
function will have a random dimension count between 4 and 25 dimensions. It will then return an A_matrix which 
represents the coefficients of the two degree variables and a B_matrix which represents the coefficients of the single
degree variables. It will also return the true solution of the function for when the minimizer is used."""


def matrices_builder(dimensions: int = None):
    if dimensions is None:
        dimensions = random.randint(4, 25)
    a_Matrix = np.random.rand(dimensions, dimensions)
    a_Matrix = np.dot(a_Matrix, a_Matrix.T) + 0.001 * np.eye(dimensions)
    b_Matrix = np.random.rand(1, dimensions)
    true_solution = - 0.5 * np.linalg.inv(a_Matrix).dot(b_Matrix.T)
    return a_Matrix, b_Matrix, true_solution


"""The bounds builder creates boundaries for the function given. It can create random bounds that do not take the true 
solution into account, random bounds that exclude the true solution, or bounds that guarantee that the true solution is 
in between the bounds."""


def bounds_builder(true_answer: np.array, dimensions: int, setting: str = None):
    lower_Bounds = np.zeros(dimensions)
    upper_Bounds = np.zeros(dimensions)
    lower_bound = None
    upper_bound = None
    if setting is None:
        for i in range(dimensions):
            while True:
                try:
                    lower_bound = -30 + true_answer[i]
                    upper_bound = 30 + true_answer[i]
                    break
                except lower_bound == upper_bound:
                    pass
            lower_Bounds[i] = lower_bound
            upper_Bounds[i] = upper_bound
        return lower_Bounds, upper_Bounds
    elif setting == "random" or setting == "r":
        for i in range(dimensions):
            while True:
                try:
                    lower_bound = random.uniform(-100, 100)
                    upper_bound = random.uniform(lower_bound, 101)
                    break
                except lower_bound == upper_bound:
                    pass
            lower_Bounds[i] = lower_bound
            upper_Bounds[i] = upper_bound
        return lower_Bounds, upper_Bounds
    elif setting == "incorrect":
        for i in range(dimensions):
            while True:
                try:
                    lower_bound = random.uniform(-100, 100)
                    upper_bound = random.uniform(lower_bound, 101)
                    break
                except lower_bound <= true_answer[i] <= upper_bound:
                    pass
            lower_Bounds[i] = lower_bound
            upper_Bounds[i] = upper_bound
        return lower_Bounds, upper_Bounds


"""This is the matrix golden search algorithm. This function receives the bounds, an index for which variable it is, 
and several other values that allow it to do the golden search, including the a and b matrices, an array called guesses
which is used for calculating the function value with the tested value for the variable. This function recurs until it 
reaches the threshold and then will return a minimum and what the lower and upper limits were."""


def golden_search_algorithm(lower_limit: float, upper_limit: float, index: int, a_Matrix: np.array, b_Matrix: np.array,
                            guesses: np.array, x_1: float = None, f_1: float = None, x_2: float = None,
                            f_2: float = None):
    # if t[index] < lower_limit or t[index] > upper_limit:
    #     raise Exception('The solution has escaped the bounds.')
    if abs(upper_limit - lower_limit) > threshold:
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
                return guesses[index], lower_limit, upper_limit
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
                return guesses[index], lower_limit, upper_limit
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
                return guesses[index], lower_limit, upper_limit
    else:
        return guesses[index], lower_limit, upper_limit


"""This utilizes the expansion algorithm, which first attempts to verify whether or not the minimum is in the given 
bounds. If it is not, it will expand and then check again until this condition is verified. Then it will search for the 
minimum using the golden search algorithm the same way."""


def non_matrix_minimizing_by_expansion(lower_limits: np.array, upper_limits: np.array, setting: str,
                                       evaluation: int = 0):
    expansions = 0
    if evaluation > 0:
        evaluation = 0

    if setting == 'one' or setting == "absolute":
        previous_guesses, current_guesses = [0], [0]
        for iteration in range(1):
            previous_guesses[iteration] = random.uniform(lower_limits[iteration], upper_limits[iteration])
            # We first check if the bounds have the local minimum in them by looking at the endpoints and the golden
            # points.
            for var in range(1):
                repeat = True
                while repeat:  # this loop will verify that the local minimum is within the endpoints given to the
                    # function.
                    randomGuess = previous_guesses[var]
                    a = lower_limits[var]
                    previous_guesses[var] = a
                    f_a = functions(previous_guesses, setting)
                    evaluation += 1
                    x_1 = upper_limits[var] - (upper_limits[var] - lower_limits[var]) / golden_ratio
                    previous_guesses[var] = x_1
                    f_x_1 = functions(previous_guesses, setting)
                    evaluation += 1
                    x_2 = lower_limits[var] + (upper_limits[var] - lower_limits[var]) / golden_ratio
                    previous_guesses[var] = x_2
                    f_x_2 = functions(previous_guesses, setting)
                    evaluation += 1
                    b = upper_limits[var]
                    previous_guesses[var] = b
                    f_b = functions(previous_guesses, setting)
                    evaluation += 1
                    if f_a == min(f_a, f_x_1, f_x_2, f_b):
                        lower_limits[var] = (golden_ratio * a - b) / (golden_ratio - 1)
                        expansions += 1
                    elif f_b == min(f_a, f_x_1, f_x_2, f_b):
                        upper_limits[var] = (golden_ratio * b - a) / (golden_ratio - 1)
                        expansions += 1
                    else:
                        repeat = False
        tolerance = accuracy(upper_limits, lower_limits)
        for j in range(1):
            current_guesses[j], lower_limits[j], upper_limits[j], eval2 = non_matrix_golden_search_algorithm(
                lower_limits[j], upper_limits[j], j, previous_guesses, evaluation, setting)
            evaluation += eval2
            previous_guesses[j] = current_guesses[j]
            tolerance = accuracy(upper_limits, lower_limits)

            if tolerance <= threshold:
                return current_guesses, expansions, evaluation
        return current_guesses, expansions, evaluation
    elif setting == 'two' or setting == 'complex' or setting == 'circle':
        previous_guesses, current_guesses = [0, 0], [0, 0]

        for iteration in range(2):
            previous_guesses[iteration] = random.uniform(lower_limits[iteration], upper_limits[iteration])
            # We first check if the bounds have the local minimum in them by looking at the endpoints and the golden
            # points.
            for var in range(2):
                repeat = True
                while repeat:  # this loop will verify that the local minimum is within the endpoints given to the
                    # function.
                    randomGuess = previous_guesses[var]
                    a = lower_limits[var]
                    previous_guesses[var] = a
                    f_a = functions(previous_guesses, setting)
                    evaluation += 1
                    x_1 = upper_limits[var] - (upper_limits[var] - lower_limits[var]) / golden_ratio
                    previous_guesses[var] = x_1
                    f_x_1 = functions(previous_guesses, setting)
                    evaluation += 1
                    x_2 = lower_limits[var] + (upper_limits[var] - lower_limits[var]) / golden_ratio
                    previous_guesses[var] = x_2
                    f_x_2 = functions(previous_guesses, setting)
                    evaluation += 1
                    b = upper_limits[var]
                    previous_guesses[var] = b
                    f_b = functions(previous_guesses, setting)
                    evaluation += 1
                    if f_a == min(f_a, f_x_1, f_x_2, f_b):
                        lower_limits[var] = (golden_ratio * a - b) / (golden_ratio - 1)
                    elif f_b == min(f_a, f_x_1, f_x_2, f_b):
                        upper_limits[var] = (golden_ratio * b - a) / (golden_ratio - 1)
                    else:
                        repeat = False
        tolerance = accuracy(upper_limits, lower_limits)
        try:
            for j in range(previous_guesses.size):
                current_guesses[j], lower_limits[j], upper_limits[j], eval2 = non_matrix_golden_search_algorithm(
                    lower_limits[j], upper_limits[j], j, previous_guesses, evaluation, setting)
                evaluation += eval2
                previous_guesses[j] = current_guesses[j]
                tolerance = accuracy(upper_limits, lower_limits)

                if tolerance <= threshold:
                    return current_guesses
            return current_guesses
        except AttributeError:
            for j in range(len(previous_guesses)):
                current_guesses[j], lower_limits[j], upper_limits[j], eval2 = non_matrix_golden_search_algorithm(
                    lower_limits[j], upper_limits[j], j, previous_guesses, evaluation, setting)
                evaluation += eval2
                previous_guesses[j] = current_guesses[j]
                tolerance = accuracy(upper_limits, lower_limits)

                if tolerance <= threshold:
                    return current_guesses
            return current_guesses


"""This is the same golden search algorithm, however, it does not take numpy arrays as inputs."""


def non_matrix_golden_search_algorithm(lower_limit: float, upper_limit: float, index: int, guesses: list, eval: int,
                                       setting: str = None, x_1: float = None, f_1: float = None, x_2: float = None,
                                       f_2: float = None):
    if abs(upper_limit - lower_limit) > threshold:
        if x_1 is None and x_2 is None:
            x_1 = upper_limit - (upper_limit - lower_limit) / golden_ratio
            x_2 = lower_limit + (upper_limit - lower_limit) / golden_ratio
            guesses[index] = x_1
            f_1 = functions(guesses, setting)
            guesses[index] = x_2
            f_2 = functions(guesses, setting)
            eval += 2
            guesses[index] = sum([x_1, x_2]) / 2
            if f_1 > f_2:
                return non_matrix_golden_search_algorithm(x_1, upper_limit, index, guesses, eval, setting, x_2, f_2,
                                                          None, None)
            elif f_1 < f_2:
                return non_matrix_golden_search_algorithm(lower_limit, x_2, index, guesses, eval, setting, None, None,
                                                          x_1, f_1)
            else:
                return guesses[index], lower_limit, upper_limit, eval
        elif x_1 is None:
            x_1 = upper_limit - (upper_limit - lower_limit) / golden_ratio
            guesses[index] = x_1
            f_1 = functions(guesses, setting)
            guesses[index] = sum([x_1, x_2]) / 2
            if f_1 > f_2:
                return non_matrix_golden_search_algorithm(x_1, upper_limit, index, guesses, eval, setting, x_2, f_2,
                                                          None, None)
            elif f_1 < f_2:
                return non_matrix_golden_search_algorithm(lower_limit, x_2, index, guesses, eval, setting, None, None,
                                                          x_1, f_1)
            else:
                return guesses[index], lower_limit, upper_limit, eval
        elif x_2 is None:
            x_2 = lower_limit + (upper_limit - lower_limit) / golden_ratio
            guesses[index] = x_2
            f_2 = functions(guesses, setting)
            guesses[index] = sum([x_1, x_2]) / 2
            if f_1 > f_2:
                return non_matrix_golden_search_algorithm(x_1, upper_limit, index, guesses, eval, setting, x_2, f_2,
                                                          None, None)
            elif f_1 < f_2:
                return non_matrix_golden_search_algorithm(lower_limit, x_2, index, guesses, eval, setting, None, None,
                                                          x_1, f_1)
            else:
                return guesses[index], lower_limit, upper_limit
    else:
        return guesses[index], lower_limit, upper_limit, eval


"""This is the minimizer by switch algorithm which utilizes the golden search to generate points, but does not 
compute all of them, rather saves one point by the order it takes to compute each function value. """


def non_matrix_minimizer_by_switch(lower_bounds_array: np.array, upper_bounds_array: np.array, setting: str):
    num_variables = lower_bounds_array.size
    guesses = num_variables * [0]
    tolerance = 1
    while tolerance > threshold:
        lower_golden_points = []
        upper_golden_points = []
        for i in range(num_variables):
            lower_golden_points.append(
                upper_bounds_array[i] - (upper_bounds_array[i] - lower_bounds_array[i]) / golden_ratio)
            upper_golden_points.append(
                lower_bounds_array[i] + (upper_bounds_array[i] - lower_bounds_array[i]) / golden_ratio)

        points = [[lower_golden_points[0], lower_golden_points[1]],
                  [upper_golden_points[0], lower_golden_points[1]],
                  [lower_golden_points[0], upper_golden_points[1]],
                  [upper_golden_points[0], upper_golden_points[1]]]

        f_min = min([functions(point, setting) for point in points])

        for i in range(num_variables):
            guesses[i] = upper_bounds_array[i] - (upper_bounds_array[i] - lower_bounds_array[i]) / golden_ratio
            # the first golden point for each dimension so for [0, 0, ... , 0] then [1, 0, ... , 0]
            # these are the first guesses.
        comparing_value = functions(guesses, setting)
        for i in range(num_variables):
            old_guess = guesses[i]
            guesses[i] = lower_bounds_array[i] + (upper_bounds_array[i] - lower_bounds_array[i]) / golden_ratio
            new_value = functions(guesses, setting)
            # print(guesses)
            if comparing_value > new_value:
                comparing_value = new_value
                lower_bounds_array[i] = old_guess
            else:
                upper_bounds_array[i] = guesses[i]
                guesses[i] = old_guess
            tolerance = accuracy(upper_bounds_array, lower_bounds_array)
            if tolerance < threshold:
                return guesses
        pred = functions(guesses, setting)
        print(f_min, pred)
        if f_min < pred:
            pass
    return guesses


"""This is the same minimizer by switch function, except that it takes in matrices instead of using one of the preset
functions."""


def minimizer_by_switch(lower_bounds_array: np.array, upper_bounds_array: np.array, a_matrix: np.array,
                        b_matrix: np.array):
    num_variables = b_matrix.size
    guesses = np.zeros(b_matrix.size)
    tolerance = 1
    # golden_points = [0] * b_matrix.size
    while tolerance > threshold:
        for i in range(num_variables):
            guesses[i] = upper_bounds_array[i] - (upper_bounds_array[i] - lower_bounds_array[i]) / golden_ratio
            # the first golden point for each dimension so for [0, 0, ... , 0] then [1, 0, ... , 0]
            # these are the first guesses.
        comparing_value = matrix_dot_product(a_matrix, b_matrix, guesses)
        for i in range(num_variables):
            old_guess = guesses[i]
            guesses[i] = lower_bounds_array[i] + (upper_bounds_array[i] - lower_bounds_array[i]) / golden_ratio
            new_value = matrix_dot_product(a_matrix, b_matrix, guesses)
            if comparing_value > new_value:
                comparing_value = new_value
                lower_bounds_array[i] = old_guess
            else:
                upper_bounds_array[i] = guesses[i]
                guesses[i] = old_guess
            tolerance = accuracy(upper_bounds_array, lower_bounds_array)
            if tolerance < threshold:
                return guesses
    return guesses


"""A function that looks at how close the lower and upper limits are for each variable in the function and takes the 
maximum difference between the lower and upper limits. This is used for finding when to stop the golden search 
algorithm. """


def accuracy(upper_limits: np.array, lower_limits: np.array):
    toleration = []
    for upper, lower in zip(upper_limits, lower_limits):
        toleration.append(abs(upper - lower))
    return max(toleration)


"""This expansion algorithm works with matrices as opposed to the other one which works with the preset functions."""


def minimizing_algorithm_by_expansion(lower_bounds_array: np.array, upper_bounds_array: np.array, a_matrix: np.array,
                                      b_matrix: np.array, evaluation: int):
    previous_guesses, current_guesses = np.zeros(b_matrix.size), np.zeros(b_matrix.size)
    expansion_counter = 0
    evaluation = 0
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
            evaluation += 1
            x_1 = upper_bounds_array[var] - (upper_bounds_array[var] - lower_bounds_array[var]) / golden_ratio
            previous_guesses[var] = x_1
            f_x_1 = matrix_dot_product(a_matrix, b_matrix, previous_guesses)
            evaluation += 1
            x_2 = lower_bounds_array[var] + (upper_bounds_array[var] - lower_bounds_array[var]) / golden_ratio
            previous_guesses[var] = x_2
            f_x_2 = matrix_dot_product(a_matrix, b_matrix, previous_guesses)
            evaluation += 1
            b = upper_bounds_array[var]
            previous_guesses[var] = b
            f_b = matrix_dot_product(a_matrix, b_matrix, previous_guesses)
            evaluation += 1
            if f_a == min(f_a, f_x_1, f_x_2, f_b):
                lower_bounds_array[var] = (golden_ratio * a - b) / (golden_ratio - 1)
                expansion_counter += 1
            elif f_b == min(f_a, f_x_1, f_x_2, f_b):
                upper_bounds_array[var] = (golden_ratio * b - a) / (golden_ratio - 1)
                expansion_counter += 1
            else:
                repeat = False
    # guesses[index] = x_1
    # f_1 = matrix_dot_product(a_Matrix, b_Matrix, guesses)
    # guesses[index] = x_2
    # f_2 = matrix_dot_product(a_Matrix, b_Matrix, guesses)

    tolerance = accuracy(upper_bounds_array, lower_bounds_array)
    for i in range(b_matrix.size):
        current_guesses[i], lower_bounds_array[i], upper_bounds_array[i], eval2 = golden_search_algorithm(
            lower_bounds_array[i], upper_bounds_array[i], i, a_matrix, b_matrix, previous_guesses)
        previous_guesses[i] = current_guesses[i]
        # accuracy = abs((t_solution - matrix_dot_product(a, b, previous_guesses)) / t_solution)
        # if accuracy < tolerance:
        #     break
        tolerance = accuracy(upper_bounds_array, lower_bounds_array)
        evaluation += eval2
        # print(f'Current solution: {current_guesses}')
        # print(f'upper limit {upper_bounds_array[i]} \n lower limit {lower_bounds_array[i]}')
        # print(f'Threshold: {tolerance}')
        if tolerance <= threshold:
            return current_guesses
    return current_guesses


"""List of nine functions that are part of the testing for this project. They scale up in dimensions. The first has 
one dimension; there are a few three dimensional and five and ten dimensional, convex functions that can also be used
for testing out these methods."""


def functions(variables: list, setting: str = None) -> float:
    if setting == 'one':  # |x| + x^2
        return abs(variables[0]) + variables[0] ** 2
    elif setting == 'absolute':  # |x|
        return abs(variables[0])
    elif setting == 'two':  # |x - y| the issue with this function is that as long as x = y, it will always reach the
        # minimum, therefore there is no 'global' minimum
        return abs(variables[0] - variables[1])
    elif setting == 'complex':  # |x - y| + x^2 + y^2. This works, because of the second degree terms of x and y which
        # force them to be zero otherwise they will not be at the minimum
        return abs(variables[0] - variables[1]) + variables[0] ** 2 + variables[1] ** 2
    elif setting == 'sphere':  # x^2 + y^2 + z^2
        return variables[0] ** 2 + variables[1] ** 2 + variables[2] ** 2
    elif setting == 'circle':  # x^2 + y^2
        return variables[0] ** 2 + variables[1] ** 2
    elif setting == 'twoNew':  # x^2 + y^2 + 7x - 8x
        return variables[0] ** 2 + variables[1] ** 2 + variables[0] * 7 - variables[1] * 8
    elif setting == 'sphereNew':  # x^2 + y^2 + z^2 + a^2 + b^2
        return variables[0] ** 2 + variables[1] ** 2 + variables[2] ** 2 + variables[3] ** 2 + variables[4] ** 2
    elif setting == 'fivetwo':  # x^2 + y^2 + z^2 + a^2 + b^2 - 7
        return variables[0] ** 2 + variables[1] ** 2 + variables[2] ** 2 + variables[3] ** 2 + variables[4] ** 2 - 7
    elif setting == 'fivethree':  # x^2 + y^2 + z^2 + a^2 + b^2 + |a|
        return variables[0] ** 2 + variables[1] ** 2 + variables[2] ** 2 + variables[3] ** 2 + variables[4] ** 2 + abs(
            variables[3])
    elif setting == 'tenone':  # a^2 + b^2 + c^2 + d^2 + e^2 + f^2 + g^2 + h^2 + j^2 + k^2
        return variables[0] ** 2 + variables[1] ** 2 + variables[2] ** 2 + variables[3] ** 2 + variables[4] ** 2 + \
               variables[5] ** 2 + variables[6] ** 2 + variables[7] ** 2 + variables[8] ** 2 + variables[9] ** 2
    elif setting == 'tentwo':  # a^2 + b^2 + c^2 + d^2 + e^2 + 2 * f^2 + g^2 + h^2 + j^2 + k^2
        return variables[0] ** 2 + variables[1] ** 2 + variables[2] ** 2 + variables[3] ** 2 + variables[4] ** 2 + \
               2 * variables[5] ** 2 + variables[6] ** 2 + variables[7] ** 2 + variables[8] ** 2 + variables[9] ** 2
    elif setting == 'tenthree':  # (a - 10)^2 + b^2 + c^2 + d^2 + e^2 + f^2 + g^2 + (h - 10)^2 + j^2 + k^2
        return (variables[0] + 10) ** 2 + variables[1] ** 2 + variables[2] ** 2 + variables[3] ** 2 + variables[
            4] ** 2 + \
               variables[5] ** 2 + variables[6] ** 2 + (variables[7] - 10) ** 2 + variables[8] ** 2 + variables[9] ** 2


# def complex_two_d_function(x: float, y: float) -> float:
#     return abs(x - y)
#
#
# def more_complex_two_d_function(x: float, y: float) -> float:
#     return abs(x - y) + x ** 2 + y ** 2


"""We are first testing the expansion algorithm, to verify whether or not the expansions properly moves into the correct
bounds if the initial bounds do not include the true solution. We select nine functions of varying numbers of dimensions
and give each one 100 random bounds. We then verify whether or not they reached the correct answer and how many times 
they had to expand the bounds until reaching a proper pair of bounds. (This could be recorded on a graph.)"""

if __name__ == '__main__':
    # First Leg of Experiments for the Expanding Algorithm, using the first function. The current threshold is 1 *
    # 10**-5 and the bounds are truly random for these 100 runs.
    print('\n'
          'First Leg of Experiment, using the first function, |x|. The current threshold is 1 * 10**-5 and the bounds '
          'are \n '
          'truly random for these 100 runs.')
    total_cpu_time = 0
    total_wait_time = 0
    for i in range(100):
        starting_cpu_time = float(t.process_time())
        starting_wait_time = float(t.time())
        print(f'True solution:        {np.array([0])}  ')
        test_two_lower_bounds, test_two_upper_bounds = bounds_builder(np.array([0]), 1, "random")
        print(f'Lower limit:          {test_two_lower_bounds} \nUpper limit:          {test_two_upper_bounds}')
        test_one_method_one_optimized_solution, test_one_expansions, evals_one = non_matrix_minimizing_by_expansion(
            test_two_lower_bounds, test_two_upper_bounds, "absolute")
        ending_cpu_time = float(t.process_time())
        ending_wait_time = float(t.time())
        evaluation_cpu_time = ending_cpu_time - starting_cpu_time
        evaluation_wait_time = ending_wait_time - starting_wait_time
        total_cpu_time += evaluation_cpu_time
        total_wait_time += evaluation_wait_time
        print(f'Optimized Solution:   {test_one_method_one_optimized_solution}')
        print(f'Expansions Count:     {test_one_expansions}')
        print(f'Evaluations:          {evals_one}')
        print(f'CPU Time:             {evaluation_cpu_time}')
        print(f'Wait Time:            {evaluation_wait_time}')
        if abs(test_one_method_one_optimized_solution[0]) <= threshold:
            print('Correct Answer \n')
        else:
            print('Incorrect Answer \n')
    print(f'Total Wait Time: {total_wait_time} seconds \nTotal CPU Time:  {total_cpu_time} seconds\n')
    print(
        'First Leg of Experiment, using the second function, x**2 + y**2. The current threshold is 1 * 10**-5 and the '
        '\nbounds '
        'are '
        'truly random for these 100 runs.')
    for i in range(100):
        print(f'True solution:        {np.array([[0.0], [0.0]]).T}  ')  # we transpose the solution here so there are no
        # issues when having it printed in the console.
        test_two_lower_bounds, test_two_upper_bounds = bounds_builder(np.array([[0.0], [0.0]]), 2, "random")
        print(f'Lower limit:          {test_two_lower_bounds} \nUpper limit:          {test_two_upper_bounds}')
        starting_cpu_time = float(t.process_time())
        starting_wait_time = float(t.time())
        test_two_method_one_optimized_solution, test_two_expansions = non_matrix_minimizing_by_expansion(
            test_two_lower_bounds, test_two_upper_bounds, "circle")
        ending_cpu_time = float(t.process_time())
        ending_wait_time = float(t.time())
        evaluation_cpu_time = ending_cpu_time - starting_cpu_time
        evaluation_wait_time = ending_wait_time - starting_wait_time
        total_cpu_time += evaluation_cpu_time
        total_wait_time += evaluation_wait_time
        print(f'Optimized Solution:   {test_two_method_one_optimized_solution}')
        print(f'Expansions Count:     {test_two_expansions}')
        #print(f'Evaluations:          {evals_two}')
        # if (test_two_method_one_optimized_solution) == np.array([[0.0],[0.0]]):
        #     print('Correct Answer \n')
        # else:
        #     print('Incorrect Answer \n')
    print(
        'First Leg of Experiment, using the third function, x**2 + y**2 + |x-y|. The current threshold is 1 * 10**-5 '
        'and the '
        '\nbounds '
        'are '
        'truly random for these 100 runs.')
    for i in range(100):
        print(f'True solution:        {np.array([[0.0], [0.0]]).T}  ')  # we transpose the solution here so there are no
        # issues when having it printed in the console.
        test_two_lower_bounds, test_two_upper_bounds = bounds_builder(np.array([[0.0], [0.0]]), 2, "random")
        print(f'Lower limit:          {test_two_lower_bounds} \nUpper limit:          {test_two_upper_bounds}')
        test_two_method_one_optimized_solution, test_two_expansions, evals_three = non_matrix_minimizing_by_expansion(
            test_two_lower_bounds, test_two_upper_bounds, "absolute")
        print(f'Optimized Solution:   {test_two_method_one_optimized_solution}')
        print(f'Expansions Count:     {test_two_expansions}')
        print(f'Evaluations:          {evals_three}')
        if abs(test_two_method_one_optimized_solution[0]) <= threshold:
            print('Correct Answer \n')
        else:
            print('Incorrect Answer \n')

# First Leg of Experiments for the Expanding Algorithm: Note that the current threshold is 1 * 10**-5
# Correct bounds
# print('First Leg of Experiment with expanding algorithm where function is one dimensional and the bounds are '
#       'correct. \n')
# for i in range(1, 100):
#     test_1_method_1_a, test_1_method_1_b, test_1_method_1_solution = matrices_builder(1)
#     print(f'True Solution:      {test_1_method_1_solution}')
#     test_one_lower_bounds, test_one_upper_bounds = bounds_builder(test_1_method_1_solution, 1)
#     print(f'Lower limit:        {test_one_lower_bounds} \nUpper limit:        {test_one_upper_bounds}')
#     test_1_method_1_optimized_solution = minimizing_algorithm_by_expansion(test_one_lower_bounds,
#                                                                            test_one_upper_bounds,
#                                                                            test_1_method_1_a, test_1_method_1_b)
#     print(f'Optimized Solution: {test_1_method_1_optimized_solution} \n')

# print(
#     'First Leg of Experiment with expanding algorithm where function is one dimensional and the bounds are '
#     'random. (The same function is used) \n')
# print(f'True Solution:      {test_1_method_1_solution}')
# test_one_lower_bounds, test_one_upper_bounds = bounds_builder(test_1_method_1_solution, 1, "incorrect")
# print(f'Lower limit:        {test_one_lower_bounds} \nUpper limit:        {test_one_upper_bounds}')
# test_1_method_1_optimized_solution = minimizing_algorithm_by_expansion(test_one_lower_bounds, test_one_upper_bounds,
#                                                                        test_1_method_1_a, test_1_method_1_b)
# print(f'Optimized Solution: {test_1_method_1_optimized_solution} \n')
#
# print('Second Leg of Experiment with expanding algorithm using exclusively x**2 and bounds are correct.\n')
# test_2_method_1_a = np.array([1])
# test_2_method_1_b = np.array([0])
# test_2_method_1_solution = np.array([0.0])
# print(f'True Solution:      {test_2_method_1_solution}')
# test_two_lower_bounds, test_two_upper_bounds = bounds_builder(test_2_method_1_solution, 1)
# print(f'Lower limit:        {test_two_lower_bounds} \nUpper limit:        {test_two_upper_bounds}')
# test_2_method_1_optimized_solution = minimizing_algorithm_by_expansion(test_two_lower_bounds, test_two_upper_bounds,
#                                                                        test_2_method_1_a, test_2_method_1_b)
# print(f'Optimized Solution: {test_2_method_1_optimized_solution}\n')
#
# print('Second Leg of Experiment with expanding algorithm using exclusively x**2 and bounds are incorrect.\n')
# print(f'True Solution:      {test_2_method_1_solution}')
# test_two_lower_bounds, test_two_upper_bounds = bounds_builder(test_2_method_1_solution, 1, 'incorrect')
# print(f'Lower limit:        {test_two_lower_bounds} \nUpper limit:        {test_two_upper_bounds}')
# test_2_method_1_optimized_solution = minimizing_algorithm_by_expansion(test_two_lower_bounds, test_two_upper_bounds,
#                                                                        test_2_method_1_a, test_2_method_1_b)
# print(f'Optimized Solution: {test_2_method_1_optimized_solution}\n')
#
# print('Third Leg of Experiment with expanding algorithm using |x| + x**2 and bounds are correct.\n')
# test_3_method_1_solution = np.array([0.0])
# print(f'True Solution:      {test_3_method_1_solution}')
# test_3_lower_bounds, test_3_upper_bounds = bounds_builder(test_3_method_1_solution, 1)
# print(f'Lower limit:        {test_3_lower_bounds} \nUpper limit:        {test_3_upper_bounds}')
# test_3_method_1_optimized_solution = non_matrix_minimizing_by_expansion(test_3_lower_bounds, test_3_upper_bounds,
#                                                                         "one")
# print(f'Optimized Solution: {test_3_method_1_optimized_solution}\n')
#
# print('Fourth Leg of Experiment with expanding algorithm using |x| + x**2 and bounds are incorrect.\n')
# test_4_method_1_solution = np.array([0.0])
# print(f'True Solution:      {test_4_method_1_solution}')
# test_4_lower_bounds, test_4_upper_bounds = bounds_builder(test_4_method_1_solution, 1, 'incorrect')
# print(f'Lower limit:        {test_4_lower_bounds} \nUpper limit:        {test_4_upper_bounds}')
# test_4_method_1_optimized_solution = non_matrix_minimizing_by_expansion(test_4_lower_bounds, test_4_upper_bounds,
#                                                                         "one")
# print(f'Optimized Solution: {test_4_method_1_optimized_solution}\n')
#
# print(f'First Leg of Experiment with switching algorithm using |x - y| + x**2 + y**2. Bounds are correct. \n')
# test_1_method_2_solution = np.array([0.0, 0.0])
# print(f'True Solution:      {test_1_method_2_solution}')
# test_1_method_2_lower_bounds, test_1_method_2_upper_bounds = bounds_builder(test_1_method_2_solution, 2)
# print(f'Lower limits:       {test_1_method_2_lower_bounds} \nUpper limits:       {test_1_method_2_upper_bounds}')
# test_1_method_2_optimized_solution = non_matrix_minimizer_by_switch(test_1_method_2_lower_bounds,
#                                                                     test_1_method_2_upper_bounds, "complex")
# print(f'Optimized Solution: {test_1_method_2_optimized_solution}\n')
#
# print(f'Second Leg of Experiment with switching algorithm using z**2 + x**2 + y**2. Bounds are correct. \n')
# test_2_method_2_solution = np.array([0.0, 0.0, 0.0])
# print(f'True Solution:      {test_2_method_2_solution}')
# test_2_method_2_lower_bounds, test_2_method_2_upper_bounds = bounds_builder(test_2_method_2_solution, 3)
# print(f'Lower limits:       {test_2_method_2_lower_bounds} \nUpper limits:       {test_2_method_2_upper_bounds}')
# test_2_method_2_optimized_solution = non_matrix_minimizer_by_switch(test_2_method_2_lower_bounds,
#                                                                     test_2_method_2_upper_bounds, "sphere")
# print(f'Optimized Solution: {test_2_method_2_optimized_solution}\n')
