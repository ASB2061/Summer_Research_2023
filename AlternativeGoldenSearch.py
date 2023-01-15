import random
import time
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

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


def minimizer_by_switch(lower_bounds_array: np.array, upper_bounds_array: np.array, a_matrix: np.array,
                        b_matrix: np.array):
    num_variables = b_matrix.size
    guesses = np.zeros(b_matrix.size)
    # golden_points = [0] * b_matrix.size
    while True:
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
                continue
            else:
                upper_bounds_array[i] = guesses[i]
                guesses[i] = old_guess


if __name__ == '__main__':
    print("testing")
