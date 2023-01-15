# Step One Recover minimization lab from CS105

import doctest


def quadratic_function(x: float, b: float, c: float):
    return x ** 2 + b * x + c


def newFunction(x: float):
    return abs(x) - 0.2 * x


def harder_function(x: float):
    return x ** 4 + abs(x) + 2 ** x


def find_minimizer(lower: float, upper: float) -> float or str:
    """
    Add the description of your code here
    The purpose of my code is to find a local minimum for the function f(x) = x^2 + -5x + 5 given the range of lower and
     upper values for the domain of the input variable for the parabolic function; in this case x. The function will
     compare the outputs of the upper and lower values for the quadratic function. If the lower is greater than the
     upper, then the minimum must be between the average of the upper and lower inputs and the upper value. The function
     will then recurse with the average value for the upper and lower as the lower and the upper value will remain the
     same until the values of lower and upper are the same. If the opposite is true, then the function will return
     itself with the same lower value and the average of the upper and lower value. This will be continued until the
     upper and lower inputs are the same.

    :param lower: Describe parameter lower here. Lower is the lowest value of the domain of x (for determining the local
    minimum). It is the initial value used for comparison to determine the location of the local minimum of x^2 + bx + c
    This parameter changes as the function recurses until the function reaches the base case
    :param upper: Describe parameter upper here. Upper is the highest value of the domain of x (for determining the
    local minimum). It is the initial value used for comparison to determine the location of the local minimum of x^2 +
    bx + c. This parameter changes as the function recurses until the function reaches the base case

    :return: Describe the function return here The function will return a float type which will tell you the location of
    the local minimum. If you take the return value and input it into the quadratic function, it will give you the value
    of the local minimum. However, if the parameters lower and upper do not fulfill the precondition of lower being less
    than upper, then the function will return "Error"

    :examples:
    >>> find_minimizer(-10.0, 10.0)
    2
    >>> find_minimizer(-2.5,3)
    2
    >>> find_minimizer(0,5) # This addresses the MVT case
    2
    >>> find_minimizer(5,5)
    5
    >>> find_minimizer(4,6)
    4
    >>> find_minimizer(6,4)
    'Error'

    """
    # Your code goes here

    b = -5
    c = 5
    # print(quadratic_function(x,b,c))
    if lower > upper:  # Base case lower must be less than upper: the precondition
        return "Error"
    elif abs(lower - upper) <= 0.009:  # My base case is if the difference of the two limits of the range is 0.009 or
        # less
        return round(lower)  # returns a rounded value thus no need for the round() in the docstrings
    elif quadratic_function(lower, b, c) < quadratic_function(upper, b, c):  # if the lower function output is lower
        # than the upper function output
        return find_minimizer(lower, (upper + lower) / 2)  # function will return itself, but the upper value will be
        # the average of the upper and lower value
    elif quadratic_function(lower, b, c) > quadratic_function(upper, b, c):  # if the lower function output is higher
        # than the upper function output
        return find_minimizer((upper + lower) / 2, upper)  # function will return itself, but the lower value will be
        # the average of the upper and lower value
    elif quadratic_function(lower, b, c) == quadratic_function(upper, b,
                                                               c):  # This is justified by the mean value theorem where
        # if the curve of a function is differentiable and if f(a) = f(b) then there must be a point f'((a+b)/2) = 0
        return round((lower + upper) / 2)


def harderFunctionMinimizer(lower: float, upper: float):
    """
    The purpose of my code is to find a local minimum for the function f(x) = 2^x + |x| + x^4 given the range of lower and
     upper values for the domain of the input variable for the function; in this case x. The function will
     compare the outputs of the upper and lower values for the function. If the lower is greater than the
     upper, then the minimum must be between the average of the upper and lower inputs and the upper value. The function
     will then recurse with the average value for the upper and lower as the lower and the upper value will remain the
     same until the values of lower and upper are the same. If the opposite is true, then the function will return
     itself with the same lower value and the average of the upper and lower value. This will be continued until the
     upper and lower inputs are the same.

    :param lower: Describe parameter lower here. Lower is the lowest value of the domain of x (for determining the local
    minimum). It is the initial value used for comparison to determine the location of the local minimum of the function
    This parameter changes as the function recurses until the function reaches the base case
    :param upper: Describe parameter upper here. Upper is the highest value of the domain of x (for determining the
    local minimum). It is the initial value used for comparison to determine the location of the local minimum of the
    function. This parameter changes as the function recurses until the function reaches the base case

    :return: Describe the function return here The function will return a float type which will tell you the location of
    the local minimum. If you take the return value and input it into the quadratic function, it will give you the value
    of the local minimum. However, if the parameters lower and upper do not fulfill the precondition of lower being less
    than upper, then the function will return "Error"
    :Examples:
    >>> harderFunctionMinimizer(5,9)
    5
    >>> harderFunctionMinimizer(-5,-2)
    -2
    >>> harderFunctionMinimizer(-1,4)
    0
    >>> harderFunctionMinimizer(4,-1)
    'Error'

    """
    if lower > upper:  # Precondition. The lower must be lower than the upper
        return "Error"  # Otherwise, you must return an Error
    elif abs(lower - upper) <= 0.009:  # Spoke with Benjamin Norris at 11:15 PM to troubleshoot and found that this
        # fixed an issue when the function was looking for 0, but the function was returning a None Type; I am curious
        # as to why the function was returning a None Type for this case. Also moved this onto my other function in this
        # file
        return round((lower + upper) / 2)  # Returns the rounded average of lower and upper
    elif harder_function(lower) < harder_function(upper):  # If the lower output is lower than the higher output
        return harderFunctionMinimizer(lower, (upper + lower) / 2)  # Recursion where the upper switches to the average
        # of the upper and lower inputs
    elif harder_function(lower) > harder_function(upper):  # If the lower output is higher than the higher output
        return harderFunctionMinimizer((upper + lower) / 2, upper)  # Recursion where the lower switches to the average
        # of the upper and lower inputs


# Uncomment the next line to run the tests
doctest.testmod()

if __name__ == '__main__':
    print(find_minimizer(-1, 2))

find_minimizer(-10.0, 10.0)
