import math

import numpy as np
import scipy.linalg

from problem1 import *
from problem2 import *
from scipy import *


def main():
    # solve problem 1:
    d = 10
    optimal = True
    q = generate_matrix_q(generate_matrix_a((d, d)))
    c = standard_normal_random_vector(d)
    init_x = standard_normal_random_vector(d)
    # apply gradient descent
    gradient_converge_f(init_x=init_x, init_q=q, init_c=c)
    # apply gradient descent with optimal hyperparameter
    gradient_converge_f(optimal=optimal, init_x=init_x, init_q=q, init_c=c)
    # apply momentum
    gradient_with_momentum_converge_f(init_x=init_x, init_q=q, init_c=c)
    # apply momentum with optimal hyperparameter
    gradient_with_momentum_converge_f(optimal=optimal, init_x=init_x, init_q=q, init_c=c)
    # apply orthogonal momentum
    gradient_with_orthogonal_momentum_converge_f(init_x=init_x, init_q=q, init_c=c)
    # apply orthogonal momentum with optimal hyperparameter
    gradient_with_orthogonal_momentum_converge_f(optimal=optimal, init_x=init_x, init_q=q, init_c=c)
    # apply a better direction
    descent_with_better_direction(optimal=optimal, init_x=init_x, init_q=q, init_c=c)

    # solve problem 2:
    # get best xi when it's real
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = get_best_x_when_all_real()
    # get solution when xi is real (which is the upper bound of F)
    real_solution = -1 * get_best_value_of_dual_function()
    # get inital lower bound of F and its corresponding x values
    # (call print(f'upper bound of F: {real_solution}, initial lower bound of F: {greedy_solution}')to see result)
    greedy_solution, xs = get_f_value_from_greedy_variable_assignment()
    # apply Branch and Bound Algorithm
    # (call print(f'final solution of F: {final_solution}') and
    # call print(f'THe number of complete variable assignments to visit to discover the optimum: {count}')
    # to see result)
    final_solution, final_xs, count = branch_and_bound()


if __name__ == '__main__':
    main()
