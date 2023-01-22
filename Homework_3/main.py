import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from sympy import *
from decimal import *


def get_undistricted_win_probability(big_n, p):
    undistricted_win_probability = 1
    for i in range(math.ceil(big_n / 2) - 1 + 1):
        undistricted_win_probability -= (Decimal(math.factorial(big_n)) /
                                         (Decimal(math.factorial(i)) * Decimal(math.factorial(big_n - i)))) * \
                                        Decimal(p**i) * Decimal((1 - p)**(big_n - i))
    return undistricted_win_probability


def get_districted_win_probability(big_n, n, p):
    # compute after analysis of p1, p2, p3
    p1 = (3 / 2) * p
    p2 = (3 / 2) * p
    if p > 2 / 3 or big_n * p > 2 * n:
        p1 = p2 = 1.0
    p3 = 3.0 * p - p1 - p2
    first_term = 0
    for i in range(math.ceil(n / 2) - 1 + 1):
        first_term += (Decimal(math.factorial(n)) /
                       (Decimal(math.factorial(i)) * Decimal(math.factorial(n - i)))) * \
                      Decimal(p1**i) * Decimal((1 - p1)**(n - i))
    second_term = 0
    for i in range(math.ceil(n / 2) - 1 + 1):
        second_term += (Decimal(math.factorial(n)) /
                        (Decimal(math.factorial(i)) * Decimal(math.factorial(n - i)))) * \
                       Decimal(p2**i) * Decimal((1 - p2)**(n - i))
    third_term = 0
    for i in range(math.ceil(n / 2) - 1 + 1):
        third_term += (Decimal(math.factorial(n)) /
                       (Decimal(math.factorial(i)) * Decimal(math.factorial(n - i)))) * \
                      Decimal(p3**i) * Decimal((1 - p3)**(n - i))
    districted_win_probability = (1 - first_term) * (1 - second_term) * third_term + \
                                 (1 - second_term) * (1 - third_term) * first_term + \
                                 (1 - first_term) * (1 - third_term) * second_term + \
                                 (1 - first_term) * (1 - second_term) * (1 - third_term)
    return districted_win_probability


def plot(probability_name, big_n, n):
    # plot two graphs
    x = np.linspace(0.0, 1.0, num=100, dtype='float64')
    y = []
    if probability_name == 'undistricted_win_probability':
        for value in x:
            y.append(get_undistricted_win_probability(big_n, value))
        plot_title = 'Undistricted Win Probability'
    else:
        for value in x:
            y.append(get_districted_win_probability(big_n, n, value))
        plot_title = 'Districted Win Probability'
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(x, y, color='steelblue')
    plt.xlabel('p')
    plt.ylabel('probability')
    plt.title(plot_title)
    plt.savefig(f'{probability_name}.png', bbox_inches='tight')
    plt.show()


def problem1():
    big_n = 303
    n = 101
    p = 0.51
    # compute the value for 3rd bullet point
    undistricted_win_probability = get_undistricted_win_probability(big_n, p)
    print(undistricted_win_probability)
    # compute the value for 7th bullet point
    districted_win_probability = get_districted_win_probability(big_n, n, p)
    print(districted_win_probability)
    # generate graph
    plot('undistricted_win_probability', big_n, n)
    plot('districted_win_probability', big_n, n)


def problem2():
    # solve with integral package
    pdf_y = lambda y: (1 / (2 * math.sqrt(2 * math.pi))) * np.exp(-1 * y**2 / 2) + \
                      (1 / (4 * math.pi)) * np.exp(-1 * y**2 / 4)

    pdf_x = lambda x: (1 / (2 * math.sqrt(2 * math.pi))) * np.exp(-1 * x**2 / 2) + \
                      (1 / (4 * math.pi)) * np.exp(-1 * x**2 / 4)

    pdf_x_y = lambda x, y: (1 / (4 * math.pi)) * np.exp(-1 * x**2 / 2 - y**2 / 2) + \
                      (1 / (8 * math.pi)) * np.exp(-1 * x**2 / 4 - y**2 / 4)

    print(dblquad(pdf_x_y, 1, np.inf, 1, np.inf)[0] / quad(pdf_y, 1, np.inf)[0])
    print(quad(pdf_x, 1, np.inf)[0])


def main():
    problem1()
    problem2()


if __name__ == '__main__':
    main()
