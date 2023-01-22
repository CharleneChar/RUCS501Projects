import copy
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

np.seterr(invalid='ignore')


def standard_normal_random_vector(d):
    # generate a vector with each element
    # selected from standard normal distribution
    # (where mean is 0 and variance is 1)
    return np.random.normal(0, 1, d).astype('float64')


def generate_matrix_a(matrix_size):
    # generate A with vectors obtained from normal_random_vector function
    return np.transpose(np.stack([(standard_normal_random_vector(matrix_size[1]))
                                  for _ in range(matrix_size[0])]))


def generate_matrix_q(matrix_a):
    # generate Q with A obtained from generate_matrix_a function
    return np.matmul(np.transpose(matrix_a), matrix_a).astype('float64')


def get_first_derivative(q, x, c):
    # compute first derivative with respect to vector x
    return (np.matmul(q, x, dtype=np.float64).astype('float64') - c).astype('float64')


def get_squared_norm_first_derivative(first_derivative):
    # compute ||first derivative with respect to vector x||^2
    return np.sum(np.square(first_derivative).astype('float64')).astype('float64')


def get_bext_x(q, c):
    # compute best x (i.e., x star)
    return np.matmul(np.linalg.inv(q).astype('float64'), c).astype('float64')


def get_norm_error(x, best_x):
    # compute ||x_k − x star|| as error
    return math.sqrt(np.sum(np.square(np.float64(x - best_x)).astype('float64')).astype('float64'))


def get_angle_of_convergence(cur_x, next_x, best_x):
    # compute angle between xk+1 − xk and x∗ − xk
    return np.dot(np.transpose((next_x - cur_x).astype('float64')), (best_x - cur_x).astype('float64')).astype(
        'float64') / \
           (math.sqrt(np.sum(np.square((next_x - cur_x).astype('float64')).astype('float64')).astype('float64')) *
            math.sqrt(np.sum(np.square((best_x - cur_x).astype('float64')).astype('float64')).astype('float64')))


def function_f(x, q, c):
    # compute F(x) where x is a vector
    return (1 / 2) * (np.dot(np.matmul(np.transpose(x), q).astype('float64'), x).astype('float64')) \
           - np.dot(np.transpose(x), c).astype('float64')


def gradient_descent(x, alpha, first_derivative):
    # compute next x (i.e., x_k+1)
    return (x - (alpha * first_derivative).astype('float64')).astype('float64')


def gradient_descent_with_momentum(cur_x, alpha, first_derivative, beta, pre_x):
    # compute next x (i.e., x_k+1) with momentum
    return (cur_x - (alpha * first_derivative).astype('float64') +
            (beta * (cur_x - pre_x).astype('float64')).astype('float64')).astype('float64')


def gradient_descent_with_orthogonal_momentum(cur_x, alpha, first_derivative, beta, m):
    # compute next x (i.e., x_k+1) with orthogonal momentum
    return (cur_x - (alpha * first_derivative).astype('float64') + (beta * m).astype('float64')).astype('float64')


def descent_for_better_direction(x, alpha, m):
    # compute next x (i.e., x_k+1) with better direction
    return (x + (alpha * m).astype('float64')).astype('float64')


def get_optimized_hyperparameter_for_gradient_descent(first_derivative, q):
    # compute optimized alpha
    return (np.dot(np.transpose(first_derivative), first_derivative).astype('float64')) / \
           (np.dot(np.matmul(np.transpose(first_derivative), q).astype('float64'),
                   first_derivative).astype('float64'))


def get_optimized_hyperparameter_for_gradient_descent_with_momentum(first_derivative, q, m):
    # compute optimized alpha and beta
    i = np.dot(np.matmul(np.transpose(m), q).astype('float64'), m).astype('float64')
    j = np.dot(np.matmul(np.transpose(first_derivative), q).astype('float64'), first_derivative).astype('float64')
    l = np.dot(np.matmul(np.transpose(first_derivative), q).astype('float64'), m).astype('float64')
    w = np.dot(np.transpose(first_derivative), first_derivative).astype('float64')
    h = np.dot(np.transpose(m), first_derivative).astype('float64')
    beta = (- j * h + w * l) / (- l * l + j * i)
    alpha = (((- j * h + w * l) * l / (- l * l + j * i)) + w) * (1 / j)
    return alpha, beta


def get_optimized_hyperparameter_for_descent_for_better_direction(cur_first_derivative, q, m):
    # compute hyperparameter for new direction
    return np.dot(np.transpose((-1 * cur_first_derivative).astype('float64')), m) / \
           np.dot(np.matmul(np.transpose(m), q).astype('float64'), m).astype('float64')


def get_beta_for_better_direction(cur_first_derivative, q, m):
    # compute hyperparameter for new direction
    return np.dot(np.matmul(np.transpose(cur_first_derivative), q).astype('float64'), m).astype('float64') / \
           np.dot(np.matmul(np.transpose(m), q).astype('float64'), m).astype('float64')


def get_orthogonal_momentum(first_derivative):
    # compute orthogonal momentum term
    m = standard_normal_random_vector(first_derivative.size)
    proj_m = (np.dot(np.transpose(m), first_derivative).astype('float64')
              / np.sum(np.square(first_derivative).astype('float64')).astype('float64')) \
             * first_derivative
    m -= proj_m.astype('float64')
    return m.astype('float64')


def get_direction(m, beta, cur_first_derivative):
    # compute new direction
    return ((-1 * cur_first_derivative).astype('float64') + (beta * m).astype('float64')).astype('float64')


def plot(stats, name):
    # plot graph to display statistics
    plt.figure(figsize=(8, 6), dpi=100)
    colors_1 = ['palevioletred', 'steelblue', 'teal']
    for i, loss_stats in enumerate(stats):
        plt.plot(stats[loss_stats],
                 label=f'{loss_stats}', color=colors_1[i])
    plt.xlabel('Time t')
    plt.ylim((-5, 5))
    plt.yticks(np.arange(-5, 6, 1))
    plt.ylabel('Error and Angle')
    plt.title('Error and Angle History')
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.show()


def gradient_converge_f(optimal=False, init_x=None, init_q=None, init_c=None):
    # is used to store error of ||cur x - best x|| for plotting purpose
    stats = {'log(cur_error)': [], 'cur_angle': []}
    # indicate alpha
    alpha = 0.01
    # indicate dimension
    d = 10
    # get vector c
    if init_c is None:
        c = standard_normal_random_vector(d)
    else:
        c = init_c
    # get matrix Q
    if init_q is None:
        q = generate_matrix_q(generate_matrix_a((d, d)))
    else:
        q = init_q
    # get best x (i.e., x star)
    best_x = get_bext_x(q, c)
    # initial x
    if init_x is None:
        init_x = cur_x = standard_normal_random_vector(d)
    else:
        cur_x = init_x
    # get norm of error
    norm_error = get_norm_error(cur_x, best_x)
    stats['log(cur_error)'].append(math.log(norm_error))
    # get current F(x) (i.e., x_k)
    cur_f = function_f(cur_x, q, c)
    # get gradient with respect to current x
    first_derivative = get_first_derivative(q, cur_x, c)
    squared_norm_first_derivative = get_squared_norm_first_derivative(first_derivative)
    threshold = get_squared_norm_first_derivative(get_first_derivative(q, best_x, c))
    # keep the process of minimizing F(x) until it converges
    step = 0
    while squared_norm_first_derivative > 0.000001 and step < 1000000:
        print(f'derivative: {first_derivative}')
        print(f'x: {cur_x}')
        print(f'||∇xF(xt)||^2: {squared_norm_first_derivative}')
        print(f'||xk − x∗||: {norm_error}')
        if optimal is True:
            # get optimized alpha
            alpha = get_optimized_hyperparameter_for_gradient_descent(first_derivative, q)
        print(f'alpha: {alpha}')
        # get newly updated x, called x_k+1
        next_x = gradient_descent(cur_x, alpha, first_derivative)
        # get convergence angle
        angle = get_angle_of_convergence(cur_x, next_x, best_x)
        stats['cur_angle'].append(angle)
        print(f'angle between xk+1 − xk and x∗ − xk: {angle}')
        # get newly norm of error
        cur_x = next_x
        norm_error = get_norm_error(cur_x, best_x)
        if norm_error > 0:
            stats['log(cur_error)'].append(math.log(norm_error))
        else:
            stats['log(cur_error)'].append(float('-Inf'))
        # get newly updated F(x), called F(x_k+1)
        cur_f = function_f(cur_x, q, c)
        first_derivative = get_first_derivative(q, cur_x, c)
        squared_norm_first_derivative = get_squared_norm_first_derivative(first_derivative)
        step += 1
    print(f'||∇xF(xt)||^2: {squared_norm_first_derivative}')
    print(f'||xk − x∗||: {norm_error}')
    eigenvalues, eigenvectors = np.linalg.eig(q)
    print(f'init_x: {init_x}')
    print(f'eigenvalues of Q: {eigenvalues}')
    print(f'c: {c}')
    print(f'threshold: {threshold}, step: {step}')
    # plot error of ||cur x - best x|| over time
    plot(stats, f'gradient_descent_f_and_error_optimal{optimal}.png')


def gradient_with_momentum_converge_f(optimal=False, init_x=None, init_q=None, init_c=None):
    # is used to store error of ||cur x - best x|| for plotting purpose
    stats = {'log(cur_error)': [], 'cur_angle': []}
    # indicate alpha and beta
    alpha = 0.01
    beta = 0.1
    # indicate dimension
    d = 10
    # get vector c
    if init_c is None:
        c = standard_normal_random_vector(d)
    else:
        c = init_c
    # get matrix Q
    if init_q is None:
        q = generate_matrix_q(generate_matrix_a((d, d)))
    else:
        q = init_q
    # get best x (i.e., x star)
    best_x = get_bext_x(q, c)
    # initial x
    if init_x is None:
        init_x = cur_x = pre_x = standard_normal_random_vector(d)
    else:
        cur_x = pre_x = init_x
    initial = True
    # get norm of error
    norm_error = get_norm_error(cur_x, best_x)
    stats['log(cur_error)'].append(math.log(norm_error))
    # get current F(x) (i.e., x_k)
    cur_f = function_f(cur_x, q, c)
    # get gradient with respect to current x
    first_derivative = get_first_derivative(q, cur_x, c)
    squared_norm_first_derivative = get_squared_norm_first_derivative(first_derivative)
    threshold = get_squared_norm_first_derivative(get_first_derivative(q, best_x, c))
    # keep the process of minimizing F(x) until it converges
    step = 0
    while squared_norm_first_derivative > 0.000001 and step < 1000000:
        print(f'||∇xF(xt)||^2: {squared_norm_first_derivative}')
        print(f'||xk − x∗||: {norm_error}')
        # indicate momentum term
        m = (cur_x - pre_x).astype('float64')
        if optimal:
            if initial:
                # get optimized alpha
                beta = 0
                alpha = get_optimized_hyperparameter_for_gradient_descent(first_derivative, q)
                initial = False
            else:
                # get optimized alpha and beta
                alpha, beta = get_optimized_hyperparameter_for_gradient_descent_with_momentum(first_derivative, q, m)
        print(f'alpha: {alpha}; beta: {beta}')
        # get newly updated x, called x_k+1
        next_x = gradient_descent_with_momentum(cur_x, alpha, first_derivative, beta, pre_x)
        pre_x = cur_x
        # get convergence angle
        angle = get_angle_of_convergence(cur_x, next_x, best_x)
        stats['cur_angle'].append(angle)
        print(f'angle between xk+1 − xk and x∗ − xk: {angle}')
        # get newly norm of error
        cur_x = next_x
        norm_error = get_norm_error(cur_x, best_x)
        if norm_error > 0:
            stats['log(cur_error)'].append(math.log(norm_error))
        else:
            stats['log(cur_error)'].append(float('-Inf'))
        # get newly updated F(x), called F(x_k+1)
        cur_f = function_f(cur_x, q, c)
        first_derivative = get_first_derivative(q, cur_x, c)
        squared_norm_first_derivative = get_squared_norm_first_derivative(first_derivative)
        step += 1
    print(f'||∇xF(xt)||^2: {squared_norm_first_derivative}')
    print(f'||xk − x∗||: {norm_error}')
    eigenvalues, eigenvectors = np.linalg.eig(q)
    print(f'init_x: {init_x}')
    print(f'eigenvalues of Q: {eigenvalues}')
    print(f'c: {c}')
    print(f'threshold: {threshold}, step: {step}')
    # plot error of ||cur x - best x|| over time
    plot(stats, f'gradient_descent_with_momentum_f_and_error_optimal{optimal}.png')


def gradient_with_orthogonal_momentum_converge_f(optimal=False, init_x=None, init_q=None, init_c=None):
    # is used to store error of ||cur x - best x|| for plotting purpose
    stats = {'log(cur_error)': [], 'cur_angle': []}
    # indicate alpha and beta
    alpha = 0.01
    beta = 0.000001
    # indicate dimension
    d = 10
    # get vector c
    if init_c is None:
        c = standard_normal_random_vector(d)
    else:
        c = init_c
    # get matrix Q
    if init_q is None:
        q = generate_matrix_q(generate_matrix_a((d, d)))
    else:
        q = init_q
    # get best x (i.e., x star)
    best_x = get_bext_x(q, c)
    # initial x
    if init_x is None:
        init_x = cur_x = standard_normal_random_vector(d)
    else:
        cur_x = init_x
    # get norm of error
    norm_error = get_norm_error(cur_x, best_x)
    stats['log(cur_error)'].append(math.log(norm_error))
    # get current F(x) (i.e., x_k)
    cur_f = function_f(cur_x, q, c)
    # get gradient with respect to current x
    first_derivative = get_first_derivative(q, cur_x, c)
    squared_norm_first_derivative = get_squared_norm_first_derivative(first_derivative)
    threshold = get_squared_norm_first_derivative(get_first_derivative(q, best_x, c))
    # keep the process of minimizing F(x) until it converges
    step = 0
    while squared_norm_first_derivative > 0.000001 and step < 1000000:
        print(f'||∇xF(xt)||^2: {squared_norm_first_derivative}')
        print(f'||xk − x∗||: {norm_error}')
        # indicate momentum term
        m = get_orthogonal_momentum(first_derivative)
        if optimal:
            # get optimized alpha and beta
            alpha, beta = get_optimized_hyperparameter_for_gradient_descent_with_momentum(first_derivative, q, m)
        print(f'alpha: {alpha}; beta: {beta}')
        # get newly updated x, called x_k+1
        next_x = gradient_descent_with_orthogonal_momentum(cur_x, alpha, first_derivative, beta, m)
        # get convergence angle
        angle = get_angle_of_convergence(cur_x, next_x, best_x)
        stats['cur_angle'].append(angle)
        print(f'angle between xk+1 − xk and x∗ − xk: {angle}')
        # get newly norm of error
        cur_x = next_x
        norm_error = get_norm_error(cur_x, best_x)
        if norm_error > 0:
            stats['log(cur_error)'].append(math.log(norm_error))
        else:
            stats['log(cur_error)'].append(float('-Inf'))
        # get newly updated F(x), called F(x_k+1)
        cur_f = function_f(cur_x, q, c)
        first_derivative = get_first_derivative(q, cur_x, c)
        squared_norm_first_derivative = get_squared_norm_first_derivative(first_derivative)
        step += 1
    print(f'||∇xF(xt)||^2: {squared_norm_first_derivative}')
    print(f'||xk − x∗||: {norm_error}')
    eigenvalues, eigenvectors = np.linalg.eig(q)
    print(f'init_x: {init_x}')
    print(f'eigenvalues of Q: {eigenvalues}')
    print(f'c: {c}')
    print(f'Q: {q}')
    print(f'threshold: {threshold}, step: {step}')
    # plot error of ||cur x - best x|| over time
    plot(stats, f'gradient_descent_with_orthogonal_momentum_f_and_error_optimal{optimal}.png')


def descent_with_better_direction(optimal=False, init_x=None, init_q=None, init_c=None):
    # is used to store error of ||cur x - best x|| for plotting purpose
    stats = {'log(cur_error)': [], 'cur_angle': []}
    # indicate alpha
    alpha = 0.01
    # indicate dimension
    d = 10
    # get vector c
    if init_c is None:
        c = standard_normal_random_vector(d)
    else:
        c = init_c
    # get matrix Q
    if init_q is None:
        q = generate_matrix_q(generate_matrix_a((d, d)))
    else:
        q = init_q
    # get best x (i.e., x star)
    best_x = get_bext_x(q, c)
    # initial x
    if init_x is None:
        init_x = cur_x = standard_normal_random_vector(d)
    else:
        cur_x = init_x
    initial = True
    # get norm of error
    norm_error = get_norm_error(cur_x, best_x)
    stats['log(cur_error)'].append(math.log(norm_error))
    # get current F(x) (i.e., x_k)
    cur_f = function_f(cur_x, q, c)
    # get gradient with respect to current x
    m = cur_first_derivative = get_first_derivative(q, cur_x, c)
    squared_norm_first_derivative = get_squared_norm_first_derivative(cur_first_derivative)
    threshold = get_squared_norm_first_derivative(get_first_derivative(q, best_x, c))
    beta = 0
    # keep the process of minimizing F(x) until it converges
    step = 0
    while squared_norm_first_derivative > 0.000001 and step < 1000000:
        print(f'||∇xF(xt)||^2: {squared_norm_first_derivative}')
        print(f'||xk − x∗||: {norm_error}')
        # indicate descent direction term
        if initial:
            m = (-1 * cur_first_derivative).astype('float')
            initial = False
        else:
            m = get_direction(m, beta, cur_first_derivative)
        if optimal:
            # get optimized alpha
            alpha = get_optimized_hyperparameter_for_descent_for_better_direction(cur_first_derivative, q, m)
        print(f'alpha: {alpha}')
        # get newly updated x, called x_k+1
        next_x = descent_for_better_direction(cur_x, alpha, m)
        # get convergence angle
        angle = get_angle_of_convergence(cur_x, next_x, best_x)
        stats['cur_angle'].append(angle)
        print(f'angle between xk+1 − xk and x∗ − xk: {angle}')
        # get newly norm of error
        cur_x = next_x
        norm_error = get_norm_error(cur_x, best_x)
        if norm_error > 0:
            stats['log(cur_error)'].append(math.log(norm_error))
        else:
            stats['log(cur_error)'].append(float('-Inf'))
        # get newly updated F(x), called F(x_k+1)
        cur_f = function_f(cur_x, q, c)
        cur_first_derivative = get_first_derivative(q, cur_x, c)
        beta = get_beta_for_better_direction(cur_first_derivative, q, m)
        squared_norm_first_derivative = get_squared_norm_first_derivative(cur_first_derivative)
        step += 1
    print(f'||∇xF(xt)||^2: {squared_norm_first_derivative}')
    print(f'||xk − x∗||: {norm_error}')
    eigenvalues, eigenvectors = np.linalg.eig(q)
    print(f'init_x: {init_x}')
    print(f'eigenvalues of Q: {eigenvalues}')
    print(f'c: {c}')
    print(f'Q: {q}')
    print(f'threshold: {threshold}, step: {step}')
    # plot error of ||cur x - best x|| over time
    plot(stats, f'descent_with_better_direction_f_and_error_optimal{optimal}.png')
