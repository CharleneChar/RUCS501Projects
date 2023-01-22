import random
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


MEAN = 0
VARIANCE = 1
ALPHA = 0.001


def normal_random_matrix_a(mean, variance, matrix_size):
    # generate A ̃ with each element selected as a standard normal random variable
    # (where mean is 0 and variance is 1)
    return np.random.normal(mean, variance, matrix_size)


def generate_matrix_b(matrix_a):
    # generate B with A ̃ obtained from normal random matrix a function
    return np.matmul(np.transpose(matrix_a), matrix_a)


def loss(matrix_a_est, matrix_b):
    # compute L(A)
    return np.sum(
        np.square(
            matrix_b -
            np.matmul(np.transpose(matrix_a_est), matrix_a_est)))


def gradient_descent(matrix_a_est, matrix_b, alpha):
    # compute A_new
    return matrix_a_est - \
           alpha * 4 * \
           np.matmul(matrix_a_est,
                     (np.matmul(np.transpose(matrix_a_est), matrix_a_est)
                      - matrix_b))


def plot(stats, name):
    # plot graph to display statistics
    plt.figure(figsize=(8, 6), dpi=100)
    colors = ['palevioletred', 'steelblue', 'teal']
    if name == 'one_matrix_10x10_loss.png':
        for i, loss_stats in enumerate(stats):
            plt.plot(stats[loss_stats], label=f'{loss_stats}', color=colors[i])
        plt.xlabel('Epoch')
        plt.ylim((0, 20))
        plt.ylabel('Loss')
        plt.title('Loss History')
    elif name == 'many_matrices_10x10_difference.png':
        for i, difference_stats in enumerate(stats):
            plt.plot([j + 1 for j in range(len(stats['cur_difference']))],
                     stats[difference_stats],
                     'o',
                     label=f'{difference_stats}', color=colors[-1])
        plt.xticks(np.arange(1, len(stats['cur_difference']) + 1, 1.0))
        plt.xlabel('ith Recovered Matrix A (where i > 0), i.e., Ai')
        plt.ylim((0, 20))
        plt.ylabel('Difference')
        plt.title('Difference from first recovered matrix A0 History')
    else:
        colors = [random.choice(list(mcolors.CSS4_COLORS))
                  for _ in range(len(mcolors.CSS4_COLORS))]
        for i, loss_stats in enumerate(stats):
            plt.plot(stats[loss_stats],
                     label=f'{loss_stats}', color=colors[i])
        plt.xlabel('Epoch')
        plt.ylim((0, 8))
        plt.ylabel('Loss')
        plt.title('Loss History')
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.show()


def sub_problem1(matrix_size):
    # get A tilde
    matrix_a = normal_random_matrix_a(MEAN, VARIANCE, matrix_size)
    # get A
    matrix_a_est = normal_random_matrix_a(MEAN, VARIANCE, matrix_size)
    # get B
    matrix_b = generate_matrix_b(matrix_a)
    # get L(A)
    cur_loss = loss(matrix_a_est, matrix_b)
    # is used to store L(A) for plotting purpose
    stats = {'cur_loss': [], 'log(cur_loss)': []}
    count = 0
    # keep the process of minimizing L(A) until it converges
    # or count i s over 150
    # ( since I observed that generally when L(A) will converge ,
    # it will converge within 150 epochs)
    while cur_loss > 0 and count < 150:
        count += 1
        stats['cur_loss'].append(cur_loss)
        stats['log(cur_loss)'].append(math.log(cur_loss))
        if math.log(cur_loss) < 0:
            break
        # get newly updated A, called A_new
        matrix_a_est = gradient_descent(matrix_a_est, matrix_b, ALPHA)
        # get newly updated L(A), called L(A_new)
        cur_loss = loss(matrix_a_est, matrix_b)
    # plot L(A) over time
    plot(stats, f'one_matrix_10x10_loss.png')


def sub_problem2(matrix_size, num_matrix_a_est):
    # get A tilde
    matrix_a = normal_random_matrix_a(MEAN, VARIANCE, matrix_size)
    # get B
    matrix_b = generate_matrix_b(matrix_a)
    # is used to store ∥first recovered A0 − later recovered Ai∥^2
    # for plotting purpose
    stats = {'cur_difference': [], 'log(cur_difference)': []}
    # is used to store A0 which is to be computed later
    target_matrix_a = None
    # get recovered A0
    # and recovered different Ai ’s with 20 different initial starting points
    # (where 1 ≤ i ≤ 20)
    for i in range(1 + num_matrix_a_est):
        # get A0 or Ai
        matrix_a_est = normal_random_matrix_a(MEAN, VARIANCE, matrix_size)
        # get L(A0) or L(Ai)
        cur_loss = loss(matrix_a_est, matrix_b)
        count = 0
        # keep minimizing L(A0) or L(Ai)
        while cur_loss > 0 and count < 150:
            count += 1
            if math.log(cur_loss) < 0:
                break
            # get newly updated A0 or Ai
            matrix_a_est = gradient_descent(matrix_a_est, matrix_b, ALPHA)
            # get newly updated L(A0) or L(Ai)
            cur_loss = loss(matrix_a_est, matrix_b)
        if i == 0:
            # get recovered A0
            target_matrix_a = matrix_a_est
        else:
            # get ∥first recovered A0 − later recovered Ai∥^2
            # to check whether Ai can be recovered to A0
            cur_difference = np.sum(np.square(target_matrix_a - matrix_a_est))
            stats['cur_difference'].append(cur_difference)
            stats['log(cur_difference)'].append(math.log(cur_difference))
    plot(stats, f'many_matrices_10x10_difference.png')


def sub_problem3(matrix_size, k_start, k_end):
    # get A tilde
    matrix_a = normal_random_matrix_a(MEAN, VARIANCE, matrix_size)
    # get B
    matrix_b = generate_matrix_b(matrix_a)
    # is used to store L(A) for each different k for plotting purpose
    stats = {f'k={i}_log(cur_loss)': [] for i in range(k_start, k_end + 1)}
    # get A’s with different k’s (from k start to k end) to spot true k for A tilde
    for i in range(k_start, k_end + 1, 1):
        matrix_size = (i, matrix_size[1])
        # get A where k=i
        matrix_a_est = normal_random_matrix_a(MEAN, VARIANCE, matrix_size)
        # get L(A) where k=i
        cur_loss = loss(matrix_a_est, matrix_b)
        count = 0
        # keep minimizing L(A) where k = i
        while cur_loss > 0 and count < 150:
            count += 1
            # print(cur_loss, math.log(cur_loss))
            if math.log(cur_loss) < 0:
                break
            # get newly updated A where k = i
            matrix_a_est = gradient_descent(matrix_a_est, matrix_b, ALPHA)
            # get newly updated L(A) where k = i
            cur_loss = loss(matrix_a_est, matrix_b)
            stats[f'k={i}_log(cur_loss)'].append(math.log(cur_loss))
    plot(stats, f'many_matrices_kx10_loss.png')


def main():
    # show L(A) decreases over time
    sub_problem1((10, 10))
    # show whether same A gets recovered every time
    sub_problem2((10, 10), 20)
    # show L(A) over time for different values of k from k=1 to k=15
    sub_problem3((10, 5), 1, 15)


if __name__ == '__main__':
    main()
