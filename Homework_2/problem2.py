import math
import copy

INVALID = 1.1


def objective_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    # compute F(x)
    return 104 * x1 + 128 * x2 + 135 * x3 + 139 * x4 + 150 * x5 + 153 * x6 + \
           162 * x7 + 168 * x8 + 195 * x9 + 198 * x10


def neg_objective_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    # compute -F(x)
    return -(104 * x1 + 128 * x2 + 135 * x3 + 139 * x4 + 150 * x5 + 153 * x6 +
             162 * x7 + 168 * x8 + 195 * x9 + 198 * x10)


def constraint_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    # compute g(x)
    return 9 * x1 ** 2 + 8 * x2 ** 2 + 7 * x3 ** 2 + 7 * x4 ** 2 + 6 * x5 ** 2 + 6 * x6 ** 2 + \
           5 * x7 ** 2 + 2 * x8 ** 2 + x9 ** 2 + x10 ** 2 - 68644


def lagragian_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, mu):
    # compute L(x, μ)
    return neg_objective_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) + \
           mu * constraint_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)


def get_best_x_gradient_of_lagragian_function(mu,
                                              r_x1=INVALID, r_x2=INVALID, r_x3=INVALID, r_x4=INVALID,
                                              r_x5=INVALID, r_x6=INVALID, r_x7=INVALID, r_x8=INVALID,
                                              r_x9=INVALID, r_x10=INVALID):
    # compute best x by setting ∇L(x, μ) (with respect to x) to zero
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = \
        104 / (18 * mu), 128 / (16 * mu), 135 / (14 * mu), 139 / (14 * mu), \
        150 / (12 * mu), 153 / (12 * mu), 162 / (10 * mu), 168 / (4 * mu), \
        195 / (2 * mu), 198 / (2 * mu)
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = \
        r_x1 if r_x1 is not INVALID else x1, r_x2 if r_x2 is not INVALID else x2, r_x3 if r_x3 is not INVALID else x3, \
        r_x4 if r_x4 is not INVALID else x4, r_x5 if r_x5 is not INVALID else x5, r_x6 if r_x6 is not INVALID else x6, \
        r_x7 if r_x7 is not INVALID else x7, r_x8 if r_x8 is not INVALID else x8, \
        r_x9 if r_x9 is not INVALID else x9, r_x10 if r_x10 is not INVALID else x10
    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10


def get_best_mu_of_dual_function(r_x1=INVALID, r_x2=INVALID, r_x3=INVALID, r_x4=INVALID,
                                 r_x5=INVALID, r_x6=INVALID, r_x7=INVALID, r_x8=INVALID,
                                 r_x9=INVALID, r_x10=INVALID):
    # compute best μ by setting ∇Q(μ) (with respect to μ) to zero
    mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, mu_x6, mu_x7, mu_x8, mu_x9, mu_x10 = \
        280 * 104 ** 2, 315 * 128 ** 2, 360 * 135 ** 2, 360 * 139 ** 2, \
        420 * 150 ** 2, 420 * 153 ** 2, 504 * 162 ** 2, 1260 * 168 ** 2, \
        2520 * 195 ** 2, 2520 * 198 ** 2
    mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, mu_x6, mu_x7, mu_x8, mu_x9, mu_x10 = \
        0 if r_x1 is not INVALID else mu_x1, 0 if r_x2 is not INVALID else mu_x2, 0 if r_x3 is not INVALID else mu_x3, \
        0 if r_x4 is not INVALID else mu_x4, 0 if r_x5 is not INVALID else mu_x5, 0 if r_x6 is not INVALID else mu_x6, \
        0 if r_x7 is not INVALID else mu_x7, 0 if r_x8 is not INVALID else mu_x8, \
        0 if r_x9 is not INVALID else mu_x9, 0 if r_x10 is not INVALID else mu_x10
    a_x1, a_x2, a_x3, a_x4, a_x5, a_x6, a_x7, a_x8, a_x9, a_x10 = \
        0 if r_x1 is INVALID else 9 * r_x1 ** 2, 0 if r_x2 is INVALID else 8 * r_x2 ** 2, 0 if r_x3 is INVALID else 7 * r_x3 ** 2, \
        0 if r_x4 is INVALID else 7 * r_x4 ** 2, 0 if r_x5 is INVALID else 6 * r_x5 ** 2, 0 if r_x6 is INVALID else 6 * r_x6 ** 2, \
        0 if r_x7 is INVALID else 5 * r_x7 ** 2, 0 if r_x8 is INVALID else 2 * r_x8 ** 2, \
        0 if r_x9 is INVALID else r_x9 ** 2, 0 if r_x10 is INVALID else r_x10 ** 2
    return math.sqrt((mu_x1 + mu_x2 + mu_x3 + mu_x4 +
                      mu_x5 + mu_x6 + mu_x7 + mu_x8 +
                      mu_x9 + mu_x10) /
                     ((68644 - a_x1 - a_x2 - a_x3 -
                       a_x4 - a_x5 - a_x6 - a_x7 - a_x8 - a_x9 - a_x10) *
                      2 * 9 * 2 * 8 * 7 * 5)
                     )


def get_best_value_of_dual_function(r_x1=INVALID, r_x2=INVALID, r_x3=INVALID, r_x4=INVALID,
                                    r_x5=INVALID, r_x6=INVALID, r_x7=INVALID, r_x8=INVALID,
                                    r_x9=INVALID, r_x10=INVALID):
    # compute largest value for dual function by plugging in the best mu
    mu = get_best_mu_of_dual_function(r_x1=r_x1, r_x2=r_x2, r_x3=r_x3, r_x4=r_x4,
                                      r_x5=r_x5, r_x6=r_x6, r_x7=r_x7, r_x8=r_x8,
                                      r_x9=r_x9, r_x10=r_x10)
    mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, mu_x6, mu_x7, mu_x8, mu_x9, mu_x10 = \
        104 / (18 * mu), 128 / (16 * mu), 135 / (14 * mu), \
        139 / (14 * mu), 150 / (12 * mu), 153 / (12 * mu), \
        162 / (10 * mu), 168 / (4 * mu), 195 / (2 * mu), \
        198 / (2 * mu)
    mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, mu_x6, mu_x7, mu_x8, mu_x9, mu_x10 = \
        0 if r_x1 is not INVALID else mu_x1, 0 if r_x2 is not INVALID else mu_x2, 0 if r_x3 is not INVALID else mu_x3, \
        0 if r_x4 is not INVALID else mu_x4, 0 if r_x5 is not INVALID else mu_x5, 0 if r_x6 is not INVALID else mu_x6, \
        0 if r_x7 is not INVALID else mu_x7, 0 if r_x8 is not INVALID else mu_x8, \
        0 if r_x9 is not INVALID else mu_x9, 0 if r_x10 is not INVALID else mu_x10
    a_x1, a_x2, a_x3, a_x4, a_x5, a_x6, a_x7, a_x8, a_x9, a_x10 = \
        0 if r_x1 is INVALID else 9 * r_x1 ** 2, 0 if r_x2 is INVALID else 8 * r_x2 ** 2, 0 if r_x3 is INVALID else 7 * r_x3 ** 2, \
        0 if r_x4 is INVALID else 7 * r_x4 ** 2, 0 if r_x5 is INVALID else 6 * r_x5 ** 2, 0 if r_x6 is INVALID else 6 * r_x6 ** 2, \
        0 if r_x7 is INVALID else 5 * r_x7 ** 2, 0 if r_x8 is INVALID else 2 * r_x8 ** 2, \
        0 if r_x9 is INVALID else r_x9 ** 2, 0 if r_x10 is INVALID else r_x10 ** 2
    return -(104 * mu_x1 + 128 * mu_x2 + 135 * mu_x3 + 139 * mu_x4 +
             150 * mu_x5 + 153 * mu_x6 + 162 * mu_x7 + 168 * mu_x8 +
             195 * mu_x9 + 198 * mu_x10) + \
           (9 * mu * mu_x1 ** 2 + 8 * mu * mu_x2 ** 2 +
            7 * mu * mu_x3 ** 2 + 7 * mu * mu_x4 ** 2 +
            6 * mu * mu_x5 ** 2 + 6 * mu * mu_x6 ** 2 +
            5 * mu * mu_x7 ** 2 + 2 * mu * mu_x8 ** 2 +
            mu * mu_x9 ** 2 + mu * mu_x10 ** 2 -
            (68644 - a_x1 - a_x2 - a_x3 - a_x4 - a_x5 -
             a_x6 - a_x7 - a_x8 - a_x9 - a_x10) * mu)


def get_best_x_when_all_real(r_x1=INVALID, r_x2=INVALID, r_x3=INVALID, r_x4=INVALID,
                             r_x5=INVALID, r_x6=INVALID, r_x7=INVALID, r_x8=INVALID,
                             r_x9=INVALID, r_x10=INVALID):
    # compute x when taken to be real values instead of integers
    if r_x1 != INVALID and r_x2 != INVALID and r_x3 != INVALID and r_x4 != INVALID and r_x5 != INVALID \
            and r_x6 != INVALID and r_x7 != INVALID and r_x8 != INVALID and r_x9 != INVALID and r_x10 != INVALID:
        return r_x1, r_x2, r_x3, r_x4, r_x5, r_x6, r_x7, r_x8, r_x9, r_x10
    mu = get_best_mu_of_dual_function(r_x1=r_x1, r_x2=r_x2, r_x3=r_x3, r_x4=r_x4,
                                      r_x5=r_x5, r_x6=r_x6, r_x7=r_x7, r_x8=r_x8,
                                      r_x9=r_x9, r_x10=r_x10)
    return get_best_x_gradient_of_lagragian_function(mu=mu,
                                                     r_x1=r_x1, r_x2=r_x2, r_x3=r_x3, r_x4=r_x4,
                                                     r_x5=r_x5, r_x6=r_x6, r_x7=r_x7, r_x8=r_x8,
                                                     r_x9=r_x9, r_x10=r_x10)


def get_x_from_greedy_variable_assignment():
    # compute x by using floor of real value of x
    integer_values = []
    for x in get_best_x_when_all_real():
        integer_values.append(math.floor(x))
    return tuple(integer_values)


def get_f_value_from_greedy_variable_assignment():
    # compute the lower bound of objective function
    # and lowest possible value for each x to reach
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = get_x_from_greedy_variable_assignment()
    is_constraint_satisfied = constraint_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) <= 0
    while not is_constraint_satisfied:
        x1 -= 1
        is_constraint_satisfied = constraint_function(x1 + 1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
    return objective_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10), (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)


def get_highest_possible_value_for_each_x():
    # compute the highest possible value for each x to reach
    return math.floor(math.sqrt(68644 / 9)), math.floor(math.sqrt(68644 / 8)), math.floor(math.sqrt(68644 / 7)), \
           math.floor(math.sqrt(68644 / 7)), math.floor(math.sqrt(68644 / 6)), math.floor(math.sqrt(68644 / 6)), \
           math.floor(math.sqrt(68644 / 5)), math.floor(math.sqrt(68644 / 2)), math.floor(math.sqrt(68644)), \
           math.floor(math.sqrt(68644))


def update_lower_bound(lower_bound, i1=INVALID, i2=INVALID, i3=INVALID,
                       i4=INVALID, i5=INVALID, i6=INVALID, i7=INVALID,
                       i8=INVALID, i9=INVALID, i10=INVALID):
    # update lower bound when newly computed F(x) is higher than the lower bound
    # (and surely not larger than the higher bound from using real values of x)
    cur_x_value = get_best_x_when_all_real(r_x1=i1, r_x2=i2,
                                           r_x3=i3, r_x4=i4,
                                           r_x5=i5, r_x6=i6,
                                           r_x7=i7, r_x8=i8,
                                           r_x9=i9, r_x10=i10)
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = cur_x_value
    cur_f_value = objective_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
    if cur_f_value < lower_bound:
        return lower_bound, True, None
    lower_bound = cur_f_value
    x1_b, x2_b, x3_b, x4_b, x5_b, x6_b, x7_b, x8_b, x9_b, x10_b = \
        get_best_x_when_all_real(r_x1=i1, r_x2=i2,
                                 r_x3=i3, r_x4=i4,
                                 r_x5=i5, r_x6=i6,
                                 r_x7=i7, r_x8=i8,
                                 r_x9=i9, r_x10=i10)
    return lower_bound, False, (x1_b, x2_b, x3_b, x4_b, x5_b, x6_b, x7_b, x8_b, x9_b, x10_b)


def branch_and_bound():
    # get upper bound on possible solutions to original problem
    upper_bound = -1 * get_best_value_of_dual_function()
    # get lower bound on possible solutions and possible_value_for_each_x
    lower_bound, (x1_l, x2_l, x3_l, x4_l, x5_l, x6_l, x7_l, x8_l, x9_l, x10_l) = \
        get_f_value_from_greedy_variable_assignment()
    max_f = lower_bound
    # get highest possible value for each x
    x1_h, x2_h, x3_h, x4_h, x5_h, x6_h, x7_h, x8_h, x9_h, x10_h = get_highest_possible_value_for_each_x()
    # best x (which will later get updated to be the final solution's corresponding x values)
    x1_b, x2_b, x3_b, x4_b, x5_b, x6_b, x7_b, x8_b, x9_b, x10_b = \
        x1_l, x2_l, x3_l, x4_l, x5_l, x6_l, x7_l, x8_l, x9_l, x10_l
    # set search range for each x
    x1_values = list(range(x1_l, x1_h + 1))
    x2_values = list(range(x2_l, x2_h + 1))
    x3_values = list(range(x3_l, x3_h + 1))
    x4_values = list(range(x4_l, x4_h + 1))
    x5_values = list(range(x5_l, x5_h + 1))
    x6_values = list(range(x6_l, x6_h + 1))
    x7_values = list(range(x7_l, x7_h + 1))
    x8_values = list(range(x8_l, x8_h + 1))
    x9_values = list(range(x9_l, x9_h + 1))
    x10_values = list(range(x10_l, x10_h + 1))
    mapping = {1: x1_values, 2: x2_values, 3: x3_values, 4: x4_values,
               5: x5_values, 6: x6_values, 7: x7_values, 8: x8_values,
               9: x9_values, 10: x10_values}
    is_pruning_needed = False
    first_node = [INVALID for _ in range(11)]
    first_node[0] = 0
    stack = [(first_node, 0)]
    # record number of complete variable assignments visited
    count = 0
    while len(stack):
        cur_node, child_index = stack.pop()
        # if child_index == 0:
        #     count += 1
        if cur_node != first_node:
            x1 = x1_values[cur_node[1]] if cur_node[1] != INVALID else 0
            x2 = x2_values[cur_node[2]] if cur_node[2] != INVALID else 0
            x3 = x3_values[cur_node[3]] if cur_node[3] != INVALID else 0
            x4 = x4_values[cur_node[4]] if cur_node[4] != INVALID else 0
            x5 = x5_values[cur_node[5]] if cur_node[5] != INVALID else 0
            x6 = x6_values[cur_node[6]] if cur_node[6] != INVALID else 0
            x7 = x7_values[cur_node[7]] if cur_node[7] != INVALID else 0
            x8 = x8_values[cur_node[8]] if cur_node[8] != INVALID else 0
            x9 = x9_values[cur_node[9]] if cur_node[9] != INVALID else 0
            x10 = x10_values[cur_node[10]] if cur_node[10] != INVALID else 0
            # check if the subtree below should get pruned
            if constraint_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) <= 0:
                x1 = x1_values[cur_node[1]] if cur_node[1] != INVALID else INVALID
                x2 = x2_values[cur_node[2]] if cur_node[2] != INVALID else INVALID
                x3 = x3_values[cur_node[3]] if cur_node[3] != INVALID else INVALID
                x4 = x4_values[cur_node[4]] if cur_node[4] != INVALID else INVALID
                x5 = x5_values[cur_node[5]] if cur_node[5] != INVALID else INVALID
                x6 = x6_values[cur_node[6]] if cur_node[6] != INVALID else INVALID
                x7 = x7_values[cur_node[7]] if cur_node[7] != INVALID else INVALID
                x8 = x8_values[cur_node[8]] if cur_node[8] != INVALID else INVALID
                x9 = x9_values[cur_node[9]] if cur_node[9] != INVALID else INVALID
                x10 = x10_values[cur_node[10]] if cur_node[10] != INVALID else INVALID
                x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = get_best_x_when_all_real(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
                cur_f_value = objective_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
                if cur_f_value < lower_bound:
                    is_pruning_needed = True
            else:
                is_pruning_needed = True
        if not is_pruning_needed:
            if INVALID in cur_node:
                x_index = cur_node.index(INVALID)
            else:
                x1 = x1_values[cur_node[1]] if cur_node[1] != INVALID else INVALID
                x2 = x2_values[cur_node[2]] if cur_node[2] != INVALID else INVALID
                x3 = x3_values[cur_node[3]] if cur_node[3] != INVALID else INVALID
                x4 = x4_values[cur_node[4]] if cur_node[4] != INVALID else INVALID
                x5 = x5_values[cur_node[5]] if cur_node[5] != INVALID else INVALID
                x6 = x6_values[cur_node[6]] if cur_node[6] != INVALID else INVALID
                x7 = x7_values[cur_node[7]] if cur_node[7] != INVALID else INVALID
                x8 = x8_values[cur_node[8]] if cur_node[8] != INVALID else INVALID
                x9 = x9_values[cur_node[9]] if cur_node[9] != INVALID else INVALID
                x10 = x10_values[cur_node[10]] if cur_node[10] != INVALID else INVALID
                final_f = objective_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
                count += 1
                if upper_bound > final_f > lower_bound:
                    lower_bound = final_f
                if upper_bound > final_f > max_f:
                    max_f = final_f
                    x1_b, x2_b, x3_b, x4_b, x5_b, x6_b, x7_b, x8_b, x9_b, x10_b = \
                        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10
                continue
            if child_index < len(mapping[x_index]):
                stack.append((copy.deepcopy(cur_node), child_index + 1))
                child_node = copy.deepcopy(cur_node)
                child_node[x_index] = child_index
                stack.append((child_node, 0))
        is_pruning_needed = False
    return objective_function(x1_b, x2_b, x3_b, x4_b, x5_b, x6_b, x7_b, x8_b, x9_b, x10_b), \
           (x1_b, x2_b, x3_b, x4_b, x5_b, x6_b, x7_b, x8_b, x9_b, x10_b), count

