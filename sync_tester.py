import numpy as np
from qutip import *
from math import sqrt
from math import factorial
import scipy.sparse
from numba import jit, njit


#Not all functions are used here. Each scipt generally contains all functions used to obtain all results.
def spin_variance_operator(spin, num_spins):
    """
    Constructs a Qobj that corresponds to the operator R is defined in in text equation used in Eq. 3 and can be equivalently defined as
    R = S(S+1) - (1/N^2)(S_tot)^2.

    :param spin: The spin quantum number
    :param num_spins: The number of spins in the system
    :return: A qobj that corresponds to the operator R
    """
    dimensions = int(2 * spin + 1)
    raising_op_single = spin_Jp(spin)
    z_op_single = spin_Jz(spin)

    # First set operator R equal to S(S+1)
    variance_op = spin * (spin + 1) * (num_spins - 1) * qeye([dimensions] * num_spins) / num_spins

    # The followinging two loops will subtract (1/N^2)(S_tot)^2 from the initial value of R above. The double loop
    # results from expanding (S_tot)^2 = ||\sum_n=1^N \vec{S}_n ||^2 in terms of individual spin operators.
    for i in range(num_spins):
        current_raising_op_left = tensor(
            [qeye(dimensions)] * i + [raising_op_single] + [qeye(dimensions)] * (num_spins - i - 1))
        current_z_op_left = tensor([qeye(dimensions)] * i + [z_op_single] + [qeye(dimensions)] * (num_spins - i - 1))
        for j in range(i + 1, num_spins):
            current_raising_op_right = tensor(
                [qeye(dimensions)] * j + [raising_op_single] + [qeye(dimensions)] * (num_spins - j - 1))
            current_z_op_right = tensor(
                [qeye(dimensions)] * j + [z_op_single] + [qeye(dimensions)] * (num_spins - j - 1))
            phase_coherence = current_raising_op_left * current_raising_op_right.dag()
            variance_op -= 2 * (
                        0.5 * (phase_coherence + phase_coherence.dag()) + current_z_op_left * current_z_op_right) / (
                                       num_spins * num_spins)
    return variance_op


def negativity(rho, size):
    """
    This is the function to calculate the negativity of a density matrix, which is used for entanglement qualification.
    :param rho: the density matrix of 2 systems at any dimensions.
    :param size: the size of 2 systems in the form: for a m dimension and n dimension system, the size is [[m, n], [m, n]].
    :return: the negativity of the density matrix.
    """
    m = size[0][0]
    n = size[0][1]
    partial_transposed_rho = np.zeros((m * n, m * n), dtype=np.complex128)

    # get the partially transposed density matrix of the original 2-body density matrix
    for i in range(0, m):
        for j in range(0, m):
            for k in range(0, n):
                for l in range(0, n):
                    partial_transposed_rho[n * i + k][n * j + l] = rho[n * i + l][n * j + k]

    # find the eigenvalue of the partial transposed density matrix
    d, f = np.linalg.eig(partial_transposed_rho)
    return np.real((np.sum(np.abs(d)) - np.sum(d)) / 2)

def to_decimal(combo):
    first_one_pos = 0
    not_found = combo[first_one_pos] == 0
    while not_found:
        first_one_pos += 1
        not_found = combo[first_one_pos] == 0
    max_power = len(combo) - first_one_pos - 1
    val = np.longlong(0)
    for i in range(len(combo) - first_one_pos):
        val += combo[i + first_one_pos] * (2 ** (max_power - i))
    return val


def to_binary(dec_val, string_len=None):
    power_list = []
    val_total = 0
    while val_total != dec_val:
        curr_power = 0
        curr_val = dec_val - val_total
        while curr_val > 1:
            curr_val //= 2
            curr_power += 1
        power_list.append(curr_power)
        val_total += 2 ** curr_power
    if val_total == 0:
        bin_list = [0]
    else:
        bin_list = [0] * (power_list[0] + 1)
    for power in power_list:
        bin_list[power] = 1
    if string_len is not None and string_len > len(bin_list):
        bin_list = bin_list + [0] * (string_len - len(bin_list))
    return bin_list[::-1]


def iterate_combo(combo):
    num_ones = 0
    counter = 0
    for digit in combo:
        if digit == 1:
            num_ones += 1
    dec_combo = to_decimal(combo)
    not_found = True
    while not_found:
        dec_combo += np.longlong(1)
        bin_combo = to_binary(dec_combo)
        counter = 0
        for digit in bin_combo:
            if digit == 1:
                counter += 1
        not_found = counter != num_ones
    new_combo = to_binary(dec_combo)
    if len(new_combo) > len(combo):
        return [-1]
    else:
        new_combo = to_binary(dec_combo, string_len=len(combo))
        return new_combo


def multi_clebsch(uncoupled_spin_numbers, uncoupled_ms_numbers, coupled_spin_numbers, coupled_M):
    if len(coupled_spin_numbers) == 1:
        return clebsch(uncoupled_spin_numbers[0], uncoupled_spin_numbers[1], coupled_spin_numbers[0],
                       uncoupled_ms_numbers[0], uncoupled_ms_numbers[1], coupled_M)
    else:
        multi_cleb = 0
        dims = int(2 * coupled_spin_numbers[-2] + 1)
        for i in range(dims):
            ms_curr = i - coupled_spin_numbers[-2]
            multi_cleb += multi_clebsch(uncoupled_spin_numbers[:-1], uncoupled_ms_numbers[:-1],
                                        coupled_spin_numbers[:-1], ms_curr) * clebsch(coupled_spin_numbers[-2],
                                                                                      uncoupled_spin_numbers[-1],
                                                                                      coupled_spin_numbers[-1], ms_curr,
                                                                                      uncoupled_ms_numbers[-1],
                                                                                      coupled_M)
        return multi_cleb


# Assumes all spins are half
def check_if_zero_state_exists(num_spins, coupled_spin_numbers):
    ms_binary_list = [0] * (num_spins // 2) + [1] * (num_spins // 2)
    while ms_binary_list[0] != -1:
        uncoupled_ms_numbers = np.array(ms_binary_list) - 0.5
        if not np.isclose(multi_clebsch([0.5] * num_spins, uncoupled_ms_numbers, coupled_spin_numbers, 0), 0):
            return True
        iterate_combo(ms_binary_list)
    return False


# Assumes that all spins are half and num_spins is >=2
# Finds all coupled basis states that have a total coupled spin quantum number coupled_spin_number and returns the degenerate quantum numbers.
def get_all_coupled_spin_combos(num_spins, coupled_spin_number):
    valid_combos = []
    coupled_number_bounds = []
    bound = 1
    for i in range(num_spins - 1):
        coupled_number_bounds.append(bound)
        bound += 0.5
    curr_str = [0] * (num_spins - 1)
    while curr_str != -1:
        is_valid = curr_str[-1] == coupled_spin_number + 0.5 or curr_str[-1] == coupled_spin_number - 0.5
        counter = 2
        while is_valid and counter <= len(curr_str):
            is_valid = curr_str[-counter] == curr_str[-counter + 1] + 0.5 or curr_str[-counter] == curr_str[
                -counter + 1] - 0.5
            counter += 1
        if is_valid:
            valid_combos.append(curr_str[1:] + [coupled_spin_number])
        curr_str = iterate_spin_numbers(curr_str, coupled_number_bounds)
    return valid_combos


def iterate_spin_numbers(spin_numbers, coupled_number_bounds):
    max_val = convert_spin_str_to_num(coupled_number_bounds, coupled_number_bounds)
    curr_num = convert_spin_str_to_num(spin_numbers, coupled_number_bounds) + 1
    if curr_num > max_val:
        return -1
    else:
        return convert_num_to_spin_str(curr_num, coupled_number_bounds)


def convert_spin_str_to_num(spin_numbers, coupled_number_bounds):
    num_spins = len(spin_numbers)
    curr_num = 0
    curr_dims = 1
    for i in range(num_spins):
        curr_num += int(2 * spin_numbers[num_spins - 1 - i]) * curr_dims
        curr_dims *= int(2 * coupled_number_bounds[num_spins - 1 - i] + 1)
    return curr_num


def convert_num_to_spin_str(num, coupled_number_bounds):
    dim_list = np.array(2 * np.array(coupled_number_bounds) + 1, dtype=int)
    curr_dims = np.prod(dim_list) // dim_list[0]
    ind_list = [num // curr_dims]
    curr_num = num - (curr_dims * ind_list[0])
    ind_list[0] /= 2
    for k in range(1, len(dim_list)):
        curr_dims //= dim_list[k]
        ind_list.append(curr_num // curr_dims)
        curr_num -= ind_list[k] * curr_dims
        ind_list[k] /= 2
    return ind_list


def get_target_state_spin_half(num_spins):
    dims = 2
    state = Qobj(np.zeros((dims ** num_spins)), dims=[[dims] * num_spins, [1] * num_spins])
    if num_spins % 2 == 0:
        ms_inds = np.array([-0.5] * (num_spins // 2) + [0.5] * (num_spins // 2))
    else:
        ms_inds = np.array([-0.5] * (num_spins // 2) + [0.5] * (num_spins // 2 + 1))
    combo = [1]
    for i in range(num_spins - 2):
        combo.append(combo[-1] + 0.5)
    while ms_inds[0] != -1:
        if num_spins % 2 == 0:
            clebsch_coeff = multi_clebsch([0.5] * num_spins, ms_inds, combo, 0)
        else:
            clebsch_coeff = multi_clebsch([0.5] * num_spins, ms_inds, combo, 0.5)
        curr_term = Qobj(clebsch_coeff)
        for i in range(num_spins):
            if ms_inds[i] == -0.5:
                curr_term = tensor(curr_term, basis(2, 0))
            else:
                curr_term = tensor(curr_term, basis(2, 1))
        curr_term.dims = state.dims
        state += curr_term
        ms_inds = iterate_combo(np.array(ms_inds + 0.5, dtype=int))
        if ms_inds[0] != -1:
            ms_inds = np.array(ms_inds) - 0.5
    return state


def symm_lc_sync_upper_manifold(num_spins, omega_list, gamma_g, gamma_d):
    spin = 0.5
    Sz = spin_Jz(spin)
    Splus = spin_Jp(spin)
    dims = int(2 * spin + 1)
    Jplus = Qobj(np.zeros((dims ** num_spins, dims ** num_spins)), dims=[[dims] * num_spins, [dims] * num_spins])
    Jz = Qobj(np.zeros((dims ** num_spins, dims ** num_spins)), dims=[[dims] * num_spins, [dims] * num_spins])
    ham = Qobj(np.zeros((dims ** num_spins, dims ** num_spins)), dims=[[dims] * num_spins, [dims] * num_spins])
    c_ops = []
    Sp_list = []
    for i in range(num_spins):
        ham += omega_list[i] * tensor([qeye(dims)] * i + [Sz] + [qeye(dims)] * (num_spins - 1 - i))
        Jplus += tensor([qeye(dims)] * i + [Splus] + [qeye(dims)] * (num_spins - 1 - i))
        Jz += tensor([qeye(dims)] * i + [Sz] + [qeye(dims)] * (num_spins - 1 - i))
        Sp_list.append(tensor([qeye(dims)] * i + [Splus] + [qeye(dims)] * (num_spins - 1 - i)))
    if num_spins % 2 == 0:
        spin_range = np.array(range(int(num_spins * spin)))
    else:
        spin_range = np.array(range(int(num_spins * spin))) + 0.5
    for curr_spin in spin_range:
        combo_list = get_all_coupled_spin_combos(num_spins, curr_spin)
        # print(combo_list)
        for combo in combo_list:
            c_op = Qobj(np.zeros((dims ** num_spins, dims ** num_spins)), dims=[[dims] * num_spins, [dims] * num_spins])
            if num_spins % 2 == 0:
                ms_inds = np.array([-0.5] * (num_spins // 2) + [0.5] * (num_spins // 2))
            else:
                # ms_inds = np.array([-0.5]*(num_spins//2) + [0.5]*(num_spins//2 + 1))
                ms_inds = np.array([-0.5] * (num_spins // 2 + 1) + [0.5] * (num_spins // 2))
            # We are only interested in the total M=0 or M=0.5 for an odd number of spins
            while ms_inds[0] != -1:
                # print(ms_inds)
                if num_spins % 2 == 0:
                    clebsch_coeff = multi_clebsch([0.5] * num_spins, ms_inds, combo, 0)
                else:
                    clebsch_coeff = multi_clebsch([0.5] * num_spins, ms_inds, combo, -0.5)
                if clebsch_coeff != 0:
                    curr_term = clebsch_coeff * tensor([qeye(2)] * num_spins)
                    for i in range(num_spins):
                        if ms_inds[i] == 0.5:
                            curr_term *= Sp_list[i]
                    c_op += curr_term
                ms_inds = iterate_combo(np.array(ms_inds + 0.5, dtype=int))
                if ms_inds[0] != -1:
                    ms_inds = np.array(ms_inds) - 0.5
            c_ops.append(c_op)
    if num_spins % 2 == 0:
        c_ops.append(np.sqrt(gamma_g) * Jplus * Jz)
        c_ops.append(np.sqrt(gamma_d) * Jplus.dag() * Jz)
    else:
        c_ops.append(np.sqrt(gamma_g) * Jplus * (Jz - 0.5 * tensor([qeye(2)] * num_spins)))
        c_ops.append(np.sqrt(gamma_d) * Jplus.dag() * (Jz - 0.5 * tensor([qeye(2)] * num_spins)))
    return ham, c_ops


def get_coupled_basis_state(spin, num_spins, J, M, d):
    dims = int(2 * spin + 1)
    combo = get_all_coupled_spin_combos(num_spins, J)[d]
    state = np.zeros((dims ** num_spins, 1))
    total_dims = dims ** num_spins
    for i in range(total_dims):
        ms_inds = 2 * np.array(convert_num_to_spin_str(i, [spin] * num_spins)) - 0.5
        state[total_dims - i - 1] += multi_clebsch([spin] * num_spins, ms_inds, combo, M)
    return Qobj(state, dims=[[dims] * num_spins, [1] * num_spins])


def get_rho_subspace(rho, J1, d1, J2, d2):
    num_spins = len(rho.dims[0])
    dims = rho.dims[0][0]
    spin = 0.5 * (rho.dims[0][0] - 1)

    num_rows = int(2 * J1 + 1)
    num_cols = int(2 * J2 + 1)
    rho_sub = np.zeros((num_rows, num_cols), dtype=np.complex128)

    combo1 = get_all_coupled_spin_combos(num_spins, J1)[d1]
    combo2 = get_all_coupled_spin_combos(num_spins, J2)[d2]
    for i in range(num_rows):
        M1 = i - J1
        for j in range(dims ** num_spins):
            ms_ind1 = 2 * np.array(convert_num_to_spin_str(j, [spin] * num_spins)) - 0.5
            # print(ms_ind1)
            for k in range(num_cols):
                M2 = k - J2
                for l in range(dims ** num_spins):
                    ms_ind2 = 2 * np.array(convert_num_to_spin_str(l, [spin] * num_spins)) - 0.5
                    # print(ms_ind2)
                    rho_sub[i, k] += rho[j, l] * multi_clebsch([spin] * num_spins, ms_ind1, combo1, M1) * multi_clebsch(
                        [spin] * num_spins, ms_ind2, combo2, M2)
                    # print(multi_clebsch([spin]*num_spins, ms_ind1, combo1, M1))
    return rho_sub


def spin_q_func(rho, theta_vals, phi_vals):
    """
    Returns an ndarray with values of the Husimi Q distribution correspoinding to the density operator of a single spin
    using the wigner D matrix to calculate its entries. This function can be slightly more efficient than using the q_func from
    get_spin_q_func to calculate such an array.

    input:
        rho - Qobj representing a single spin state
        theta_vals - An array containing the values of theta at which the q function should be evaluated
        phi_vals - an array containing the values of phi at which the q function should be evaluated
    returns:
        q_func - An ndarray array with dimension len(theta_vals) x len(phi_vals) containing the corresponding q function values
    """

    dims = rho.dims[0][0]
    matrix_elements = np.zeros((dims, dims))
    for i in range(dims):
        d_mi = sqrt(factorial(dims - 1) / (factorial(dims - i - 1) * factorial(i)))
        for j in range(dims):
            matrix_elements[i, j] = d_mi * sqrt(factorial(dims - 1) / (factorial(dims - j - 1) * factorial(j)))
            matrix_elements[j, i] = matrix_elements[i, j]
        matrix_elements[i, i] = d_mi * d_mi
    q_func = np.zeros((len(theta_vals), len(phi_vals)))
    for n in range(len(theta_vals)):
        theta = theta_vals[n]
        for m in range(len(phi_vals)):
            phi = phi_vals[m]
            for i in range(dims):
                d_mi = (np.cos(0.5 * theta) ** (dims - i - 1)) * (np.sin(0.5 * theta) ** i)
                for j in range(i + 1, dims):
                    q_func[n, m] += 2 * np.real(rho[i, j] * np.exp(1j * (j - i) * phi)) * d_mi * (
                                np.cos(0.5 * theta) ** (dims - j - 1)) * (np.sin(0.5 * theta) ** j) * matrix_elements[
                                        i, j]
                q_func[n, m] += np.real(rho[i, i]) * d_mi * d_mi * matrix_elements[i, i]
            q_func[n, m] *= dims / (4 * np.pi)
    return q_func


def spin_q_func_diag(rho, theta_vals, phi_vals):
    """
    Returns an ndarray with values of the Husimi Q distribution correspoinding to the density operator of a single spin
    using the wigner D matrix to calculate its entries. This function can be slightly more efficient than using the q_func from
    get_spin_q_func to calculate such an array.

    input:
        rho - Qobj representing a single spin state
        theta_vals - An array containing the values of theta at which the q function should be evaluated
        phi_vals - an array containing the values of phi at which the q function should be evaluated
    returns:
        q_func - An ndarray array with dimension len(theta_vals) x len(phi_vals) containing the corresponding q function values
    """

    dims = rho.dims[0][0]
    matrix_elements = np.zeros((dims, dims))
    for i in range(dims):
        d_mi = sqrt(factorial(dims - 1) / (factorial(dims - i - 1) * factorial(i)))
        for j in range(dims):
            matrix_elements[i, j] = d_mi * sqrt(factorial(dims - 1) / (factorial(dims - j - 1) * factorial(j)))
            matrix_elements[j, i] = matrix_elements[i, j]
        matrix_elements[i, i] = d_mi * d_mi
    q_func = np.zeros((len(theta_vals), len(phi_vals)))
    for n in range(len(theta_vals)):
        theta = theta_vals[n]
        val = 0
        for i in range(dims):
            d_mi = (np.cos(0.5 * theta) ** (dims - i - 1)) * (np.sin(0.5 * theta) ** i)
            val += np.real(rho[i, i]) * d_mi * d_mi * matrix_elements[i, i]
        val *= dims / (4 * np.pi)
        for m in range(len(phi_vals)):
            q_func[n, m] = val
    return q_func


def relative_phase_distribution(rho, subsys_a, subsys_b, phi_dist):
    """
    Returns an nd.array of real double values whose values are the relative phase distribtuion S(phi) as defined in https://arxiv.org/abs/2208.12766

    input:
    rho - A Qobj that represents a system of n coupled spins.
    subsys_a - index of the first subsystem with which will be measured by the relative phase distribution
    subsys_b - index of the second subsystem which will be measured by the relative phase distribution
    stepsize - The stepsize of phi in the relative phase distribution

    returns:
    An ndarray of doubles whose values correspond to S(phi) where phi ranges from -pi to pi
    """

    rho_reduced = rho.ptrace([subsys_a, subsys_b])
    # print(rho_reduced[0,0])
    dims = rho_reduced.dims[0][0]
    spin = 0.5 * (rho.dims[0][0] - 1)
    phase_dist = np.zeros((len(phi_dist)))
    matrix_elements = np.zeros((dims, dims))
    for i in range(dims):
        for j in range(dims):
            if i == j:
                matrix_elements[i, j] = 0
            else:
                matrix_elements[i, j] = ((2 * spin + 1) / (4 * np.pi)) * ((2 * factorial(dims - 1) * gamma(
                    1 + int(2 * spin) - 0.5 * (i + j)) * gamma(1 + 0.5 * (i + j))) /
                                                                          (sqrt(factorial(dims - 1 - i) * factorial(
                                                                              i) * factorial(dims - 1 - j) * factorial(
                                                                              j)) * factorial(dims)))
    # Now calculate the phi distribution
    for i in range(len(phi_dist)):
        phi_current = phi_dist[i]
        # print(phi_current)
        for m_a in range(dims):
            for n_a in range(m_a + 1, dims):
                for m_b in range(dims + m_a - n_a):
                    n_b = n_a - m_a + m_b
                    phase_dist[i] += (4 * np.pi * np.real(
                        np.exp(1j * (n_a - m_a) * phi_current) * rho_reduced[m_a * dims + n_b, n_a * dims + m_b]) *
                                      matrix_elements[m_a, n_a] * matrix_elements[m_b, n_b])
    return phase_dist


def get_arc_length_distribution(rho, len_vals):
    total_dims = rho.shape[0]
    dim_a = rho.dims[0][0]
    dim_b = rho.dims[0][1]
    num_spins = 2
    # data = rho.data.as_scipy().tocoo().data
    data = scipy.sparse.coo_array(rho.full()).data
    # rows, cols = rho.data.as_scipy().tocoo().coords
    rows, cols = scipy.sparse.coo_array(rho.full()).coords

    matrix_elements_a = np.zeros((dim_a, dim_a))
    matrix_elements_b = np.zeros((dim_b, dim_b))
    for i in range(dim_a):
        d_mi_a = sqrt(factorial(dim_a - 1) / (factorial(dim_a - i - 1) * factorial(i)))
        for j in range(dim_a):
            matrix_elements_a[i, j] = d_mi_a * sqrt(factorial(dim_a - 1) / (factorial(dim_a - j - 1) * factorial(j)))
            matrix_elements_a[j, i] = matrix_elements_a[i, j]
        matrix_elements_a[i, i] = d_mi_a * d_mi_a
    for i in range(dim_b):
        d_mi_b = sqrt(factorial(dim_b - 1) / (factorial(dim_b - i - 1) * factorial(i)))
        for j in range(dim_b):
            matrix_elements_b[i, j] = d_mi_b * sqrt(factorial(dim_b - 1) / (factorial(dim_b - j - 1) * factorial(j)))
            matrix_elements_b[j, i] = matrix_elements_b[i, j]
        matrix_elements_b[i, i] = d_mi_b * d_mi_b
    total_dims = dim_a * dim_b
    upper_triangle_inds = []
    row_inds = []
    col_inds = []
    for l in range(len(data)):
        if rows[l] <= cols[l]:
            current_dims = total_dims // dim_a
            row_ind_list = [rows[l] // current_dims, -1]
            col_ind_list = [cols[l] // current_dims, -1]
            current_row_ind = rows[l] - (current_dims * row_ind_list[0])
            current_col_ind = cols[l] - (current_dims * col_ind_list[0])
            current_dims //= dim_b
            row_ind_list[1] = current_row_ind // current_dims
            col_ind_list[1] = current_col_ind // current_dims
            # current_row_ind -= (row_ind_list[1]*current_dims)
            # current_col_ind -= (col_ind_list[1]*current_dims)
            upper_triangle_inds.append(l)
            row_inds.append(row_ind_list)
            col_inds.append(col_ind_list)
    upper_triangle_inds = np.array(upper_triangle_inds, dtype=np.int_)
    row_inds = np.array(row_inds, dtype=np.int_)
    col_inds = np.array(col_inds, dtype=np.int_)
    # return get_arc_length_distribution_numba(data, rows, cols, num_spins, dim_a, dim_b, matrix_elements_a, matrix_elements_b)
    return get_arc_length_distribution_numba(len_vals, data, rows, cols, upper_triangle_inds, row_inds, col_inds,
                                             num_spins, dim_a, dim_b, matrix_elements_a, matrix_elements_b)


@njit
def get_arc_length_distribution_numba(len_vals, data, rows, cols, upper_triangle_inds, row_inds, col_inds, num_spins,
                                      dim_a, dim_b, matrix_elements_a, matrix_elements_b):
    # print("Starting")
    total_dims = dim_a * dim_b
    step = 5e-2
    step_prod = step * step * step
    theta_vals = np.arange(0, np.pi, step)
    phi_vals = np.arange(0, 2 * np.pi, step)
    v_phi_vals = np.arange(0, 2 * np.pi, step)
    # q_func = np.zeros([len(theta_vals)]*num_spins + [len(phi_vals)]*num_spins)
    distribution = np.zeros(len(len_vals))
    len_len = len(len_vals)
    len_theta = len(theta_vals)
    len_phi = len(phi_vals)
    len_v_phi = len(v_phi_vals)
    len_triangle = len(upper_triangle_inds)
    for i in range(len_len):
        # print(100*i/len_len)
        len_val = len_vals[i]
        # for theta in theta_vals:
        for j in range(len_theta):
            theta = theta_vals[j]
            vol_element_a = np.sin(theta) * step_prod
            # for phi in phi_vals:
            for k in range(len_phi):
                phi = phi_vals[k]
                # for v_phi in v_phi_vals:
                for m in range(len_v_phi):
                    v_phi = v_phi_vals[m]
                    distribution[i] += dist_lower_level(len_val, theta, phi, v_phi, vol_element_a, step, data, rows,
                                                        cols, upper_triangle_inds, row_inds, col_inds, num_spins, dim_a,
                                                        dim_b, matrix_elements_a, matrix_elements_b)
    return distribution


@njit(cache=True)
def dist_lower_level(len_val, theta, phi, v_phi, vol_element_a, step, data, rows, cols, upper_triangle_inds, row_inds,
                     col_inds, num_spins, dim_a, dim_b, matrix_elements_a, matrix_elements_b):
    total_dims = dim_a * dim_b
    x = np.cos(theta) * np.cos(phi) * np.sin(len_val) * np.cos(v_phi) + np.sin(theta) * np.cos(phi) * np.cos(
        len_val) - np.sin(phi) * np.sin(len_val) * np.sin(v_phi)
    y = np.cos(theta) * np.sin(phi) * np.sin(len_val) * np.cos(v_phi) + np.sin(theta) * np.sin(phi) * np.cos(
        len_val) + np.cos(phi) * np.sin(len_val) * np.sin(v_phi)
    z = -np.sin(theta) * np.sin(len_val) * np.cos(v_phi) + np.cos(theta) * np.cos(len_val)

    theta_b = np.arccos(z)
    phi_b = np.arctan2(y, x)
    val = 0
    for l in range(len(upper_triangle_inds)):
        row = rows[upper_triangle_inds[l]]
        col = cols[upper_triangle_inds[l]]
        if row < col:
            curr_element = 2 * data[upper_triangle_inds[l]]
            d_mi_a = (np.cos(0.5 * theta) ** (row_inds[l][0])) * (np.sin(0.5 * theta) ** (dim_a - row_inds[l][0] - 1))
            d_mi_b = (np.cos(0.5 * theta_b) ** (row_inds[l][1])) * (
                        np.sin(0.5 * theta_b) ** (dim_b - row_inds[l][1] - 1))
            curr_element *= np.exp(1j * (col_inds[l][0] - row_inds[l][0]) * phi) * d_mi_a * (
                        np.cos(0.5 * theta) ** (col_inds[l][0])) * (
                                        np.sin(0.5 * theta) ** (dim_a - col_inds[l][0] - 1)) * matrix_elements_a[
                                row_inds[l][0], col_inds[l][0]]
            curr_element *= np.exp(1j * (col_inds[l][1] - row_inds[l][1]) * phi_b) * d_mi_b * (
                        np.cos(0.5 * theta_b) ** (col_inds[l][1])) * (
                                        np.sin(0.5 * theta_b) ** (dim_b - col_inds[l][1] - 1)) * matrix_elements_b[
                                row_inds[l][1], col_inds[l][1]]
            val += (np.real(curr_element) * total_dims / (16 * np.pi * np.pi)) * vol_element_a
        else:
            curr_element = np.real(data[upper_triangle_inds[l]])
            d_mi_a = (np.cos(0.5 * theta) ** (row_inds[l][0])) * (np.sin(0.5 * theta) ** (dim_a - row_inds[l][0] - 1))
            d_mi_b = (np.cos(0.5 * theta_b) ** (row_inds[l][1])) * (
                        np.sin(0.5 * theta_b) ** (dim_b - row_inds[l][1] - 1))
            curr_element *= d_mi_a * d_mi_a * matrix_elements_a[row_inds[l][0], col_inds[l][0]]
            curr_element *= d_mi_b * d_mi_b * matrix_elements_b[row_inds[l][1], col_inds[l][1]]
            val += (curr_element * total_dims / (16 * np.pi * np.pi)) * vol_element_a
    return val


# This script is fairly basic. Its mainly meant to be an easily accesible way to test the results in the paper
# One should test this across a variety of num_spins, gamma_g, gamma_d, and omegas to verify claims.
if __name__ == '__main__':
    # Inputs
    num_spins = 4
    spin = 0.5
    omegas = [1] * num_spins
    gamma_g = 1
    gamma_d = 10

    # Construction of the Hamiltonian and collapse operators
    ham, c_ops = symm_lc_sync_upper_manifold(num_spins, omegas, gamma_g, gamma_d)
    # Steady state solver
    rho_ss = steadystate(ham, c_ops=c_ops)

    # Obtain the measure r from Eq. 3 in the main text
    r_val = (np.real((rho_ss * spin_variance_operator(spin, num_spins)).tr()) - (
                (num_spins - 1) * spin / num_spins)) / (spin * (spin + 1) - ((num_spins - 1) * spin) / num_spins)
    print("Synchonization measure: r = {}".format(r_val))

    # Obtain the negativity of the first spin with resepect to the remaining spins
    neg = negativity(rho_ss, [[2, 2 ** (num_spins - 1)], [2, 2 ** (num_spins - 1)]])
    print("Negativity: N = {}".format(neg))

    # Print the fidelty of the target synchronous state with respect to |NS, K><NS, K|
    target = get_target_state_spin_half(num_spins)
    target = tensor(target, target.dag())
    target.dims = rho_ss.dims
    print("Fidelty between target and rho_ss: F = {}".format(fidelity(rho_ss, target)))