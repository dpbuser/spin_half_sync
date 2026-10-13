import numpy as np
import matplotlib.pyplot as plt
from qutip import *

#Helper functions. Not all will be used.

def all2all_adjacency_matrix(N):
    graph = np.ones((N, N)) - np.diag(np.ones(N))
    return graph

def anti_symmetric_ring_matrix(N):
    graph = np.zeros((N, N))
    graph[0, -1] = 1
    graph[-1, 0] = -1
    for i in range(N - 1):
        j = i + 1
        graph[i, j] = -1.0
        graph[j, i] = 1.0
    return graph

def symmetric_ring_matrix(N):
    graph = np.zeros((N, N))
    graph[0, -1] = 1
    graph[-1, 0] = 1
    for i in range(N - 1):
        j = i + 1
        graph[i, j] = 1.0
        graph[j, i] = 1.0
    return graph

def directed_ring_matrix(N):
    graph = np.zeros((N, N))
    graph[0, -1] = 1
    graph[-1, 0] = 0
    for i in range(N - 1):
        j = i + 1
        graph[i, j] = 0
        graph[j, i] = 1
    return graph

def spin_variance_operator(spin, num_spins):
    """
    Constructs a Qobj that corresponds to the operator R is defined in Eq. S9 and can be equivalently defined as
    R = S(S+1) - (1/N^2)(S_tot)^2.

    :param spin: The spin quantum number
    :param num_spins: The number of spins in the system
    :return: A qobj that corresponds to the operator R in Eq. 7 in the main article
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


# def negativity(rho, size):
#     """
#     This is the function to calculate the negativity of a density matrix, which is used for entanglement qualification.
#     :param rho: the density matrix of 2 systems at any dimensions.
#     :param size: the size of 2 systems in the form: for a m dimension and n dimension system, the size is [[m, m], [n, n]].
#     :return: the negativity of the density matrix.
#     """
#     m = size[0][0]
#     n = size[0][1]
#     partial_transposed_rho = np.zeros((m * n, m * n), dtype=np.complex128)

#     # get the partially transposed density matrix of the original 2-body density matrix
#     for i in range(0, m):
#         for j in range(0, m):
#             for k in range(0, n):
#                 for l in range(0, n):
#                     partial_transposed_rho[n * i + k][n * j + l] = rho[n * i + l][n * j + k]

#     # find the eigenvalue of the partial transposed density matrix
#     d, f = np.linalg.eig(partial_transposed_rho)
#     return np.real((np.sum(np.abs(d)) - np.sum(d)) / 2)

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

def symm_lc_sync_upper_manifold_sz_coupling_diss(num_spins, omega_list, gamma_g, gamma_d, adj_matrix):
    spin = 0.5
    Sz = spin_Jz(spin)
    Splus = spin_Jp(spin)
    dims = int(2*spin + 1)
    Jplus = Qobj(np.zeros((dims**num_spins, dims**num_spins)), dims=[[dims]*num_spins, [dims]*num_spins])
    Jz = Qobj(np.zeros((dims**num_spins, dims**num_spins)), dims=[[dims]*num_spins, [dims]*num_spins])
    ham = Qobj(np.zeros((dims**num_spins, dims**num_spins)), dims=[[dims]*num_spins, [dims]*num_spins])
    c_ops = []
    Sp_list = []
    for i in range(num_spins):
        ham += omega_list[i] * tensor([qeye(dims)]*i + [Sz] + [qeye(dims)]*(num_spins-1-i))
        Jplus += tensor([qeye(dims)]*i + [Splus] + [qeye(dims)]*(num_spins-1-i))
        Jz += tensor([qeye(dims)]*i + [Sz] + [qeye(dims)]*(num_spins-1-i))
        Sp_list.append(tensor([qeye(dims)]*i + [Splus] + [qeye(dims)]*(num_spins-1-i)))
    if num_spins % 2 == 0:
        spin_range = np.array(range(int(num_spins * spin)))
    else:
        spin_range = np.array(range(int(num_spins * spin))) + 0.5
    for curr_spin in spin_range:
        combo_list = get_all_coupled_spin_combos(num_spins, curr_spin)
        # print(combo_list)
        for combo in combo_list:
            c_op = Qobj(np.zeros((dims**num_spins, dims**num_spins)), dims=[[dims]*num_spins, [dims]*num_spins])
            if num_spins % 2 == 0:
                ms_inds = np.array([-0.5]*(num_spins//2) + [0.5]*(num_spins//2))
            else:
                # ms_inds = np.array([-0.5]*(num_spins//2) + [0.5]*(num_spins//2 + 1))
                ms_inds = np.array([-0.5]*(num_spins//2 + 1) + [0.5]*(num_spins//2))
            # We are only interested in the total M=0 or M=0.5 for an odd number of spins
            while ms_inds[0] != -1:
                # print(ms_inds)
                if num_spins % 2 == 0:
                    clebsch_coeff = multi_clebsch([0.5]*num_spins, ms_inds, combo, 0)
                else:
                    clebsch_coeff = multi_clebsch([0.5]*num_spins, ms_inds, combo, -0.5)
                if clebsch_coeff != 0:
                    curr_term = clebsch_coeff*tensor([qeye(2)]*num_spins)
                    for i in range(num_spins):
                        if ms_inds[i] == 0.5:
                            curr_term *= Sp_list[i]
                    c_op += curr_term
                ms_inds = iterate_combo(np.array(ms_inds+0.5, dtype=int))
                if ms_inds[0] != -1:
                    ms_inds = np.array(ms_inds) - 0.5
            c_ops.append(c_op)
    if num_spins %2 == 0:
        c_ops.append(np.sqrt(gamma_g) * Jplus*Jz)
        c_ops.append(np.sqrt(gamma_d) * Jplus.dag()*Jz)
    else:
        c_ops.append(np.sqrt(gamma_g) * Jplus*(Jz-0.5*tensor([qeye(2)]*num_spins)))
        c_ops.append(np.sqrt(gamma_d) * Jplus.dag()*(Jz-0.5*tensor([qeye(2)]*num_spins)))

    sz = spin_Jz(spin)
    adj_herm = 0.5*(adj_matrix + np.conjugate(adj_matrix.T))
    adj_anti = 0.5*(adj_matrix - np.conjugate(adj_matrix.T))
    # print(adj_anti)
    # print(adj_anti)
    if len(adj_matrix > 1):
        eig_vals, eig_vecs = np.linalg.eigh(np.real(-1j*adj_anti))
    for i in range(num_spins):
        sz_i = tensor([qeye(dims)] * i + [sz] + [qeye(dims)] * (num_spins - i - 1))
        ham += -adj_herm[i,i]*sz_i*sz_i
        for j in range(i+1,num_spins):
            sz_i_sz_j = sz_i * tensor([qeye(dims)] * j + [sz] + [qeye(dims)] * (num_spins - j - 1))
            ham += -(adj_herm[i,j] * sz_i_sz_j + np.conjugate(adj_herm[i,j])*sz_i_sz_j.dag())
        if(np.isclose(eig_vals[i], 0)):
            c_op_current = Qobj(np.zeros((dims ** num_spins, dims ** num_spins)),
                                dims=[[dims] * num_spins, [dims] * num_spins])
            if eig_vals[i]>0:
                for j in range(num_spins):
                    c_op_current += np.conjugate(eig_vecs[j,i]) * tensor(
                        [qeye(dims)] * j + [sz] + [qeye(dims)] * (num_spins - j - 1))
                c_ops.append(np.sqrt(2*eig_vals[i]) * c_op_current)
            else:
                for j in range(num_spins):
                    c_op_current += eig_vecs[j, i] * tensor(
                        [qeye(dims)] * j + [sz] + [qeye(dims)] * (num_spins - j - 1))
                c_ops.append(np.sqrt(-2*eig_vals[i]) * c_op_current)
    return ham, c_ops

if __name__ == "__main__":
    # First the case of all2all coupling will be considered.

    #Define system parameters. Note that omega and coupling_strength will be changed throughout the script.
    spin = 0.5
    num_spins = 4
    time_arr = np.arange(-100, 100, 0.01)
    omega = 2
    gamma_d = 1
    gamma_g = 1
    coupling_strength = 0
    adj_matrix = coupling_strength * all2all_adjacency_matrix(num_spins)
    ham, c_ops = symm_lc_sync_upper_manifold_sz_coupling_diss(num_spins, [omega] * num_spins, gamma_g, gamma_d,
                                                              adj_matrix)
    #Find the steady state
    rho_ss = steadystate(ham, c_ops)
    omega_vals_1 = [0] * num_spins
    w_vals_1 = []
    spec_vals_1 = []

    # Here we calculate the frequency spectrum for all spins. The argmax of each is then compared to ensure
    # each spin has the same observed frequency
    for curr_spin in range(num_spins):
        s_plus = tensor(
            [qeye(2 * spin + 1)] * (curr_spin) + [spin_Jp(spin)] + [qeye(2 * spin + 1)] * (num_spins - 1 - curr_spin))
        # This calculates the two time correlation function which is in the integrand of Eq. 10
        corr = correlation_2op_1t(ham, rho_ss, time_arr, c_ops, s_plus, s_plus.dag(), solver='es')
        # This takes the Fourier transform of the two time correlation function
        w_list, spec = spectrum_correlation_fft(time_arr, corr)
        w_vals_1.append(w_list)
        spec_vals_1.append(spec)
        omega_vals_1[curr_spin] = w_list[np.argmax(spec)]
    # Inspect the observed frequency of each spin.
    print(omega_vals_1)

    #Repeat the same procedure now with omega = 0.
    spin = 0.5
    num_spins = 4
    time_arr = np.arange(-100, 100, 0.01)
    omega = 0
    gamma_d = 1
    gamma_g = 1
    coupling_strength = 0
    adj_matrix = coupling_strength * all2all_adjacency_matrix(num_spins)
    ham, c_ops = symm_lc_sync_upper_manifold_sz_coupling_diss(num_spins, [omega] * num_spins, gamma_g, gamma_d,
                                                              adj_matrix)

    rho_ss = steadystate(ham, c_ops)
    omega_vals_2 = [0] * num_spins
    w_vals_2 = []
    spec_vals_2 = []

    for curr_spin in range(num_spins):
        s_plus = tensor(
            [qeye(2 * spin + 1)] * (curr_spin) + [spin_Jp(spin)] + [qeye(2 * spin + 1)] * (num_spins - 1 - curr_spin))
        # print(s_plus)
        corr = correlation_2op_1t(ham, rho_ss, time_arr, c_ops, s_plus, s_plus.dag(), solver='es')
        w_list, spec = spectrum_correlation_fft(time_arr, corr)
        w_vals_2.append(w_list)
        spec_vals_2.append(spec)
        omega_vals_2[curr_spin] = w_list[np.argmax(spec)]
    print(omega_vals_2)

    # Now coupling_strength (A) is nonzero.
    spin = 0.5
    num_spins = 4
    time_arr = np.arange(-100, 100, 0.01)
    omega = 0
    gamma_d = 1
    gamma_g = 1
    coupling_strength = 5
    adj_matrix = coupling_strength * all2all_adjacency_matrix(num_spins)
    ham, c_ops = symm_lc_sync_upper_manifold_sz_coupling_diss(num_spins, [omega] * num_spins, gamma_g, gamma_d,
                                                              adj_matrix)

    rho_ss = steadystate(ham, c_ops)
    omega_vals_3 = [0] * num_spins
    w_vals_3 = []
    spec_vals_3 = []

    for curr_spin in range(num_spins):
        s_plus = tensor(
            [qeye(2 * spin + 1)] * (curr_spin) + [spin_Jp(spin)] + [qeye(2 * spin + 1)] * (num_spins - 1 - curr_spin))
        corr = correlation_2op_1t(ham, rho_ss, time_arr, c_ops, s_plus, s_plus.dag(), solver='es')
        w_list, spec = spectrum_correlation_fft(time_arr, corr)
        w_vals_3.append(w_list)
        spec_vals_3.append(spec)
        omega_vals_3[curr_spin] = w_list[np.argmax(spec)]
    print(omega_vals_3)

    #Plot results
    scale = 3
    fontsize = 15
    ticksize = 15
    textsize = 15
    figsize = 5

    max_1 = np.max(spec_vals_1[0])
    max_2 = np.max(spec_vals_2[0])
    max_3 = np.max(spec_vals_3[0])

    fontsize = 20
    ax_ticks_scale = 1
    ax_label_scale = 1  # 1.5
    cfontsize = ax_ticks_scale * fontsize
    fig = plt.figure()
    curr_ax = fig.add_subplot(1,1,1)

    curr_ax.plot(w_vals_1[0], spec_vals_1[0], color="#ffcc00", label=r"$\omega=2, A=0$")
    curr_ax.plot(w_vals_2[0], spec_vals_2[0], color="#ff6600", ls=":", label=r"$\omega=0, A=0$")
    curr_ax.plot(w_vals_3[0], spec_vals_3[0], color="#ff0000", ls="--", label=r"$\omega=0, A=5$")

    peak1 = w_vals_1[0][np.argmax(spec_vals_1[0])]
    peak2 = w_vals_2[0][np.argmax(spec_vals_2[0])]
    peak3 = w_vals_3[0][np.argmax(spec_vals_3[0])]
    plt.plot(peak1, max_1, "*", color="#ffcc00", markersize=10, zorder=100)
    plt.plot(peak2, max_2, "*", color="#ff6600", markersize=10, zorder=100)
    plt.plot(peak3, max_3, "*", color="#ff0000", markersize=10, zorder=100)

    curr_ax.set_xlim(-2, 6)
    curr_ax.set_ylim(0, 0.45)
    curr_ax.set_xticks(list(plt.xticks()[0]))
    curr_ax.tick_params(labelsize=ticksize)
    curr_ax.set_title("Frequency Spectrum for Complete Coupling")
    curr_ax.set_xlabel(r"$\omega_p$", fontsize=fontsize)
    curr_ax.set_ylabel(r"$S_n(\omega_p)$", fontsize=fontsize)
    curr_ax.legend()

    # Save resulting figure
    plt.savefig("oscillation_revival_all2all.png", bbox_inches="tight")

############################################################################################################

# Now repeat the whole procedure above for a ring adjacency matrix
spin = 0.5
num_spins = 4
time_arr = np.arange(-100, 100, 0.01)
omega = 2
gamma_d = 1
gamma_g = 1
coupling_strength = 0
adj_matrix = coupling_strength*symmetric_ring_matrix(num_spins)

ham, c_ops = symm_lc_sync_upper_manifold_sz_coupling_diss(num_spins, [omega]*num_spins, gamma_g, gamma_d, adj_matrix)

rho_ss = steadystate(ham, c_ops)
omega_vals_1_ring = [0]*num_spins
w_vals_1_ring = []
spec_vals_1_ring = []

for curr_spin in range(num_spins):
    s_plus = tensor([qeye(2*spin+1)]*(curr_spin) + [spin_Jp(spin)] + [qeye(2*spin+1)]*(num_spins-1-curr_spin))
    corr = correlation_2op_1t(ham, rho_ss, time_arr, c_ops, s_plus, s_plus.dag(), solver='es')
    w_list_ring_1, spec_ring_1 = spectrum_correlation_fft(time_arr, corr)
    w_vals_1_ring.append(w_list_ring_1)
    spec_vals_1_ring.append(spec_ring_1)
    omega_vals_1_ring[curr_spin] = w_list_ring_1[np.argmax(spec_ring_1)]
print(omega_vals_1_ring)

spin = 0.5
num_spins = 4
time_arr = np.arange(-100, 100, 0.01)
omega = 0
gamma_d = 1
gamma_g = 1
coupling_strength = 0
adj_matrix = coupling_strength*symmetric_ring_matrix(num_spins)

ham, c_ops = symm_lc_sync_upper_manifold_sz_coupling_diss(num_spins, [omega]*num_spins, gamma_g, gamma_d, adj_matrix)

rho_ss = steadystate(ham, c_ops)
omega_vals_2_ring = [0]*num_spins
w_vals_2_ring = []
spec_vals_2_ring = []

for curr_spin in range(num_spins):
    s_plus = tensor([qeye(2*spin+1)]*(curr_spin) + [spin_Jp(spin)] + [qeye(2*spin+1)]*(num_spins-1-curr_spin))
    # print(s_plus)
    corr = correlation_2op_1t(ham, rho_ss, time_arr, c_ops, s_plus, s_plus.dag(), solver='es')
    w_list_ring_2, spec_ring_2 = spectrum_correlation_fft(time_arr, corr)
    w_vals_2_ring.append(w_list_ring_2)
    spec_vals_2_ring.append(spec_ring_2)
    omega_vals_2_ring[curr_spin] = w_list_ring_2[np.argmax(spec_ring_2)]
print(omega_vals_2_ring)

spin = 0.5
num_spins = 4
time_arr = np.arange(-100, 100, 0.01)
omega = 0
gamma_d = 1
gamma_g = 1
coupling_strength = 1
adj_matrix = coupling_strength*symmetric_ring_matrix(num_spins)

ham, c_ops = symm_lc_sync_upper_manifold_sz_coupling_diss(num_spins, [omega]*num_spins, gamma_g, gamma_d, adj_matrix)

rho_ss = steadystate(ham, c_ops)
omega_vals_3_ring = [0]*num_spins
w_vals_3_ring = []
spec_vals_3_ring = []

for curr_spin in range(num_spins):
    s_plus = tensor([qeye(2*spin+1)]*(curr_spin) + [spin_Jp(spin)] + [qeye(2*spin+1)]*(num_spins-1-curr_spin))
    corr = correlation_2op_1t(ham, rho_ss, time_arr, c_ops, s_plus, s_plus.dag(), solver='es')
    w_list_ring_3, spec_ring_3 = spectrum_correlation_fft(time_arr, corr)
    w_vals_3_ring.append(w_list_ring_3)
    spec_vals_3_ring.append(spec_ring_3)
    omega_vals_3_ring[curr_spin] = w_list_ring_3[np.argmax(spec_ring_3)]
print(omega_vals_3_ring)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

max_1 = np.max(spec_vals_1_ring[0])
max_2 = np.max(spec_vals_2_ring[0])
max_3 = np.max(spec_vals_3_ring[0])

ax.plot(w_vals_1_ring[0], spec_vals_1_ring[0], color="#ffcc00", label=r"$\omega=2, A=0$")
ax.plot(w_vals_2_ring[0], spec_vals_2_ring[0], color="#ff6600" , ls=":", label=r"$\omega=0, A=0$")
ax.plot(w_vals_3_ring[0], spec_vals_3_ring[0], color="#ff0000", ls="--", label=r"$\omega=0, A=1$")

peak1 = w_vals_1_ring[0][np.argmax(spec_vals_1_ring[0])]
peak2= w_vals_2_ring[0][np.argmax(spec_vals_2_ring[0])]
peak3 = w_vals_3_ring[0][np.argmax(spec_vals_3_ring[0])]
plt.plot(peak1, max_1, "*",color="#ffcc00", markersize=10, zorder=100)
plt.plot(peak2, max_2, "*",color="#ff6600", markersize=10, zorder=100)
plt.plot(peak3, max_3, "*",color="#ff0000", markersize=10, zorder=100)

ax.set_xlim(-1, 3)
ax.set_ylim(0, 0.45)
ax.set_xticks(list(plt.xticks()[0]))
ax.tick_params(labelsize=ticksize)

ax.set_title("Frequency Spectrum for Ring Coupling")
ax.set_xlabel(r"$\omega_p$", fontsize=fontsize)
ax.set_ylabel(r"$S_n(\omega_p)$", fontsize=fontsize)
plt.legend()

plt.savefig("oscillation_revival_ring.png", bbox_inches="tight")