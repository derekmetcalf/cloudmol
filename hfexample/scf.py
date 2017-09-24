#!/usr/bin/python3

####################################
#
# SELF CONSISTENT FIELD METHOD 
#
####################################

import numpy as np


# Put Fock matrix in Orthonormal AO basis
def f_prime(x, f):
    return np.dot(np.transpose(x), np.dot(f, x))


# Diagonalize a matrix. Return Eigenvalues
# and non orthogonal Eigenvectors in separate 2D arrays.
def diagonalize(a, s_inv_sqrt):
    e, c_prime = np.linalg.eigh(a)
    c = np.dot(s_inv_sqrt, c_prime)
    return e, c


# Make Density Matrix
# and store old one to test for convergence
def make_density(coefficients, density, num_basis_func, num_electrons):
    old_density = np.zeros((num_basis_func, num_basis_func))
    for mu in range(num_basis_func):
        for nu in range(num_basis_func):
            old_density[mu, nu] = density[mu, nu]
            density[mu, nu] = 0.0e0
            # num_electrons // 2
            for m in range(num_electrons // 2):
                density[mu, nu] = density[mu, nu] + (2 * coefficients[mu, m] * coefficients[nu, m])
    return density, old_density


# Make Fock Matrix
def make_fock(h_core, density, num_basis_func, eri_data):
    fock = np.zeros((num_basis_func, num_basis_func))
    for i in range(num_basis_func):
        for j in range(num_basis_func):
            fock[i, j] = h_core[i, j]
            for k in range(num_basis_func):
                for l in range(num_basis_func):
                    i_prime, j_prime, k_prime, l_prime = (num_basis_func - 1 - a for a in [i, j, k, l])
                    a = eri_data[i_prime, j_prime, k_prime, l_prime]
                    b = eri_data[i_prime, k_prime, j_prime, l_prime]
                    f = fock[i, j]
                    p = density[k, l]
                    fock[i, j] = f + p * (a - 0.5 * b)
    return fock


# Calculate change in density matrix
# def calculate_density_change(density, old_density, num_basis_func):
#     delta = 0.0
#     for i in range(num_basis_func):
#         for j in range(num_basis_func):
#             delta = delta + ((density[i, j] - old_density[i, j]) ** 2)
#     delta = (delta / 4) ** 0.5
#     return delta


# Calculate energy at iteration
def get_current_energy(density, h_core, fock, num_basis_func):
    energy = 0.0
    for i in range(num_basis_func):
        for j in range(num_basis_func):
            energy = energy + 0.5 * density[i, j] * (h_core[i, j] + fock[i, j])
    return energy


# get raw data from dat files
def get_raw_data():
    enuc = float(np.load('../ammonia_data/enuc.npy'))
    s = np.load('../ammonia_data/s.npy')
    t = np.load('../ammonia_data/t.npy')
    v = np.load('../ammonia_data/v.npy')
    eri = np.load('../ammonia_data/eri.npy')
    return enuc, s, t, v, eri


def run_scf():
    num_electrons = 10  # The number of electrons in our system

    nuclear_repulsion, overlap_integrals, kinetic_energy, potential_energy, eri_data = get_raw_data()
    # num_basis_func is the number of basis functions
    num_basis_func = len(overlap_integrals)
    #num_basis_func = int((np.sqrt(8 * len(overlap_integrals + 1) - 1) / 2))
    h_core = kinetic_energy + potential_energy
    h_core = h_core[::-1, ::-1]

    overlap_eigenvalue, overlap_eigenvector = np.linalg.eig(overlap_integrals)
    overlap_eigenvalue_inv_sqrt = (np.diag(overlap_eigenvalue ** (-0.5)))
    overlap_inv_sqrt = np.dot(overlap_eigenvector,
                              np.dot(overlap_eigenvalue_inv_sqrt, np.transpose(overlap_eigenvector)))

    density_matrix = np.zeros((num_basis_func, num_basis_func))  # P is density matrix, set intially to zero.
    delta = 1.0
    convergence = 1e-07
    fock_matrix = None
    i = 0
    electronic_energy = 0.0
    while delta > convergence:
        fock_matrix = make_fock(h_core, density_matrix, num_basis_func, eri_data)

        fock_prime = f_prime(overlap_inv_sqrt, fock_matrix)

        energy, coefficient_matrix = diagonalize(fock_prime, overlap_inv_sqrt)

        density_matrix, old_density = make_density(coefficient_matrix, density_matrix, num_basis_func, num_electrons)

        # test for convergence. if meets criteria, exit loop and calculate properties of interest

        # delta = calculate_density_change(density_matrix, old_density, num_basis_func)
        i += 1
        old_electronic = electronic_energy
        electronic_energy = get_current_energy(density_matrix, h_core, fock_matrix, num_basis_func)
        delta = abs(electronic_energy - old_electronic)
        print(electronic_energy + nuclear_repulsion, i)

    electronic_energy = get_current_energy(density_matrix, h_core, fock_matrix, num_basis_func)
    return electronic_energy + nuclear_repulsion


if __name__ == "__main__":
    total_energy = run_scf()
    print("Total Energy: {}".format(total_energy))