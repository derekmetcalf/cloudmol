#!/usr/bin/python3

####################################
#
# SELF CONSISTENT FIELD METHOD 
#
####################################

import numpy as np
from numpy import genfromtxt


# Symmetrize a matrix given a triangular one
def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


# Return compound index given four indices
def get_compound_index(a, b, c, d):
    if a > b:
        ab = a * (a + 1) / 2 + b
    else:
        ab = b * (b + 1) / 2 + a
    if c > d:
        cd = c * (c + 1) / 2 + d
    else:
        cd = d * (d + 1) / 2 + c
    if ab > cd:
        abcd = ab * (ab + 1) / 2 + cd
    else:
        abcd = cd * (cd + 1) / 2 + ab
    return abcd


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
            for m in range(num_electrons // 2):
                density[mu, nu] = density[mu, nu] + 2 * coefficients[mu, m] * coefficients[nu, m]
    return density, old_density


# Make Fock Matrix
def make_fock(h_core, density, num_basis_func, eri_data):
    fock = np.zeros((num_basis_func, num_basis_func))
    for i in range(num_basis_func):
        for j in range(num_basis_func):
            for k in range(num_basis_func):
                for l in range(num_basis_func):
                    a = eri_data[i, j, k, l]
                    b = eri_data[i, k, j, l]
                    fock[i, j] = h_core[i, j] + density[k, l] * (a - 0.5 * b)
    return fock


# Calculate change in density matrix
def calculate_density_change(density, old_density, num_basis_func):
    delta = 0.0
    for i in range(num_basis_func):
        for j in range(num_basis_func):
            delta = delta + ((density[i, j] - old_density[i, j]) ** 2)
    delta = (delta / 4) ** 0.5
    return delta


# Calculate energy at iteration
def get_current_energy(density, h_core, fock, num_basis_func):
    energy = 0.0
    for i in range(num_basis_func):
        for j in range(num_basis_func):
            energy = energy + 0.5 * density[i, j] * (h_core[i, j] + fock[i, j])
    return energy


# get raw data from dat files
def get_raw_data():
    enuc = float(np.load('../nitrido_data/enuc.npy'))
    s = np.load('../nitrido_data/s.npy')
    t = np.load('../nitrido_data/t.npy')
    v = np.load('../nitrido_data/v.npy')
    eri = np.load('../nitrido_data/eri.npy')
    return enuc, s, t, v, eri


def run_scf():
    num_electrons = 58  # The number of electrons in our system

    nuclear_repulsion, overlap_integrals, kinetic_energy, potential_energy, eri_data = get_raw_data()
    # num_basis_func is the number of basis functions
    # num_basis_func = int((np.sqrt(8 * len(s_data) + 1) - 1) / 2)
    num_basis_func = len(overlap_integrals)

    for i in range(num_basis_func):
        for j in range(num_basis_func):
            if i == j: continue
            for k in range(num_basis_func):
                if k in [i,j]: continue
                for l in range(num_basis_func):
                    if l in [i,j,k]: continue

                    eri_data[i, i, j, k] = 0.0
                    eri_data[i, j, k, k] = 0.0
    # Initialize integrals, and put them in convenient Numpy array format
    # overlap_integrals = np.zeros((num_basis_func, num_basis_func))
    # kinetic_energy = np.zeros((num_basis_func, num_basis_func))
    # potential_energy = np.zeros((num_basis_func, num_basis_func))
    #
    # for i in s_data:
    #     overlap_integrals[i[0] - 1, i[1] - 1] = i[2]
    # for i in t_data:
    #     kinetic_energy[i[0] - 1, i[1] - 1] = i[2]
    # for i in v_data:
    #     potential_energy[i[0] - 1, i[1] - 1] = i[2]

    # overlap_integrals = symmetrize(overlap_integrals)
    # kinetic_energy = symmetrize(kinetic_energy)
    # potential_energy = symmetrize(potential_energy)

    h_core = kinetic_energy + potential_energy

    # two_electron_dict = {get_compound_index(row[0], row[1], row[2], row[3]): row[4] for row in eri_data}

    overlap_eigenvalue, overlap_eigenvector = np.linalg.eig(overlap_integrals)
    overlap_eigenvalue_inv_sqrt = (np.diag(overlap_eigenvalue ** (-0.5)))
    overlap_inv_sqrt = np.dot(overlap_eigenvector,
                              np.dot(overlap_eigenvalue_inv_sqrt, np.transpose(overlap_eigenvector)))

    density_matrix = np.zeros((num_basis_func, num_basis_func))  # P is density matrix, set intially to zero.
    delta = 1.0
    convergence = 1e-08
    fock_matrix = None
    i = 0
    electronic_energy = None
    while delta > convergence:
        fock_matrix = make_fock(h_core, density_matrix, num_basis_func, eri_data)

        fock_prime = f_prime(overlap_inv_sqrt, fock_matrix)

        energy, coefficient_matrix = diagonalize(fock_prime, overlap_inv_sqrt)

        density_matrix, old_density = make_density(coefficient_matrix, density_matrix, num_basis_func, num_electrons)

        # test for convergence. if meets criteria, exit loop and calculate properties of interest

        delta = calculate_density_change(density_matrix, old_density, num_basis_func)
        i += 1
        electronic_energy = get_current_energy(density_matrix, h_core, fock_matrix, num_basis_func)
        print(electronic_energy + nuclear_repulsion, i)

    return electronic_energy + nuclear_repulsion


if __name__ == "__main__":
    total_energy = run_scf()
    print("Total Energy: {}".format(total_energy))
