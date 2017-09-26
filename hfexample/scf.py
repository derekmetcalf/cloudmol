#!/usr/bin/python3

####################################
#
# SELF CONSISTENT FIELD METHOD 
#
####################################

import numpy as np


# Make Density Matrix
# and store old one to test for convergence
def make_density(coefficients, density, num_electrons):
    n = len(density)
    old_density = np.zeros((n, n))
    for mu in range(n):
        for nu in range(n):
            old_density[mu, nu] = density[mu, nu]
            density[mu, nu] = 0.0e0
            for m in range(num_electrons // 2):
                density[mu, nu] = density[mu, nu] + (2 * coefficients[mu, m] * coefficients[nu, m])
    return density, old_density


# Make Fock Matrix
def make_fock(h_core, density, eri_data):
    n = len(h_core)
    fock = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            fock[i, j] = h_core[i, j]
            for k in range(n):
                for l in range(n):
                    a = eri_data[i, j, k, l]
                    b = eri_data[i, k, j, l]
                    f = fock[i, j]
                    p = density[k, l]
                    fock[i, j] = f + p * (a - 0.5 * b)
    return fock


# Calculate change in density matrix
def calculate_density_change(density, old_density):
    delta = 0.0
    n = len(density)
    for i in range(n):
        for j in range(n):
            delta = delta + ((density[i, j] - old_density[i, j]) ** 2)
    delta = (delta / 4) ** 0.5
    return delta


# get raw data from dat files
def get_raw_data(data_dir):
    enuc = float(np.load(data_dir + 'enuc.npy'))
    s = np.load(data_dir + 's.npy')
    t = np.load(data_dir + 't.npy')
    v = np.load(data_dir + 'v.npy')
    eri = np.load(data_dir + 'eri.npy')
    return enuc, s, t, v, eri


def run_scf():
    num_electrons = 2  # The number of electrons in our system
    data_dir = "../helium_data/"

    nuclear_repulsion, overlap_integrals, kinetic_energy, potential_energy, eri_data = get_raw_data(data_dir)

    h_core = kinetic_energy + potential_energy

    overlap_eigenvalue, overlap_eigenvector = np.linalg.eig(overlap_integrals)
    overlap_eigenvalue_inv_sqrt = (np.diag(overlap_eigenvalue ** (-0.5)))
    overlap_inv_sqrt = np.dot(overlap_eigenvector,
                              np.dot(overlap_eigenvalue_inv_sqrt, np.transpose(overlap_eigenvector)))

    density = np.zeros((len(kinetic_energy), len(kinetic_energy)))  # P is density matrix, set intially to zero.
    delta = 1.0
    convergence = 1e-07
    fock_matrix = None
    i = 0
    energy = nuclear_repulsion
    while delta > convergence:
        energy = nuclear_repulsion
        fock_matrix = make_fock(h_core, density, eri_data)

        fock_prime = np.dot(np.transpose(overlap_inv_sqrt), np.dot(fock_matrix, overlap_inv_sqrt))

        _, c_prime = np.linalg.eigh(fock_prime)
        coefficients = np.dot(overlap_inv_sqrt, c_prime)

        energy = energy + 0.5 * np.sum(density * (h_core + fock_matrix))

        density, old_density = make_density(coefficients, density, num_electrons)

        density = 0.5 * density + 0.5 * old_density

        delta = calculate_density_change(density, old_density)
        i += 1
        print(energy, i)

    return energy


if __name__ == "__main__":
    total_energy = run_scf()
    print("Total Energy: {}".format(total_energy))
