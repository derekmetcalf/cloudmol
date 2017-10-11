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

#Construct DIIS vector for interpolation
def append_diis_vec(fock_matrix,fock_history,error_matrix,j,old_density,overlap_integrals):
    fock_history[j,:,:] = fock_matrix
    for i in range(j):
        error_matrix[i,:,:] = np.dot(fock_history[i,:,:],np.dot(old_density,overlap_integrals) -
                                     np.dot(old_density, np.dot(overlap_integrals,fock_history[i, :, :])))
    return fock_history, error_matrix

# Start interpolation at chosen diis_start_iter iteration
def diis_interpolate(fock_history,error_matrix,m,h_core):

    coeff = np.zeros(m+1)                  #initialize coefficients
    b_matrix = np.zeros((m+1, m+1))     #initialize b_matrix matrix

    for i in range(m):
        b_matrix[m, i] = -1
        b_matrix[i, m] = -1
        for j in range(m):
            b_matrix[i, j] = np.linalg.norm(np.dot(error_matrix[i,:,:], error_matrix[j,:,:]))
    b_matrix[m, m] = 0                      #compute b_matrix matrix
    fock_prime = np.zeros((len(h_core),len(h_core)))
    fock_prime_set = np.zeros((m,len(h_core),len(h_core)))
    solver_vector = np.zeros(m+1)
    solver_vector[m] = -1
    coeff = np.linalg.tensorsolve(b_matrix,solver_vector)  #compute coefficients

    for i in range(m):
        fock_prime_set[i, :, :] = coeff[i] * fock_history[i, :, :]
        fock_prime = np.add(fock_prime,fock_prime_set[i,:,:])
        # for j in range(m):
        #     fock_prime[i,j] = np.sum(fock_prime_set[i,:,j])    #compute new fock prime matrix

    return fock_prime

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
    num_electrons = 58  # The number of electrons in our system
    data_dir = "../nitrido_data/"
    diis_start_iter = 6
    nuclear_repulsion, overlap_integrals, kinetic_energy, potential_energy, eri_data = get_raw_data(data_dir)

    h_core = kinetic_energy + potential_energy

    overlap_eigenvalue, overlap_eigenvector = np.linalg.eig(overlap_integrals)
    overlap_eigenvalue_inv_sqrt = (np.diag(overlap_eigenvalue ** (-0.5)))
    overlap_inv_sqrt = np.dot(overlap_eigenvector,
                              np.dot(overlap_eigenvalue_inv_sqrt, np.transpose(overlap_eigenvector)))

    diis_errors = []
    density = np.zeros((len(kinetic_energy), len(kinetic_energy)))  # P is density matrix, set initially to zero.
    delta = 1.0
    convergence = 1e-07
    fock_matrix = np.zeros((len(h_core), len(h_core)))
    fock_history = np.zeros((diis_start_iter, len(h_core), len(h_core)))
    error_matrix = np.zeros((diis_start_iter, len(h_core), len(h_core)))

    i = 0       #total SCF iterations
    j = 0       #iterations since last DIIS interpolation

    energy = nuclear_repulsion
    while delta > convergence:

        fock_matrix = make_fock(h_core, density, eri_data)

        fock_prime = np.dot(np.transpose(overlap_inv_sqrt), np.dot(fock_matrix, overlap_inv_sqrt))

        _, c_prime = np.linalg.eigh(fock_prime)
        coefficients = np.dot(overlap_inv_sqrt, c_prime)

        energy = nuclear_repulsion + 0.5 * np.sum(density * (h_core + fock_matrix))

        density, old_density = make_density(coefficients, density, num_electrons)

        density = 0.5 * density + 0.5 * old_density

        delta = calculate_density_change(density, old_density)
        if j < diis_start_iter:
            fock_history, diis_errors = append_diis_vec(fock_matrix,fock_history,error_matrix,j,old_density,overlap_integrals)
        if j == diis_start_iter:
            fock_prime = diis_interpolate(fock_history, error_matrix, j, h_core)
            
            _, c_prime = np.linalg.eigh(fock_prime)
            coefficients = np.dot(overlap_inv_sqrt, c_prime)

            energy = energy + 0.5 * np.sum(density * (h_core + fock_matrix))

            density, old_density = make_density(coefficients, density, num_electrons)

            density = 0.5 * density + 0.5 * old_density

            delta = calculate_density_change(density, old_density)

            # Re-initialize vectors
            j = -1

        i += 1
        j +=1



        print(energy, i)

    return energy


if __name__ == "__main__":
    total_energy = run_scf()
    print("Total Energy: {}".format(total_energy))
