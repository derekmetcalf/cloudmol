#!/usr/bin/python3

####################################
#
# SELF CONSISTENT FIELD METHOD
#
####################################

import numpy as np
from timeit import default_timer as timer


class SCF(object):
    def __init__(self, num_electrons, data_dir):
        self.num_electrons = num_electrons
        self.nuclear_repulsion, self.overlap_integrals, self.kinetic_energy, self.potential_energy, self.eri_data = self.get_raw_data(
            data_dir)
        self.h_core = self.kinetic_energy + self.potential_energy

        overlap_eigenvalues, overlap_eigenvectors = np.linalg.eig(self.overlap_integrals)
        overlap_eigenvalue_rsqrt = np.diag(overlap_eigenvalues ** (-0.5))
        self.overlap_rsqrt = overlap_eigenvectors @ (overlap_eigenvalue_rsqrt @ overlap_eigenvectors.transpose())

        n = len(self.kinetic_energy)
        self.density = np.zeros((n, n))
        self.delta = 1.0

        self.fock_matrix = np.zeros((n, n))
        a = self.eri_data
        b = np.transpose(self.eri_data, (0, 2, 1, 3))
        self.fock_coefficient = a - 0.5 * b

        self.convergence = 1e-07
        self.damping_factor = 0.5

    # get raw data from dat files
    @staticmethod
    def get_raw_data(data_dir):
        enuc = float(np.load(data_dir + 'enuc.npy'))
        s = np.load(data_dir + 's.npy')
        t = np.load(data_dir + 't.npy')
        v = np.load(data_dir + 'v.npy')
        eri = np.load(data_dir + 'eri.npy')
        return enuc, s, t, v, eri

    def run(self, log_iterations=False):
        i = 0
        energy = self.nuclear_repulsion
        while self.delta > self.convergence:
            if log_iterations:
                start_loop = timer()
            energy = self.nuclear_repulsion
            self.update_fock_matrix()

            fock_prime = self.overlap_rsqrt.transpose() @ (self.fock_matrix @ self.overlap_rsqrt)

            _, c_prime = np.linalg.eigh(fock_prime)
            coefficients = self.overlap_rsqrt @ c_prime

            energy = energy + 0.5 * np.sum(self.density * (self.h_core + self.fock_matrix))

            old_density = self.update_density(coefficients)

            self.density = self.damping_factor * self.density + (1 - self.damping_factor) * old_density

            self.delta = self.get_density_change(old_density)
            i += 1
            if log_iterations:
                end_loop = timer()
                elapsed = end_loop - start_loop
                print("energy: {}, i: {}, iteration time: {:.4f} sec".format(energy, i, elapsed))

        return energy

    # Make Fock Matrix
    def update_fock_matrix(self):
        n = len(self.h_core)
        p = np.broadcast_to(self.density, self.eri_data.shape)
        f = p * self.fock_coefficient
        f = np.reshape(f, (n, n, n ** 2))
        self.fock_matrix = self.h_core + np.sum(f, 2)

    # Make Density Matrix
    # and return old one to test for convergence
    def update_density(self, coefficients):
        n = len(self.density)
        num_orbitals = self.num_electrons // 2
        old_density = np.copy(self.density)
        c = coefficients[:, :num_orbitals]
        self.density = 2 * c @ c.transpose()
        return old_density

    # Calculate change in density matrix
    def get_density_change(self, old_density):
        delta = np.sum((self.density - old_density) ** 2)
        delta = (delta / 4) ** 0.5
        return delta


if __name__ == "__main__":
    scf = SCF(num_electrons=58, data_dir="../data/nitrido/")
    start = timer()
    total_energy = scf.run()
    end = timer()
    print("Total Energy: {}".format(total_energy))
    print("Execution Time: {:.4f} sec".format(end - start))
