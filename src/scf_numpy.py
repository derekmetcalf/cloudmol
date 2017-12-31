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

        overlap_eigenvalue, overlap_eigenvector = np.linalg.eig(self.overlap_integrals)
        self.overlap_eigenvalue_inv_sqrt = (np.diag(overlap_eigenvalue ** (-0.5)))
        self.overlap_inv_sqrt = np.dot(overlap_eigenvector,
                                       np.dot(self.overlap_eigenvalue_inv_sqrt, np.transpose(overlap_eigenvector)))

        self.density = np.zeros(
            (len(self.kinetic_energy), len(self.kinetic_energy)))
        self.delta = 1.0
        self.convergence = 1e-07
        self.fock_matrix = None

    # get raw data from dat files
    @staticmethod
    def get_raw_data(data_dir):
        enuc = float(np.load(data_dir + 'enuc.npy'))
        s = np.load(data_dir + 's.npy')
        t = np.load(data_dir + 't.npy')
        v = np.load(data_dir + 'v.npy')
        eri = np.load(data_dir + 'eri.npy')
        return enuc, s, t, v, eri

    def run(self):
        i = 0
        energy = self.nuclear_repulsion
        while self.delta > self.convergence:
            energy = self.nuclear_repulsion
            self.make_fock()

            fock_prime = np.dot(np.transpose(self.overlap_inv_sqrt), np.dot(self.fock_matrix, self.overlap_inv_sqrt))

            _, c_prime = np.linalg.eigh(fock_prime)
            coefficients = np.dot(self.overlap_inv_sqrt, c_prime)

            energy = energy + 0.5 * np.sum(self.density * (self.h_core + self.fock_matrix))

            old_density = self.make_density(coefficients)

            self.density = 0.5 * self.density + 0.5 * old_density

            self.delta = self.calculate_density_change(old_density)
            i += 1
            print(energy, i)

        return energy

    # Make Fock Matrix
    def make_fock(self):
        n = len(self.h_core)
        fock = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                fock[i, j] = self.h_core[i, j]
                for k in range(n):
                    for l in range(n):
                        a = self.eri_data[i, j, k, l]
                        b = self.eri_data[i, k, j, l]
                        f = fock[i, j]
                        p = self.density[k, l]
                        fock[i, j] = f + p * (a - 0.5 * b)
        self.fock_matrix = fock

    # Make Density Matrix
    # and return old one to test for convergence
    def make_density(self, coefficients):
        n = len(self.density)
        old_density = np.zeros((n, n))
        for mu in range(n):
            for nu in range(n):
                old_density[mu, nu] = self.density[mu, nu]
                self.density[mu, nu] = 0.0e0
                for m in range(self.num_electrons // 2):
                    self.density[mu, nu] = self.density[mu, nu] + (2 * coefficients[mu, m] * coefficients[nu, m])
        return old_density

    # Calculate change in density matrix
    def calculate_density_change(self, old_density):
        delta = 0.0
        n = len(self.density)
        for i in range(n):
            for j in range(n):
                delta = delta + ((self.density[i, j] - old_density[i, j]) ** 2)
        delta = (delta / 4) ** 0.5
        return delta


if __name__ == "__main__":
    scf = SCF(num_electrons=2, data_dir="../data/helium/")
    start = timer()
    total_energy = scf.run()
    end = timer()
    print("Total Energy: {}".format(total_energy))
    print("Execution Time: {:.4f} sec".format(end - start))
