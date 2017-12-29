#!/usr/bin/python3

####################################
#
# SELF CONSISTENT FIELD METHOD
#
####################################

import numpy as np
import torch
from timeit import default_timer as timer

torch.set_num_threads(4)


class SCF(object):
    def __init__(self, num_electrons, data_dir):
        self.num_electrons = num_electrons
        self.nuclear_repulsion, self.overlap_integrals, self.kinetic_energy, self.potential_energy, self.eri_data = self.get_raw_data(
            data_dir)
        self.h_core = self.kinetic_energy + self.potential_energy

        overlap_eigenvalues, overlap_eigenvectors = torch.eig(self.overlap_integrals, eigenvectors=True)
        overlap_eigenvalues = overlap_eigenvalues[:, 0]
        overlap_eigenvalue_rsqrt = torch.diag(torch.rsqrt(overlap_eigenvalues))
        self.overlap_rsqrt = overlap_eigenvectors @ (overlap_eigenvalue_rsqrt @ overlap_eigenvectors.t())

        n = len(self.kinetic_energy)
        self.density = torch.zeros((n, n)).double()
        self.delta = 1.0
        self.convergence = 1e-07
        self.fock_matrix = torch.zeros((n, n)).double()
        self.damping_factor = 0.5

    # get raw data from npy files
    @staticmethod
    def get_raw_data(data_dir):
        enuc = float(np.load(data_dir + 'enuc.npy'))
        s = torch.from_numpy(np.load(data_dir + 's.npy')).double()
        t = torch.from_numpy(np.load(data_dir + 't.npy')).double()
        v = torch.from_numpy(np.load(data_dir + 'v.npy')).double()
        eri = torch.from_numpy(np.load(data_dir + 'eri.npy')).double()
        return enuc, s, t, v, eri

    def run(self):
        i = 0
        energy = self.nuclear_repulsion
        while self.delta > self.convergence:
            start_loop = timer()
            energy = self.nuclear_repulsion
            self.update_fock_matrix()

            fock_prime = self.overlap_rsqrt.t() @ (self.fock_matrix @ self.overlap_rsqrt)
            _, c_prime = torch.symeig(fock_prime, eigenvectors=True)
            coefficients = self.overlap_rsqrt @ c_prime
            energy = energy + 0.5 * torch.sum(self.density * (self.h_core + self.fock_matrix))

            old_density = self.update_density(coefficients)
            self.density = self.damping_factor * self.density + (1 - self.damping_factor) * old_density
            self.delta = self.get_density_change(old_density)

            i += 1
            end_loop = timer()
            elapsed = end_loop - start_loop
            print("energy: {}, i: {}, iteration time: {:.4f} sec".format(energy, i, elapsed))

        return energy

    def update_fock_matrix(self):
        n = len(self.h_core)
        p = self.density.expand_as(self.eri_data)
        a = self.eri_data
        b = torch.transpose(self.eri_data, 1, 2)
        f = p * (a - 0.5 * b)
        f = f.view(n, n, n ** 2)
        self.fock_matrix = self.h_core + torch.sum(f, 2)

    # update density
    # return old density to test for convergence
    def update_density(self, coefficients):
        n = len(self.density)
        num_orbitals = self.num_electrons // 2
        old_density = self.density.clone()
        c = coefficients[:, :num_orbitals]
        self.density = 2 * c @ c.t()
        return old_density

    # Calculate change in density matrix
    def get_density_change(self, old_density):
        delta = torch.sum((self.density - old_density) ** 2)
        delta = (delta / 4) ** 0.5
        return delta


if __name__ == "__main__":
    scf = SCF(num_electrons=2, data_dir="../data/helium/")
    start = timer()
    total_energy = scf.run()
    end = timer()
    print("Total Energy: {}".format(total_energy))
    print("Execution Time: {:.4f} sec".format(end - start))
