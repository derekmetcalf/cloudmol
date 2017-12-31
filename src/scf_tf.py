#!/usr/bin/python3

####################################
#
# SELF CONSISTENT FIELD METHOD
#
####################################

import numpy as np
import tensorflow as tf
from timeit import default_timer as timer


def get_len(t):
    s = t.get_shape()
    return s[0].value


class SCF:
    def __init__(self, num_electrons, data_dir):
        self.num_electrons = num_electrons
        self.nuclear_repulsion, \
        self.overlap_integrals, \
        self.kinetic_energy, \
        self.potential_energy, \
        self.eri_data = self.get_raw_data(data_dir)
        self.h_core = self.kinetic_energy + self.potential_energy

        overlap_eigenvalues, overlap_eigenvectors = tf.self_adjoint_eig(self.overlap_integrals)
        overlap_eigenvalue_rsqrt = tf.diag(tf.rsqrt(overlap_eigenvalues))
        self.overlap_rsqrt = overlap_eigenvectors @ (overlap_eigenvalue_rsqrt @ tf.transpose(overlap_eigenvectors))
        self.overlap_rsqrt_t = tf.transpose(self.overlap_rsqrt)

        a = self.eri_data
        b = tf.transpose(self.eri_data, perm=[0, 2, 1, 3])
        self.fock_coefficient = a - 0.5 * b

        self.convergence = tf.constant(1e-07, dtype=tf.float64)
        self.damping_factor = tf.constant(0.5, dtype=tf.float64)

    @staticmethod
    def get_raw_data(data_dir):
        enuc = tf.constant(float(np.load(data_dir + 'enuc.npy')), dtype=tf.float64)
        s = tf.convert_to_tensor(np.load(data_dir + 's.npy'), dtype=tf.float64)
        t = tf.convert_to_tensor(np.load(data_dir + 't.npy'), dtype=tf.float64)
        v = tf.convert_to_tensor(np.load(data_dir + 'v.npy'), dtype=tf.float64)
        eri = tf.convert_to_tensor(np.load(data_dir + 'eri.npy'), dtype=tf.float64)
        return enuc, s, t, v, eri

    def run(self):
        n = get_len(self.kinetic_energy)
        init_energy = tf.Variable(self.nuclear_repulsion, name='energy')
        init_density = tf.Variable(tf.zeros([n, n], dtype=tf.float64, name='density'))
        init_delta = tf.Variable(1.0, dtype=tf.float64, name='delta')
        init_fock_matrix = tf.Variable(tf.zeros([n, n], dtype=tf.float64), name='fock_matrix')

        def cond(fock_matrix, density, delta, energy):
            return delta > self.convergence

        def body(fock_matrix, density, delta, energy):
            energy = self.nuclear_repulsion
            fock_matrix = self.update_fock_matrix(density)

            fock_prime = self.overlap_rsqrt_t @ (fock_matrix @ self.overlap_rsqrt)
            _, c_prime = tf.self_adjoint_eig(fock_prime)
            coefficients = self.overlap_rsqrt @ c_prime
            energy = energy + 0.5 * tf.reduce_sum(density * (self.h_core + fock_matrix))

            density, old_density = self.update_density(density, coefficients)

            density = self.damping_factor * density + (1 - self.damping_factor) * old_density
            delta = self.get_density_change(density, old_density)
            return [fock_matrix, density, delta, energy]

        loop_vars = [init_fock_matrix, init_density, init_delta, init_energy]

        scf_loop = tf.while_loop(cond, body, loop_vars)
        sess = tf.Session()
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            result_vars = sess.run(scf_loop)
        total_energy = result_vars[3]
        return total_energy

    def update_fock_matrix(self, density):
        n = get_len(self.h_core)
        shape = tf.ones_like(self.eri_data)
        p = shape * density
        f = p * self.fock_coefficient
        f = tf.reshape(f, [n, n, n ** 2])
        fock_matrix = self.h_core + tf.reduce_sum(f, 2)
        return fock_matrix

    def update_density(self, density, coefficients):
        num_orbitals = self.num_electrons // 2
        old_density = tf.identity(density)
        c = coefficients[:, :num_orbitals]
        density = 2 * c @ tf.transpose(c)
        return density, old_density

    # Calculate change in density matrix
    def get_density_change(self, density, old_density):
        delta = tf.reduce_sum((density - old_density) ** 2)
        delta = (delta / 4) ** 0.5
        return delta


if __name__ == "__main__":
    scf = SCF(num_electrons=2, data_dir="../data/helium/")
    start = timer()
    total_energy = scf.run()
    end = timer()
    print("Total Energy: {}".format(total_energy))
    print("Execution Time: {:.4f} sec".format(end - start))
