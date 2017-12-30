#!/usr/bin/python3

####################################
#
# SELF CONSISTENT FIELD METHOD
#
####################################

import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
from collections import namedtuple

Context = namedtuple('Context',
                     ['num_electrons',
                      'nuclear_repulsion',
                      'overlap_integrals',
                      'kinetic_energy',
                      'potential_energy',
                      'eri_data',
                      'h_core',
                      'overlap_rsqrt',
                      'overlap_rsqrt_t',
                      'convergence',
                      'damping_factor',
                      'fock_coefficient']
                     )


def get_len(t):
    s = t.get_shape()
    return s[0].value


def get_raw_data(data_dir):
    enuc = tf.constant(float(np.load(data_dir + 'enuc.npy')), dtype=tf.float64)
    s = tf.convert_to_tensor(np.load(data_dir + 's.npy'), dtype=tf.float64)
    t = tf.convert_to_tensor(np.load(data_dir + 't.npy'), dtype=tf.float64)
    v = tf.convert_to_tensor(np.load(data_dir + 'v.npy'), dtype=tf.float64)
    eri = tf.convert_to_tensor(np.load(data_dir + 'eri.npy'), dtype=tf.float64)
    return enuc, s, t, v, eri


def create_context(num_electrons, data_dir):
    num_electrons = num_electrons
    nuclear_repulsion, overlap_integrals, kinetic_energy, potential_energy, eri_data = get_raw_data(
        data_dir)
    h_core = kinetic_energy + potential_energy

    overlap_eigenvalues, overlap_eigenvectors = tf.self_adjoint_eig(overlap_integrals)
    overlap_eigenvalue_rsqrt = tf.diag(tf.rsqrt(overlap_eigenvalues))
    overlap_rsqrt = overlap_eigenvectors @ (overlap_eigenvalue_rsqrt @ tf.transpose(overlap_eigenvectors))
    overlap_rsqrt_t = tf.transpose(overlap_rsqrt)

    a = eri_data
    b = tf.transpose(eri_data, perm=[0, 2, 1, 3])
    fock_coefficient = a - 0.5 * b

    convergence = tf.constant(1e-07, dtype=tf.float64)

    damping_factor = tf.constant(0.5, dtype=tf.float64)
    context = Context(
        num_electrons,
        nuclear_repulsion,
        overlap_integrals,
        kinetic_energy,
        potential_energy,
        eri_data,
        h_core,
        overlap_rsqrt,
        overlap_rsqrt_t,
        convergence,
        damping_factor,
        fock_coefficient)
    return context


def run_scf(context):
    i = tf.Variable(0, dtype=tf.int32)
    n = get_len(context.kinetic_energy)
    init_energy = tf.Variable(context.nuclear_repulsion, name='energy')
    init_density = tf.Variable(tf.zeros([n, n], dtype=tf.float64, name='density'))
    init_delta = tf.Variable(1.0, dtype=tf.float64, name='delta')
    init_fock_matrix = tf.Variable(tf.zeros([n, n], dtype=tf.float64), name='fock_matrix')

    def cond(context, fock_matrix, density, delta, energy, i):
        return tf.greater(delta, context.convergence)

    def body(context, fock_matrix, density, delta, energy, i):
        energy = context.nuclear_repulsion
        fock_matrix = update_fock_matrix(context, density)

        fock_prime = context.overlap_rsqrt_t @ (fock_matrix @ context.overlap_rsqrt)
        _, c_prime = tf.self_adjoint_eig(fock_prime)
        coefficients = context.overlap_rsqrt @ c_prime
        energy = energy + 0.5 * tf.reduce_sum(density * (context.h_core + fock_matrix))

        density, old_density = update_density(context, density, coefficients)

        density = context.damping_factor * density + (1 - context.damping_factor) * old_density
        delta = get_density_change(context, density, old_density)
        i += 1
        return [context, fock_matrix, density, delta, energy, i]

    loop_vars = [context, init_fock_matrix, init_density, init_delta, init_energy, i]

    scf_loop = tf.while_loop(cond, body, loop_vars)

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        result_vars = sess.run(scf_loop)
    total_energy = result_vars[4]
    return total_energy


def update_fock_matrix(context, density):
    n = get_len(context.h_core)
    shape = tf.ones_like(context.eri_data)
    p = shape * density
    f = p * context.fock_coefficient
    f = tf.reshape(f, [n, n, n ** 2])
    fock_matrix = context.h_core + tf.reduce_sum(f, 2)
    return fock_matrix


def update_density(context, density, coefficients):
    num_orbitals = context.num_electrons // 2
    old_density = tf.identity(density)
    c = coefficients[:, :num_orbitals]
    density = 2 * c @ tf.transpose(c)
    return density, old_density


# Calculate change in density matrix
def get_density_change(context, density, old_density):
    delta = tf.reduce_sum((density - old_density) ** 2)
    delta = (delta / 4) ** 0.5
    return delta


if __name__ == "__main__":
    scf_context = create_context(num_electrons=2, data_dir="../data/helium/")
    start = timer()
    total_energy = run_scf(scf_context)
    end = timer()
    print("Total Energy: {}".format(total_energy))
    print("Execution Time: {:.4f} sec".format(end - start))
