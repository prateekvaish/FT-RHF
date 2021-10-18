#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A driver program to perform thermal Hartree-Fock calculation for H20 molecule in the
STO-3G basis.

@author: prateekvaish
@email: prateek_vaish@brown.edu
"""

from pyscf_rhf import SCF
import pyscf


mol = pyscf.gto.Mole()
mol.atom = '''
O
H 1 0.96
H 1 0.96 2 109.5
'''
mol.basis = "STO-3G"
mol.unit = "angstrom"
mol.build()


rhf = SCF()
rhf.read_integrals(mol)

beta = 1.0
mu0 = -1.0
delmu = 5.0

mu_converged = rhf.optimize_mu(beta = beta, mu0 = mu0, delmu = delmu, scf_iter = 1000, \
                mu_conv = 1.0e-5,  do_LDM = True, mix_dm = 0.5, mu_iter = 50)

if mu_converged:
    print("Occupancy = {:.4f}".format(rhf.get_occupancy()))
    
    print("Internal energy = {:.4f} Hartree".format(rhf.get_Energy()))

else:
    print("Occupancy not converged")