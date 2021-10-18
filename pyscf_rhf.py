#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program performs restricted thermal Hartree-Fock calculations for closed-shell
molecular systems. Molecular orbitals are optimized for a given temperature and 
chemical potential.

@author: Prateek Vaish
@email: prateek_vaish@brown.edu
"""


import numpy as np
import math
import pyscf


class SCF:
    
    def __init__(self):
        """
        initialize a SCF object
        """
        self.S = None
        self.V = None
        self.T = None
        self.Hcore = None
        self.enuc = None
        self.nelec = None
        self.norb = None
        self.twoe = None
        
        self.X = None
        self.F = None
        self.Fprime = None
        self.P = None
        self.Coeff = None
        self.oenergies = None
        self.Pprime = None
        self.mu = None
        
    def read_integrals(self, mol):
        """
        Parameters
        ----------
        mol : Mole pyscf object
        """
        
        self.S = mol.intor("int1e_ovlp")
        self.T = mol.intor("int1e_kin")
        self.V = mol.intor("int1e_nuc")
        self.enuc = mol.energy_nuc()
        self.norb = self.V.shape[0]
        self.nelec = sum(mol.nelec)
        self.Hcore = self.T + self.V 
        self.twoe = mol.intor("int2e")
        

    def orthogonalize(self):
        '''
        Returns
        -------
        S_minhalf: 2D numpy array of dimension (norb X norb)
            Unitary matrix S_minhalf used to tranform the Fock operator in orthonormal AO basis.

        '''
        SVAL, SVEC = np.linalg.eigh(self.S)
        SVAL_minhalf = (np.diag(SVAL**(-0.5)))
        S_minhalf = np.dot(SVEC,np.dot(SVAL_minhalf,np.transpose(SVEC)))
        return S_minhalf
    
    
    def get_Fock(self, P):
        '''
        computes the Fock matrix in AO basis
        
        Parameters
        ----------
        P : 2D numpy array of dimension (norb X norb)
            Density matrix.

        Returns
        -------
        F : 2D numpy array of dimension (norb X norb)
            Fock matrix.

        '''
        dim = self.norb
        F = np.zeros((dim,dim))
        for i in range(dim):
            for j in range(dim):
                F[i,j] = self.Hcore[i, j]
                for k in range(dim):
                    for l in range(dim):
                        F[i,j] = F[i,j] + P[k, l] * (self.twoe[i, j, l, k]- \
                                                 0.5 * self.twoe[i, k, j, l])
        return F  
    

    def get_density(self, C, P, beta, pot, zt = False):
        '''
        computes the average SCF energy at a given temperature and chemical potential

        Parameters
        ----------
        C : 2D numpy array of dimension (norb X norb)
            Matrix of orbital coefficients.
        P : 2D numpy array of dimension (norb X norb)
            Density Matrix.
        beta : Float
            inverse of temperature. (1 / kT)
        pot : Float
            Chemical potential.
        zt : Boolean, optional
            Whether to calculate the density matrix at zero temperture. The default is False.

        Returns
        -------
        P : 2D numpy array of dimension (norb X norb)
            Density matrix guess for next iteration.
        old_P : 2D numpy array of dimension (norb X norb)
            store old density matrix to check convergence.

        '''
        dim = self.norb
        if zt:
            dim2 = self.nelec // 2
        else:
            dim2 = self.norb
        
        old_P = np.zeros((dim, dim))
        for mu in range(dim):
            for nu in range(dim):
                old_P[mu, nu] = P[mu, nu]
                P[mu, nu] = 0.0
                for l in range(dim2):
                    try:
                        temp = math.exp(beta * (self.oenergies[l] - pot)) 
                        fermi = 1.0/ (temp + 1.0)
                    except OverflowError:
                        fermi = 0.0
                        
                    if zt:
                        fermi = 1.0
                    P[mu, nu] += 2 * fermi * C[mu, l] * C[nu, l]
                  
        return P, old_P

    def get_Energy(self):  
        '''
        Returns
        -------
        Float
            Average internal energy at a given temperature and chemical potential.

        '''
        dim = self.norb
        EN = 0.0 
        F = self.get_Fock(self.P)
        for mu in range(0,dim):  
            for nu in range(0,dim):  
                EN += 0.5 * self.P[mu,nu] * (self.Hcore[mu, nu] + F[mu,nu])  
        return EN + self.enuc



    
    def run_scf(self, beta, pot, scf_conv = 1.0e-6, scf_iter = 1000, do_DIIS = False, num_e = 10, \
                do_LDM = True, mix_dm = 0.5, zt = False):
        '''
        Performs a restricted Hartree Fock SCF calculation at a given temp and chemical potential.
        
        Parameters
        ----------
        beta : Float
            inverse of temperature. (1 / kT).
        pot : Float
            chemical potential.
        scf_conv : Float, optional
            Convergence threshold of SCF procedure. The default is 1.0e-6.
        scf_iter : Integer, optional
            Maximum number of SCF iterations. The default is 1000.
        do_DIIS : Boolean, optional
            Whether to perform DIIS(Pulay mixing). Needs more testing. The default is False.
        num_e : Integer, optional
            DIIS parameter. The default is 10.
        do_LDM : Boolean, optional
            Whether to perform Linear Density Mixing. The default is True.
        mix_dm : Float, optional
            A number between 0 and 1 which generates a new density matrix by mixing old and 
            current density matrix. The default is 0.5. Reduce the value at high beta values to avoid 
            oscillations.
        zt : Boolean, optional
            Whether the SCF calculation is at zero temperature. The default is False.

        Returns
        -------
        Boolean
            True : SCF converged, otherwise returns False.

        '''
        dim = self.norb
        self.oenergies = np.zeros(self.norb)
        
        # Calculate guess density matrix
        tempe, tempc = np.linalg.eigh(self.Hcore)
        tempp =  np.zeros((dim, dim))
        self.P, tempe = self.get_density(tempc, tempp, beta, pot, zt)

        self.X = self.orthogonalize()
        ErrorSet = []
        FockSet = []
        

        for j in range(scf_iter):
            self.F = self.get_Fock(self.P)
            
            if do_DIIS == True:
                if j > 0:
                    error = ((np.dot(self.F,np.dot(self.P, self.S)) - np.dot(self.S,np.dot(self.P, self.F))))
                    if len(ErrorSet) < num_e:
                        FockSet.append(self.F)
                        ErrorSet.append(error)
                    elif len(ErrorSet) >= num_e:
                        del FockSet[0]
                        del ErrorSet[0]
                        FockSet.append(self.F) 
                        ErrorSet.append(error)
                NErr = len(ErrorSet)
                if NErr >= 2:
                    Bmat = np.zeros((NErr+1,NErr+1))
                    ZeroVec = np.zeros((NErr+1))
                    ZeroVec[-1] = -1.0
                    for a in range(0,NErr):
                        for b in range(0,a+1):
                            Bmat[a,b] = Bmat[b,a] = np.trace(np.dot(ErrorSet[a].T,ErrorSet[b]))
                            Bmat[a,NErr] = Bmat[NErr,a] = -1.0
                    try:
                        coeff = np.linalg.solve(Bmat,ZeroVec)
                    except np.linalg.linalg.LinAlgError as err:
                        if 'Singular matrix' in err.message:
                            print('\tSingular B matrix, turing off DIIS')
                            do_DIIS = False
                    else:
                        self.F = 0.0
                        for i in range(0,len(coeff)-1):
                            self.F += coeff[i]*FockSet[i]
            	
            
            Fprime = np.transpose(self.X) @ self.F @ self.X
            self.oenergies, self.Cprime = np.linalg.eigh(Fprime)  
            self.Coeff = self.X @ self.Cprime
            self.P, old_P = self.get_density(self.Coeff, self.P, beta, pot, zt)
            
            if np.allclose(self.P, old_P, atol = scf_conv):
                self.Pprime, tempp = self.get_density(self.Cprime, tempc, pot, zt)
                return True
                break
            
            if do_LDM:
                self.P = mix_dm * self.P + (1 - mix_dm) * old_P
                
            if j == scf_iter - 1:
                print("SCF not converged")
                break

        return False
        
    
    def optimize_mu(self, beta, mu0, delmu, scf_iter = 1000, mu_conv = 1.0e-5, do_LDM = True, \
                    verbose = False, mix_dm = 0.5, mu_iter = 50, num_e = 6, zt = False):
        '''
        optimizes the chemical potential until the occupancy is equal to the number of electrons

        Parameters
        ----------
        beta : Float
            inverse of temperature. (1 / kT).
        mu0 : Float
            initial guess of chemical potential.
        delmu : Float
            step size for chemical potential.
        scf_iter : Integer, optional
            Maximum number of SCF iterations. The default is 1000.
        mu_conv : Float, optional
            Convergence threshold for occupancy. The default is 1.0e-5.
        do_LDM : Boolean, optional
            Whether to perform Linear Density Mixing. The default is True.
        mix_dm : Float, optional
            A number between 0 and 1 which generates a new density matrix by mixing old and 
            current density matrix. The default is 0.5. Reduce the value at high beta values to avoid 
            oscillations.
        verbose : Boolean, optional
            DESCRIPTION. The default is False.
        mu_iter : Integer, optional
            Max number of iterations for occupancy convergence. The default is 50.
        num_e : Integer, optional
            DIIS parameter. The default is 6.
        zt : Boolean, optional
            Whether the SCF calculation is at zero temperature. The default is False.

        Returns
        -------
        bool
            True: if the occupancy is converged, otherwise returns False.
        '''
        mu_high = mu0 + delmu
        mu_low = mu0 - delmu
        occ_diff = 1.0
        
        for it in range(mu_iter):
            
            mu = 0.5 * (mu_high + mu_low)
            converged = self.run_scf( beta, mu, scf_iter = scf_iter, do_DIIS = False, num_e = num_e, do_LDM = do_LDM, mix_dm = mix_dm, zt = zt)
            
            if converged:
                total_occ = self.get_occupancy()
                occ_diff = total_occ - self.nelec
                if abs(occ_diff) < mu_conv:
                    if verbose:
                        print("Final mu = {}".format(mu))
                    self.mu = mu
                    self.F = self.get_Fock(self.P)
                    return True
                
                if occ_diff > 0:
                    mu_high = mu
                else:
                    mu_low = mu
            else:
                print("SCF not converged at mu = {}".format(mu))
                print("Try lowering the value of mix_dm parameter.")
                return False
            
        print("Mu not converged)")
        return False
            
    
    def get_occupancy(self):
        '''
        calculates the occupancy calculated by Trace(P.S)

        Returns
        -------
        Float
            Occupancy at a given chemical potential.

        '''
        temp = self.P @ self.S
        return sum([temp[i, i] for i in range(self.norb)])
    
if __name__ == "__main__":
    rhf = SCF()
    mol = pyscf.gto.Mole()
    mol.atom = '''
    C
    C 1 1.262
    '''

    
    mol.basis = "STO-6G"
    mol.unit = "angstrom"
    mol.build()
    rhf.read_integrals(mol)
    
    beta = 5.0
    mu0 = -1.0
    delmu = 5.0
    mix = 0.2
    rhf.run_scf(beta = beta, pot = mu0, mix_dm = mix)
    rhf.optimize_mu(beta, mu0, delmu, mix_dm = mix)
    print(rhf.get_occupancy())
    print(rhf.get_Energy())

    