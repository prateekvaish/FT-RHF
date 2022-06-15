# FT-RHF

This program performs restricted Hartree-Fock calculations at finite temperture for closed-shell molecular systems. One and two-electron integrals are obtained using pyscf package. Within the program, the chemical potential can be tuned such that occupancy equals the number of electrons in the molecule. 

## Known issues

- Pulay mixing (DIIS) is not working at finite temperatures. 
- Linear density mixing is extremely inefficient at low temperatures. Lowering the mix_dm parameter converges the SCF procedure but increases the number of SCF iterations.

## References

[1]  A. Szabo and N. S. Ostlund, *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory*, Dover Publications (1996). 

[2] N. David Mermin. Stability of the Thermal Hartree-Fock Approximation. *Annals  of  Physics*, 21(1):99–121, 1963. doi:10.1016/0003-4916(63)90226-4.

[3] J. Sokoloff. Some consequences of the thermal Hartree-Fock Approximation at zero temperature. *Annals  of  Physics*, 45(2):186–190, 1967.ISSN 1096035X. doi:10.1016/0003-4916(67)90122-4.
