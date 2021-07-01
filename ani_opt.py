# -*- coding: utf-8 -*-
"""

This example is modified from the official `home page` and
`Constant temperature MD`_ to use the ASE interface of TorchANI as energy
calculator.

More information about the original code can be found at the links below
.. _home page:
    https://wiki.fysik.dtu.dk/ase/
.. _Constant temperature MD:
    https://wiki.fysik.dtu.dk/ase/tutorials/md/md.html#constant-temperature-md

The modified version of the code uses the BFSG algorithm to optimize a single structure
and return the associated energy.
"""
import os
import docopt

from ase.optimize import BFGS
from ase import units, atoms
from ase.io import read, write
from ase.units import Bohr, Rydberg, kJ, kB, fs, Hartree, mol, kcal
from ase.constraints import FixInternals
import torch
import torchani

###############################################################################
## ANI CONFIGURATION ##
# Possible alternative hack to automate cuda detection #
# print(f'Cuda Avail: {torch.cuda.is_available()}')
# calculator = torchani.models.ANI2x().ase()

# The option above tried to intuit  Cuda presence but was faulty.
# Hardcoding means this script can only run on machines with CUDA,

# change to 'cpu' if needed
# device = torch.device('cuda')

# Models https://aiqm.github.io/torchani/api.html?highlight=ani1#torchani.models.ANI1ccx
# ANI2x : supports (H, C, N, O, F, Cl, S)
# ANI1x : wB97X/6-31G(d) equivalent, it predicts energies on HCNO
# ANI1ccx : CCSD(T)*/CBS (CCSD(T) using the DPLNO-CCSD(T) method). It predicts energies on HCNO
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchani.models.ANI2x(periodic_table_index=True).to(device).double()
    print(f'Did I find TorchDevice using CUDA: {torch.cuda.is_available()}', flush=True)
except RuntimeError:  # if the thing is going to bomb out just get it done.
    device = torch.device('cpu')
    model = torchani.models.ANI2x(periodic_table_index=True).to(device).double()
    print('Did I find TorchDevice using CUDA: FALSE', flush=True)

calculator = model.ase()


## END ANI CONFIGURATION ##
##############################################################################

def opt(mol: atoms.Atoms()):
    '''
    xyz: str
        string from open-eye xyz conversion

    Returns
    -------
    energy: float
        energy calculated from the provisioned structure with the specified constraints

    '''

    # Now let's set the calculator for ``mol``:
    mol.set_calculator(calculator)

    # Now let's minimize the structure:
    print("Begin minimizing...")
    opt = BFGS(mol)
    # The threshold of 0.01 was tested on 3-4 torsions, use at your own risk
    opt.run(fmax=0.01)
    return (mol, mol.get_potential_energy())  #



if __name__ == "__main__":
    '''This script should be run in a conda environment with the following packages and dependencies installed:
    
    ASE: https://wiki.fysik.dtu.dk/ase/install.html
    ANI:  https://aiqm.github.io/torchani/start.html
    pip install docopt

    `python ani_opt.py --file myxyz.xyz`

    will return the optimized xyz structure.  Additional information and constraints can be found in the links above
    '''

    __doc__ = """ani_torsion.py
         Usage:
           ani_torsion.py -f myxyz.xyz

          -f, --file=<file>   Path to xyz file
    """
    arguments = docopt.docopt(__doc__)
    file = os.path.abspath(arguments['--file'])
    mol = read(os.path.join(file), format='xyz')
    mol_opted, energy = opt(mol)
    mol_opted.write(os.path.join(file + "_opted.xyz"), format='xyz')
