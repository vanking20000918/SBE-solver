# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:23:53 2023

@author: Van king
"""

# Input parameters, all units is atomic unit unless otherwise stated

from constants import au2eV, fs2au, mathpi

num_kpoints = 400; # number of k points; better to be odd number so that it can include the Gamma point.
       
lattice_constant = 5.32; # lattice constant;
conduction_band_param = [0.0898, -0.0814, -0.0024, -0.0048, -0.0003, -0.0009]; # CB expansion coefficients
valence_band_param = [-0.0928, 0.0705, 0.0200, -0.0012, 0.0029, 0.0006]; # VB expansion coefficients
conduction_band_shift = 3.3 / au2eV; # CB Energy shift
valence_band_shift = 0 / au2eV; # VB Energy shift
       
# laser_type = 1; # 1 means input efield, 2 means input afield.
max_amplitude = 0.003; # laser electric/vector field max amplitude
cycle = 10.9 * fs2au; # laser optical cycle
photon_energy = 2* mathpi /cycle * au2eV; # laser photon energy, eV
simulate_time = 40 * cycle; # simulation time
num_time_step = 4361*1; 
time_interval = simulate_time / (num_time_step - 1)
envelope_center = simulate_time / 2; # the center of enveloped laser 
FWHM = 10 * cycle; # laser full width half maximum in efield, FWHM=2*sqrt(2ln2)*σ=2.355*σ for gaussion efield; FWHM=2/3*simulate_time for sine square afield
CEP = 0 * 2* mathpi # carrier envelpoed phase  

T2 = cycle /4; # polarization's dephasing time
T1 = simulate_time * 1000 # carriers occupation's relaxation time  
Order = 40; # order of FFT

method = 'scipy' # the method of solving the SBEs
# method = ['AdaptiveSolverBase', 'ExplicitMPISolver','explicit','explicit_mpi', 'implicit', 'scipy']
# 'scipy' is 4th or 5th Runge-Kutta 

MPI_switch = 'True' # 'True' means open MPI to parallel compute, 'False' means not
FFT_switch = 'True' # 'True' means it will run fast_fourier_transforme.py to get HHG, 'False' means not
plot_switch = 'True' # 'True' means it will plot and get the important physical quantities, 'False' means not
moive_switch = 'False' # 'True' means it will produce a movie of time varying density matrix, 'False' means not
