# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:09:08 2023

@author: Van king
"""

from pde import FieldCollection, PDEBase, PlotTracker, ScalarField,plot_kymographs, CartesianGrid, VectorField, MemoryStorage,ProgressTracker
from pde.tools.numba import jit
# from pde.tools import FF
#from pde.tools.mpi import MPI
from mpi4py import MPI
from constants import h, hbar, e, m_e, a_B, epsilon_0, mathpi, fs2au, au2eV
from input_parameters import num_kpoints,lattice_constant,band_params,band_shifts   
from input_parameters import max_amplitude,cycle,photon_energy,simulate_time,num_time_step,time_interval,envelope_center,FWHM,CEP,T2,T1,Order,FFT_switch,plot_switch,MPI_switch,method,moive_switch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import integrate
import subprocess
import itertools
import time
import sys
import os
# Initialize the MPI communication context for the entire MPI program.
comm = MPI.COMM_WORLD # This includes all the MPI processes that are part of the execution.

rank = comm.Get_rank() # Get the rank of the current process within the communicator.
size = comm.Get_size() # Get the total number of processes in the communicator.

if rank == 0: # run in main process
    begin_time = time.time() # time begins
    print("Time begins")
    
def func_gen_derivative(y_array, x_array):
    """
    Parameters
    ----------
    y_array : array
        dependent variable.
    x_array : TYPE
        independent variable.

    Returns
    -------
    derivative : array
        The first derivative of y with respect to x.

    """
    derivative = np.empty(y_array.shape)
    dx = x_array[1] - x_array[0]
    for i in range(1, len(x_array) - 1):
        derivative[i] = (y_array[i + 1] - y_array[i - 1]) / (2 * dx)
        derivative[0] = (y_array[1] - y_array[0]) / dx
        derivative[-1] = (y_array[-1] - y_array[-2]) / dx
    
    return derivative
        
    
def time_dependent_E_field(t):
    """
    

    Parameters
    ----------
    t : float
        time point.

    Returns
    -------
    float
        time dependent electric field value.

    """
    
    return max_amplitude * np.exp(-(t-envelope_center)**2 / (FWHM/(2 * (np.sqrt(2 * np.log(2)))))**2 / 2) * np.cos(2*np.pi/cycle*t + CEP);

def time_dependent_A_field(t):
    """
    

    Parameters
    ----------
    t : float
        time point.

    Returns
    -------
    float
        time dependent vector field value.
        
    Take care of the begin of intergral is 0!!!    
    """
    result, err =  integrate.quad(time_dependent_E_field, 0, t)
    
    return -1 * result 

def func_group_velocity(grid, band):
    """
    

    Parameters
    ----------
    grid : CartesianGrid
        one dimension.
    band : array
        band energy of corresponding kpoints.

    Returns
    -------
    vel : array
        group velocity of corresponding kpoints.

    """
    x=grid.axes_coords[0]
    vel = np.zeros(len(band))
    k_interval = x[1] - x[0]
    vel[0] = (band[1] - band[-2])/2/k_interval
    for i in range(1,len(band)-1):
       vel[i] = (band[i+1] - band[i-1])/2/k_interval	
    vel[-1] = vel[0]
    return vel

def func_gen_tight_binding_band_1D(lattice_constant, kpoint_coord, gen_param, energy_shift, band_index):
	"""
	Generate band energy at given kpoint from cosine function expansion.
	
	Args:
		lattice_constant: single value; the lattice constant.
		kpoint_coord: single value; the reciprocal coordinate of kpoint.
		gen_param: 1D numpy array; the j-th coefficient of cos( j * k * a ) where a = lattice constant.
		energy_shift: single value; extra energy added on the cosine expansion; e.g., the band gap.
	
	Return:
		The band energy as a single value
	"""
	energy = 0
	for param_index, param in enumerate(gen_param[band_index]):
		energy += param * np.cos( param_index * kpoint_coord * lattice_constant )
	energy += energy_shift[band_index]
	return energy
    
def space_dependent_Eg(grid, band_param1, band_shift1, band_param2, 
                       band_shift2, band_index1, band_index2):
    """
    

    Parameters
    ----------
    grid : CartesianGrid
        one dimension

    Returns
    -------
    float
        energy gap at grid.

    """
    x = grid.axes_coords[0]
    band_energy1 = func_gen_tight_binding_band_1D(lattice_constant, 
                                                            x, band_param1, band_shift1, band_index1)
    band_energy2 = func_gen_tight_binding_band_1D(lattice_constant,
                                                         x, band_param2, band_shift2, band_index2)
    
    return  band_energy2 - band_energy1

def func_gen_transition_dipole_moment_1D(grid, TDM_index):
    """
    

    Parameters
    ----------
    grid : CartesianGrid
        one dimension

    Returns
    -------
    TDM : complex array
        transition dipole moment of corresponding kpoints

    """
    x = grid.axes_coords[0]
    # print(x)
    transition_dipole_moment = np.loadtxt("transition_dipole_moment.dat")
    # if transition_dipole_moment.shape[0] != num_kpoints: # check if the match between file and TDM
    #     print("Size does not match! exit")
    #     sys.exit()
    # else:
    # if len(x) != transition_dipole_moment.shape[0]:
    #         print("Length of x and transition_dipole_moment does not match.")
    #         print("Length of x:", len(x))
    #         print("Rows in transition_dipole_moment:", transition_dipole_moment.shape[0])
            
            # sys.exit()
            
     # choose kpoints index according to MPI's subprocess kpoints's coordinates

    start_index = np.searchsorted(grid.axes_coords[0], x[0])
    end_index = start_index + len(x)
    selected_tdm = transition_dipole_moment[start_index:end_index]

    TDM_real_fit = interp1d(x, selected_tdm[:, TDM_index * 2], kind='linear')
    TDM_imag_fit = interp1d(x, selected_tdm[:, TDM_index * 2 + 1], kind='linear')

    TDM_real = TDM_real_fit(x)
    TDM_imag = TDM_imag_fit(x)
    
    TDM = TDM_real + 1j * TDM_imag
    return TDM  

class oneD_SBEsPDE(PDEBase):
    """Semiconductor Bloch Equations in Bolch basis, n Bands, 1D, omititing coloumb interation"""

    def __init__(
        self, Eg_function, TDM_function,  Efield_function=None,  bc="periodic"
    ):
        super().__init__()
        self.Eg_function = Eg_function  # space-dependent energy gap
        self.TDM_function = TDM_function
        self.Efield_function = Efield_function  # electric field
        self.bc = bc  # boundary condition       

    def get_state(self, dm):
        """prepare a useful initial state"""
        
        for i, field in enumerate(dm):
            if i < num_pol:
                field.label = f"microspolarization {i}" # the first num_pol dm is microspolarization
            else:
                field.label = f"occupation {i-num_pol}" # the left dm is occupation
        
        return FieldCollection(dm,copy_fields="False")

    def evolution_rate(self, state, t=0):
        assert state.grid.dim == 1 # 1D
        Efield = self.Efield_function(t)
        dm = [state[i] for i in range(num_bands + num_pol)]
        Eg, TDM, omega = {}, {}, {} 
        for i, (band1, band2) in enumerate(itertools.combinations(range(num_bands), 2)):
            Eg[i] = self.Eg_function(state.grid, band_params, band_shifts, band_params, band_shifts, band1, band2)
            TDM[i] = self.TDM_function(state.grid, i)
            omega[i] = Efield * TDM[i]
        
        rhs = state.copy()          
        # microspolarization
        rhs[0] = -1j * ( (Eg[0] - 1j / T2) * dm[0] - omega[0] * (1 - dm[3] - dm[4]) + 1j * Efield * dm[0].gradient(self.bc)[0] + (omega[1] * np.conjugate(dm[2]) - omega[2] * dm[1]) )
        rhs[1] = -1j * ( (Eg[1] - 1j / T2) * dm[1] - omega[1] * (1 - dm[3] - dm[5]) + 1j * Efield * dm[1].gradient(self.bc)[0] + (omega[1] * dm[2]               - omega[2] * dm[1]) )
        rhs[2] = -1j * ( (Eg[2] - 1j / T2) * dm[2] - omega[2] * (dm[5] - dm[4])     + 1j * Efield * dm[2].gradient(self.bc)[0] + (omega[0] * dm[1]               - omega[1] * np.conjugate(dm[1])) )
        # occupation
        rhs[3] = -2 * np.imag( omega[0] * np.conjugate(dm[0]) + omega[1] * np.conjugate(dm[1]) ) + Efield * dm[3].gradient(self.bc)[0] - dm[3] / T1
        rhs[4] = -2 * np.imag( omega[0] * np.conjugate(dm[0]) + omega[2] * np.conjugate(dm[2]) ) + Efield * dm[4].gradient(self.bc)[0] - dm[4] / T1
        rhs[5] = -2 * np.imag( omega[1] * np.conjugate(dm[1]) + omega[2] * np.conjugate(dm[2]) ) + Efield * dm[5].gradient(self.bc)[0] - dm[5] / T1
 
        # dp_dt = -1j * Eg * p + 1j * omega * (1 - fv - fc) + Efield * p.gradient(self.bc)[0] - p / T2
        # dfv_dt = -2 * np.imag( omega * np.conjugate(p) ) + Efield * fv.gradient(self.bc)[0] - fv / T1
        # dfc_dt = dfv_dt
        return FieldCollection(rhs,copy_fields="False")
  
num_bands = len(band_shifts) # number of bands
num_pol = int((num_bands) * (num_bands -1 ) /2) # number of polarization
num_CB = sum(1 for x in band_shifts if x > 0)
num_VB = num_bands - num_CB  
eq = oneD_SBEsPDE(Eg_function=space_dependent_Eg, Efield_function=time_dependent_E_field, TDM_function=func_gen_transition_dipole_moment_1D)

if rank == 0:
# indices of micropolarization and occupation
    print("---------------------------------------\nIndices of density matrix are as follows:\n---------------------------------------")
    for i, (band1, band2) in enumerate(itertools.combinations(range(num_bands), 2)):
        print(f"micropolarization {i}: between band {band1} and band {band2}")
    print("---------------------------------------")
    for i in range(num_VB):
        print(f"occupation {i}: holes in band {i}")
    for i in range(num_VB, num_bands):
        print(f"occupation {i}: electrons in band {i}")
    print("---------------------------------------")

# initialize state
grid = CartesianGrid([[-1 * np.pi / lattice_constant, np.pi / lattice_constant] ], num_kpoints, periodic= [True])
dm = [ScalarField(grid, 0, dtype=np.complex_) for _ in range(num_bands + num_pol)]
state = eq.get_state(dm)

#storge data
storage = MemoryStorage()
if moive_switch == 'True': # judge if you want to generate moive
    tracker = [storage.tracker(interval=time_interval), "progress",
               PlotTracker(interval=100, movie="density_matrix.mp4",              
                       title="Density Matrix at step: {time:.2f}")]    # storage data every time_interval, movie each 10 steps(different unit)
else:
    tracker = [storage.tracker(interval=time_interval),"progress"]

# simulate the pde
if MPI_switch == 'True': # choose 'explicit_mpi' in method if parallel is open 
    method = 'explicit_mpi'

if method == 'explicit_mpi':
    sol = eq.solve(state, t_range=[0,simulate_time], tracker=tracker, method=method, scheme='rk', adaptive=True, tolerance=1e-2/num_kpoints)
elif method == 'scipy':
    sol = eq.solve(state, t_range=[0,simulate_time], tracker=tracker, method=method)
else:     
    sol = eq.solve(state, t_range=[0,simulate_time], tracker=tracker, method=method, scheme='rk', adaptive=True, tolerance=1e-2/num_kpoints)

# schme = 'euler' or 'rk'

# Make sure all processes have finished their simulation work
comm.Barrier()

if rank == 0:
    # plot_kymographs(storage) # show field with colorbar
    tspan = np.array(storage.times) # extract time
    data_array = np.array(storage.data) # transform data from list to array for indexing conveniently 

# extract data from storged solutions
    kmesh = grid.axes_coords[0]
    p_array = np.zeros((num_pol,len(tspan), num_kpoints),dtype=np.complex_)
    f_array = np.zeros((num_bands,len(tspan), num_kpoints),dtype=np.float_)
    for i in range(num_pol):
        p_array[i] = np.array([item[i].data for item in storage.data])/num_kpoints # p in each time point for all kpoints
    for i in range(num_pol, num_bands + num_pol):
        f_array[i-num_pol] = np.real(np.array([item[i].data for item in storage.data]))/num_kpoints # fband in each time point for all kpoints
#     fv_array = np.real(np.array([item[1].data for item in storage.data]))/num_kpoints # fv in each time point for all kpoints
#     fc_array = np.real(np.array([item[2].data for item in storage.data]))/num_kpoints # fc in each time point for all kpoints

    p_tot_array = np.sum(p_array, (0,2)) # p in each time point of the sum of all kpoints
    f_tot_array = np.sum(f_array, (0,2)) # f in each time point of the sum of all kpoints
#     fv_tot_array = np.sum(fv_array, 1) # fc in each time point of the sum of all kpoints

# current and HHG
# intraband current
    band_energy = np.zeros((num_bands,num_kpoints)) # band energy of each
    band_group_vel = np.zeros((num_bands,num_kpoints)) # band group velocity of each
    Jra_singlek = np.zeros((num_bands,len(tspan),num_kpoints)) # intraband current of each band, single kpoint
    Jra = np.zeros((num_bands,len(tspan)))# intraband current of each band
    Jra_tot = np.zeros((len(tspan), num_kpoints)) # intraband current of all bands
    # Jra_CB = np.zeros((num_CB,len(tspan),num_kpoints))
    for i in range(num_bands):
        band_energy[i] = func_gen_tight_binding_band_1D(lattice_constant, kmesh, band_params, band_shifts, i)
        band_group_vel[i] = func_group_velocity(grid, band_energy[i])
    for i in range(num_bands):
        if i < num_VB:
            # print(i,"1")
            Jra_singlek [i] = -1 * f_array[i] * band_group_vel[i]
        else:
            # print(i)
            Jra_singlek [i] = f_array[i] * band_group_vel[i]
    Jra = np.sum(Jra_singlek, 2)
    Jra_tot = np.sum(Jra, 0)        

#interband current
    p_tot = np.sum(p_array, 2) # micropolarization of all kpoints
    TDM_array = np.zeros((num_pol, num_kpoints),dtype=np.complex_) # transition dipole moment of each band pair
    P_array = np.zeros((num_pol,len(tspan)),dtype=np.float_) # macropolarization of each band pair
    Jer = np.zeros((num_pol, len(tspan)), dtype=np.float_) # interband current of each band pair
    Jer_tot = np.zeros(len(tspan), dtype=np.float_) # interband current of all band pairs
    for i in range(num_pol):
        TDM_array[i] = func_gen_transition_dipole_moment_1D(grid, i)
        P_array[i] = np.sum(2 * np.real(TDM_array[i] * p_array[i]), 1)
        Jer[i] = func_gen_derivative(P_array[i], tspan)
    Jer_tot = np.sum(Jer, 0)

    # circulate electric field and vector field  
    Efield = np.zeros(len(tspan))
    Afield = np.zeros(len(tspan))
    for i in range(len(tspan)):
        Efield[i] = time_dependent_E_field(tspan[i])
        Afield[i] = time_dependent_A_field(tspan[i])
   
    # save data into folder "data"
    data_values = np.column_stack([tspan / fs2au] + [Jra_tot, Jer_tot] +  [Jra[i] for i in range(len(Jra))] + [Jer[i] for i in range(len(Jer))] + [Efield, Afield])  # time(fs), Jra_tot(a.u.), Jer_tot(a.u.), Efield(a.u.), Afield(a.u.)
    data_names = ["time(fs)"] + ["Jra_tot(a.u.)", "Jer_tot(a.u.)"] + [f"Jra_{i}(a.u.)" for i in range(len(Jra))] + [f"Jer_{i}(a.u.)" for i in range(len(Jer))] + ["Efield(a.u.)", "Afield(a.u.)"]
    
    folder_name = 'data'
    if not os.path.exists(folder_name):# check if folder exists, create it if not
        os.makedirs(folder_name)

    # save each data array with header 
    filename = os.path.join(folder_name, "data.dat")   
    header_string = '\t'.join(data_names)
    np.savetxt(filename, data_values, header=header_string, fmt='%f', comments="#") # Use np.savetxt to write data with header
    
    # run "fast_fourier_transform.py" to get HHG if you switch on
    if FFT_switch == 'True': 
        subprocess.run(["python", "fast_fourier_transform.py"])  
   
    # run "plot.py" to get picture of all important physical quantities if you switch on
    if plot_switch == 'True':
       
        # plot electric field
        plt.figure(1)
        plt.plot( tspan/ cycle,  Efield, color="red")
        plt.xlabel("Time (cycles)")
        plt.ylabel(" Efield (a.u.)")
        plt.title('Laser E field')
        plt.savefig('Laser E field.png')

        # plot vector field
        plt.figure(2)
        plt.plot( tspan / cycle, Afield , color="blue")
        plt.xlabel("Time (cycles)")
        plt.ylabel(" Afield (a.u.)")
        plt.title('Laser A field')
        plt.savefig('Laser A field.png')
       
        # plot bandstructrue
        plt.figure(3)
        for i in range(num_bands):  
            if i < num_VB:
                plt.plot(kmesh / (2 * mathpi / lattice_constant),  au2eV * band_energy[i], color="blue")
            else:
                plt.plot(kmesh / (2 * mathpi / lattice_constant),  au2eV * band_energy[i], color="red")
        plt.xlabel("k (reciprocal vector)")
        plt.ylabel(" Energy (eV)")
        plt.title('Tight-binding bands')
        plt.savefig('Tight-binding bands.png')
        
        # plot transition dipole moment
        plt.figure(4,layout = 'constrained')
        plt.subplot(2, 1, 1)
        for i in range(num_pol):  # 假设 TDM_array 是一个数组或列表
            plt.plot(kmesh, np.real(TDM_array[i]),  label=f"TDM {i}")
        plt.xlabel("k (reciprocal vector)")
        plt.ylabel("Real (transition dipole moment) (a.u.)")
        plt.legend(loc='best')  # 显示图例       
        plt.subplot(2, 1, 2)
        for i in range(num_pol):
            plt.plot(kmesh, np.imag(TDM_array[i]),  label=f"TDM {i}")
        plt.xlabel("k (reciprocal vector)")
        plt.ylabel("Imag (transition dipole moment) (a.u.)")
        plt.legend(loc='best')  # 显示图例
        plt.suptitle('Transition Dipole Moment')
        fig = plt.gcf()  # 获取当前图形
        fig.set_size_inches(6, 8)  # 设置图像尺寸为 8x6 英寸
        plt.savefig('Transition dipole moment.png')

        # plot p, f
        plt.figure(5, figsize=(10, 6)) # plot real(p), imag(p)
        plt.subplot(2, 1, 1)
        for i in range(num_pol):
            plt.plot(tspan/cycle,  np.real(p_tot[i]),  label=f"micropolarization {i}")
        plt.title('Real part of micropolarization')
        plt.xlabel('Time (cycles)')
        plt.ylabel('Real (p)')
        plt.legend(loc='best')  # 显示图例
        plt.subplot(2, 1, 2)
        for i in range(num_pol):
            plt.plot(tspan/cycle,  np.imag(p_tot[i]),  label=f"micropolarization {i}")
        plt.title('Imaginary part of micropolarization')
        plt.xlabel('Time (cycles)')
        plt.ylabel('Imag (p)')
        plt.legend(loc='best')  # 显示图例
        plt.tight_layout()
        plt.savefig('Micropolarization.png', dpi=300)
        


        plt.figure(6, figsize=(10, 6)) # plot fv
        plt.subplot(2, 1, 1)
        for i in range(num_VB):
            plt.plot(tspan/cycle,  np.sum(f_array[i],1),  label=f"occupation {i}")
        plt.title('Holes occupation of VB')
        plt.xlabel('Time (cycles)')
        plt.ylabel('fv (h+/unit)')
        plt.legend(loc='best')  # 显示图例
        plt.subplot(2, 1, 2) # plot fc
        for i in range(num_CB):
            plt.plot(tspan/cycle,  np.sum(f_array[i+num_VB],1),  label=f"occupation {i+num_VB}")
        plt.title('Electrons occupation of CB')
        plt.xlabel('Time (cycles)')
        plt.ylabel('fc (e-/unit)')
        plt.legend(loc='best')  # 显示图例
        plt.tight_layout()
        plt.savefig('fv & fc.png', dpi=300)
        
        # plot intraband and interband current
        plt.figure(7, figsize=(10, 6)) # plot real(p), imag(p)
        plt.subplot(2, 1, 1)
        plt.plot(tspan/cycle,  Jra_tot,  color='red')
        plt.title('Intraband current')
        plt.xlabel('Time (cycles)')
        plt.ylabel('Jra(a.u.)')
        plt.subplot(2, 1, 2)
        plt.plot(tspan/cycle, Jer_tot ,  color='blue')
        plt.title('Interband current')
        plt.xlabel('Time (cycles)')
        plt.ylabel('Jer(a.u.)')
        plt.tight_layout()
        plt.savefig('Jra & Jer.png', dpi=300) 
        
    end_time = time.time() # time ends
    elapsed_time = end_time - begin_time
    print("Total elapsed time is {:.2f} seconds".format(elapsed_time)) # print elapsed time

# # close MPI environment
# #MPI.Finalize()