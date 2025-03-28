# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:09:08 2023

@author: Van king
"""
from pde import FieldCollection, PDEBase, PlotTracker, ScalarField, plot_kymographs, CartesianGrid, VectorField, MemoryStorage, FieldBase
from pde.tools.numba import jit
#from pde.tools.mpi import MPI
from mpi4py import MPI
from constants import h, hbar, e, m_e, a_B, epsilon_0, mathpi, fs2au, au2eV
from input_parameters import num_kpoints,lattice_constants,conduction_band_params,valence_band_params,conduction_band_shift,valence_band_shift   
from input_parameters import max_amplitude,cycle,photon_energy,simulate_time,num_time_step,time_interval,envelope_center,FWHM,CEP,T2,T1,Order,FFT_switch,plot_switch,MPI_switch,method,moive_switch,TDM_from_expression_switch,TDM_expression
import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
from scipy import integrate
import subprocess
import time
import sys
import os

# Initialize 
#the MPI communication context for the entire MPI program.
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

def time_dependent_E_field(grid, t):
    """
    Generate a time-dependent 3D electric field.

    Args:
        grid: Grid object defining the spatial discretization.
        t: float, time point.

    Returns:
        VectorField: 3D electric field vector at time t.
    """
    max_amplitude_array = np.array(max_amplitude)
    
    # Assuming uniform electric field over the entire grid
    Efield_data = np.zeros((3,) + grid.shape)  # Shape: (3,Nx, Ny, Nz) , 3 means Ex,Ey,Ez from 0 to 2; Nx,Ny,Nz assure the kpoint position
    Efield_value = max_amplitude_array * np.exp(-(t-envelope_center)**2 / (FWHM/(2 * np.sqrt(2 * np.log(2))))**2 / 2) * np.cos(2 * np.pi / cycle * t + CEP)
    
    Efield_data[0,:] = Efield_value[0]  # x-component of the electric field
    Efield_data[1,:] = Efield_value[1]  # y-component
    Efield_data[2,:] = Efield_value[2]  # z-component

    return VectorField(grid, Efield_data)

def time_dependent_A_field(grid, t):
    """
    Generate a time-dependent 3D vector field.

    Parameters
    ----------
    grid : Grid
        The spatial grid.
    t : float
        The time point at which the vector potential is calculated.

    Returns
    -------
    VectorField
        The vector potential at time t.
    """
    A_field_value = np.zeros(3)  

    # Define the electric field integral function of each component individually
    def integrand_x(t):
        return time_dependent_E_field(grid, t).data[0,0,0,0]

    def integrand_y(t):
        return time_dependent_E_field(grid, t).data[1,0,0,0]

    def integrand_z(t):
        return time_dependent_E_field(grid, t).data[2,0,0,0]

    # Integrate each direction component
    A_field_value[0], _ = integrate.quad(integrand_x, 0, t)
    A_field_value[1], _ = integrate.quad(integrand_y, 0, t)
    A_field_value[2], _ = integrate.quad(integrand_z, 0, t)
  
    A_field_data = np.tile(-A_field_value[:, np.newaxis, np.newaxis, np.newaxis], grid.shape)

    return VectorField(grid, A_field_data)


def band_energy_grid_field(grid, lattice_constants, band_params, energy_shift):
    """
    Generate the energy of a specific band (conduction or valence) across a 3D grid
    using a tight-binding model in 3D.

    Args:
        grid: CartesianGrid
            A 3D grid representing the space.
        lattice_constants: list of floats
            The lattice constants for each dimension.
        band_params: list of 1D numpy arrays
            The j-th coefficient arrays for cos(j * k * a) for each dimension.
        energy_shift: float
            Extra energy added on the cosine expansion, e.g., the band gap or band bottom.
    
    Return:
        A 3D scalar field with all grid points energy energy
    """
    x_coords, y_coords, z_coords = grid.coordinate_arrays
    band_energy = np.zeros(grid.shape)

    for x_index in range(x_coords.shape[0]):
        for y_index in range(y_coords.shape[1]):
            for z_index in range(z_coords.shape[2]):
                energy = [0, 0, 0]
                for dim in range(3):
                    lattice_constant = lattice_constants[dim]
                    kpoint_coord = [x_coords[x_index, y_index, z_index], 
                                    y_coords[x_index, y_index, z_index], 
                                    z_coords[x_index, y_index, z_index]][dim]

                    for param_index, param in enumerate(band_params[dim]):
                        energy[dim] += param * np.cos(param_index * kpoint_coord * lattice_constant)
                
                band_energy[x_index, y_index, z_index] = np.sum(energy) + energy_shift
    
    return ScalarField(grid, band_energy)

def func_gen_transition_dipole_moment(dim):
    """
    Parameters
    ----------
    dim : int
        dimension length.

    Returns
    -------
    final_array : array
        transition dipole moment, dim* num_kpoints[0]*num_kpoints[1]*num_kpoints[2].

    """
    #dim = 3 # dimension length    
    shape = ((dim,) +  tuple(num_kpoints))# Define the shape and data type for the array 
    data = np.loadtxt("transition_dipole_moment.dat") # read from file "transition_dipole_moment.dat"

    # where odd columns are real parts and even columns are imaginary parts
    real_parts = data[:, 0::2]
    imag_parts = data[:, 1::2]
    # Combine the real and imaginary parts to form complex numbers
    complex_data = real_parts + 1j * imag_parts
    # Reshape along rows of the complex data to form the desired array of shape 
    final_array = complex_data.T.reshape(shape, order = 'C')
    
    return final_array
    

#     """
    

#     Parameters
#     ----------
#     grid : CartesianGrid
#         one dimension

#     Returns
#     -------
#     TDM : complex array
#         transition dipole moment of corresponding kpoints

#     """
    
#     x = grid.axes_coords[0]
#     # print(x)
#     transition_dipole_moment = np.loadtxt("transition dipole moment.txt")
#     # if transition_dipole_moment.shape[0] != num_kpoints: # check if the match between file and TDM
#     #     print("Size does not match! exit")
#     #     sys.exit()
#     # else:
#     # if len(x) != transition_dipole_moment.shape[0]:
#     #         print("Length of x and transition_dipole_moment does not match.")
#     #         print("Length of x:", len(x))
#     #         print("Rows in transition_dipole_moment:", transition_dipole_moment.shape[0])
            
#             # sys.exit()
            
#      # choose kpoints index according to MPI's subprocess kpoints's coordinates

#     start_index = np.searchsorted(grid.axes_coords[0], x[0])
#     end_index = start_index + len(x)
#     selected_tdm = transition_dipole_moment[start_index:end_index]

#     TDM_real_fit = interp1d(x, selected_tdm[:, 0], kind='linear')
#     TDM_imag_fit = interp1d(x, selected_tdm[:, 1], kind='linear')

#     TDM_real = TDM_real_fit(x)
#     TDM_imag = TDM_imag_fit(x)
    
#     TDM = TDM_real + 1j * TDM_imag
#     return TDM  

class threeD_SBEsPDE(PDEBase):
    """Semiconductor Bloch Equations in Bolch basis, 2 Bands, 3D, omititing coloumb interation"""

    def __init__(
        self, band_energy_function,  Efield_function=None,  bc="periodic", TDM_from_expression=TDM_from_expression_switch 
    ): #, TDM_function
        super().__init__()
        self.band_energy_function = band_energy_function  # space-dependent energy gap
        # self.TDM_function = TDM_function
        self.Efield_function = Efield_function  # electric field     
        self.bc = bc  # boundary condition       
        self.TDM_from_expression= TDM_from_expression_switch  # Assignment TDM_from_expression_switch
    def get_state(self, p, fv, fc):
        """prepare initial state"""
        
        p.label = "mipolarization"
        fv.label = "occupation of holes in VB"
        fc.label = "occupation of electrons in CB"
        
        return FieldCollection([p, fv, fc])
        
    def evolution_rate(self, state, t=0):
        # assert state.grid.dim == 3 # 3D
        p, fv, fc = state # micropolarization, hole occupation, electron occupation
        Efield = self.Efield_function(state.grid,t) # electric field
        conduction_band_energy = self.band_energy_function(state.grid,lattice_constants,conduction_band_params,conduction_band_shift).data 
        valence_band_energy = self.band_energy_function(state.grid,lattice_constants,valence_band_params,valence_band_shift).data 
        Eg = conduction_band_energy - valence_band_energy # band energy difference between CB and VB
        
        # if self.TDM_from_expression:
            # TDM read from TDM_expression
        TDM = VectorField.from_expression(state.grid, TDM_expression) # transition dipole moment field               
        # else:
        #     # TDM read from file "transition_dipole_moment.dat"
        #     TDM = VectorField(grid, func_gen_transition_dipole_moment(3)) # 3D
            
        omega =  VectorField.dot(Efield, TDM) # Rabi frequency         
        dp_dt = -1j * Eg * p + 1j * omega * (1 - fv - fc) + VectorField.dot(Efield, p.gradient(self.bc)) - p / T2
        dfv_dt = -2 * np.imag( omega * np.conjugate(p) ) + VectorField.dot(Efield, fv.gradient(self.bc)) - fv / T1
        dfc_dt = dfv_dt
        return FieldCollection([dp_dt, dfv_dt, dfc_dt],copy_fields="False")
    
    # try to use numba to accerate but fail
    # def _make_pde_rhs_numba(self, state, t=0):      
    #     Efield = self.Efield_function(t)
    #     Eg = self.Eg_function(state.grid)
    #     TDM = self.TDM_function(state.grid)
    #     omega = Efield * TDM
    #     gradient = state.grid.make_operator("gradient", bc=self.bc)
        
    #     @jit(nopython=True,parallel=True)
    #     def pde_rhs(state_data, t):
    #         p = state_data[0]
    #         fv = state_data[1]
    #         fc = state_data[2]
            
    #         rate = np.empty_like(state_data)
    #         rate[0] = -1j * Eg * p + 1j * omega * (1 - fv - fc) + Efield * gradient(p) - p / T2
    #         rate[1] = -2 * np.imag( omega * np.conjugate(p) ) + Efield * gradient(fv) - fv / T1
    #         rate[2] = rate[1]
    #         return rate
    #     return pde_rhs

    
eq = threeD_SBEsPDE(band_energy_function=band_energy_grid_field, Efield_function=time_dependent_E_field)

# initialize state
grid = CartesianGrid([[-1 * np.pi / lattice_constants[0], np.pi / lattice_constants[0]], 
                      [-1 * np.pi / lattice_constants[1], np.pi / lattice_constants[1]], 
                      [-1 * np.pi / lattice_constants[2], np.pi / lattice_constants[2]]], 
                      num_kpoints, 
                      periodic=[True, True, True])  
p = ScalarField(grid, 0, dtype=np.complex_)
fv = ScalarField(grid, 0, dtype=np.complex_)
fc = ScalarField(grid, 0, dtype=np.complex_)
state = eq.get_state(p, fv, fc)

#storge data
storage = MemoryStorage()
#sol = eq.solve(state, t_range=[0,simulate_time], tracker=storage.tracker(1e-2), dt=time_interval, method=method)
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
    sol = eq.solve(state, t_range=[0,simulate_time], tracker=tracker, method=method, scheme='rk', adaptive=True, tolerance=1e-2/grid.num_cells)
elif method == 'scipy':
    sol = eq.solve(state, t_range=[0,simulate_time], tracker=tracker, method=method)
else:     
    sol = eq.solve(state, t_range=[0,simulate_time], tracker=tracker, method=method, scheme='rk', adaptive=True, tolerance=1e-2/grid.num_cells)

# schme = 'euler' or 'rk'

# Make sure all processes have finished their simulation work
comm.Barrier()

if rank == 0:
# plot_kymographs(storage) # show field with colorbar
    t = storage.times # extract time
    #data_array = np.array(storage.data) # transform data from list to array for indexing conveniently 

# extract data from storged solutions
    x_coords, y_coords, z_coords = grid.coordinate_arrays
    tspan = np.array(t)
    
    p_array = np.array([item[0].data for item in storage.data])/grid.num_cells # p in each time point for all kpoints
    fv_array = np.real(np.array([item[1].data for item in storage.data]))/grid.num_cells # fv in each time point for all kpoints
    fc_array = np.real(np.array([item[2].data for item in storage.data]))/grid.num_cells # fc in each time point for all kpoints

    p_tot_array = np.sum(p_array, (1,2,3)) # p in each time point of the sum of all kpoints
    fc_tot_array = np.sum(fc_array, (1,2,3)) # fv in each time point of the sum of all kpoints
    fv_tot_array = np.sum(fv_array, (1,2,3)) # fc in each time point of the sum of all kpoints

# current and HHG
# intraband current
    valence_band_group_vel = np.empty((3,) +grid.shape)
    conduction_band_group_vel = np.empty((3,) +grid.shape)
    Jra_x = np.empty((len(tspan),) +grid.shape)
    Jra_y = np.empty((len(tspan),) +grid.shape)
    Jra_z = np.empty((len(tspan),) +grid.shape)
    Jra_x_tot = np.empty(len(tspan))
    Jra_y_tot = np.empty(len(tspan))
    Jra_z_tot = np.empty(len(tspan))
    Jra_tot = np.empty(len(tspan))
    conduction_band_energy_field = band_energy_grid_field(grid, lattice_constants, 
                                                          conduction_band_params, conduction_band_shift)
    valence_band_energy_field = band_energy_grid_field(grid, lattice_constants, 
                                                       valence_band_params, valence_band_shift)
    valence_band_group_vel = valence_band_energy_field.gradient(bc="periodic").data # valence band group velocity
    conduction_band_group_vel = conduction_band_energy_field.gradient(bc="periodic").data # conduction band group velocity
    Jra_x = -1 * fv_array * valence_band_group_vel[0] + fc_array * conduction_band_group_vel[0] # Jra along x orientation
    Jra_y = -1 * fv_array * valence_band_group_vel[1] + fc_array * conduction_band_group_vel[1] # Jra along y orientation
    Jra_z = -1 * fv_array * valence_band_group_vel[2] + fc_array * conduction_band_group_vel[2] # Jra along z orientation
    
    Jra_x_tot = np.sum(Jra_x, (1,2,3)) # total intraband current in each time point of all kpoints along x orientation
    Jra_y_tot = np.sum(Jra_y, (1,2,3)) # total intraband current in each time point of all kpoints along y orientation
    Jra_z_tot = np.sum(Jra_z, (1,2,3)) # total intraband current in each time point of all kpoints along z orientation
    Jra_tot = np.sqrt(Jra_x_tot**2 + Jra_y_tot**2 + Jra_z_tot**2) # total mode intraband current in each time point of all kpoints and 3 orientations 
        
#interband current
    P_array_x = np.empty((len(tspan),) +grid.shape)
    P_array_y = np.empty((len(tspan),) +grid.shape)
    P_array_z = np.empty((len(tspan),) +grid.shape)
    P_array_tot = np.empty(len(tspan))
    P_array_x_tot = np.empty(len(tspan))
    P_array_y_tot = np.empty(len(tspan))
    P_array_z_tot = np.empty(len(tspan))
    Jer_tot = np.empty(len(tspan))
    # Jer_x_tot = np.empty(len(tspan))
    # Jer_y_tot = np.empty(len(tspan))
    # Jer_z_tot = np.empty(len(tspan))
    
    TDM_field = VectorField.from_expression(grid, ["3.46","3.46","3.94"]) # transition dipole moment field
    # TDM = VectorField.from_file("transition_dipole_moment.hdf5")
    TDM_array = TDM_field.data
    # TDM = func_gen_transition_dipole_moment_3D(grid)
    
    P_array_x = 2 * np.real(p_array * TDM_array[0]) # macropolarization in each time point for all kpoints along x orientation
    P_array_y = 2 * np.real(p_array * TDM_array[1]) # macropolarization in each time point for all kpoints along y orientation
    P_array_z = 2 * np.real(p_array * TDM_array[2]) # macropolarization in each time point for all kpoints along z orientation

    P_array_x_tot = np.sum(P_array_x, (1,2,3)) # total macropolarization in each time point of all kpoints along x orientation
    P_array_y_tot = np.sum(P_array_y, (1,2,3)) # total macropolarization in each time point of all kpoints along y orientation
    P_array_z_tot = np.sum(P_array_z, (1,2,3)) # total macropolarization in each time point of all kpoints along z orientation
    
    Jer_x_tot = func_gen_derivative(P_array_x_tot, tspan) # dP/dt
    Jer_y_tot = func_gen_derivative(P_array_y_tot, tspan)
    Jer_z_tot = func_gen_derivative(P_array_z_tot, tspan)
    # for i in range(1, len(tspan) - 1):
    #     Jer_x_tot[i] = (P_array_x_tot[i + 1] - P_array_x_tot[i - 1]) / 2/(tspan[1] - tspan[0])  
    #     Jer_y_tot[i] = (P_array_y_tot[i + 1] - P_array_y_tot[i - 1]) / 2/(tspan[1] - tspan[0]) 
    #     Jer_z_tot[i] = (P_array_z_tot[i + 1] - P_array_z_tot[i - 1]) / 2/(tspan[1] - tspan[0]) 
    # Jer_x_tot[0] = (P_array_x_tot[1] - P_array_x_tot[0]) / (tspan[1] - tspan[0])
    # Jer_y_tot[0] = (P_array_y_tot[1] - P_array_y_tot[0]) / (tspan[1] - tspan[0])
    # Jer_z_tot[0] = (P_array_z_tot[1] - P_array_z_tot[0]) / (tspan[1] - tspan[0])
    # Jer_x_tot[-1] = (P_array_x_tot[-1] - P_array_x_tot[-2]) / (tspan[1] - tspan[0]) # total interband current in each time point of all kpoints along x orientation 
    # Jer_y_tot[-1] = (P_array_y_tot[-1] - P_array_y_tot[-2]) / (tspan[1] - tspan[0]) # total interband current in each time point of all kpoints along y orientation  
    # Jer_z_tot[-1] = (P_array_z_tot[-1] - P_array_z_tot[-2]) / (tspan[1] - tspan[0]) # total interband current in each time point of all kpoints along z orientation  
    
    Jer_tot = np.sqrt(Jer_x_tot**2 + Jer_y_tot**2 + Jer_z_tot**2) # total interband current mode in each time point of all kpoints and 3 orientations 
    
    # circulate electric field and vector field  
    Efield = np.zeros((len(tspan),3))
    Afield = np.zeros((len(tspan),3))
    Efield_x = np.zeros(len(tspan))
    Efield_y = np.zeros(len(tspan))
    Efield_z = np.zeros(len(tspan))
    Afield_x = np.zeros(len(tspan))
    Afield_y = np.zeros(len(tspan))
    Afield_z = np.zeros(len(tspan))
    Efield_tot = np.zeros(len(tspan))
    Afield_tot = np.zeros(len(tspan))
    for i in range(len(tspan)):
        Efield[i] = time_dependent_E_field(grid,tspan[i]).data[:,0,0,0] # take the Efield value of (0,0,0) position  as efield cause the spatial uniformity of efield
        Afield[i] = time_dependent_A_field(grid,tspan[i]).data[:,0,0,0] # take the Efield value of (0,0,0) position  as efield cause the spatial uniformity of afield
        Efield_tot[i] = np.sqrt(Efield[i,0]**2 + Efield[i,1]**2 + Efield[i,2]**2) # mode of Efield in each time point
        Afield_tot[i] = np.sqrt(Afield[i,0]**2 + Afield[i,1]**2 + Afield[i,2]**2) # mode of Afield in each time point
    Efield_x = Efield[:,0] # Efield along x
    Efield_y = Efield[:,1] # Efield along y
    Efield_z = Efield[:,2] # Efield along z
    Afield_x = Afield[:,0] # Afield along x
    Afield_y = Afield[:,1] # Afield along y
    Afield_z = Afield[:,2] # Afield along z
    
    # save data into folder "data"
    data_values = np.column_stack([tspan / fs2au, Jra_tot, Jer_tot, Efield_tot, Afield_tot, 
                                   Jra_x_tot, Jra_y_tot, Jra_z_tot, Jer_x_tot, Jer_y_tot, Jer_z_tot,
                                   Efield_x,Efield_y,Efield_z,Afield_x,Afield_y,Afield_z])  
    # time(fs), Jra_tot(a.u.), Jer_tot(a.u.), Efield_tot(a.u.), Afield_tot(a.u.), Jra_x(a.u.),Jra_y(a.u.),Jra_z(a.u.),Jer_x(a.u.),Jer_y(a.u.),Jer_z(a.u.),
    # Efield_x(a.u.),Efield_y(a.u.),Efield_z(a.u.),Afield_x(a.u.),Afield_y(a.u.),Afield_z(a.u.)

    data_names = ["time(fs)", "Jra_tot(a.u.)", "Jer_tot(a.u.)", "Efield_tot(a.u.)", "Afield_tot(a.u.)", 
                  "Jra_x(a.u.)","Jra_y(a.u.)","Jra_z(a.u.)","Jer_x(a.u.)","Jer_y(a.u.)","Jer_z(a.u.)",
                  "Efield_x(a.u.)","Efield_y(a.u.)","Efield_z(a.u.)","Afield_x(a.u.)","Afield_y(a.u.)",
                  "Afield_z(a.u.)"]
    
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
        plt.figure(1,figsize=(8, 6))  
        plt.subplot(3, 1, 1)
        plt.plot(tspan / cycle, Efield_x, color="red")
        plt.xlabel("Time (cycles)")
        plt.ylabel("Efield of x part (a.u.)")
        plt.subplot(3, 1, 2)
        plt.plot(tspan / cycle, Efield_y, color="blue")
        plt.xlabel("Time (cycles)")
        plt.ylabel("Efield of y part (a.u.)")
        plt.subplot(3, 1, 3)
        plt.plot(tspan / cycle, Efield_z, color="black")
        plt.xlabel("Time (cycles)")
        plt.ylabel("Efield of z part (a.u.)")
        plt.suptitle('Laser E field')
        plt.savefig('Laser E field.png')

        # plot vector field
        plt.figure(2,figsize=(8, 6))  
        plt.subplot(3, 1, 1)
        plt.plot(tspan / cycle, Afield_x, color="red")
        plt.xlabel("Time (cycles)")
        plt.ylabel("Afield of x part (a.u.)")
        plt.subplot(3, 1, 2)
        plt.plot(tspan / cycle, Afield_y, color="blue")
        plt.xlabel("Time (cycles)")
        plt.ylabel("Afield of y part (a.u.)")
        plt.subplot(3, 1, 3)
        plt.plot(tspan / cycle, Afield_z, color="black")
        plt.xlabel("Time (cycles)")
        plt.ylabel("Afield of z part (a.u.)")
        plt.suptitle('Laser A field')
        plt.savefig('Laser A field.png')
        # plt.show()
        # plot bandstructrue
        # plt.figure(3)
        # slice at z=0
        conduction_z_slice = conduction_band_energy_field.slice({'z': 0})
        valence_z_slice = valence_band_energy_field.slice({'z': 0})
        X, Y = np.meshgrid(y_coords[0, :, 0]/ (2 * mathpi / lattice_constants[1]), x_coords[:, 0, 0]/ (2 * mathpi / lattice_constants[0]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, au2eV * conduction_z_slice.data, cmap='viridis')
        ax.plot_surface(X, Y, au2eV * valence_z_slice.data, cmap='viridis')
        ax.set_xlabel("ky (reciprocal vector)")
        ax.set_ylabel("kx (reciprocal vector)")
        ax.set_zlabel('Energy (eV)')
        ax.set_title("Bandstructure at z=0")
        plt.savefig('Tight-binding bands.png')
        # plt.show()

        # plot transition dipole moment
        #plt.figure(4)
        # TDM_field.plot(kind='auto',title='Transition dipole moment',filename='Transition dipole moment.png')
        

        # plot p, fv, fc
        plt.figure(5, figsize=(10, 6),layout = 'constrained') # plot real(p), imag(p)
        plt.subplot(2, 1, 1)
        plt.plot(tspan/cycle,  np.real(p_tot_array),  color='red')
        plt.title('Real part of micropolarization')
        plt.xlabel('Time (cycles)')
        plt.ylabel('Real (p)')
        plt.subplot(2, 1, 2)
        plt.plot(tspan/cycle,  np.imag(p_tot_array),  color='blue')
        plt.title('Imaginary part of micropolarization')
        plt.xlabel('Time (cycles)')
        plt.ylabel('Imag (p)')
        plt.savefig('Micropolarization.png', dpi=300)

        plt.figure(6, figsize=(10, 6),layout = 'constrained') # plot fv
        plt.subplot(2, 1, 1)
        plt.plot(tspan/cycle,  np.array(fv_tot_array), color='red')
        plt.title('Holes occupation of VB')
        plt.xlabel('Time (cycles)')
        plt.ylabel('fv (h+/unit)')
        plt.subplot(2, 1, 2) # plot fc
        plt.plot(tspan/cycle,  np.array(fc_tot_array), color='blue')
        plt.title('Electrons occupation of CB')
        plt.xlabel('Time (cycles)')
        plt.ylabel('fc (e-/unit)')
        plt.savefig('fv & fc.png', dpi=300)

        # plot intraband and interband current
        plt.figure(7, figsize=(10, 6),layout = 'constrained') # plot real(p), imag(p)
        plt.subplot(2, 1, 1)
        plt.plot(tspan/cycle,  Jra_x_tot,  color='red')
        plt.title('Intraband current')
        plt.xlabel('Time (cycles)')
        plt.ylabel('Jra(a.u.)')
        plt.subplot(2, 1, 2)
        plt.plot(tspan/cycle, Jer_x_tot ,  color='blue')
        plt.title('Interband current')
        plt.xlabel('Time (cycles)')
        plt.ylabel('Jer(a.u.)')
        plt.savefig('Jra & Jer.png', dpi=300) 

    end_time = time.time() # time ends
    elapsed_time = end_time - begin_time
    print("Total elapsed time is {:.2f} seconds".format(elapsed_time)) # print elapsed time

# # close MPI environment
# MPI.Finalize()