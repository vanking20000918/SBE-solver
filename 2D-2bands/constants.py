# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:15:12 2023

@author: Van king
"""

# define common constant and unit transform coefficients

from scipy import constants

h = constants.h  # Planck constant in joule-second
hbar = constants.hbar  # Reduced Planck constant (h/2pi) in joule-second
e = constants.e  # Elementary charge in coulombs
m_e = constants.m_e  # Electron mass in kilograms
a_B = constants.value("Bohr radius")  # Bohr radius in meters
epsilon_0 = constants.epsilon_0  # Vacuum permittivity in farads per meter
mathpi = constants.pi  # Mathematical constant pi

fs2au = 1 / (m_e * a_B**2 / hbar) * constants.femto  # Conversion coefficients from femtoseconds to atomic units of time
au2eV = e / (4 * mathpi * epsilon_0 * a_B)  # Conversion coefficients from atomic units of energy to electron volts
