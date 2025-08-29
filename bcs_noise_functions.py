"""
BCS Noise Functions
This module provides functions for evaluating the enhancement of current and magnetic field noise in superconductors, 
based on BCS theory. All energies are measured in Kelvin, and momenta are measured with respect to the Fermi surface.
Functions:
----------
nf(Ek, Beta)
    Computes the Fermi-Dirac distribution for a given energy and inverse temperature.
ukvks(k_tilde, kmq_tilde, gap, Ef)
    Calculates the quasi-particle weights (u_k, u_{k-q}, v_k, v_{k-q}) for given momenta, gap, and Fermi energy.
coherence_factor(k_tilde, kmq_tilde, gap, Ef)
    Computes the BCS coherence factor squared for given momenta, gap, and Fermi energy.
density_factor(Ek, hbar_omega, beta)
    Calculates the density factor for transitions between quasi-particle states.
kinetic_constraint(q_tilde, k_tilde, sq, Ek, hbar_omega, gap, Ef)
    Solves for the angle between the noise wavevector and the particle/hole momentum, 
    returning the cosine of the angle and the squared shifted momentum.
DOSA_vfq(k_tilde, costheta_qk, Ekmq, gap)
    Computes the density of scattering angles (DOSA) for given parameters.
HSintegrand(Ek, q_tilde, hbar_omega, gap, kbT, Ef)
    Evaluates the integrand for current noise enhancement as a function of energy and other parameters.
weights(Ek, sk, sq, q_tilde, hbar_omega, gap, Ef)
    Returns the quasi-particle weights for given parameters and prints intermediate values for debugging.
current_noise_enhancement(q_tilde, hbar_omega, gap, kbT, Ef)
    Calculates the enhancement of current noise relative to a normal metal using numerical integration.
rate_enhancement(d, qs, RJs)
    Computes the enhancement of the relaxation rate relative to a normal metal, 
    given distance, wavevectors, and spectral densities.
Notes:
------
- The code assumes all energies are in Kelvin and momenta are normalized to the Fermi surface.
- Some approximations are made for large gap values; results may not be accurate for small gaps.
"""
import numpy as np
from scipy.integrate import quad

def nf(Ek, Beta):
    """Fermi-Dirac distribution."""
    return 1 / (np.exp(Ek * Beta) + 1)

def ukvks(k_tilde, kmq_tilde, gap, Ef):
    """Calculate quasi-particle weights (u_k, u_{k-q}, v_k, v_{k-q})."""
    xi_k   = k_tilde**2*Ef-Ef
    xi_kmq = kmq_tilde**2*Ef-Ef

    Ek   = np.sqrt(xi_k**2+gap**2)
    Ekmq = np.sqrt(xi_kmq**2+gap**2)
    
    theta_k   = np.arctan2(gap/Ek, xi_k/Ek)
    theta_kmq = np.arctan2(gap/Ekmq, xi_kmq/Ekmq)

    uk   = np.cos(theta_k/2)
    ukmq = np.cos(theta_kmq/2)
    vk   = np.sin(theta_k/2)
    vkmq = np.sin(theta_kmq/2)

    return (uk,ukmq,vk,vkmq)

def coherence_factor(k_tilde, kmq_tilde, gap, Ef):
    """Compute BCS coherence factor squared."""
    xi_k   = k_tilde**2*Ef-Ef
    xi_kmq = kmq_tilde**2*Ef-Ef

    Ek   = np.sqrt(xi_k**2+gap**2)
    Ekmq = np.sqrt(xi_kmq**2+gap**2)
    
    theta_k   = np.arctan2(gap/Ek, xi_k/Ek)
    theta_kmq = np.arctan2(gap/Ekmq, xi_kmq/Ekmq)

    uk   = np.cos(theta_k/2)
    ukmq = np.cos(theta_kmq/2)
    vk   = np.sin(theta_k/2)
    vkmq = np.sin(theta_kmq/2)

    return (uk*ukmq+vk*vkmq)**2

def bcs_transition_density(Ek, hbar_omega, beta):
    """Density factor for transitions between quasi-particle states."""
    return nf(Ek, beta) * (1 - nf(Ek + hbar_omega, beta)) + \
           (1 - nf(Ek, beta)) * nf(Ek + hbar_omega, beta)

def kinetic_constraint(q_tilde, k_tilde, sq, Ek, hbar_omega, gap, Ef):
    """Solve for angle and squared shifted momentum for kinetic constraint."""
    cos_theta_kmq = (k_tilde**2+q_tilde**2-1+sq*1/Ef*np.sqrt((hbar_omega+Ek)**2-gap**2))/2/k_tilde/q_tilde
    kmq_tilde_sqrd = k_tilde**2 + q_tilde**2 - 2*k_tilde*q_tilde*cos_theta_kmq
    return (cos_theta_kmq, kmq_tilde_sqrd)

def density_of_scattering_angles(k_tilde, costheta_qk, Ekmq, gap):
    """Density of scattering angles (DOSA)."""
    if costheta_qk**2<1:
        DOSAvfq = Ekmq/np.sqrt(Ekmq**2-gap**2)/k_tilde*np.sqrt(1-costheta_qk**2)
    else:
        DOSAvfq = 0
    return DOSAvfq

def current_r1_enhancement_integrand(Ek, q_tilde, hbar_omega, gap, kbT, Ef):
    """Integrand for current noise enhancement."""
    result = 0
    for sk in [-1,1]:
        for sq in [-1,1]:
            k_tilde = np.sqrt(1+sk/Ef*np.sqrt(Ek**2-gap**2))
            (cos_theta_kmq, kmq_tilde) = kinetic_constraint(q_tilde,k_tilde,sq, Ek, hbar_omega, gap, Ef)
            Ekmq = Ek+hbar_omega
            DOSAvfq = density_of_scattering_angles(k_tilde, cos_theta_kmq, Ekmq, gap)
            cf = coherence_factor(k_tilde, kmq_tilde, gap, Ef)
            if cos_theta_kmq <= 1:
                result += DOSAvfq*Ek/np.sqrt(Ek**2-gap**2)*cf*bcs_transition_density(Ek, hbar_omega, 1/kbT)
    return result 

def current_noise_enhancement(q_tilde, hbar_omega, gap, kbT, Ef):
    """Enhancement of current noise relative to normal metal."""
    result, _ = quad(current_r1_enhancement_integrand, gap , np.inf, args=(q_tilde, hbar_omega, gap, kbT, Ef), epsabs=1e-12, epsrel=1e-12)
    return result * 1/kbT/2

def rate_enhancement(d, qs, RJs):
    """
    Calculate the enhancement of the relaxation rate relative to a normal metal.

    This function computes the enhancement factor of the relaxation rate for a nitrogen-vacancy (NV) center
    located at a distance `d` from a superconductor, based on the spectral density `RJs` and corresponding
    wavevectors `qs`. The calculation follows Equation ** from the relevant theoretical framework.

    Parameters
    ----------
    d : float
        Distance of the NV center from the superconductor.
    qs : array_like
        Array of wavevector values.
    RJs : array_like
        Array of spectral density values corresponding to each wavevector in `qs`.

    Returns
    -------
    float
        The enhancement factor of the relaxation rate relative to a normal metal.
    """
    return 2*d*np.trapz( RJs*np.exp(-2*d*qs), qs)
