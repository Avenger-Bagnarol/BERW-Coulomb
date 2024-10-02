'''
    Complementary utility for the script "busch_coulomb.py"
    See the paper 
    "Accurate calculation of low energy scattering phase shifts of charged particles in a harmonic oscillator trap"
    of M. Bagnarol, N. Barneaa, M. Rojikb, M. Schafer
    for further details.

    Copyright (C) 2024  Mirko Bagnarol, Nir Barnea, Martin Schafer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from math import factorial
import numpy as np
from numpy.linalg import inv
from scipy.special import psi, gamma, spherical_jn, spherical_yn, erf
import scipy.integrate as integrate
from mpmath import whitm, whitw, acot



#############
# Constants #
#############

hbarc = 197.3269804 # MeV fm
e2 = 1.4399764 # MeV fm


#############
# Functions #
#############
 
#______________________________________________________________________________________
def momentum(mass, E): # mass [MeV], energy [MeV]
    return np.sqrt(2.*mass*E / hbarc**2) # fm^-1

#______________________________________________________________________________________
def eta(Z1, Z2, mu, k): # mu in MeV, k in fm^-1
    return Z1*Z2*e2*mu/(k*hbarc**2) # eta must be dimensionless

#______________________________________________________________________________________
def sommerfeld(l, eta):
    return 2**l * np.abs(gamma(l+1 + 1.j*eta )) * np.exp(- 0.5*np.pi*eta) / factorial(2*l+1) # pure

#______________________________________________________________________________________
def h_ere (eta): # function called h in the effective range expansion, wants the dimensionless eta
    return 0.5*( psi(1 + 1.j*eta) + psi(1 - 1.j*eta) ) - np.log(eta)

#______________________________________________________________________________________
def V_coulomb(Z1, Z2, r): # r in fm
    return Z1*Z2*e2/r # MeV
#______________________________________________________________________________________
def V_HO(mu, om, r): # r in fm
    return 0.5*mu*om**2*r**2/hbarc**2 # MeV
#______________________________________________________________________________________
def hankel_first(l, x):
    return spherical_jn(l, x) + 1.j*spherical_yn(l, x) # pure

#______________________________________________________________________________________
def G_free_mat(l, mu, k, rs): # mu in MeV, rs in fm, k in fm^-1

    jn = spherical_jn(l, rs*k)
    hn = hankel_first(l, rs*k)

    M = np.outer(jn,hn)
    G = -2.j*mu*k*(np.tril(M.T,-1)+np.triu(M,0))/hbarc**2

    return(G)

#______________________________________________________________________________________
def G_trap_mat(l, mu, omega, E, rs): # omega and E must be in MeV, r and rp in fm

    e_2om = E/(2.*omega)

    wl = np.array([float(whitm(e_2om, 0.5*l + 0.25, mu*omega*r**2/hbarc**2))/r**1.5 for r in rs]) 
    wh = np.array([float(whitw(e_2om, 0.5*l + 0.25, mu*omega*r**2/hbarc**2))/r**1.5 for r in rs]) 

    M = np.outer(wl,wh)

    factor = -( gamma(0.5*l + 0.75 - e_2om) / gamma(l + 1.5) ) / omega # MeV^-1 fm ^-3

    G = factor*(np.tril(M.T,-1)+np.triu(M,0))

    return(G)
#______________________________________________________________________________________
def Ftrap_coulomb_inv(Z1, Z2, mu, l, om, e, rs, fit_diagonal):


    rs_length = len(rs)

    # calculate integration weights
    print(f"Ftrap_coulomb: number of grid points={rs_length:4d}")
    
    integration_wghts = np.zeros(rs_length)
    integration_wghts[ 0] = 0.5*(rs[ 1]-rs[ 0])
    integration_wghts[-1] = 0.5*(rs[-1]-rs[-2])
    
    for idx in range(1, rs_length-1):
        integration_wghts[idx] = 0.5*(rs[idx + 1] - rs[idx - 1])

    Wintegral = np.diag(rs**2 * V_coulomb(Z1, Z2, rs) * integration_wghts)

    k = momentum(mu, e)
        
    # initialization of Gtrap and Gfree for the first couple omega, E

    Gfree = G_free_mat(l, mu, k, rs)
    Gtrap = G_trap_mat(l, mu, om, e, rs)
    
    identity = np.identity(rs_length)
    
    current_Gcoul = np.matmul(inv(identity-np.matmul(Gfree,Wintegral)),Gfree)
    current_GtrapC = np.matmul(inv(identity-np.matmul(Gtrap,Wintegral)),Gtrap)
    
    print("Matrix inversion:")
    print(np.real(current_Gcoul[0, 0]),"\t",np.real(current_GtrapC[0, 0]))

    this_eta = eta(Z1, Z2, mu, k)
    som = sommerfeld(l, this_eta )

    
    if not fit_diagonal:
        # The limit is computed directly
        green_diff_limit = np.real( current_Gcoul[0, 0] - current_GtrapC[0, 0] ) / rs[0]**(2*l)
    else:
        # Interpolation
        # I look for the value closes to 0.5, the maximum I allow to push 
        begin_idx = np.argmin(np.abs(rs - 0.05))
        end_idx = np.argmin(np.abs(rs - 0.5)) 

        if end_idx - begin_idx < 6: 
            # too few points for a meaningful fit
            green_diff_limit = np.real( current_Gcoul[0, 0] - current_GtrapC[0, 0] ) / rs[0]**(2*l)
            print("Two few points to fit, resort to first point approximation!")
        else:
            # Fitting six points between 0.05 and 0.5
            idxs = np.arange(begin_idx, end_idx + 1, (end_idx - begin_idx)/6.)
            idxs = np.floor(idxs).astype(int)

            rs_star = rs[idxs]
            diffs_star = np.array( [np.real( current_Gcoul[j, j] - current_GtrapC[j, j] ) / rs[j]**(2*l) for j in idxs] )

            fit = np.polyfit(rs_star, diffs_star, 3)

            green_diff_limit = fit[-1]
            

    k_cot_som2 = hbarc**2/ (2. * mu) * green_diff_limit

    ftrap = k_cot_som2 /( k**(2*l+1) * som**2)
    phase_shift = float(np.real(180*acot(ftrap)/np.pi))

    ere = k_cot_som2 + 2. * this_eta * h_ere(this_eta) * k**(2*l+1) * som**2 / ( sommerfeld(0, this_eta ) )**2

    return(phase_shift, ere, current_GtrapC, current_Gcoul, 0, 0)

#______________________________________________________________________________________
def Ftrap_coulomb(Z1, Z2, mu, l, om, e, rs, mix_eta, delta, NO_OF_ITERS, print_lines, fit_diagonal):

    rs_length = len(rs)

    # calculate integration weights
    print(f"  Ftrap_coulomb: number of grid points={rs_length:4d}")
    
    integration_wghts = np.zeros(rs_length)
    integration_wghts[ 0] = 0.5*(rs[ 1]-rs[ 0])
    integration_wghts[-1] = 0.5*(rs[-1]-rs[-2])
    
    for idx in range(1, rs_length-1):
        integration_wghts[idx] = 0.5*(rs[idx + 1] - rs[idx - 1])

    Wintegral = np.diag(rs**2 * V_coulomb(Z1, Z2, rs) * integration_wghts)

    k = momentum(mu, e)
        
    # initialization of Gtrap and Gfree for the first couple omega, E

    Gfree = G_free_mat(l, mu, k, rs)
    Gtrap = G_trap_mat(l, mu, om, e, rs)

    current_GtrapC = Gtrap
    current_Gcoul = Gfree

    count = 0

    # Will return flags separtely for G_C and G_CT
    # 0 converged
    # 1 exceeded max. number of iterations
    # 2 convergense is failing -> calculation aborted
    flag_G_C  = 1
    flag_G_CT = 1
    
    phase_shift =0  
    ere = 0
    
    # iteration loop
    while (count < NO_OF_ITERS +1): # iteration of the freedhoml equation
    
        loop_GtrapC = np.zeros(shape=(rs_length, rs_length))
        loop_Gcoul = np.zeros(shape=(rs_length, rs_length), dtype=np.complex128)
            
        if flag_G_CT == 1 :
            
            #Iteration
            loop_GtrapC = Gtrap + np.matmul(Gtrap,np.matmul(Wintegral,current_GtrapC))
            #Integral of the difference between two successive diagonals
            diffT = np.abs( np.sum( integration_wghts* (np.diag(loop_GtrapC) - np.diag(current_GtrapC)) ) )
            
            if (count > 10 and diffT < delta) :
                flag_G_CT=0
                
            if (diffT > 1e2): 
                flag_G_CT =2
        
        if flag_G_C == 1 :

            #Iteration
            loop_Gcoul = Gfree + np.matmul(Gfree,np.matmul(Wintegral,current_Gcoul))
            #Integral of the difference between two successive diagonals
            diffC = np.abs( np.sum( integration_wghts* (np.diag(loop_Gcoul) - np.diag(current_Gcoul)) ) )
            
            if (count > 10 and diffC < delta) :
                flag_G_C =0
                
            if (diffC > 1e2):
                flag_G_C =2

        #printing differencess Delta
        line = f"  {count:3d}"
        line+= f"  Delta Gtrap diag={diffT:14.10f}"
        line+= f"  Delta Gcoul diag={diffC:14.10f}"

        if (count%print_lines == 0): print(line) 
        
        #break conditions:
            
        if (flag_G_C == 2 and flag_G_CT == 2):
            print(line)
            print("\nSTATUS: G_C GREEN FUNCTION AN G_CT ARE FAILING TO CONVERGE! ABORT CALCULATION!\n")     
            break
        
        #Fully Converged
        if (flag_G_C==0 and flag_G_CT==0) :
            print(line)
            print("\nSTATUS: BOTH G_C and G_CT GREEN FUNCTION CONVERGED!")
            break
            
        #Only G_C green function converged
        if (flag_G_C==0 and flag_G_CT==2) :
            print(line)
            print("\nSTATUS: G_C GREEN FUNCTION CONVERGED BUT G_CT IS FAILING! ABORT CALCULATION! \n")
            break
            
        #Only G_CT green function converged
        if (flag_G_CT==0 and flag_G_C==2) :
            print(line)
            print("\nSTATUS: G_CT GREEN FUNCTION CONVERGED BUT G_C IS FAILING! ABORT CALCULATION! \n")
            break

        #Max number of iterations reached
        if (count >= NO_OF_ITERS):
            print("\nSTATUS: MAX NUMBERS OF ITERATIONS REACHED!\n")

        #Feedback mixing
        if flag_G_CT == 1 :
            current_GtrapC = mix_eta*loop_GtrapC + (1. - mix_eta)*current_GtrapC
        if flag_G_C  == 1 :
            current_Gcoul = mix_eta*loop_Gcoul + (1. - mix_eta)*current_Gcoul
        
        count +=1
   
    #Phase shift and Coulombic ERE calculation
    if (flag_G_C!=2 or flag_G_CT!=2) :
        
        this_eta = eta(Z1, Z2, mu, k)
        som = sommerfeld(l, this_eta)
        
        if not fit_diagonal:
            # The limit is computed directly
            green_diff_limit = np.real( current_Gcoul[0, 0] - current_GtrapC[0, 0] ) / rs[0]**(2*l)
        else:
            # Interpolation
            # I look for the value closes to 0.2, the maximum I allow to push 
            begin_idx = np.argmin(np.abs(rs - 0.05))
            end_idx = np.argmin(np.abs(rs - 0.5)) 

            if end_idx - begin_idx < 6: 
                # too few points for a meaningful fit
                green_diff_limit = np.real( current_Gcoul[0, 0] - current_GtrapC[0, 0] ) / rs[0]**(2*l)
                print("Two few points to fit, resort to first point approximation!")
            else:
                # Fitting six points between 0.05 and 0.5
                idxs = np.arange(begin_idx, end_idx + 1, (end_idx - begin_idx)/6.)
                idxs = np.floor(idxs).astype(int)

                rs_star = rs[idxs]
                diffs_star = np.array( [np.real( current_Gcoul[j, j] - current_GtrapC[j, j] ) / rs[j]**(2*l) for j in idxs] )

                fit = np.polyfit(rs_star, diffs_star, 3)

                green_diff_limit = fit[-1]
            

        k_cot_som2 = hbarc**2/ (2. * mu) * green_diff_limit

        ftrap = k_cot_som2 /( k**(2*l+1) * som**2)
        phase_shift = float(np.real(180*acot(ftrap)/np.pi))

        ere = k_cot_som2 + 2. * this_eta * h_ere(this_eta) * k**(2*l+1) * som**2 / ( sommerfeld(0, this_eta ) )**2

    return(phase_shift, ere, current_GtrapC, current_Gcoul, flag_G_C, flag_G_CT)
#  -----------------------------------------------------------------------------
