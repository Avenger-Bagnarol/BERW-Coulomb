'''
    Script which calculates scattering phase shifts of charged core-fragment nulcear or atomic scattering.
    See the paper 
    "Accurate calculation of low energy scattering phase shifts of charged particles in a harmonic oscillator trap"
    of M. Bagnarol, N. Barneaa, M. Rojikb, M. Schafer
    for further details.

    Copyright (C) 2024  Mirko Bagnarol, Nir Barnea, Martin Shafer

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

import sys
import os
import numpy as np

import argparse

from busch_coulomb_input import *
from busch_coulomb_utility import *


#################
# PARSING INPUT	#
#################

argv = sys.argv[1:]

parser = create_parser()

args = parser.parse_args(argv)

if args.input_file != "": # reads json input file
    main_file, threshold_file, outfile_phsh, print_lines, print_green, outfile_green, unit_mass, mass_threshold, mass_fragment, Z1, Z2, l, method, fit_diagonal, NO_OF_ITERS, N, q, r_min, max_factor, mix_eta, delta = read_parser_file(args.input_file)

else: # standart input parameter per parameter
    main_file, threshold_file, outfile_phsh, print_lines, print_green, outfile_green, unit_mass, mass_threshold, mass_fragment, Z1, Z2, l, method, fit_diagonal, NO_OF_ITERS, N, q, r_min, max_factor, mix_eta, delta = read_parser_args(args)

o_p = open(outfile_phsh, 'w')
mu = mass_threshold*mass_fragment/(mass_threshold + mass_fragment)

fact = 1
for i in range (1, N):
    fact += q**i

#####################
## FILES READING   ##
#####################

main, threshold = read_files(main_file, threshold_file)

#########################
## ACTUAL CALCULATUION ##
#########################

result = [] 

print(f"Reduced mass = {mu} MeV, angular momentum = {l}, charges {Z1}e and {Z2}e")

# loop on E, omega
for i in range( len(main) ): # those are lists [ [cutoff1, [traplengths at cutoff1], [energy at cutoff1]], [cutoff2, ...] ]
        
    cutoff = main[i][0]

    # defining the containers to give back in a similar fashion [ [cut1, [...]], [cut2, [...]] ]
    energies = []
    omegas = []
    phshs = []
    eres = []

    flags_G_C = []
    flags_G_CT = []
    
    G_Cs = []
    G_CTs = []

    for j in range ( len(main[i][1]) ):

        if len(threshold) == 0:
            e = main[i][2][j]  # MeV
        else:
            e = main[i][2][j] - threshold[i][2][j] # MeV

        # NOW THE SCRIPT EXPECTS OMEGA IN THE FIRST COLUMN! 
        om = main[i][1][j] # MeV
        b = hbarc / np.sqrt( mu * om) # fm 

        # building rs given b
        end = b * max_factor
        step = (end - r_min) / fact
        curr_r = step
        rs = [r_min]

        for ir in range(1, N):
            rs.append(rs[ir - 1] + step * q**ir )

        rs = np.array(rs)

        print("\n=====================================================================")
        print(f"b:\t{b} fm\nenergy:\t{e} MeV\nomega:\t{om} MeV\ne_om:\t{e/om}\n")

        # computes the phase shift and reports if successfull
        if method == 1:
            phsh, ere, Gtrap, Gcoul, flag_G_C, flag_G_CT = Ftrap_coulomb(Z1, Z2, mu, l, om, e, rs, mix_eta, delta, NO_OF_ITERS, print_lines, fit_diagonal)
        else:
            phsh, ere, Gtrap, Gcoul, flag_G_C, flag_G_CT = Ftrap_coulomb_inv(Z1, Z2, mu, l, om, e, rs, fit_diagonal)
        
        energies.append(e)
        omegas.append(om)
        phshs.append(phsh)
        eres.append(np.real(ere))
        flags_G_C.append(flag_G_C)
        flags_G_CT.append(flag_G_CT)
        G_Cs.append(np.real(Gcoul[0][0]))
        G_CTs.append(Gtrap[0][0])

        if print_green:

            # computing the Sommerfeld factor also here to print it out
            k = momentum(mu, e)
            this_eta = eta(Z1, Z2, mu, k)
            som = sommerfeld(l, this_eta )

            o_g = open(outfile_green, 'w')

            o_g.write(f"# Cutoff: {cutoff}, Omega: {om}, Energy: {e} MeV, Mass: {mu} MeV, Sommerfeld: {som}\n")
            o_g.write("# r [fm]\tG_omega(r,r) [MeV^-1 fm^-3]\tG_Coul(r,r) [MeV^-1 fm^-3]\n")
            for ir, r in enumerate(rs):
                o_g.write(f"{r}\t{Gtrap[ir][ir]}\t{Gcoul[ir][ir]}\n")
            o_g.write("\n")


        sys.stdout.flush()

    result.append( [ cutoff, energies, omegas, phshs, eres, flags_G_C, flags_G_CT, G_Cs, G_CTs] )

############
## OUTPUT ##
############

o_p.write("# CUTOFF\tENERGY [MeV]\tOMEGA [MeV]\tPHSH [deg]\tERE [fm^-(2l+1)]\tFlag G_C\tFlag G_CT\tG_C(rmin,rmin) [MeV^-1 fm^-3]\t G_CT(rmin,rmin) [MeV^-1 fm^-3]\n")
for i in range( len(result)):
    for j in range( len(result[i][1] ) ): 

        string = str(result[i][0])
        for k in range( 1, len(result[i] ) ):
            string += "\t" + str(result[i][k][j]) 
            
        o_p.write( string + "\n")  

