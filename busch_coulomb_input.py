'''
    Complementary input handling for the script "busch_coulomb.py"
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

import json
import argparse
import numpy as np


#________________________________________________________________________________
def create_parser():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_file", type=str, required=False,
                        help="File path of the json inputs. Absolute or relative. If given, all the other inputs are going to be ignored.", default="")

    parser.add_argument("-ma", "--main_file", type=str, required=False,
                        help="File path of the SVM results. Absolute or relative.", default="")
    parser.add_argument("-t", "--threshold", type=str, required=False,
                        help="File path of the SVM results for the threshold. Absolute or relative.", default="")

    parser.add_argument("-m", "--mass", type=float, required=False,
                        help="Mass (in units of unit_mass parameter) of the core", default=1)
    parser.add_argument("-fm", "--mass_fragment", type=float, required=False,
                        help="Mass (in units of unit_mass parameter) of the fragment", default=1)
    parser.add_argument("-um", "--unit_mass", type=float, required=False,
                        help="All masses will be multiplied by this value", default= 938.918)

    parser.add_argument("-c", "--charge", type=float, required=False,
                        help="Charge (in units of e) of the core", default=1)
    parser.add_argument("-fc", "--charge_fragment", type=float, required=False,
                        help="Charge (in units of e) of the fragment", default=1)

    parser.add_argument("-l", "--lang", type=int, required=False,
                        help="Angular momentum of the system", default=0)

    parser.add_argument("-im", "--max_iters", type=int, required=False,
                        help="Maximum numbers of iterations", default=1000)

    parser.add_argument("-mt", "--method", type=int, required=False,
                        help="Method to calculate Green functions (1 Leuville-Neumann series, any other number matrix inversion)", default=0)
    parser.add_argument("-fd", "--fit_diagonal", type=int, required=False,
                        help="Wether to calculate the limit with a fit or to compute it numerically with the lowest point.", default=0)

    parser.add_argument("-np", "--no_of_points", type=int, required=False,
                        help="Total number of points until r max", default=200)
    parser.add_argument("-q", "--ratio", type=float, required=False,
                        help="Incremental factor in the step", default=1.05)
    parser.add_argument("-rmi", "--r_min", type=float, required=False,
                        help="Initial value of the point grid", default= 0.001)
    parser.add_argument("-rm", "--r_max_fact", type=float, required=False,
                        help="Factor that determines the endpoint in fm of the r and r' range, such that r_max = b*r_max_factor", default=30.)

    parser.add_argument("-e", "--eta", type=float, required=False,
                        help="Mixing variable between the old and new Green f. in the Dyson eq.", default=1.)
    parser.add_argument("-d", "--delta", type=float, required=False,
                        help="Precision for the Green function to exit the loop", default=1.e-8)

    parser.add_argument("-pl", "--print_lines", type=int, required=False,
                        help="The convergence status is printed every print_lines iterations", default= 5)

    parser.add_argument("-op", "--output_phsh", type=str, required=False,
                        help="File path of the output of the phase shifts computed. Absolute or relative.", default= "busch_coulomb_phsh.txt")

    parser.add_argument("-pg", "--print_green", type=bool, required=False,
                        help="Whether to print or not the Green function", default= False)
    parser.add_argument("-og", "--output_green", type=str, required=False,
                        help="File path of the output of the diagonal Green functions. Absolute or relative.", default= "busch_coulomb_green.txt")

    return parser

#_______________________________________________________________________
def read_parser_file(infile):

    # returns JSON object as a dictionary
    indata = json.load(open(infile))

    main_file = indata['input_output']['main_file']
    threshold_file = indata['input_output']['threshold_file']
    
    outfile_phsh = indata['input_output']['output_phase_shifts']

    print_lines = indata['input_output']['print_lines']

    print_green = indata['input_output']['print_green']
    outfile_green = indata['input_output']['output_green']

    unit_mass = indata['physics']['unit_mass']

    mass_threshold = indata['physics']['core_mass'] * unit_mass           
    mass_fragment = indata['physics']['fragment_mass'] * unit_mass

    Z1 = indata['physics']['core_charge']
    Z2 = indata['physics']['fragment_charge']

    l = indata['physics']['angular_momentum']

    method = indata['numerics']['method']
    fit_diagonal = indata['numerics']['fit_diagonal']

    NO_OF_ITERS = indata['numerics']['max_iterations']

    # range of r
    N = indata['numerics']['no_of_points']
    q = indata['numerics']['q']
    r_min = indata['numerics']['r_min']
    max_factor = indata['numerics']['max_radius_factor']

    mix_eta = indata['numerics']['eta']
    if mix_eta <= 0:
        mix_eta = 0.1
    if mix_eta > 1.:
        mix_eta = 1.

    delta = indata['numerics']['delta']

    return main_file, threshold_file, outfile_phsh, print_lines, print_green, outfile_green, unit_mass, mass_threshold, mass_fragment, Z1, Z2, l, method, fit_diagonal, NO_OF_ITERS, N, q, r_min, max_factor, mix_eta, delta
           
#_______________________________________________________________________
def read_parser_args(args):

    main_file = args.main_file
    threshold_file = args.threshold
    
    unit_mass = args.unit_mass

    mass_threshold = args.mass * unit_mass           
    mass_fragment = args.mass_fragment * unit_mass
    
    l = args.lang

    Z1 = args.charge
    Z2 = args.charge_fragment

    NO_OF_ITERS = args.max_iters
    
    method = args.method
    fit_diagonal = args.fit_diagonal

    # range of r
    q = args.ratio
    r_min = args.r_min
    max_factor = args.r_max_fact
    N = args.no_of_points

    mix_eta = args.eta
    if mix_eta <= 0:
        mix_eta = 0.1
    if mix_eta > 1.:
        mix_eta = 1.

    delta = args.delta

    outfile_phsh = args.output_phsh

    print_lines = args.print_lines

    print_green = args.print_green
    outfile_green = args.output_green

    return main_file, threshold_file, outfile_phsh, print_lines, print_green, outfile_green, unit_mass, mass_threshold, mass_fragment, Z1, Z2, l, method, fit_diagonal, NO_OF_ITERS, N, q, r_min, max_factor, mix_eta, delta

#_______________________________________________________________________
def read_files(main_file, threshold_file):

    main= []
    threshold = []

    with open(main_file, 'r') as minput:
        lines = minput.readlines()

        for line in lines:
            if line[0] != '#' and not line.isspace(): # ignores the header
                main.append([float(v) for v in line.split()])

    main = sorted(main, key=lambda x:(float(x[0]),x[1]) ) # sorting for cutoff and then traplength


    if len(threshold_file) > 0:
        with open(threshold_file, 'r') as tinput: # exactly the same of before
            lines = tinput.readlines()

            for line in lines:
                if line[0] != '#' and not line.isspace():
                    threshold.append([float(v) for v in line.split()])


        threshold = sorted(threshold, key=lambda x:(float(x[0]),x[1]) )

    # checking that main and threshold match on cutoff and omega after ordering
    if len(threshold) > 0:
        for m, t in zip(main, threshold): # those are lists [cutoff, omega, energy]

            # checking same cutoff
            if (m[0] != t[0] ):
                print(m[0], t[0])
                print("Not same cutoff for threshold")
                exit()

            # checking same omega
            if (m[1] != t[1] ):
                print(m[1], t[1])
                print("Not same omega for threshold")
                exit()

    # Reshaping data lists in a more comfortable way
    cutoffs, cut_index, cut_counts = np.unique( [ v[0] for v in main ], return_index = True, return_counts = True )
    main = list(map(list, zip(*main)))
    new_main = []

    for cut, idx, count in zip(cutoffs, cut_index, cut_counts):

        temp = [ cut,  main[1][idx:idx+count], main[2][idx:idx+count]] 
        new_main.append( temp )

    main = new_main

    if len(threshold) > 0:
        cutoffs, cut_index, cut_counts = np.unique( [ v[0] for v in threshold ], return_index = True, return_counts = True )
        threshold = list(map(list, zip(*threshold)))
        new_threshold = []

        for cut, idx, count in zip(cutoffs, cut_index, cut_counts):
            new_threshold.append( [ cut,  threshold[1][idx:idx+count], threshold[2][idx:idx+count] ] )

        threshold = new_threshold

    return main, threshold
    