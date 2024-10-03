# BERW-Coulomb
See the paper "Accurate calculation of low energy scattering phase shifts of charged particles in a harmonic oscillator trap" by M. Bagnarol, N. Barnea, M. Rojik and M. Schafer 

## BUSCH FORMULA WITH COULOMB APPLICATION  
### Written by Mirko Bagnarol, Nir Barnea and Martin Schafer

For a direct application see the paper Accurate calculation of low energy scattering phase shifts of charged particles in a harmonic oscillator trap

Type "python3 busch_coulomb.py -h" for info on each input

Either fill a .json input file and run as:

python3 busch_coulomb.py -i /path/to/input.json

Or use the script in a similar fashion of a bash comand, such as:

python3 busch_coulomb.py -ma /path/to/main [-t /path/to/threshold] [-m core_mass] [-fm fragment_mass] [-um unit_mass] [-c core_charge] [-fc fragment_charge] [-l angular_momentum] [-im max_iterations] [-mt method] [-fd fit_diagonal] [-np grid_points] [-q points_ratio] [-rmi minimum_r] [-rm rmax_factor] [-e eta] [-d delta] [-pl print_lines] [-op /path/to/output] [-pg print_green] [-og /path/to/green_output]

The input must be in three columns, being
1) Cutoff (or any number differentiating batches of data)
2) Omega  in MeV
3) Energy in MeV

The masses must be in multiples of the unit mass parameter and the charges in multiples of the elementary charge.
In order to insert the masses directly just set unit_mass to 1. 

hbar and e^2 are hard coded as\
hbarc = 197.3269804  MeV fm\
e2    = 1.4399764    MeV fm\
If you want to change them, just modify the first lines of busch_coulomb_utility.py

The grid is defined such that R_max = b * rmax_factor and such that r_n = r_(n-1) + step * points_ratio^n, with r_0 = minimum_r. Avoid choosing less than 1e-3 for rmin.
Step is calculated such that after grid_points we reach R_max. For equidistant points just choose points_ratio = 1.

The script has two methods\
any number =/= 1 -> Direct inversion, being quick and precise, converging everywhere but in the poles E/omega = l + 1.5 + 2n, n natural number\
1 -> Leuville-Neumann series, which solves the Dyson equation iteratively.\
It does not converge for positive phase shifts after the poles E/omega = l + 1.5 + 2*n, n natural number, and for very low energies (E << 0.1 MeV)\
It uses the parameters eta and delta, defined here:

Eta is the mixing parameter between the old Green function and the new green function.\
eta ~ 1 -> fast convergence, unstable numerically\
eta ~ 0 -> slow convergence, stable numerically

Each point is considered converged when Int dr |Green_n (r, r) - Green_(n-1) (r, r)| < delta for both Green functions (Trap and Coulomb). \
Delta should be at least 1e-3 or lower.

The output is going to be 9 columns:\
1 - CUTOFF\
2 - ENERGY 			[MeV]\
3 - OMEGA  			[MeV]\
4 - PHSH   			[deg]\
5 - EFFECTIVE RANGE EXPANSION   [fm^-(2l+1)]\
6 - FLAG G_COULOMB\
7 - FLAG G_COULOMB,TRAP\
8 - G_COULOMB      (RMIN,RMIN)  [MeV^-1 fm^-3]\
9 - G_COULOMB,TRAP (RMIN,RMIN)  [MeV^-1 fm^-3]
