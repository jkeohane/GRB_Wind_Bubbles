import numpy as np
import matplotlib.pyplot as plt
from VegasAfterglow import TophatJet, Observer, Radiation, Model, Medium
import os
#make directory if needed
os.makedirs("assets", exist_ok=True)
pi = 3.1415926

## Global constants
mp = 1.67e-24 # proton mass in gram
year = 3.16E7 # seconds
Msun = 1.989e33 # grams
kb = 1.3806488e-16

## Global fit parameters
M_dot = 1E-5 * Msun/year # g/s
v_w = 1E8 # cm/s
n_ism = 1 # per cm^3
P_ism = 10**4.5 * kb  ### LMC Pressures
f_sh = 0.05 ## fraction of bubble size of the shell.

## The termination shock radius
rho_ism = n_ism*mp
R_t =(M_dot * v_w /(4*pi*P_ism ))**2
dR = f_sh*R_t
A_star = M_dot/(4*pi*v_w)

# Define a custom density profile function
def density(phi, theta, r):# r in cm, phi and theta in radians [scalar]
    global A_star, v_w, rho_ism, R_t,dR
    if r < R_t:
        rho = A_star * r**-2
    elif r < R_t + dR:
        rho = (R_t*rho_ism)/(3*dr)
    else:
        rho = rho_ism
    return rho
    #return whatever density profile (g*cm^-3) you want as a function of phi, theta, and r

# Create a user-defined medium
medium = Medium(rho=density)

# 2. Configure the jet structure (top-hat with opening angle, energy, and Lorentz factor)
jet = TophatJet(theta_c=0.1, E_iso=1e52, Gamma0=300) # in cgs unit

# 3. Set observer parameters (distance, redshift, viewing angle)
obs = Observer(lumi_dist=1e26, z=0.1, theta_obs=0) # in cgs unit

# 4. Define radiation microphysics parameters
rad = Radiation(eps_e=1e-1, eps_B=1e-3, p=2.3)

## Now make a list of models with different ISM densities
## Now make a list of models with different ISM densities

#Light Curve Calculation

# 1. Create logarithmic time array from 10² to 10⁸ seconds (100s to ~3yrs)
times = np.logspace(2, 8, 200)
times_days = times/86400

# 2. Define observing frequencies (radio, optical, X-ray bands in Hz)
bands = np.array([1e9, 1e14, 1e17])

band_names = ["Radio", "Optical", "X-ray"]

n_isms = np.array([1,10,100,1000,1E4]) # atoms/cc

results = []
for n_ism in n_isms:
    rho_ism = n_ism * mp
    medium = Medium(rho=density)
    model = ( Model(jet=jet, medium=medium, observer=obs, fwd_rad=rad) )
    results.append( model.flux_density_grid(times, bands) ) # cgs units

# print(len(mediums)) ; print(len(n_isms));print(len(models)) ; print(len(results))
# 4. Visualize the multi-wavelength light curves


# 5. Plot each frequency band as a different plot
for j, band_name in enumerate(band_names):
    plt.figure(figsize=(4.8, 3.6), dpi=200)
    for i, n_ism in enumerate(n_isms):
        exp = int(np.floor(np.log10(n_ism)))
        base = n_ism / 10 ** exp
        if base == 1.0:
           label = fr'$10^{{{exp}}}\,\mathrm{{cm}}^{{-3}}$'
        else:
            label = fr'${base:.1f} \times 10^{{{exp}}}\,\mathrm{{cm}}^{{-3}}$'
        plt.loglog(times_days, results[i].total[j,:]*1E23, label=label)

    plt.xlabel('Time (days)')
    plt.ylabel('Flux Density (Jy)')
    plt.legend()
    plt.title('Constant ISM Density ' + band_name + ' Light Curves')
    plt.tight_layout()
    plt.savefig('assets/'+ band_name + '-lc.png', dpi=300)
    plt.show()   # adds an interactive plot window

quit()

### Make  spectral plots

# 1. Define broad frequency range (10⁵ to 10²² Hz)
frequencies = np.logspace(5, 22, 200)

# 2. Select specific time epochs for spectral snapshots
epochs = np.array([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])  # in seconds
epochs_days = epochs/86400

# 3. Calculate spectra at each epoch and density
results = []
for i, model in enumerate(models):
    results.append( model.flux_density_grid(epochs, frequencies) )
 # 4. Plot broadband spectra at each density
    plt.figure(figsize=(4.8, 3.6),dpi=200)
    colors = plt.cm.viridis(np.linspace(0,1,len(n_isms)))

for j, epoch in enumerate(epochs):

    for i, n_ism in enumerate(n_isms):
         ### to make the labels
        exp = int(np.floor(np.log10(n_ism)))
        base = n_ism / 10 ** exp
        if base == 1.0:
           label = fr'$10^{{{exp}}}\,\mathrm{{cm}}^{{-3}}$'
        else:
            label = fr'${base:.1f} \times 10^{{{exp}}}\,\mathrm{{cm}}^{{-3}}$'

        plt.loglog(frequencies, results[i].total[:,j], color=colors[i], label=label)

        # 5. Add vertical lines marking the bands from the light curve plot
        for k, band in enumerate(bands):
            plt.axvline(band, ls='--', color=f'C{k}')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Flux Density (erg/cm²/s/Hz)')
    plt.legend(ncol=2)
    exp = int(np.floor(np.log10(epoch)))
    base = epoch / 10 ** exp
    if base == 1.0:
        label = fr'$10^{{{exp}}}\,\mathrm{{s}}$'
    else:
        label = fr'${base:.1f} \times 10^{{{exp}}}\,\mathrm{{s}}$'

    plt.title(r'Synchrotron Spectra at '+ label)
    plt.tight_layout()
    plt.savefig('assets/'+ str(exp) +'-spec.png', dpi=300)
    plt.show()

quit()
