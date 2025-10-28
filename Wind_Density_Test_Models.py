import numpy as np
import matplotlib.pyplot as plt
from VegasAfterglow import ISM, Wind, TophatJet, Observer, Radiation, Model
import os
#make directory if needed
os.makedirs("assets", exist_ok=True)

## Initial Parameter Definitions

# 2. Configure the jet structure (top-hat with opening angle, energy, and Lorentz factor)
jet = TophatJet(theta_c=0.1, E_iso=1e52, Gamma0=300) # in cgs unit

# 3. Set observer parameters (distance, redshift, viewing angle)
obs = Observer(lumi_dist=1e26, z=0.1, theta_obs=0) # in cgs unit

# 4. Define radiation microphysics parameters
rad = Radiation(eps_e=1e-1, eps_B=1e-3, p=2.3)

## Now make a list of models with different ISM densities

n_isms = np.array([1,10,100,1000,1E4,1E5]) # atoms/cc

mediums = []
for n_ism in n_isms:
    mediums.append(Wind(A_star=1.0,n_ism=n_ism) )

models = []
for medium in mediums:
    models.append( Model(jet=jet, medium=medium, observer=obs, fwd_rad=rad) )

#Light Curve Calculation

# 1. Create logarithmic time array from 10² to 10⁸ seconds (100s to ~3yrs)
times = np.logspace(2, 8, 200)
times_days = times/86400

# 2. Define observing frequencies (radio, optical, X-ray bands in Hz)
bands = np.array([1e9, 1e14, 1e17])

band_names = ["Radio", "Optical", "X-ray"]

# 3. Calculate the afterglow emission at each time and frequency
results = []
for i, model in enumerate(models):
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
    plt.title('Wind and ISM Density ' + band_name + ' Light Curves')
    plt.tight_layout()
    plt.savefig('assets/Wind_ISM_'+ band_name + '-lc.png', dpi=300)
    plt.show()   # adds an interactive plot window


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

        plt.loglog(frequencies, results[i].total[:,j]*1E23, color=colors[i], label=label)

        # 5. Add vertical lines marking the bands from the light curve plot
        for k, band in enumerate(bands):
            plt.axvline(band, ls='--', color=f'C{k}')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Flux Density (Jy)')
    plt.legend(ncol=2)
    exp = int(np.floor(np.log10(epoch)))
    base = epoch / 10 ** exp
    if base == 1.0:
        label = fr'$10^{{{exp}}}\,\mathrm{{s}}$'
    else:
        label = fr'${base:.1f} \times 10^{{{exp}}}\,\mathrm{{s}}$'
    if epoch < 60:
        title = 'Synchrotron Spectra at ' + label
    elif epoch < 60*60*24:
        title = 'Synchrotron Spectra at ' + label + \
                ' (~' + str(int(10**round(np.log10(epoch/60), 0))) + ' min)'
    elif epoch < 2.16E7:
        title = 'Synchrotron Spectra at ' + label + \
                ' (~' + str(int(10 ** round(np.log10(epochs_days[j]), 0))) + ' days)'
    else:
        title = 'Synchrotron Spectra at ' + label + \
                ' (~' + str(int(10 ** round(np.log10(epochs_days[j]/365.25), 0))) + ' years)'

    plt.title(title)
    plt.tight_layout()
    plt.savefig('assets/Wind_ISM'+ str(exp) +'-spec.png', dpi=300)
    plt.show()

quit()
