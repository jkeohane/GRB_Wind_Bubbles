import numpy as np
import matplotlib.pyplot as plt
from VegasAfterglow import ISM, TophatJet, Observer, Radiation, Model
import os
#make directory if needed
os.makedirs("assets", exist_ok=True)

## Initial Parameter Definitions

# 1. Define the circumburst environment (constant density ISM)
medium = ISM(n_ism=1) # in cgs unit

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
    mediums.append( ISM(n_ism=n_ism) )

models = []
for medium in mediums:
    models.append( Model(jet=jet, medium=medium, observer=obs, fwd_rad=rad) )


#Light Curve Calculation

# 1. Create logarithmic time array from 10² to 10⁸ seconds (100s to ~3yrs)
times = np.logspace(2, 8, 200)

# 2. Define observing frequencies (radio, optical, X-ray bands in Hz)
#bands = np.array([1e9, 1e14, 1e17])
# optical only
bands = np.array([1E14])

# 3. Calculate the afterglow emission at each time and frequency
results = []
for i, model in enumerate(models):
    results.append( model.flux_density_grid(times, bands) ) # cgs units

print(len(mediums)) ; print(len(n_isms));print(len(models)) ; print(len(results))
# 4. Visualize the multi-wavelength light curves
plt.figure(figsize=(4.8, 3.6), dpi=200)

# 5. Plot each frequency band
for i, n_ism in enumerate(n_isms):
    exp = int(np.floor(np.log10(n_ism)))
    base = n_ism / 10 ** exp
    if base == 1.0:
       label = fr'$10^{{{exp}}}\,\mathrm{{cm}}^{{-3}}$'
    else:
        label = fr'${base:.1f} \times 10^{{{exp}}}\,\mathrm{{cm}}^{{-3}}$'
    plt.loglog(times, results[i].total[0,:]*1E23, label=label)

plt.xlabel('Time (s)')
plt.ylabel('Flux Density (Jy)')
plt.legend()
plt.title('Constant ISM Density Optical Light Curves')
plt.tight_layout()
plt.savefig('assets/quick-lc.png', dpi=300)
plt.show()   # adds an interactive plot window

quit()
