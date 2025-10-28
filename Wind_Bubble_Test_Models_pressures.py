
import numpy as np
import matplotlib.pyplot as plt
from VegasAfterglow import TophatJet, Observer, Radiation, Model, Medium
import os

# ---- constants (cgs) ----
pi   = 3.141592653589793
mp   = 1.67262192369e-24
kb   = 1.3806488e-16
year = 3.15576e7
Msun = 1.98847e33

# ---- WR wind defaults ----
M_dot = 1e-5 * Msun / year      # g/s
v_w   = 1e8                     # cm/s  (≈ 1000 km/s; use 2e8 for 2000 km/s if you like)

# ---- ambient baseline ----
n_ism = 100                    # cm^-3  (you vary P_ism below; n_ism fixes rho_ism)
rho_ism = n_ism * mp
f_sh  = 0.05                    # fractional shell thickness (ΔR = f_sh * R_b or here: f_sh * R_t as your code assumed)

# helpful constant for free wind region
A_star = M_dot / (4.0 * pi * v_w)  # g/cm

# make output folder
os.makedirs("assets", exist_ok=True)

# --- build a *new* density function for each model ---
def make_density_callable(P_ism, use_shell=True):
    """
    Return a rho(phi, theta, r) function (g/cm^3) that encodes this model's R_t and shell density.
    Late-time, pressure-confined simplification:
      - free wind for r < R_t
      - thin 'shell' of thickness dR just outside R_t (your original placement),
        with density set to produce a robust column
      - ambient ISM beyond
    """
    global A_star, v_w, rho_ism, R_t, dR
    # Correct termination shock:
    R_t = (M_dot * v_w / (4.0 * pi * P_ism))**0.5

    # If you want the shell *at R_t* (as your code did):
    dR = f_sh * R_t
    # robust column idea (normally uses R_b, but if you place shell at R_t, keep it consistent)
    # N_sh ≈ (1/3) * rho_ism * R_t  ->  rho_shell ≈ N_sh / dR
    # This just gives you a geometrically thin, denser zone.
    if use_shell:
        rho_shell = ( (1.0/3.0) * rho_ism * R_t ) / dR
    else:
        rho_shell = rho_ism  # no special shell

    def density(phi, theta, r):
        # Expecting scalar r; if VA passes arrays later, you can vectorize easily
        if r < R_t:
            return A_star * r**-2
        elif r < R_t + dR and use_shell:
            return rho_shell
        else:
            return rho_ism
    return density

# ---- jet, observer, radiation ----
jet = TophatJet(theta_c=0.1, E_iso=1e52, Gamma0=300)
obs = Observer(lumi_dist=1e26, z=0.1, theta_obs=0)
rad = Radiation(eps_e=1e-1, eps_B=1e-3, p=2.3)

# ---- time and bands ----
times = np.logspace(2, 8, 200)          # s
times_days = times / 86400.0
bands = np.array([1e9, 1e14, 1e17])     # Hz
band_names = ["Radio", "Optical", "X-ray"]

# ---- sweep in ambient pressure (K cm^-3 * k_B) ----
P_isms = (10.0**np.arange(4, 6)) * kb   # 1e2 .. 1e9 K cm^-3 * k_B

# Radius grid: 10^15–10^20 cm (~0.003–100 pc)
r = np.logspace(16, 22, 600)
results = []
rho_profiles = []
R_ts = []
for P_ in P_isms:
    rho_fn = make_density_callable(P_)
    print(R_t)
    R_ts.append(R_t)
    medium = Medium(rho=rho_fn)
    model = Model(jet=jet, medium=medium, observer=obs, fwd_rad=rad)
    results.append(model.flux_density_grid(times, bands))
    rho_profiles.append(np.array([rho_fn(0, 0, ri) for ri in r]))



plt.figure(figsize=(5,3.6), dpi=200)

colors = plt.cm.viridis(np.linspace(0, 1, len(P_isms)))  # same colormap for consistency

for i, rho_profile in enumerate(rho_profiles):
    n = rho_profile / mp
    P_ = P_isms[i]
    exp = int(np.floor(np.log10(P_ / kb)))
    base = (P_ / kb) / 10 ** exp
    if np.isclose(base, 1.0):
        label = fr'$10^{{{exp}}}\,\mathrm{{K\,cm}}^{{-3}}$'
    else:
        label = fr'${base:.1f}\!\times\!10^{{{exp}}}\,\mathrm{{K\,cm}}^{{-3}}$'

    plt.loglog(r / 3.086e18, n, color=colors[i], lw=1.3, label=label)
    plt.axvline(R_ts[i]/ 3.086e18, ls='--', color=f'C{i}')



plt.xlabel('Radius (pc)')
plt.ylabel(r'n(r) [cm$^{-3}$]')
plt.title('Late-time WR Bubble Density Profiles')
plt.legend(ncol=2, fontsize=7)
plt.tight_layout()
plt.show()

# ---- plotting ----
for j, band_name in enumerate(band_names):
    plt.figure(figsize=(4.8, 3.6), dpi=200)
    for i, P_ in enumerate(P_isms):
        exp = int(np.floor(np.log10(P_ / kb)))
        base = (P_ / kb) / 10 ** exp
        if np.isclose(base, 1.0):
            label = fr'$10^{{{exp}}}\,\mathrm{{K\,cm}}^{{-3}}$'
        else:
            label = fr'${base:.1f}\times10^{{{exp}}}\,\mathrm{{K\,cm}}^{{-3}}$'
        plt.loglog(times_days, results[i].total[j, :] * 1e23, label=label)

    plt.xlabel('Time (days)')
    plt.ylabel('Flux Density (Jy)')
    plt.legend(ncol=2, fontsize=7)
    plt.title('Wind Bubble ' + band_name + ' Light Curves')
    plt.tight_layout()
    plt.savefig(f'assets/{band_name}-lc.png', dpi=300)
    plt.show()



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
