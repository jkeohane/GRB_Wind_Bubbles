# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from VegasAfterglow import TophatJet, Observer, Radiation, Model, Medium, Wind
from os import makedirs
import astropy as ap

# ---------- setup ----------
makedirs("assets", exist_ok=True)

debug = True

# constants (cgs)
pi   = np.pi
mp   = 1.67262192369e-24  #  g
mu   = 1.3
m_mol = mu * mp
kb   = 1.3806488e-16  # erg/K
day  = 86400.0   # s
year = 3.15576e7 # s
Msun = 1.98847e33 # g
pc_to_cm = 3.086e18 # cm

# ---------- PARAMETERS----------
E_iso = 3.97485e+51
Gamma0 = 307.554
theta_c = 0.101333
eps_e = 0.730569
eps_B = 0.121648
p = 2.26557
R_t = 3.17642e+17
log10_n_t = -1.19935
log10_n_ism = -1.91702

n_ism   = 10**(log10_n_ism)          # cm^-3
n_t = 10**(log10_n_t)             # Density just inside the termination shock

z = 0.1
lumi_dist_cm = 1e+26
theta_obs = 0.0
mu = 1.3
nwalkers = 64
burn = 1000
nsteps = 3000
num_nu_grid = 32
#  ------------------------------------------------------------------

def rho(phi, theta, r):
    """
    Returns: rho(phi,theta,r) [g/cm^3],
    Very simple shell assuming it is mixed
    """
    # Parameters to be fit
    global R_t, n_t, n_ism
    n_sh = max(4 * n_t, 4 * n_ism)  ## strong shock factor of 4 in density & guarantees n_sh > n_ism
    # region boundary
    # R1 would be R_t   R2 us outside of shell
    R2 = R_t * ((n_sh + 3*n_t) / (n_sh - n_ism)) ** 0.333  # shell outer --> ISM
    # Notice the limits.  Very important that it is continuous if n_t = n_ism
    #   if n_t > n_ism then:    R2 = ( 7 n_t )/(4 n_t - n_ism)  R_t
    #   if n_t >> n_ism   R2 ~ (7/4)R_t  + (7/16) (n_ism/n_t)
    #   if n_t >>> n_ism        R2 ~ (7/4)R_t    ### 7/4 = 1.75
    #   if n_t >== n_ism        R2 ~ (7/3)R_t    ### 7/3 = 2.33
    #   if n_t <== n_ism        R2 ~ (7/3)R_t --- CHECK!!!  No jumps in parameter space
    #   if n_t < n_ism then:    R2 = ( (4/3) + n_t/n_ism) R_t
    #   if n_t << n_ism then:   R2 ~ (4/3) R_t  ### 4/3 = 1.33
    #   Notice the bubble is biggest, compared to R_t, if n_t and n_ism are near each other
    #   Also, this makes the most physical sense if n_t > n_ism.
    #   Remember the model assumes n_ism kT_ism = m_p n_t v_wind**2 at R_t
    #   This model assumes that the same amount of mass that is in the wind,
    #   is also in the shell.
    #   After radiative cooling occurs, the mass of the shell will stay the
    #   same, but it will become thinner.  Also, the WR stage would probably have started
    #   and we probably would have a superwind.

    ## convert to density in cgs
    rho_t = n_t * m_mol
    rho_sh = n_sh * m_mol
    rho_ism = n_ism * m_mol

    if debug:
        meta = {"R_sh": 0.5 * (R2 - R_t), "dR": R2 - R_t, "n_t": n_t,
                "rho_t": rho_t, "n_shell": n_sh,
                "rho_shell": rho_sh, "n_ism": n_ism, "rho_ism": rho_ism}
        print(meta)

    # free wind
    if r < R_t:
        return  rho_t * (R_t/r)**2
    # thin shell
    elif r < R2:
        return rho_sh
    # ism
    else:
        return rho_ism

# build one medium (tweak parameters here as desired)

bubble = Medium(rho=rho)
A = m_mol*n_t*R_t*R_t  # in cgs  ## VA assumes a mu of 1.3 I guess depends on the helium too
wind = Wind(A_star=A/5E11)
wind_2 = Wind(A_star=A/5E11,n_ism=n_ism*mu)  ## Vegas Afterglow is not consistent on mean molecular weight

# ---------- model (jet/observer/radiation) ----------

jet = TophatJet(theta_c=theta_c, E_iso=E_iso, Gamma0=Gamma0)
obs = Observer(lumi_dist=lumi_dist_cm, z=z, theta_obs=0)
rad = Radiation(eps_e=eps_e, eps_B=eps_B, p=p)

models = []  ; model_names = []
model = Model(jet=jet, medium=bubble, observer=obs, fwd_rad=rad)
model_names.append("Simple Bubble")
models.append(model)
model = Model(jet=jet, medium=wind, observer=obs, fwd_rad=rad)
model_names.append("Simple Wind")
models.append(model)
model = Model(jet=jet, medium=wind_2, observer=obs, fwd_rad=rad)
model_names.append("Stratified Wind")
models.append(model)

# ---------- density profile plot with twin axes ----------
# ---------- density profile from the model's medium ----------

r = np.logspace(16, 20, 600)  # cm

# vectorized evaluation using the model's medium (returns g/cm^3)

#rho_profile = np.array([medium_rho(model, 0.0, 0.0, ri) for ri in r])

fig, ax1 = plt.subplots(figsize=(5, 3.6), dpi=200)

for i,model in enumerate(models):
    rho_profile = np.asarray(model.medium(0.0, 0.0, r), dtype=float)
    # convert to number density if you want n(r)
    n_profile = rho_profile / (m_mol)  # cm^-3
    ax1.loglog(r / pc_to_cm, n_profile, lw=1.5, label=model_names[i],
               alpha=0.6/(i+1))


ax1.set_xlabel('Radius (pc)')
ax1.set_ylabel(r'n(r) [cm$^{-3}$]')
ax1.set_title('Model Medium Density')
ax1.legend(fontsize=6)
# top x-axis in cm
def pc_to_cm_f(x):  return x * pc_to_cm
def cm_to_pc_f(x):  return x / pc_to_cm
ax2 = ax1.secondary_xaxis('top', functions=(pc_to_cm_f, cm_to_pc_f))
ax2.set_xlabel('Radius (cm)')

# right y-axis in ρ
def n_to_rho(y):  return y * (m_mol)
def rho_to_n(y):  return y / (m_mol)
ax3 = ax1.secondary_yaxis('right', functions=(n_to_rho, rho_to_n))
ax3.set_ylabel(r'$\rho(r)$ [g cm$^{-3}$]')

plt.tight_layout()
plt.savefig("assets/density_profile.png", dpi=300)
plt.show()


# ---------- light curves (multi-band) ----------
# ---------- combined multi-band light curves with dual x- and y-axes ----------
times = np.logspace(2, 8, 200)           # seconds
times_days = times / day
bands = np.array([1e9, 1e14, 1e17])      # Hz (radio, optical, X-ray)
band_names = ["Radio", "Optical", "X-ray"]
lcs = []
for model in models:
    lcs.append( model.flux_density_grid(times, bands) ) # erg cm^-2 s^-1 Hz^-1

fig, ax1 = plt.subplots(figsize=(5.8, 3.8), dpi=200)

i =0
# --- main plot (bottom axis: seconds, flux in Jy) ---
for j, (name, nu) in enumerate(zip(band_names, bands)):
    exp = int(np.log10(nu))
    for k, lc in enumerate(lcs):
        label = fr'{name} ($10^{{{exp}}}$ Hz)' + " " + str(model_names[k])
        if model_names[k] == "Simple Bubble":
            alpha = 0.75
        else:
            alpha
        ax1.loglog(times, lc.total[j, :] * 1e23,
                   color=f'C{3*j+k}', lw=1.6, label=label, alpha=alpha)
        i = i+1

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Flux Density (Jy)')
ax1.set_title('Afterglow Light Curves in Multiple Bands')

# --- legend (LaTeX formatted) ---
ax1.legend(ncol=len(bands), fontsize=3)

# --- top axis: time in days ---
def s_to_days(x):  return x / day
def days_to_s(x):  return x * day
ax2 = ax1.secondary_xaxis('top', functions=(s_to_days, days_to_s))
ax2.set_xlabel('Time (days)')

# --- right y-axis: flux density in cgs (erg/cm^2/s/Hz) ---
def Jy_to_cgs(y):  return y * 1e-23
def cgs_to_Jy(y):  return y / 1e-23
ax3 = ax1.secondary_yaxis('right', functions=(Jy_to_cgs, cgs_to_Jy))
ax3.set_ylabel(r'Flux Density [erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$]')

plt.tight_layout()
plt.savefig('assets/lightcurves_all_bands.png', dpi=300)
plt.show()

###
# ---------- spectra at selected epochs (all on one figure) ----------
frequencies = np.logspace(5, 22, 300)    # Hz
epochs      = np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])  # s
epochs_days = epochs / day

# Compute spectral evolution: Fν(ν, t)
spec_grids = []
for model in models:
    spec_grids.append(model.flux_density_grid(epochs, frequencies)  ) # shape ≈ [len(freq), len(time)]

# --- Plot all epochs on one figure ---
plt.figure(figsize=(5.5, 3.8), dpi=200)
colors = plt.cm.plasma(np.linspace(0, 1, 3*len(epochs)))
# Mark radio / optical / X-ray bands
for k, nu in enumerate(bands):
    plt.axvline(nu, ls='--', color=f'C{k}', alpha=0.6)

for j, tsec in enumerate(epochs):
    # nice epoch label
    exp = int(np.floor(np.log10(tsec)))
    base = tsec / 10**exp
    if np.isclose(base, 1.0):
        label = fr'$10^{{{exp}}}\,\mathrm{{s}}$'
    else:
        label = fr'${base:.1f}\times10^{{{exp}}}\,\mathrm{{s}}$'
    for k, spec_grid in enumerate(spec_grids):
        new_label = label + ' ' + model_names[k]
        plt.loglog(frequencies, spec_grid.total[:, j] * 1e23,
                   color=colors[3*j+k], lw=1.5, label=new_label, alpha=0.3)

# Labels and legend
plt.xlabel('Frequency (Hz)')
plt.ylabel('Flux Density (Jy)')
plt.title('Synchrotron Spectra at Multiple Epochs')
plt.legend(ncol=len(epochs), fontsize=(3*7/len(epochs)), loc='lower center')

plt.tight_layout()
plt.savefig('assets/spectra_all_epochs.png', dpi=300)
plt.show()
