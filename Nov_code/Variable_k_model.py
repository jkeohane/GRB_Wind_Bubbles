# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from VegasAfterglow import TophatJet, Observer, Radiation, Model, Medium, Wind
from os import makedirs

# ---------- setup ----------
makedirs("assets", exist_ok=True)

# constants (cgs)
pi   = np.pi
mp   = 1.67262192369e-24
kb   = 1.3806488e-16
year = 3.15576e7
Msun = 1.98847e33
pc_to_cm = 3.086e18

# ---------- two-stage (slow → WR) density ----------
def make_variable_k_density(
    A=1.5E11,   # g/s  if k=2
    k=2    # wind structure
):
    """
    Returns: rho(phi,theta,r) [g/cm^3], meta dict
    Inner WR wind, thin swept shell at R_sh, slow wind out to R_slow, ISM beyond.
    """

    def rho(phi, theta, r):
        return A*r**(-k)
    meta = {"A": A, "k": k}
    return rho, meta

# build one medium (tweak parameters here as desired)
rho_fn, meta = make_variable_k_density(
    A=1.5E11,
    k=2
)
medium = Medium(rho=rho_fn)
medium = Wind(A_star=1)  ###  Testing to see if output is same

# ---------- model (jet/observer/radiation) ----------
jet = TophatJet(theta_c=0.1, E_iso=1e52, Gamma0=300)
obs = Observer(lumi_dist=1e26, z=0.1, theta_obs=0)
rad = Radiation(eps_e=1e-1, eps_B=1e-3, p=2.3)
model = Model(jet=jet, medium=medium, observer=obs, fwd_rad=rad)

# ---------- density profile plot with twin axes ----------
r = np.logspace(16, 20, 600)                     # cm
rho_profile = model.medium(0,0,r)
n_profile   = rho_profile / mp

fig, ax1 = plt.subplots(figsize=(5, 3.6), dpi=200)
ax1.loglog(r / pc_to_cm, n_profile, color='C0', lw=1.5)
ax1.set_xlabel('Radius (pc)')
ax1.set_ylabel(r'n(r) [cm$^{-3}$]')
ax1.set_title('Two-Stage (LBV/RSG → WR) Bubble Density')

# top x-axis in cm
def pc_to_cm_f(x):  return x * pc_to_cm
def cm_to_pc_f(x):  return x / pc_to_cm
ax2 = ax1.secondary_xaxis('top', functions=(pc_to_cm_f, cm_to_pc_f))
ax2.set_xlabel('Radius (cm)')

# right y-axis in ρ
def n_to_rho(y):  return y * mp
def rho_to_n(y):  return y / mp
ax3 = ax1.secondary_yaxis('right', functions=(n_to_rho, rho_to_n))
ax3.set_ylabel(r'$\rho(r)$ [g cm$^{-3}$]')

plt.tight_layout()
plt.savefig("assets/density_profile.png", dpi=300)
plt.show()

# ---------- light curves (multi-band) ----------
# ---------- combined multi-band light curves with dual x- and y-axes ----------
times = np.logspace(2, 8, 200)           # seconds
times_days = times / 86400.0
bands = np.array([1e9, 1e14, 1e17])      # Hz (radio, optical, X-ray)
band_names = ["Radio", "Optical", "X-ray"]

lc = model.flux_density_grid(times, bands)  # erg cm^-2 s^-1 Hz^-1

fig, ax1 = plt.subplots(figsize=(5.8, 3.8), dpi=200)

# --- main plot (bottom axis: seconds, flux in Jy) ---
for j, (name, nu) in enumerate(zip(band_names, bands)):
    exp = int(np.log10(nu))
    label = fr'{name} ($10^{{{exp}}}$ Hz)'
    ax1.loglog(times, lc.total[j, :] * 1e23,
               color=f'C{j}', lw=1.6, label=label)

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Flux Density (Jy)')
ax1.set_title('Afterglow Light Curves in Multiple Bands')

# --- legend (LaTeX formatted) ---
ax1.legend(title='Band', ncol=1, fontsize=9, title_fontsize=10)

# --- top axis: time in days ---
def s_to_days(x):  return x / 86400.0
def days_to_s(x):  return x * 86400.0
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

# ---------- spectra at selected epochs ----------
frequencies = np.logspace(5, 22, 300)    # Hz
epochs      = np.array([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])  # s
epochs_days = epochs / 86400.0

spec_grid = model.flux_density_grid(epochs, frequencies)  # shape ~ [len(freq), len(time)]

###
# ---------- spectra at selected epochs (all on one figure) ----------
frequencies = np.logspace(5, 22, 300)    # Hz
epochs      = np.array([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])  # s
epochs_days = epochs / 86400.0

# Compute spectral evolution: Fν(ν, t)
spec_grid = model.flux_density_grid(epochs, frequencies)  # shape ≈ [len(freq), len(time)]

# --- Plot all epochs on one figure ---
plt.figure(figsize=(5.5, 3.8), dpi=200)
colors = plt.cm.plasma(np.linspace(0, 1, len(epochs)))

for j, tsec in enumerate(epochs):
    # nice epoch label
    exp = int(np.floor(np.log10(tsec)))
    base = tsec / 10**exp
    if np.isclose(base, 1.0):
        label = fr'$10^{{{exp}}}\,\mathrm{{s}}$'
    else:
        label = fr'${base:.1f}\times10^{{{exp}}}\,\mathrm{{s}}$'

    plt.loglog(frequencies, spec_grid.total[:, j] * 1e23,
               color=colors[j], lw=1.5, label=label)

# Mark radio / optical / X-ray bands
for k, nu in enumerate(bands):
    plt.axvline(nu, ls='--', color=f'C{k}', alpha=0.6)

# Labels and legend
plt.xlabel('Frequency (Hz)')
plt.ylabel('Flux Density (Jy)')
plt.title('Synchrotron Spectra at Multiple Epochs')
plt.legend(title='Epoch (s)', ncol=2, fontsize=8, title_fontsize=9, loc='best')

plt.tight_layout()
plt.savefig('assets/spectra_all_epochs.png', dpi=300)
plt.show()
