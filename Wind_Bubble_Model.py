# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from VegasAfterglow import TophatJet, Observer, Radiation, Model, Medium
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
def make_two_stage_density(
    # WR (inner, fast)
    Mdot_WR = 6e-6 * Msun / year,   # g/s  (≈ 6e-6 Msun/yr)
    v_WR    = 2e8,                  # cm/s (≈ 2000 km/s)
    t_WR    = 1e4 * year,           # s    (10^4 yr)

    # Slow phase (LBV/RSG)
    Mdot_slow = 3e-5 * Msun / year, # g/s  (≈ 3e-5 Msun/yr)
    v_slow    = 2.5e6,              # cm/s (≈ 25 km/s → RSG; use 1e7 for LBV 100 km/s)
    t_slow    = 5e4 * year,         # s    (5×10^4 yr laid down pre-WR)

    # ISM
    n_ism   = 1.0,                  # cm^-3
    rho_ism = None,                 # g/cm^3; if None, set to n_ism*mp

    # shell placement & mass loading
    shell_mode    = "kinematic",    # "kinematic" or "fractional"
    frac_Rsh      = 0.5,            # if "fractional": R_sh = frac_Rsh * R_slow
    f_sweep       = 0.7,            # fraction of slow-wind mass (inside R_sh) swept into shell
    rel_thickness = 0.03            # ΔR / R_sh (geometrically thin)
):
    """
    Returns: rho(phi,theta,r) [g/cm^3], meta dict
    Inner WR wind, thin swept shell at R_sh, slow wind out to R_slow, ISM beyond.
    """
    if rho_ism is None:
        rho_ism = n_ism * mp

    # A-parameters (g cm^-1)
    A_WR   = Mdot_WR   / (4.0 * pi * v_WR)
    A_slow = Mdot_slow / (4.0 * pi * v_slow)

    # slow wind extent (laid down before WR)
    R_slow = max(1e15, v_slow * t_slow)

    # shell radius
    if shell_mode.lower().startswith("kin"):
        R_sh = max(1e15, min(R_slow, v_slow * t_WR))
    else:
        R_sh = max(1e15, min(R_slow, frac_Rsh * R_slow))

    # shell thickness
    dR = max(1e-3 * R_sh, rel_thickness * R_sh)

    # mass of slow wind interior to R_sh (for ρ∝r^-2: M(<R)=(Mdot/v)*R)
    M_slow_in = (Mdot_slow / v_slow) * R_sh
    M_shell   = f_sweep * M_slow_in

    # uniform shell density (mass / volume of thin spherical shell)
    rho_shell = M_shell / (4.0 * pi * R_sh**2 * dR)

    # region boundaries
    R1 = R_sh        # WR wind → shell inner
    R2 = R_sh + dR   # shell outer
    R3 = R_slow      # slow wind outer

    def rho(phi, theta, r):
        # free WR wind
        if r < R1:
            return A_WR / (r**2)
        # thin shell
        if r < R2:
            return rho_shell
        # slow wind zone
        if r < R3:
            return A_slow / (r**2)
        # ambient ISM
        return rho_ism

    meta = {"R_sh": R_sh, "dR": dR, "R_slow": R_slow,
            "rho_shell": rho_shell, "A_WR": A_WR, "A_slow": A_slow,
            "n_ism": n_ism, "rho_ism": rho_ism}
    return rho, meta

# build one medium (tweak parameters here as desired)
rho_fn, meta = make_two_stage_density(
    # choose LBV or RSG character by v_slow; other knobs above
    v_slow=2.5e6,        # 25 km/s (RSG-like)
    n_ism=1.0,           # cm^-3
    f_sweep=0.7,
    rel_thickness=0.03
)
medium = Medium(rho=rho_fn)

# ---------- model (jet/observer/radiation) ----------
jet = TophatJet(theta_c=0.1, E_iso=1e52, Gamma0=300)
obs = Observer(lumi_dist=1e26, z=0.1, theta_obs=0)
rad = Radiation(eps_e=1e-1, eps_B=1e-3, p=2.3)
model = Model(jet=jet, medium=medium, observer=obs, fwd_rad=rad)

# ---------- density profile plot with twin axes ----------
r = np.logspace(16, 20, 600)                     # cm
rho_profile = np.array([rho_fn(0, 0, ri) for ri in r])
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
