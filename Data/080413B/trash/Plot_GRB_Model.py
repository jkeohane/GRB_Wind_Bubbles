#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import toml
import glob
import os
from VegasAfterglow import TophatJet, Observer, Radiation, Model, Medium

# ===============================================================
#  CONFIGURATION
# ===============================================================
DATA_DIR  = "/Users/jkeohane/GRBs/GRB_Wind_Bubbles/Data/080413B"
PARAMFILE = os.path.join(DATA_DIR, "fit_results.toml")
OUTDIR    = DATA_DIR

os.makedirs(OUTDIR, exist_ok=True)

# ===============================================================
#  READ PARAMETERS
# ===============================================================
pars = toml.load(PARAMFILE)["best"]

E_iso   = pars["E_iso"]
Gamma0  = pars["Gamma0"]
theta_c = pars["theta_c"]
eps_e   = pars["eps_e"]
eps_B   = pars["eps_B"]
p       = pars["p"]
R_t     = pars["R_t"]
log_n_t   = pars["log10_n_t"]
log_n_ism = pars["log10_n_ism"]

n_t   = 10.0**log_n_t
n_ism = 10.0**log_n_ism

print("Loaded parameters:")
for k,v in pars.items():
    print(f"  {k:12s} = {v}")

# ===============================================================
#  READ DATA FILE
# ===============================================================
csvfiles = glob.glob(os.path.join(DATA_DIR, "*.csv"))
if len(csvfiles) != 1:
    raise RuntimeError("Expected exactly one CSV photometry file.")
datafile = csvfiles[0]

print("\nReading data:", datafile)

raw = np.genfromtxt(datafile, delimiter=",", names=True, dtype=None, encoding=None)

t_data   = raw["Time"]
nu_data  = raw["Wave"]      # Hz
f_data   = raw["Value"]     # mJy
err_data = raw["ValueUpper"]  # mJy

bands = raw["Filter"]

unique_bands = np.unique(bands)
print("Bands found:", unique_bands)

# Convert mJy → cgs: 1 mJy = 1e-26 erg/s/cm^2/Hz
f_data_cgs   = f_data * 1e-26
err_data_cgs = err_data * 1e-26

# ===============================================================
#  BUILD USER MEDIUM (simple bubble)
# ===============================================================
def rho_user(r):
    """User-defined density profile: uniform ISM for now"""
    return n_ism * 1.6726219e-24  # g cm^-3

medium = Medium(rho=rho_user)

# ===============================================================
#  BUILD MODEL
# ===============================================================
jet = TophatJet(theta_c=theta_c, E_iso=E_iso, Gamma0=Gamma0)
obs = Observer(lumi_dist=1e26, z=0.1, theta_obs=0.0)

rad = Radiation(eps_e=eps_e, eps_B=eps_B, p=p)

model = Model(jet=jet, medium=medium, observer=obs, fwd_rad=rad)

# ===============================================================
#  LIGHT CURVE PLOTTING
# ===============================================================
def plot_lightcurves():
    plt.figure(figsize=(8,6))

    colors = plt.cm.tab10(np.linspace(0,1,len(unique_bands)))

    for i,band in enumerate(unique_bands):
        mask = (bands == band)
        t = t_data[mask]
        nu = nu_data[mask][0]  # each band has one central frequency
        f  = f_data_cgs[mask]
        e  = err_data_cgs[mask]

        # Model flux
        f_mod = model.flux(t, nu, nu, 1).total[:,0]  # cgs

        plt.errorbar(t, f, yerr=e, fmt="o", color=colors[i], label=band)
        plt.plot(t, f_mod, "-", color=colors[i])

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Time (s)")
    plt.ylabel("Flux density (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)")
    plt.title("Light curves by filter")
    plt.legend()

    outfile = os.path.join(OUTDIR, "lightcurves_model.png")
    plt.savefig(outfile, dpi=200)
    plt.close()
    print("Saved:", outfile)

# ===============================================================
#  SPECTRA PLOTTING
# ===============================================================
def plot_spectra():
    # Pick epochs from the data
    # Here: choose 5 logarithmically spaced times within data range
    tmin, tmax = np.min(t_data), np.max(t_data)
    epochs = np.logspace(np.log10(tmin), np.log10(tmax), 5)

    # Frequency grid for spectra (radio → X-ray)
    nu_grid = np.logspace(8, 19, 300)

    plt.figure(figsize=(8,6))

    for t in epochs:
        F = model.flux(np.array([t]), nu_grid, nu_grid, len(nu_grid))
        spec = F.total[0,:]  # cgs
        plt.plot(nu_grid, spec, label=f"{t:.1f} s")

    # vertical lines for data bands
    for band in unique_bands:
        mask = (bands == band)
        nu0 = nu_data[mask][0]
        plt.axvline(nu0, color="gray", alpha=0.3)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Flux density (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)")
    plt.title("Spectra at selected epochs")
    plt.legend()

    outfile = os.path.join(OUTDIR, "spectra_model.png")
    plt.savefig(outfile, dpi=200)
    plt.close()
    print("Saved:", outfile)

# ===============================================================
#  MAIN EXECUTION
# ===============================================================
plot_lightcurves()
plot_spectra()

print("\nAll products saved in:", OUTDIR)
