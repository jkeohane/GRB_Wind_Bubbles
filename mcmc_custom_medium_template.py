#!/usr/bin/env python3
"""
MCMC example: fitting a user-defined Medium density function in VegasAfterglow.

What this shows
---------------
1) How to make a custom `density(phi, theta, r)` that depends on global parameters.
2) How to expose those parameters to an MCMC sampler (emcee).
3) How to connect sampled parameters to the VegasAfterglow Model.
4) How to run the sampler, recover best-fit values, and plot a quick check.

Replace the synthetic data block with your real data to do an actual fit.
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
from astropy.cosmology import Planck18 as cosmo

# You will need VegasAfterglow installed and importable in this environment.
# pip install VegasAfterglow  (or your local editable install)
from VegasAfterglow import Medium, TophatJet, Observer, Radiation, Model

# ----------------------------
# 0) Cosmology helper
# ----------------------------
def luminosity_distance_cgs(z):
    """
    Return luminosity distance in cm.
    Tries astropy first.  Falls back to a flat LCDM approximation (H0=70, Om0=0.3).
    """
    try:
        from astropy.cosmology import Planck18 as cosmo
        return cosmo.luminosity_distance(z).cgs.value  # cm
    except Exception:
        c = 2.99792458e10  # cm/s
        H0 = 70.0 * 1.0e5 / 3.0856775814913673e24  # s^-1  (70 km/s/Mpc)
        Om0 = 0.3
        Ol0 = 0.7
        q0 = 0.5 * Om0 - Ol0  # ≈ -0.55
        z2 = z * z
        dC = (c / H0) * (z - 0.5 * (1.0 - q0) * z2)  # O(z^2)
        dL = (1.0 + z) * dC
        return dL

# ----------------------------
# 1) Observation setup  FAKE OBSERVATION FOR TESTING
# ----------------------------
# Example: single observing band at nu = 1e14 Hz, sampled at times from 1e3 to 1e7 s
rng = np.random.default_rng(42)
t_obs = np.geomspace(1.0e3, 1.0e7, 50)     # seconds
nu_min = 1.0e14                              # Hz
nu_max = 1.0e14                              # Hz (single-frequency)
num_nu = 1

# ----------------------------
# 2) Global fit parameters used by density()
# ----------------------------
# Initialize with some guesses.  These will be overwritten by the sampler at runtime.
rho0_fit  = 1.0e-24     # g cm^-3 at r = r0_fit
r0_fit    = 1.0e17      # cm  reference radius
slope_fit = 2.0         # dimensionless power-law slope

def density(phi, theta, r):
    """
    User-defined density profile, units g cm^-3.
    Uses global parameters that the sampler will update.
    """
    global rho0_fit, r0_fit, slope_fit
    r_safe = np.maximum(r, 1e-20)  # ensure safe behavior for r <= 0
    return rho0_fit * (r_safe / r0_fit)**(-slope_fit)

# ----------------------------
# 3) Model construction helper
# ----------------------------
def make_model(params):
    """
    Build a VegasAfterglow Model from a dict of parameters.
    """
    # Update globals for the density() function
    global rho0_fit, r0_fit, slope_fit
    rho0_fit  = float(params["rho0_fit"])
    r0_fit    = float(params["r0_fit"])
    slope_fit = float(params["slope_fit"])

    # Medium
    medium = Medium(rho=density)

    # Jet
    jet = TophatJet(
        theta_c = float(params["theta_c"]),
        E_iso   = float(params["E_iso"]),
        Gamma0  = float(params["Gamma0"]),
    )

    # Observer
    z_val = float(params["z"])
    d_L   = luminosity_distance_cgs(z_val)
    obs = Observer(
        lumi_dist = d_L,
        z         = z_val,
        theta_obs = float(params["theta_obs"]),
        # phi_obs defaults to 0
    )

    # Radiation (forward shock)
    rad = Radiation(
        eps_e = float(params["eps_e"]),
        eps_B = float(params["eps_B"]),
        p     = float(params["p"]),
    )

    # Model (API expects observer= and fwd_rad=)
    model = Model(
        jet       = jet,
        medium    = medium,
        observer  = obs,
        fwd_rad   = rad,
        rvs_rad   = None,
    )
    return model

# ----------------------------
# Helper for consistent flux extraction (Option B)
# ----------------------------
def first_band(pyflux_obj):
    """
    Coerce PyFlux.total to at least 2-D and return the first band as a 1-D array (num_t,).
    This is robust whether the library returns shape (num_t,), (1, num_t), or (num_nu, num_t).
    """
    arr = np.asarray(pyflux_obj.total)
    arr2 = np.atleast_2d(arr)
    return arr2[0, :]

# ----------------------------
# 4) Generate synthetic data (replace with your real data)
# ----------------------------
true = dict(
    rho0_fit  = 5.0e-25,
    r0_fit    = 3.0e17,
    slope_fit = 2.1,
    theta_c   = 0.03,
    E_iso     = 3.0e52,
    Gamma0    = 120.0,
    z         = 0.5,
    theta_obs = 0.01,   # observer angle Craps out when this is zero.
    eps_e     = 0.1,
    eps_B     = 2.0e-3,
    p         = 2.2,
)

model_true = make_model(true)
pyflux = model_true.flux(t_obs, nu_min, nu_max, num_nu)  # returns PyFlux
flux_true = first_band(pyflux)                           # (num_t,)

# Add 10% Gaussian noise
sigma_frac = 0.10
sigma = np.maximum(sigma_frac * flux_true, 1e-99)  # guard against zeros
data = flux_true + sigma * rng.standard_normal(size=flux_true.size)

# ----------------------------
# 5) Priors and parameter bounds
# ----------------------------
bounds = {
    "rho0_fit":  (1e-27, 1e-22),
    "r0_fit":    (1e16,  1e19),
    "slope_fit": (0.0,   3.5),
    "theta_c":   (0.01,  0.5),
    "E_iso":     (1e50,  1e54),
    "Gamma0":    (5.0,   500.0),
    "z":         (1.4,   1.5),
    "theta_obs": (0.0,   0.5),
    "eps_e":     (1e-4,  0.5),
    "eps_B":     (1e-6,  0.1),
    "p":         (2.01,  3.2),
}

# Choose which parameters to sample.  You can reduce this list for a lighter demo.
param_names = list(bounds.keys())

def in_bounds(theta):
    for x, name in zip(theta, param_names):
        lo, hi = bounds[name]
        if not (lo <= x <= hi):
            return False
    return True

def log_prior(theta):
    # Flat priors in the given bounds
    return 0.0 if in_bounds(theta) else -np.inf

def model_flux_from_theta(theta):
    params = {k: float(v) for k, v in zip(param_names, theta)}
    m = make_model(params)
    pf = m.flux(t_obs, nu_min, nu_max, num_nu)
    return first_band(pf)  # (num_t,)

def log_likelihood(theta):
    f = model_flux_from_theta(theta)
    chi2 = np.sum(((data - f) / sigma)**2)
    return -0.5 * chi2

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta)
    return lp + ll

# ----------------------------
# 6) Initialize walkers near truth (or near your guesses)
# ----------------------------
ndim = len(param_names)
nwalkers = 32

theta0 = np.array([true[k] for k in param_names], dtype=float)  # start near the true values
p0 = theta0[None, :] * (1.0 + 1e-3 * rng.standard_normal(size=(nwalkers, ndim)))  # small ball

# ----------------------------
# 7) Run emcee  -- need bigger n later
# ----------------------------
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
n_burn = 50   # keep small for demo
n_run  = 100

print("Running burn-in...")
state = sampler.run_mcmc(p0, n_burn, progress=True)
sampler.reset()

print("Running production...")
sampler.run_mcmc(state, n_run, progress=True)

# ----------------------------
# 8) Report results
# ----------------------------
samples = sampler.get_chain(flat=True)
med = np.median(samples, axis=0)
lo  = np.percentile(samples, 16, axis=0)
hi  = np.percentile(samples, 84, axis=0)

print("\nPosterior medians and 1-sigma intervals:")
for i, name in enumerate(param_names):
    print(f"  {name:9s} = {med[i]:.6e}  (+{hi[i]-med[i]:.2e}, -{med[i]-lo[i]:.2e})   truth={true[name]:.6e}")

# ----------------------------
# 9) Quick visual check vs. data
# ----------------------------
f_med = model_flux_from_theta(med)

plt.figure()
plt.loglog(t_obs, data, 'o', label='data')
plt.loglog(t_obs, f_med, '-', label='model (median)')
plt.xlabel('Time  [s]')
plt.ylabel('Flux density  [arbitrary units]')
plt.legend()
plt.tight_layout()
plt.savefig('mcmc_custom_medium_check.png', dpi=150)
print("\nSaved plot: mcmc_custom_medium_check.png")
plt.show()

# Optional: trace plot for a few parameters
plt.figure(figsize=(10, 6))
for j, name in enumerate(param_names[:5]):  # show first 5 traces
    ax = plt.subplot(5, 1, j+1)
    ax.plot(sampler.get_chain()[:, :, j], alpha=0.5)
    ax.set_ylabel(name)
plt.xlabel('Step')
plt.tight_layout()
plt.savefig('mcmc_traces.png', dpi=150)
print("Saved plot: mcmc_traces.png")
plt.show()




quit()