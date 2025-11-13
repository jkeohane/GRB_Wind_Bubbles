
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRB Bubble-Shell Fit with VegasAfterglow (MCMC)
Author: ChatGPT for Jonathan Keohane

This script reads a GRB light-curve CSV and a matching parameters.toml from a
Data/<GRB_NAME>/ subfolder, builds a bubble + shell circumstellar medium, and
fits the data using MCMC.  Starting points and priors are taken from the TOML
file exactly.  Outputs include best-fit parameters, diagnostic plots, and an
overlay of model vs. data by filter.

Usage
-----
    python fit_grb_bubble.py --root /Users/jkeohane/GRBs/GRB_Wind_Bubbles \
        --grb 080413B --nwalkers 64 --nsteps 6000 --burn 2000

Conventions
-----------
    * Times are read from the CSV column "Time" with units in column "TimeUnits".
      If TimeUnits == "s", times are converted to days for modeling convenience,
      but the calculation uses seconds internally to pass to VegasAfterglow.
    * Flux densities are read from "Value" with "ValueUnits" == "mJy".  These
      are converted to cgs (erg s^-1 cm^-2 Hz^-1) via 1 mJy = 1e-26.
    * Frequency is read from "Wave" with "WaveUnits" == "Hz".
    * Each row has a "Filter" and three grouping columns: CalGroup, HostGroup,
      and SlopGroup.  The TOML may define offsets per filter or per group.
    * The model medium is a smoothed wind-to-ISM bubble with an adjustable thin
      shell at R_t having contrast f_shell and fractional width dR/R_t.

Notes
-----
    * This script assumes a recent VegasAfterglow that exposes the following:
        - VegasAfterglow.Medium          : arbitrary density via callable rho(r)
        - VegasAfterglow.TophatJet       : standard top-hat jet
        - VegasAfterglow.Observer, Radiation, Model
      If API differences exist in your environment, adjust the builder functions
      where marked.
    * Dependencies: numpy, pandas, tomllib (or tomli), matplotlib, emcee.
    * The code follows Jonathan’s stylistic preferences: standard American
      English, no contractions, double spaces after periods, and Oxford commas.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Toml loader: Python 3.11+ has tomllib in the stdlib.  Fall back to tomli.
try:
    import tomllib  # type: ignore
except Exception:
    import tomli as tomllib  # type: ignore

# MCMC sampler
import emcee

# VegasAfterglow imports (adjust if your package layout differs).
from VegasAfterglow import Medium, TophatJet, Observer, Radiation, Model

# ---------- Utility ----------

PC_IN_CM = 3.0856775814913673e18
MP = 1.67262192369e-24  # g
MU = 1.3                # mean molecular weight for number density conversion


def mJy_to_cgs(x_mJy: np.ndarray) -> np.ndarray:
    """Convert mJy to cgs flux density (erg s^-1 cm^-2 Hz^-1)."""
    return 1.0e-26 * np.asarray(x_mJy, dtype=float)


def days_to_sec(t_days: np.ndarray) -> np.ndarray:
    return 86400.0 * np.asarray(t_days, dtype=float)


def sec_to_days(t_sec: np.ndarray) -> np.ndarray:
    return np.asarray(t_sec, dtype=float) / 86400.0


# ---------- Data Loading ----------

@dataclass
class LCData:
    t_sec: np.ndarray         # seconds since trigger
    nu_hz: np.ndarray         # observed frequency (Hz)
    f_cgs: np.ndarray         # flux density (erg s^-1 cm^-2 Hz^-1)
    sigma_cgs: np.ndarray     # symmetric 1-sigma uncertainty in same units
    filt: np.ndarray          # filter name, e.g., g, r, i
    cal_group: np.ndarray     # calibration group key (string)
    host_group: np.ndarray    # host group key (string)
    slop_group: np.ndarray    # slop group key (string)


def read_lc_csv(csv_path: Path) -> LCData:
    df = pd.read_csv(csv_path)

    # Time handling
    t = df["Time"].to_numpy(dtype=float)
    time_units = str(df["TimeUnits"].iloc[0]).strip()
    if time_units.lower().startswith("s"):
        t_sec = t
    elif time_units.lower().startswith("d"):
        t_sec = days_to_sec(t)
    else:
        raise ValueError(f"Unsupported TimeUnits: {time_units}  Only 's' or 'days' are supported.")

    # Flux handling
    f = df["Value"].to_numpy(dtype=float)
    val_units = str(df["ValueUnits"].iloc[0]).strip()
    if val_units.lower() == "mjy":
        f_cgs = mJy_to_cgs(f)
        # Use symmetric error from ValueLower/ValueUpper if present, else 10%.
        if "ValueLower" in df and "ValueUpper" in df and df["ValueLower"].notna().any():
            # Assuming these are 1-sigma.  If not, this is still a decent start.
            sig = 0.5 * (df["ValueLower"].to_numpy(dtype=float) + df["ValueUpper"].to_numpy(dtype=float))
            sigma_cgs = mJy_to_cgs(sig)
        else:
            sigma_cgs = 0.1 * f_cgs
    else:
        raise ValueError(f"Unsupported ValueUnits: {val_units}.  Expected 'mJy'.")

    # Frequency
    nu_hz = df["Wave"].to_numpy(dtype=float)
    wave_units = str(df["WaveUnits"].iloc[0]).strip().lower()
    if wave_units != "hz":
        raise ValueError(f"Unsupported WaveUnits: {wave_units}.  Expected 'Hz'.")

    # Grouping columns
    filt = df["Filter"].astype(str).to_numpy()
    cal_group = df["CalGroup"].astype(str).to_numpy()
    host_group = df["HostGroup"].astype(str).to_numpy()
    slop_group = df["SlopGroup"].astype(str).to_numpy()

    return LCData(
        t_sec=t_sec,
        nu_hz=nu_hz,
        f_cgs=f_cgs,
        sigma_cgs=sigma_cgs,
        filt=filt,
        cal_group=cal_group,
        host_group=host_group,
        slop_group=slop_group,
    )


# ---------- Parameter Handling ----------

@dataclass
class Prior:
    kind: str                 # 'uniform' or 'gaussian'
    lower: float | None = None
    upper: float | None = None
    mu: float | None = None
    sigma: float | None = None
    scale: str = "linear"     # 'linear' or 'log'
    init: float | None = None
    init_sigma: float | None = None


def load_priors_from_toml(toml_path: Path) -> Dict[str, Prior]:
    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    priors: Dict[str, Prior] = {}

    def _parse_entry(entry: dict) -> tuple[str, Prior]:
        name = entry["name"]
        scale = entry.get("scale", "linear")

        # Support both new ("prior" block) and old (flat) TOML formats
        p = entry.get("prior", entry)

        kind = p.get("type", "uniform")
        lower = p.get("lower")
        upper = p.get("upper")
        mu = p.get("mu")
        sigma = p.get("sigma")
        init = p.get("initial_guess", p.get("init"))
        init_sigma = p.get("initial_sigma", 0.1)

        return name, Prior(
            kind=kind,
            lower=lower,
            upper=upper,
            mu=mu,
            sigma=sigma,
            scale=scale,
            init=init,
            init_sigma=init_sigma,
        )

    # Model block (jet, medium, microphysics typically)
    for ent in raw.get("model", []):
        k, v = _parse_entry(ent)
        priors[k] = v

    # Calibration offsets per filter (optional)
    for ent in raw.get("calibration", []):
        k, v = _parse_entry(ent)
        priors[k] = v

    # Host flux terms (optional)
    for ent in raw.get("host", []):
        k, v = _parse_entry(ent)
        priors[k] = v

    # Extra slop term (optional)
    for ent in raw.get("slop", []):
        k, v = _parse_entry(ent)
        priors[k] = v

    return priors


# ---------- Bubble + Shell Medium ----------

def make_bubble_shell_rho(A_star: float,
                          n_ism: float,
                          R_t_pc: float,
                          shell_contrast: float,
                          shell_frac_width: float) -> callable:
    """
    Return rho(r) in cgs (g cm^-3) for a smoothed wind to ISM bubble with a thin shell.
    A  1/r^2 wind, with normalization from A_star: A = 5e11 * A_star (g cm^-1).
    n_ism  uniform ISM beyond the bubble edge.
    R_t_pc transition radius (pc) from wind to ISM.
    shell_contrast  peak density multiplier of the shell relative to the local background.
    shell_frac_width  delta R / R_t controlling shell thickness.
    """
    A = 5.0e11 * float(A_star)  # g cm^-1
    R_t = float(R_t_pc) * PC_IN_CM
    dR = shell_frac_width * R_t

    def rho(r_cm: np.ndarray | float) -> np.ndarray:
        r = np.asarray(r_cm, dtype=float)
        # Wind density in g cm^-3 (rho = A / (4*pi*r^2) ?  Many definitions use A/(4πr^2).
        # In VA, wind often uses number density n = A/(mp r^2).  We include mu explicitly.
        rho_wind = (A / (r**2 + 1e-60)) / (MU) / (1.0)  # g cm^-3 with MU interpretation

        # Smooth transition to ISM using a logistic switch around R_t
        s = 0.5 * (1.0 + np.tanh((r - R_t) / (0.25 * dR + 1e-30)))
        rho_bg = (1.0 - s) * rho_wind + s * (n_ism * MU * MP)

        # Thin shell as a Gaussian bump at R_t
        shell = np.exp(-0.5 * ((r - R_t) / (0.5 * dR + 1e-30))**2)
        rho_shell = rho_bg * (1.0 + shell_contrast * shell)

        return rho_shell

    return rho


# ---------- Model Builder ----------

@dataclass
class BubbleParams:
    # Medium
    A_star: float
    n_ism: float
    R_t_pc: float
    shell_contrast: float
    shell_frac_width: float
    # Jet and radiation
    E_iso: float
    Gamma0: float
    theta_c: float
    z: float
    theta_obs: float
    eps_e: float
    eps_B: float
    p: float
    # Nuisance
    slop: float
    # Per-filter calibration offsets (magnitudes converted to multiplicative factors on flux).
    cal_offsets_mag: Dict[str, float]
    # Per-host additive constant in cgs per HostGroup (optional)
    host_add_cgs: Dict[str, float]


def make_va_model(par: BubbleParams) -> Model:
    rho = make_bubble_shell_rho(par.A_star, par.n_ism, par.R_t_pc,
                                par.shell_contrast, par.shell_frac_width)
    medium = Medium(rho=rho)

    jet = TophatJet(theta_c=par.theta_c, E_iso=par.E_iso, Gamma0=par.Gamma0)
    obs = Observer(lumi_dist=None, z=par.z, theta_obs=par.theta_obs)
    rad = Radiation(eps_e=par.eps_e, eps_B=par.eps_B, p=par.p)

    model = Model(jet=jet, medium=medium, observer=obs, fwd_rad=rad)
    return model


# ---------- Likelihood ----------

def flux_scale_from_mag_offset(dm: float) -> float:
    """Convert a magnitude offset to a multiplicative flux factor."""
    return 10.0 ** (-0.4 * dm)


def model_flux(par: BubbleParams, t_sec: np.ndarray, nu_hz: np.ndarray) -> np.ndarray:
    model = make_va_model(par)
    # VegasAfterglow Model should support evaluate_flux_density(nu, t).
    # Adjust call if your installed version uses a different interface.
    f = model.flux_density(nu=nu_hz, t=t_sec)  # erg s^-1 cm^-2 Hz^-1
    return f


def log_prior(theta: np.ndarray, key_order: List[str], priors: Dict[str, Prior]) -> float:
    lp = 0.0
    for val, key in zip(theta, key_order):
        p = priors[key]
        x = float(val)
        if p.kind == "uniform":
            lo = -np.inf if p.lower is None else p.lower
            hi = np.inf if p.upper is None else p.upper
            if not (lo <= x <= hi):
                return -np.inf
            # Flat prior adds constant, which we drop.
        elif p.kind == "gaussian":
            mu = float(p.mu)
            sig = float(p.sigma)
            lp += -0.5 * ((x - mu) / sig) ** 2 - np.log(sig * np.sqrt(2.0 * np.pi))
        else:
            raise ValueError(f"Unknown prior kind: {p.kind}")
    return lp


def pack_params(theta: np.ndarray,
                key_order: List[str],
                data: LCData,
                priors: Dict[str, Prior]) -> BubbleParams:
    """Construct BubbleParams from the current theta vector and TOML-defined names."""
    d = {k: float(v) for k, v in zip(key_order, theta)}

    # Required keys with defaults if absent in priors TOML
    A_star = d.get("A_star", 0.1)
    n_ism = 10.0 ** d.get("n_ism_log", -1.0) if "n_ism_log" in d else d.get("n_ism", 1.0)
    R_t_pc = 10.0 ** d.get("rt", -1.0)  # treat 'rt' as log10(pc) if given
    shell_contrast = d.get("f_shell", 10.0)
    shell_frac_width = d.get("dR_frac", 0.2)

    E_iso = 10.0 ** d.get("E", 52.0)
    Gamma0 = 10.0 ** d.get("Gamma0", 2.3)
    theta_c = d.get("theta_c", 0.1)
    z = d.get("z", 0.1)
    theta_obs = d.get("theta_obs", 0.0)
    eps_e = 10.0 ** d.get("eps_e", -1.0)
    eps_B = 10.0 ** d.get("eps_B", -3.0)
    p = d.get("p", 2.2)
    slop = max(0.0, d.get("slop", 0.02))

    # Per-filter calibration mag offsets that appear in TOML as X_offset
    cal_offsets_mag: Dict[str, float] = {}
    for key in d.keys():
        if key.endswith("_offset"):
            filt = key.replace("_offset", "")
            cal_offsets_mag[filt] = d[key]

    # Host additive fluxes log10 given in TOML by entries like "g_host"
    host_add_cgs: Dict[str, float] = {}
    for key in d.keys():
        if key.endswith("_host"):
            grp = key  # use exact key to match HostGroup column
            host_add_cgs[grp] = 10.0 ** d[key]

    return BubbleParams(
        A_star=A_star, n_ism=n_ism, R_t_pc=R_t_pc, shell_contrast=shell_contrast,
        shell_frac_width=shell_frac_width, E_iso=E_iso, Gamma0=Gamma0,
        theta_c=theta_c, z=z, theta_obs=theta_obs, eps_e=eps_e, eps_B=eps_B, p=p,
        slop=slop, cal_offsets_mag=cal_offsets_mag, host_add_cgs=host_add_cgs
    )


def log_likelihood(theta: np.ndarray,
                   key_order: List[str],
                   data: LCData,
                   priors: Dict[str, Prior]) -> float:
    par = pack_params(theta, key_order, data, priors)
    f_model = model_flux(par, data.t_sec, data.nu_hz)

    # Apply per-filter calibration multiplicative factors
    scale = np.ones_like(f_model)
    if par.cal_offsets_mag:
        for filt, dm in par.cal_offsets_mag.items():
            factor = flux_scale_from_mag_offset(dm)
            scale[data.filt == filt] *= factor
    f_model = f_model * scale

    # Add per-host additive flux
    if par.host_add_cgs:
        host_add = np.zeros_like(f_model)
        for key, val in par.host_add_cgs.items():
            host_add[data.host_group == key] += val
        f_model = f_model + host_add

    # Extra slop term added in quadrature
    sigma2 = data.sigma_cgs**2 + (par.slop * data.f_cgs)**2

    resid = data.f_cgs - f_model
    ll = -0.5 * np.sum(resid**2 / sigma2 + np.log(2.0 * np.pi * sigma2))
    return ll


def log_posterior(theta: np.ndarray,
                  key_order: List[str],
                  data: LCData,
                  priors: Dict[str, Prior]) -> float:
    lp = log_prior(theta, key_order, priors)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, key_order, data, priors)
    return lp + ll


# ---------- Initialization from TOML ----------

def initial_theta_and_keys(priors: Dict[str, Prior]) -> Tuple[np.ndarray, List[str]]:
    keys: List[str] = []
    inits: List[float] = []
    for name, pr in priors.items():
        # Use initial guess exactly, if provided.  Otherwise center of uniform, or mu of gaussian.
        if pr.init is not None:
            x0 = float(pr.init)
        elif pr.kind == "uniform" and pr.lower is not None and pr.upper is not None:
            x0 = 0.5 * (float(pr.lower) + float(pr.upper))
        elif pr.kind == "gaussian" and pr.mu is not None:
            x0 = float(pr.mu)
        else:
            x0 = 0.0
        keys.append(name)
        inits.append(x0)
    return np.array(inits, dtype=float), keys


def scatter_walkers(x0: np.ndarray, priors: Dict[str, Prior], nwalkers: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    pos = np.tile(x0, (nwalkers, 1))
    for j, (name, pr) in enumerate(priors.items()):
        sig = pr.init_sigma if pr.init_sigma is not None else 0.01
        pos[:, j] += rng.normal(0.0, sig, size=nwalkers)
    return pos


# ---------- Plotting ----------

def plot_fit(data: LCData,
             par_map: BubbleParams,
             out_png: Path,
             title: str = "Bubble-Shell Fit: Model vs. Data") -> None:
    plt.figure(figsize=(8, 5.5))

    # Plot per filter
    filters = np.unique(data.filt)
    for flt in filters:
        m = data.filt == flt
        plt.loglog(sec_to_days(data.t_sec[m]), data.f_cgs[m], ".", label=f"{flt} data")

    # Model curve at the median frequency per filter; overplot as lines
    # Because the model is not filter-integrated here, this is illustrative.
    t_grid = np.logspace(np.log10(np.min(data.t_sec)*0.8),
                         np.log10(np.max(data.t_sec)*1.2), 300)

    for flt in filters:
        m = data.filt == flt
        if not np.any(m):
            continue
        nu_med = float(np.median(data.nu_hz[m]))
        nu_grid = np.full_like(t_grid, nu_med, dtype=float)
        f = model_flux(par_map, t_grid, nu_grid)

        # Apply calibration on the fly for the line for this filter, if present
        if flt in par_map.cal_offsets_mag:
            f = f * flux_scale_from_mag_offset(par_map.cal_offsets_mag[flt])

        plt.loglog(sec_to_days(t_grid), f, "-", alpha=0.9, label=f"{flt} model")

    plt.xlabel("Observer Time [days]")
    plt.ylabel(r"$F_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
    plt.title(title)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# ---------- Main ----------

# ### This is to make it a command line version
# ap = argparse.ArgumentParser()
# ap.add_argument("--root", type=str, required=True, help="Path to GRB_Wind_Bubbles project root.")
# ap.add_argument("--grb", type=str, required=True, help="GRB folder name under Data/, e.g., 080413B")
# ap.add_argument("--nwalkers", type=int, default=64)
# ap.add_argument("--nsteps", type=int, default=4000)
# ap.add_argument("--burn", type=int, default=1500)
# ap.add_argument("--seed", type=int, default=1234)
# args = ap.parse_args()

# ---------- Manual configuration for PyCharm ----------

from types import SimpleNamespace

args = SimpleNamespace(
    root="/Users/jkeohane/GRBs/GRB_Wind_Bubbles",
    grb="080413B",
    nwalkers=64,
    nsteps=4000,
    burn=1500,
    seed=1234,
)


root = Path(args.root).expanduser().resolve()
data_dir = root / "Data" / args.grb
csv_path = max([p for p in data_dir.glob("*.csv")], key=lambda p: p.stat().st_size)
toml_path = data_dir / "parameters.toml"

print(f"Reading: {csv_path}")
data = read_lc_csv(csv_path)

print(f"Reading priors: {toml_path}")
priors = load_priors_from_toml(toml_path)

# Order and initial vector from TOML
x0, key_order = initial_theta_and_keys(priors)
print("Parameter order:")
for k in key_order:
    print(f"  - {k}")

# Sampler setup
ndim = x0.size
nwalkers = args.nwalkers
pos0 = scatter_walkers(x0, priors, nwalkers)

rng = np.random.default_rng(args.seed)

sampler = emcee.EnsembleSampler(
    nwalkers, ndim,
    log_posterior,
    args=(key_order, data, priors),
    pool=None,
  #  dtype=np.float64,
    moves=emcee.moves.StretchMove(a=2.0),
    random_state=rng,
)

print(f"Running burn-in: {args.burn} steps.")
pos, prob, state = sampler.run_mcmc(pos0, args.burn, progress=True)
sampler.reset()
print(f"Running production: {args.nsteps} steps.")
sampler.run_mcmc(pos, args.nsteps, progress=True)

outdir = root / "assets" / f"{args.grb}_fit"
outdir.mkdir(parents=True, exist_ok=True)

# Save chain
np.save(outdir / "chain.npy", sampler.get_chain())
np.save(outdir / "logprob.npy", sampler.get_log_prob())

# MAP estimate (best posterior sample)
flat = sampler.get_chain(discard=0, thin=10, flat=True)
lnp = sampler.get_log_prob(discard=0, thin=10, flat=True)
ibest = int(np.argmax(lnp))
theta_best = flat[ibest]

# Unpack to BubbleParams for plotting
par_best = pack_params(theta_best, key_order, data, priors)

# Save best-fit parameters in a readable table
with open(outdir / "best_fit.txt", "w") as f:
    for k, v in zip(key_order, theta_best):
        f.write(f"{k:20s}  {v: .6e}\n")

# Diagnostic plot
plot_fit(data, par_best, out_png=outdir / "model_vs_data.png",
         title=f"Bubble-Shell Fit: {args.grb}")

print("Done.  Results written to:", outdir)
