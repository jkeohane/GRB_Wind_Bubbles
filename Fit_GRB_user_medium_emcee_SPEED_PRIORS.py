#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fit_GRB_user_medium_emcee_SPEED_PRIORS.py

Speed-optimized afterglow fit with:
  • Schema-aware CSV loader (Time/Wave/Value/Units) like v7.
  • Log-time binning per frequency group (fast).
  • emcee sampling with fewer steps.
  • TOML priors I/O:
      - --priors_in  : read priors & starting points from a TOML file
      - --priors_out : write best-fit + chain stats back to a TOML file
"""

import sys, os, math, argparse, pickle
import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

try:
    import tomllib as toml  # py3.11+
except Exception:
    import tomli as toml

try:
    import tomli_w as toml_w
except Exception:
    toml_w = None  # we'll write by hand if missing

from VegasAfterglow import TophatJet, Observer, Radiation, Model, Medium

# ---------- constants ----------
mp = 1.67262192369e-24  # g
h  = 6.62607015e-27     # erg s
c  = 2.99792458e10      # cm s^-1

UNIT_SCALE = {'jy':1e-23,'mjy':1e-26,'ujy':1e-29,'µjy':1e-29}
FREQ_SCALE = {'hz':1.0,'khz':1e3,'mhz':1e6,'ghz':1e9,'thz':1e12}
TIME_BINS_PER_DECADE = 8
REL_FREQ_GROUP = 1e-3
MINIMUM_TIME = 5E2 # seconds
PARAM_NAMES = ["E_iso","Gamma0","theta_c","eps_e","eps_B",
               "p","R_t","log10_n_t","log10_n_ism"]
PLOT_ONLY = True
GRB_NAME = "080413B"

# ---------- config ----------
@dataclass
class FitConfig:
    z: float = 0.1
    lumi_dist: float = 1e26
    theta_obs: float = 0.0
    mu: float = 1.3
    num_nu_grid: int = 32
    nwalkers: int = 128
    nsteps: int = 3000
    burn: int = 1000
    seed: int = 1234

DEFAULT_BOUNDS = {
    "E_iso":      (1e50, 1e54),
    "Gamma0":     (10.0, 600.0),
    "theta_c":    (1e-3, 0.5),
    "eps_e":      (1e-5, 0.5),
    "eps_B":      (1e-6, 0.5),
    "p":          (2.0, 2.8),
    "R_t":        (1e14, 1e20),
    "log10_n_t":  (-6.0, 6.0),
    "log10_n_ism":(-6.0, 6.0),
}

# ---------- CSV loader (same as SPEED) ----------
def load_080413_schema(csv_path: str):
    df = pd.read_csv(csv_path)
    t = df['Time'].astype(float).to_numpy()
    if 'Wave' not in df.columns or 'WaveUnits' not in df.columns:
        raise ValueError("CSV needs Wave and WaveUnits")
    wave = df['Wave']
    wunits = df['WaveUnits'].astype(str).str.lower().str.strip().to_numpy()
    nu = np.full_like(t, np.nan, float)
    for i,(w,u) in enumerate(zip(wave,wunits)):
        try: val=float(w)
        except: continue
        if u in FREQ_SCALE:
            nu[i]=val*FREQ_SCALE[u]
        elif 'hz' in u:
            nu[i]=val
    if np.any(~np.isfinite(nu)) and 'Filter' in df.columns:
        filt = df['Filter'].astype(str).str.lower().str.strip().to_numpy()
        opt = {'u': c/(365e-7), 'b': c/(445e-7), 'v': c/(551e-7), 'r': c/(658e-7), 'i': c/(806e-7), 'g': c/(477e-7), 'z': c/(900e-7)}
        for i in range(nu.size):
            if not np.isfinite(nu[i]) and len(filt[i])>0 and filt[i][0] in opt:
                nu[i]=float(opt[filt[i][0]])
    val = df['Value'].to_numpy()
    vunits = df['ValueUnits'].astype(str).str.lower().str.strip().to_numpy()
    f = np.full_like(t, np.nan, float)
    for i,(x,u) in enumerate(zip(val,vunits)):
        try: xv=float(x)
        except: continue
        f[i]=xv*(UNIT_SCALE.get(u,1.0))
    e = np.full_like(f, np.nan, float)
    lo = df['ValueLower'].to_numpy() if 'ValueLower' in df.columns else np.full_like(f, np.nan)
    hi = df['ValueUpper'].to_numpy() if 'ValueUpper' in df.columns else np.full_like(f, np.nan)
    for i,u in enumerate(vunits):
        s=UNIT_SCALE.get(u,1.0)
        lo_i = lo[i] if isinstance(lo[i], (int,float,np.floating)) else np.nan
        hi_i = hi[i] if isinstance(hi[i], (int,float,np.floating)) else np.nan
        sigs=[s for s in [lo_i,hi_i] if np.isfinite(s) and s>0]
        if sigs: e[i]=max(sigs)*s
    mask_base = np.isfinite(t)&np.isfinite(nu)&np.isfinite(f)&(nu>1E6)&(t>MINIMUM_TIME)
    if np.any(~np.isfinite(e)&mask_base):
        idx=np.where(mask_base)[0]
        order=idx[np.argsort(nu[idx])]
        groups=[]; g=[order[0]]
        for j in order[1:]:
            if abs(nu[j]-nu[g[-1]])<=REL_FREQ_GROUP*nu[g[-1]]: g.append(j)
            else: groups.append(g); g=[j]
        groups.append(g)
        e2=e.copy()
        for g in groups:
            vals=[e[k] for k in g if np.isfinite(e[k]) and e[k]>0]
            med=float(np.median(vals)) if len(vals)>0 else None
            for k in g:
                if not (np.isfinite(e2[k]) and e2[k]>0):
                    e2[k]=med if (med and med>0) else max(1e-6,0.1*abs(f[k]))
        e=e2
    mask = np.isfinite(t)&np.isfinite(nu)&np.isfinite(f)&np.isfinite(e)&(nu>0)&(e>0)&(t>MINIMUM_TIME)
    if not np.any(mask): raise ValueError("No valid rows after parsing")
    return t[mask], nu[mask], f[mask], e[mask]

# ---------- binning ----------
def group_by_frequency(nu: np.ndarray, rel_tol: float = REL_FREQ_GROUP):
    order = np.argsort(nu)
    groups=[]; g=[order[0]]
    for idx in order[1:]:
        if abs(nu[idx]-nu[g[-1]])<=rel_tol*nu[g[-1]]: g.append(idx)
        else: groups.append(np.array(g,int)); g=[idx]
    groups.append(np.array(g,int))
    return groups

def logtime_bin_group(t, f, e, nbins):
    tpos = t[t>0]
    if tpos.size<3: return t, f, e
    lo, hi = np.log10(tpos.min()), np.log10(tpos.max())
    nb = max(2, min(nbins, tpos.size))
    edges = np.linspace(lo, hi, nb)
    idx = np.digitize(np.log10(tpos), edges) - 1
    tb, fb, eb = [], [], []
    for b in range(edges.size-1):
        m = (idx==b)
        if not np.any(m): continue
        tw = tpos[m]; fw = f[t>0][m]; ew = e[t>0][m]
        w = 1.0/np.maximum(ew,1e-30)**2
        tb.append(np.median(tw))
        fb.append(np.average(fw, weights=w))
        eb.append(1.0/np.sqrt(np.sum(w)))
    return np.array(tb), np.array(fb), np.array(eb)

def bin_data(t, nu, f, e):
    groups = group_by_frequency(nu)
    T,NU,F,E = [],[],[],[]
    for g in groups:
        nu0 = float(np.median(nu[g]))
        span = max(1,int(np.log10(nu[g].size if nu[g].size>0 else 1)+1))
        nbins = TIME_BINS_PER_DECADE*span
        tb, fb, eb = logtime_bin_group(t[g], f[g], e[g], nbins)
        T.append(tb); F.append(fb); E.append(eb); NU.append(np.full(tb.size, nu0))
    return np.concatenate(T), np.concatenate(NU), np.concatenate(F), np.concatenate(E)

# ---------- model & priors ----------
def nearest_index(arr,x): return int(np.abs(np.asarray(arr,float)-x).argmin())
def all_finite(x): return np.all(np.isfinite(x))

def build_model(theta, cfg: FitConfig):
    E_iso,Gamma0,theta_c,eps_e,eps_B,p,R_t,log10_n_t,log10_n_ism = theta[:9]
    n_t, n_ism = 10.0**log10_n_t, 10.0**log10_n_ism
    m_mol = cfg.mu*mp
    def rho_fn(phi, th, r):
        r=max(1.0,float(r))
        n_sh=max(4.0*n_t,4.0*n_ism)
        denom=(n_sh-n_ism)
        if denom<=0.0: return np.nan
        R2=float(R_t)*((n_sh+3.0*n_t)/denom)**(1.0/3.0)
        rho_t=n_t*m_mol; rho_sh=n_sh*m_mol; rho0=n_ism*m_mol
        if r<R_t: return rho_t*(R_t/r)**2
        elif r<R2: return rho_sh
        else: return rho0
    jet = TophatJet(theta_c=float(theta_c), E_iso=float(E_iso), Gamma0=float(Gamma0))
    obs = Observer(lumi_dist=float(cfg.lumi_dist), z=float(cfg.z), theta_obs=float(cfg.theta_obs))
    rad = Radiation(eps_e=float(eps_e), eps_B=float(eps_B), p=float(p))
    return Model(jet=jet, medium=Medium(rho=rho_fn), observer=obs, fwd_rad=rad)

def predict(model, times, freqs, cfg: FitConfig):
    """
    Predict model flux at arbitrary (time, frequency) pairs using
    VegasAfterglow's flux_density_grid(times, freqs).

    Returns an array with the same shape as `times` / `freqs`.
    """
    times = np.asarray(times, float)
    freqs = np.asarray(freqs, float)

    if times.shape != freqs.shape:
        raise ValueError(f"predict: times.shape={times.shape} and "
                         f"freqs.shape={freqs.shape} must match")

    # Flatten for convenience, but remember original shape
    orig_shape = times.shape
    t_flat = times.ravel()
    nu_flat = freqs.ravel()

    # Basic sanity mask
    good = (np.isfinite(t_flat) & np.isfinite(nu_flat) &
            (t_flat > 0.0) & (nu_flat > 0.0))
    if not np.any(good):
        return np.full(orig_shape, np.nan, float)

    # Replace any bad entries by medians so the grid call still works
    t_work = t_flat.copy()
    nu_work = nu_flat.copy()
    t_med = np.nanmedian(t_flat[good])
    nu_med = np.nanmedian(nu_flat[good])
    t_work[~good] = t_med
    nu_work[~good] = nu_med

    # Unique grids in time and frequency
    t_unique, t_inv = np.unique(t_work, return_inverse=True)
    nu_unique, nu_inv = np.unique(nu_work, return_inverse=True)

    # This is the API that works in Wind_Bubble_Model.py:
    # grid = model.flux_density_grid(times, freqs)
    # grid.total has shape (Nfreq, Ntime) and we index [ifreq, itime]
    grid = model.flux_density_grid(t_unique, nu_unique)
    A = np.asarray(grid.total)
    A = np.squeeze(A)

    # We expect A.shape == (len(nu_unique), len(t_unique))
    n_nu = nu_unique.size
    n_t  = t_unique.size

    if A.ndim == 1:
        # Only one time or one frequency
        if A.size == n_t and n_nu == 1:
            A = A.reshape(1, n_t)
        elif A.size == n_nu and n_t == 1:
            A = A.reshape(n_nu, 1)
        else:
            A = A.reshape(n_nu, n_t)
    elif A.ndim == 2:
        if A.shape != (n_nu, n_t):
            # Maybe the axes are swapped
            if A.shape == (n_t, n_nu):
                A = A.T
            else:
                A = A.reshape(n_nu, n_t)
    else:
        # Fallback: flatten then reshape
        A = A.reshape(n_nu, n_t)

    # Now A[nu_index, t_index]
    preds_flat = A[nu_inv, t_inv].astype(float)

    # Restore NaNs in originally bad entries
    preds_flat[~good] = np.nan

    return preds_flat.reshape(orig_shape)


def log_prior_uniform(theta, bounds_map):
    for val, name in zip(theta, PARAM_NAMES):
        lo, hi = bounds_map[name]
        if not (lo <= val <= hi):
            return -np.inf
    return 0.0

def log_prior_normal(theta, normal_map):
    # Sum of independent Gaussian log-priors (optional)
    lp = 0.0
    for val, name in zip(theta, PARAM_NAMES):
        if name in normal_map:
            mu, sig = normal_map[name]
            if sig is None or sig <= 0: 
                continue
            d = (val - mu)/sig
            lp += -0.5*(d*d) - math.log(max(sig,1e-300)) - 0.5*math.log(2*math.pi)
    return lp

def log_like(theta, t, nu, f, e, cfg: FitConfig):
    try:
        m = predict(build_model(theta,cfg), t, nu, cfg)
        if not all_finite(m) or np.any(e<=0): return -np.inf
        return -0.5*np.sum(((f-m)/e)**2)
    except Exception:
        return -np.inf

def log_prob(theta, t, nu, f, e, cfg: FitConfig, bounds_map, normal_map):
    lp_u = log_prior_uniform(theta, bounds_map)
    if not np.isfinite(lp_u): return -np.inf
    lp_n = log_prior_normal(theta, normal_map)
    ll   = log_like(theta,t,nu,f,e,cfg)
    return (lp_u + lp_n + ll) if np.isfinite(ll) else -np.inf

# ---------- priors I/O ----------
def load_priors_toml(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = toml.load(f)
    return data

def extract_bounds_from_toml(data: Dict[str,Any], defaults=DEFAULT_BOUNDS) -> Dict[str,Tuple[float,float]]:
    out = dict(defaults)
    if "bounds" in data:
        for k,v in data["bounds"].items():
            if isinstance(v, (list,tuple)) and len(v)==2:
                out[k] = (float(v[0]), float(v[1]))
    return out

def extract_normals_from_toml(data: Dict[str,Any]) -> Dict[str,Tuple[float,float]]:
    out = {}
    if "priors" in data:
        for k,v in data["priors"].items():
            if isinstance(v, dict) and v.get("type","").lower()=="normal":
                mu = float(v.get("mu"))
                sig= float(v.get("sigma"))
                out[k] = (mu, sig)
    return out

def extract_init_from_toml(data: Dict[str,Any], theta0_default: np.ndarray) -> np.ndarray:
    init = theta0_default.copy()
    def set_if(name, val):
        idx = PARAM_NAMES.index(name)
        init[idx] = float(val)
    if "init" in data and isinstance(data["init"], dict):
        for name,val in data["init"].items():
            if name in PARAM_NAMES:
                set_if(name, val)
    elif "best" in data and isinstance(data["best"], dict):
        for name,val in data["best"].items():
            if name in PARAM_NAMES:
                set_if(name, val)
    return init

def chain_summary(chain_flat: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Return median, mean, std, p16, p84 per parameter from flat samples."""
    stats = {}
    for i, name in enumerate(PARAM_NAMES):
        x = chain_flat[:, i]
        x = x[np.isfinite(x)]
        if x.size == 0:
            stats[name] = {"median": np.nan, "mean": np.nan, "std": np.nan, "p16": np.nan, "p84": np.nan}
            continue
        stats[name] = {
            "median": float(np.median(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1) if x.size>1 else 0.0),
            "p16": float(np.percentile(x, 16.0)),
            "p84": float(np.percentile(x, 84.0)),
        }
    return stats

def save_results_toml(path_out: str, best: np.ndarray, chain: np.ndarray, bounds_map, normal_map, cfg: FitConfig):
    flat_chain = chain.reshape(-1, chain.shape[-1])
    stats = chain_summary(flat_chain)

    data = {
        "meta": {
            "note": "VegasAfterglow SPEED fit; values in native units; can be used as priors for future runs.",
            "z": cfg.z,
            "lumi_dist_cm": cfg.lumi_dist,
            "theta_obs": cfg.theta_obs,
            "mu": cfg.mu,
            "nwalkers": cfg.nwalkers,
            "burn": cfg.burn,
            "nsteps": cfg.nsteps,
            "num_nu_grid": cfg.num_nu_grid,
        },
        "best": {name: float(val) for name, val in zip(PARAM_NAMES, best)},
        "init": {name: float(val) for name, val in zip(PARAM_NAMES, best)},  # next run starts at best
        "bounds": {name: [float(lo), float(hi)] for name, (lo,hi) in bounds_map.items()},
        "priors": {},
        "chain_stats": stats,
    }

    # If we have useful std from chain, write normal priors with 1σ = std (guarding small/zero)
    for name in PARAM_NAMES:
        std = stats[name]["std"]
        if std and np.isfinite(std) and std > 0:
            data["priors"][name] = {"type": "normal", "mu": data["best"][name], "sigma": std}
        elif name in normal_map:
            mu, sigma = normal_map[name]
            data["priors"][name] = {"type": "normal", "mu": mu, "sigma": sigma}

    # Write TOML
    if toml_w is not None:
        with open(path_out, "wb") as f:
            toml_w.dump(data, f)
    else:
        # Minimal manual TOML writer
        def toml_escape_key(k): return k
        lines = []
        lines.append("[meta]")
        for k,v in data["meta"].items():
            if isinstance(v, str): lines.append(f'{k} = "{v}"')
            else: lines.append(f"{k} = {v}")
        lines.append("\n[best]")
        for k,v in data["best"].items():
            lines.append(f"{k} = {v}")
        lines.append("\n[init]")
        for k,v in data["init"].items():
            lines.append(f"{k} = {v}")
        lines.append("\n[bounds]")
        for k,(lo,hi) in data["bounds"].items():
            lines.append(f'{k} = [{lo}, {hi}]')
        lines.append("\n[priors]")
        for k,v in data["priors"].items():
            lines.append(f'[{ "priors."+k }]')
            lines.append(f'type = "normal"')
            lines.append(f"mu = {v['mu']}")
            lines.append(f"sigma = {v['sigma']}")
        lines.append("\n[chain_stats]")
        for k,st in data["chain_stats"].items():
            lines.append(f'[{ "chain_stats."+k }]')
            for kk,v in st.items():
                lines.append(f"{kk} = {v}")
        with open(path_out, "w") as f:
            f.write("\n".join(lines))

# ---------- sampler ----------
def init_walkers(theta0, nwalkers, rng, bounds_map):
    ndim=theta0.size
    rel=np.array([0.02,0.05,0.02,0.05,0.05,0.01,0.05,0.05,0.05])
    step=np.maximum(np.abs(theta0),1.0)*rel
    step[2]=max(0.02,step[2]); step[5]=max(0.01,step[5]); step[7]=max(0.05,step[7]); step[8]=max(0.05,step[8])

    # QR directions for full rank
    Q,_=np.linalg.qr(rng.standard_normal((ndim,ndim)))
    p0=np.zeros((nwalkers,ndim),float)
    for i in range(nwalkers):
        trial = theta0 + 0.5*((i+1)/nwalkers)*step*Q[:,i%ndim] + rng.standard_normal(ndim)*step*0.5
        # clip to bounds
        for j,name in enumerate(PARAM_NAMES):
            lo, hi = bounds_map[name]
            trial[j] = min(max(trial[j], lo), hi)
        p0[i,:]=trial
    return p0

def run_emcee(t, nu, f, e, theta0, cfg: FitConfig,
              bounds_map, normal_map,
              state0=None, chain0=None, lnps0=None, resume=False):
    """
    Run or resume an emcee EnsembleSampler.

    If resume == False:
        - Initialize walkers around theta0
        - Run burn-in, reset, then production run of cfg.nsteps.

    If resume == True and state0 is not None:
        - Start from saved sampler.State and take another cfg.nsteps.
        - If chain0/lnps0 are provided, append new samples onto them.

    Returns:
        best       : (ndim,) best-fit parameters (max log-posterior)
        chain_all  : (nwalkers, nsteps_total, ndim) full chain
        lnps_all   : (nwalkers, nsteps_total)     full log-prob
        state      : final emcee sampler.State for future resume
    """
    rng = np.random.default_rng(cfg.seed)
    ndim = theta0.size
    if cfg.nwalkers < 2 * ndim:
        raise ValueError("nwalkers too small")

    # New sampler every time (state object carries coords & RNG)
    sampler = emcee.EnsembleSampler(
        cfg.nwalkers,
        ndim,
        log_prob,
        args=(t, nu, f, e, cfg, bounds_map, normal_map),
        moves=emcee.moves.StretchMove(a=1.8),
    )

    if resume and (state0 is not None):
        print("[run_emcee] Resuming from saved state.")
        # Continue chain from state0
        state = sampler.run_mcmc(state0, cfg.nsteps, progress=True)
        new_chain = sampler.get_chain()
        new_lnps  = sampler.get_log_prob()

        if (chain0 is not None) and (lnps0 is not None):
            # Append in time (axis=1)
            if chain0.shape[0] != new_chain.shape[0] or chain0.shape[2] != new_chain.shape[2]:
                raise ValueError("Saved chain does not match current sampler dimensions.")
            chain_all = np.concatenate([chain0, new_chain], axis=1)
            lnps_all  = np.concatenate([lnps0,  new_lnps],  axis=1)
        else:
            chain_all = new_chain
            lnps_all  = new_lnps

    else:
        print("[run_emcee] Starting fresh chain.")
        p0 = init_walkers(theta0, cfg.nwalkers, rng, bounds_map)
        state = sampler.run_mcmc(p0, cfg.burn, progress=True)
        sampler.reset()
        state = sampler.run_mcmc(state, cfg.nsteps, progress=True)
        chain_all = sampler.get_chain()
        lnps_all  = sampler.get_log_prob()

    # Best point over entire chain
    flat_chain = chain_all.reshape(-1, ndim)
    flat_lnps  = lnps_all.reshape(-1)
    ibest = int(np.argmax(flat_lnps))
    best = flat_chain[ibest, :]

    return best, chain_all, lnps_all, state

# ---------- plotting ----------
def plot_lightcurves(model, t, nu, f, e, outpath, cfg: FitConfig):
    """
    Plot model light curves + binned data.

    For each frequency group, we construct a time grid, evaluate the
    model at that (t, nu0) pair using `predict`, then overplot the
    binned data in the same color.
    """
    groups = group_by_frequency(nu)

    # Time grid for the model curves
    tmin, tmax = float(MINIMUM_TIME), float(np.nanmax(t))
    t_grid = np.logspace(
    np.log10(max(tmin, 100)),
    np.log10(tmax * 1.2),
    300,
    )

    plt.figure()

    for g in groups:
        nu0 = float(np.median(nu[g]))

        # Frequency array matching t_grid, all at nu0
        nu_vec = np.full_like(t_grid, nu0, dtype=float)

        # Evaluate model using the same machinery as log_like
        try:
            y = predict(model, t_grid, nu_vec, cfg)
        except Exception as err:
            print(f"[plot_lightcurves] Failed at nu ≈ {nu0:.3g} Hz: {err}")
            continue

        y = np.asarray(y, float)

        # Mask non-finite/nonpositive values for log–log
        m_mod = np.isfinite(y) & (y > 0.0)
        n_good = np.count_nonzero(m_mod)
        if n_good == 0:
            print(f"[plot_lightcurves] nu ≈ {nu0:.3g} Hz: no positive model flux; skipping.")
            continue

        # Draw model curve and grab its color
        (line,) = plt.loglog(t_grid[m_mod], y[m_mod],
                             lw=1.8, label=f"{nu0:.3g} Hz")
        col = line.get_color()

        # Overplot the binned data in the same color
        plt.errorbar(
            t[g], f[g], yerr=e[g],
            fmt="o", ms=4, capsize=2,
            color=col, ecolor=col, markeredgecolor=col,
            linestyle="none",
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Flux density [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")

    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend(loc="best", fontsize=8)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    print(f"[plot_lightcurves] Saved plot to {outpath}")
    plt.show()
    plt.close()


def pick_epochs(t, n=3):
    """
    Pick up to n representative epochs from the time array:
    early, middle, late in log time.
    """
    t = np.asarray(t, float)
    tpos = t[t > 0.0]
    if tpos.size == 0:
        return []
    tpos = np.unique(tpos)
    if tpos.size <= n:
        return tpos.tolist()
    qs = np.quantile(np.log10(tpos), [0.05, 0.5, 0.95])
    return (10.0 ** qs).tolist()


def plot_spectra(model, t, nu, f, e, outpath):
    """
    Plot model spectra at a few epochs + nearby data.

    Uses VegasAfterglow's flux_density_grid(times, freqs) just like in
    Wind_Bubble_Model.py.  Each chosen epoch gets one model spectrum.
    """
    epochs = pick_epochs(t, 3)
    if not epochs:
        print("[plot_spectra] No valid epochs found; skipping.")
        return

    # Data-based frequency range
    nu_min_data = float(np.nanmin(nu))
    nu_max_data = float(np.nanmax(nu))
    nu_min = float(100E6)
    nu_max = float(1E18)
    print("Min and max nu = ", nu_min_data, nu_max_data)

    # Frequency grid for spectra
    nu_grid = np.logspace(
        np.log10(nu_min * 0.8),
        np.log10(nu_max * 1.2),
        300,
    )

    epochs_arr = np.array(epochs, float)

    # One call to flux_density_grid for all epochs
    spec_grid = model.flux_density_grid(epochs_arr, nu_grid)
    A = np.asarray(spec_grid.total)
    A = np.squeeze(A)

    n_nu = nu_grid.size
    n_t  = epochs_arr.size

    # We expect shape (Nfreq, Nepochs)
    if A.ndim == 1:
        if A.size == n_nu and n_t == 1:
            A = A.reshape(n_nu, 1)
        else:
            raise ValueError(
                f"[plot_spectra] unexpected 1D total of size {A.size} "
                f"(expected {n_nu} or {n_nu * n_t})"
            )
    elif A.ndim == 2:
        if A.shape != (n_nu, n_t):
            # Maybe transposed
            if A.shape == (n_t, n_nu):
                A = A.T
            else:
                raise ValueError(
                    f"[plot_spectra] unexpected 2D total shape {A.shape}; "
                    f"expected ({n_nu}, {n_t}) or ({n_t}, {n_nu})"
                )
    else:
        raise ValueError(f"[plot_spectra] unexpected ndim={A.ndim} for total")

    plt.figure()
    for j, tj in enumerate(epochs_arr):
        y = np.asarray(A[:, j], float)
        m_mod = np.isfinite(y) & (y > 0.0)
        if not np.any(m_mod):
            print(f"[plot_spectra] Epoch t={tj:.3g} s has no positive model flux; skipping.")
            continue

        (line,) = plt.loglog(nu_grid[m_mod], y[m_mod],
                             lw=1.8, label=f"t = {tj:.3g} s")
        col = line.get_color()

        # Data within ~20% in time, same color
        m_data = (t > tj / 1.2) & (t < tj * 1.2)
        m_data &= np.isfinite(f) & np.isfinite(e) & (e > 0.0)
        if np.any(m_data):
            plt.errorbar(
                nu[m_data], f[m_data], yerr=e[m_data],
                fmt="o", ms=4, capsize=2,
                color=col, ecolor=col, markeredgecolor=col,
                linestyle="none",
            )

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Flux density [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend(loc="best", fontsize=8)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    print(f"[plot_spectra] Saved plot to {outpath}")
    plt.show()
    plt.close()

def print_parameters(theta, names=PARAM_NAMES, header="Parameters"):
    """
    Print a nicely formatted list of (name, value) pairs.
    """
    print(f"\n{header}:")
    for name, val in zip(names, theta):
        print(f"  {name:12s} = {val:.6g}")
    print()

def apply_dylan_priors(data: Dict[str, Any],
                       theta0_default: np.ndarray,
                       bounds_default: Dict[str, Tuple[float, float]],
                       cfg: FitConfig
                      ) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
    """
    Interpret Dylan's JetFit-style `parameters.toml` and map onto
    VegasAfterglow parameters.

    Returns:
      (theta, bounds), and prints only those parameters provided in Dylan's file.
    """
    theta = theta0_default.copy()
    bounds = dict(bounds_default)
    updated_params = []  # <-- track what changed

    models = data.get("model", [])
    if not isinstance(models, list):
        models = [models]

    def set_param(name_va: str, value: float):
        if name_va in PARAM_NAMES:
            idx = PARAM_NAMES.index(name_va)
            theta[idx] = float(value)
            updated_params.append(name_va)

    def set_bounds(name_va: str, lo: float, hi: float):
        if name_va in bounds:
            bounds[name_va] = (float(lo), float(hi))
            # do not track bounds updates unless you want to display them too
    for m in models:
        mname = m.get("name")
        scale = str(m.get("scale", "linear")).lower()
        prior = m.get("prior", {})
        lower = prior.get("lower")
        upper = prior.get("upper")
        ig    = prior.get("initial_guess")

        # --- EXISTING mappings: E, nt, rt, eps_e, eps_b, p, rho0, etc. ---

        if mname == "E":  # log10(E_iso/1e52)
            ...
        elif mname == "nt":
            ...
        elif mname == "rt":
            ...
        elif mname == "eps_e":
            ...
        elif mname == "eps_b":
            ...
        elif mname == "p":
            ...
        # --- NEW: cosmology from Dylan's file ---
        elif mname == "z":
            # Dylan's file has: name = "z", value = 1.1
            if "value" in m:
                cfg.z = float(m["value"])
        elif mname == "dL":
            # JetFit typically uses dL in units of 1e28 cm.
            # For GRB="080413B": dL = 2.36 → 2.36 × 10^28 cm
            if "value" in m:
                cfg.lumi_dist = float(m["value"]) * 1.0e28
        # ---------------------------------------
        elif mname == "rho0":  # some FireballModel files
            if ig is not None:
                set_param("log10_n_ism", float(ig))
            if lower is not None and upper is not None:
                set_bounds("log10_n_ism", float(lower), float(upper))


    # ------- Print only the updated parameters --------
    print("\n[apply_dylan_priors] Using Dylan's priors for:")
    if updated_params:
        for name in updated_params:
            idx = PARAM_NAMES.index(name)
            print(f"  {name:12s} = {theta[idx]:.6g}")
    else:
        print("  (No overlapping parameters found.)")

    return theta, bounds



# ---------- main ----------
def main():
    global GRB_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default="Data/"+GRB_NAME+"/"+GRB_NAME+".csv",
        help="Path to data CSV file",
    )
    ap.add_argument(
        "--grb",
        default=GRB_NAME,
        help="Path to data CSV file",
    )
    ap.add_argument(
        "--priors_in",
        default=None,
        help="TOML file with bounds/priors/init to read (default: parameters.toml next to --data)",
    )
    ap.add_argument(
        "--priors_out",
        default=None,
        help="TOML file to write best+priors (default: "+GRB_NAME+"_fit_results.toml next to --data)",
    )
    ap.add_argument(
        "--no_mcmc",
        action="store_true",
        help="Only plot prior model over data, do not run emcee",
    )
    args = ap.parse_args()

    GRB_NAME = str(args.grb)
    print(f"GRB NAME: {GRB_NAME}")

    # ----- auto-detect priors: prefer our own fit_results.toml over parameters.toml -----
    data_dir = os.path.dirname(args.data)
    print(f"data_dir: {data_dir}")
    fit_toml = data_dir+"/"+GRB_NAME+"_fit_results.toml"
    print(fit_toml)

    if os.path.exists(fit_toml):
        args.priors_in = fit_toml
        print(f"Auto-detected priors file from previous fit: {args.priors_in}")
    elif os.path.exists(dylan_toml):
        args.priors_in = dylan_toml
        print(f"Auto-detected Dylan priors file: {args.priors_in}")
    else:
        print("No fit_results.toml or parameters.toml found.  Using built-in defaults.")

    # ------------------------------------------------------------------------------

    print(f"Reading data: {args.data}")
    t, nu, f, e = load_080413_schema(args.data)
    tb, nub, fb, eb = bin_data(t, nu, f, e)
    print(f"Parsed rows: {t.size}  |  Binned rows: {tb.size}")

    # Defaults
    theta0 = np.array([2.0e52, 300.0, 0.10, 1.0e-1, 1.0e-3, 2.30, 1.0e17, 0.0, -2.0], float)
    print_parameters(theta0, names=PARAM_NAMES, header="Default Parameters")
    cfg = FitConfig()
    bounds_map = dict(DEFAULT_BOUNDS)
    normal_map: Dict[str, Tuple[float, float]] = {}

    # ----- Load priors if provided (possibly auto-detected) -----
    if args.priors_in and os.path.exists(args.priors_in):
        print(f"Reading priors from: {args.priors_in}")
        data = load_priors_toml(args.priors_in)

        # --- NEW: read cosmology from meta if present ---
        meta = data.get("meta", {})
        if "z" in meta:
            cfg.z = float(meta["z"])
            print("z=", cfg.z)
        if "lumi_dist_cm" in meta:
            cfg.lumi_dist = float(meta["lumi_dist_cm"])
            print("lumi_dist=", cfg.lumi_dist)
        # ------------------------------------------------

        # For *our* own fit_results.toml format (with [bounds], [priors], [init]/[best])
        bounds_map = extract_bounds_from_toml(data, defaults=bounds_map)
        normal_map = extract_normals_from_toml(data)
        theta0 = extract_init_from_toml(data, theta0)

        # For Dylan's JetFit-style format
        if "model" in data:
            theta0, bounds_map = apply_dylan_priors(data, theta0, bounds_map, cfg)

        print_parameters(theta0, names=PARAM_NAMES, header="Prior Parameters")

    else:
        print(f"Could not read priors from: {args.priors_in}.  Using default values.")

    # ---------- PRIOR model debug plots ----------
    model_prior = build_model(theta0, cfg)

    outdir = os.path.join(os.path.dirname(__file__), "Data/"+GRB_NAME)
    os.makedirs(outdir, exist_ok=True)

    print("THETA0_START", theta0)

    print("\nPlotting PRIOR model over binned data")
    plot_lightcurves(
        model_prior,
        tb, nub, fb, eb,
        os.path.join(outdir, "lightcurves_prior.png"),
        cfg,
    )

    # Spectra are optional for debug: never crash if they misbehave.
    # try:
    #     plot_spectra(
    #         model_prior,
    #         tb, nub, fb, eb,
    #         os.path.join(outdir, "spectra_prior.png"),
    #     )
    # except Exception as err:
    #     print(f"[plot_spectra] Could not plot PRIOR spectra: {err}")

    if args.no_mcmc:
        print("Skipping emcee (--no_mcmc set).")
        return

    # ---------------------------------------------

    # ---------- Run emcee (with optional resume) ----------
    npz_filename = os.path.join(outdir, GRB_NAME + "_fit_results.npz")
    state_filename = os.path.join(outdir, GRB_NAME + "_emcee_state.pkl")

    if PLOT_ONLY:
        print("\nReading file " + npz_filename + "  -- not fitting data because PLOT_ONLY is True")
        data = np.load(npz_filename)
        best = data["best"]  # (ndim,)
        chain = data["chain"]  # (nwalkers, nsteps, ndim)
        lnps = data["lnps"]  # (nwalkers, nsteps)

    else:
        # Optional resume
        state0 = None
        chain0 = None
        lnps0 = None

        if os.path.exists(npz_filename) and os.path.exists(state_filename):
            print("\nResuming emcee from previous state.")
            data = np.load(npz_filename)
            chain0 = data["chain"]
            lnps0 = data["lnps"]
            with open(state_filename, "rb") as f:
                state0 = pickle.load(f)
            resume = True
        else:
            resume = False

        print("\nRunning emcee.")
        best, chain, lnps, state = run_emcee(
            tb, nub, fb, eb,
            theta0,
            cfg,
            bounds_map,
            normal_map,
            state0=state0,
            chain0=chain0,
            lnps0=lnps0,
            resume=resume,
        )
        print("\nDone running emcee. -- PLOT_ONLY is False")

        print("\nWriting emcee results into file " + npz_filename + ".")
        np.savez(
            npz_filename,
            best=best,
            chain=chain,
            lnps=lnps,
        )

        print("Writing emcee state into file " + state_filename + ".")
        with open(state_filename, "wb") as f:
            pickle.dump(state, f)

    print("\nBest-fit parameters (max posterior):")
    print_parameters(best, names=PARAM_NAMES, header="Best-fit parameters (max posterior)")
    # --------------------------------

    # ---------- Save results before plotting ----------
    out_path = outdir+"/"+GRB_NAME+"_fit_results.toml"
    save_results_toml(out_path, best, chain, bounds_map, normal_map, cfg)
    print(f"Saved priors/results to: {out_path}")
    # -------------------------------------------------
    print("theta0 at start:", theta0)
    print("best at end:", best)

    # ---------- Best-fit plots ----------
    model_best = build_model(best, cfg)

    plot_lightcurves(
        model_best, tb, nub, fb, eb,
        os.path.join(outdir, "lightcurves_speed.png"), cfg
    )


    print("BEST_END", best)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: sys.exit(130)
