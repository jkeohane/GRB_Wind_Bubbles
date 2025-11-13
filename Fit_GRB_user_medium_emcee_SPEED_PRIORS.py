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

import sys, os, math, re, argparse
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

PARAM_NAMES = ["E_iso","Gamma0","theta_c","eps_e","eps_B","p","R_t","log10_n_t","log10_n_ism"]

# ---------- config ----------
@dataclass
class FitConfig:
    z: float = 0.1
    lumi_dist: float = 1e26
    theta_obs: float = 0.0
    mu: float = 1.3
    num_nu_grid: int = 32
    nwalkers: int = 64
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
    mask_base = np.isfinite(t)&np.isfinite(nu)&np.isfinite(f)&(nu>0)
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
    mask = np.isfinite(t)&np.isfinite(nu)&np.isfinite(f)&np.isfinite(e)&(nu>0)&(e>0)
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
    t_grid = np.unique(times.astype(float))
    nu_min, nu_max = float(np.nanmin(freqs)), float(np.nanmax(freqs))
    grid = model.flux(t=t_grid, nu_min=nu_min, nu_max=nu_max, num_nu=cfg.num_nu_grid)
    preds=np.empty_like(times,float)
    for k in range(times.size):
        preds[k]=float(grid.total[nearest_index(grid.nu,freqs[k]), nearest_index(grid.t,times[k])])
    return preds

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

def run_emcee(t, nu, f, e, theta0, cfg: FitConfig, bounds_map, normal_map):
    rng=np.random.default_rng(cfg.seed); ndim=theta0.size
    if cfg.nwalkers<2*ndim: raise ValueError("nwalkers too small")
    p0=init_walkers(theta0,cfg.nwalkers,rng,bounds_map)
    sampler=emcee.EnsembleSampler(cfg.nwalkers, ndim, log_prob, args=(t,nu,f,e,cfg,bounds_map,normal_map), moves=emcee.moves.StretchMove(a=1.8))
    state=sampler.run_mcmc(p0, cfg.burn, progress=True)
    sampler.reset(); sampler.run_mcmc(state, cfg.nsteps, progress=True)
    flat_lnprob=sampler.get_log_prob(flat=True); flat_chain=sampler.get_chain(flat=True)
    ibest=int(np.argmax(flat_lnprob)); best=flat_chain[ibest,:]
    return best, sampler.get_chain(), sampler.get_log_prob()

# ---------- plotting ----------
def plot_lightcurves(model, t, nu, f, e, outpath):
    """
    Plot model light curves + data.
    Each frequency group gets one model curve and matching–color data points.
    """
    groups = group_by_frequency(nu)

    tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))
    t_grid = np.logspace(np.log10(max(tmin, 1e-6)),
                         np.log10(tmax * 1.2),
                         300)

    plt.figure()
    for g in groups:
        nu0 = float(np.median(nu[g]))

        # Model curve for this frequency
        grid = model.flux(t=t_grid, nu_min=nu0, nu_max=nu0, num_nu=1)
        y = robust_total_to_curve(grid.total, expect_len=t_grid.size)

        # Draw model line and grab its color
        (line,) = plt.loglog(t_grid, y, lw=1.8, label=f"{nu0:.3g} Hz")
        col = line.get_color()
        plt.plot(np.log10(t_grid), np.log10(y), color=col)
        # Data points in the same color
        plt.errorbar(
            t[g], f[g], yerr=e[g],
            fmt="o", ms=4, capsize=2,
            color=col, ecolor=col, markeredgecolor=col,
            linestyle="none",
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Flux density [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
    plt.legend(loc="best", fontsize=8)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()
    plt.close()


def pick_epochs(t, n=3):
    tl=np.sort(np.unique(t)); 
    if tl.size<=n: return tl.tolist()
    qs=np.quantile(np.log10(tl), [0.05,0.5,0.95]); return (10**qs).tolist()

def robust_total_to_curve(grid_total, expect_len):
    A = np.asarray(grid_total); A = np.squeeze(A)
    if A.ndim == 1:
        return A if A.size==expect_len else np.ravel(A)[:expect_len]
    if A.ndim == 2:
        if A.shape[0]==expect_len and A.shape[1]==1: return A[:,0]
        if A.shape[1]==expect_len and A.shape[0]==1: return A[0,:]
        return np.ravel(A)[:expect_len]
    return np.ravel(A)[:expect_len]

def plot_spectra(model, t, nu, f, e, outpath):
    """
    Plot model spectra at a few epochs + nearby data.
    Each epoch gets one spectrum and same-color points.
    """
    epochs = pick_epochs(t, 3)

    nu_min = max(1e1, float(np.nanmin(nu)) * 0.8)
    nu_max = float(np.nanmax(nu)) * 1.2
    nu_grid = np.logspace(np.log10(nu_min), np.log10(nu_max), 300)

    plt.figure()
    for tj in epochs:
        # Model spectrum at this epoch
        grid = model.flux(t=np.array([tj]),
                          nu_min=nu_min, nu_max=nu_max,
                          num_nu=nu_grid.size)
        y = robust_total_to_curve(grid.total, expect_len=nu_grid.size)

        # Draw model line and grab its color
        (line,) = plt.loglog(nu_grid, y, lw=1.8, label=f"t = {tj:.3g} s")
        col = line.get_color()

        # Data within ~20% in time, same color
        m = (t > tj / 1.2) & (t < tj * 1.2)
        if np.any(m):
            plt.errorbar(
                nu[m], f[m], yerr=e[m],
                fmt="o", ms=4, capsize=2,
                color=col, ecolor=col, markeredgecolor=col,
                linestyle="none",
            )

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Flux density [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
    plt.legend(loc="best", fontsize=8)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
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
                       bounds_default: Dict[str, Tuple[float, float]]
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

        if mname == "E":  # log10(E_iso/1e52)
            if ig is not None:
                set_param("E_iso", 10.0**float(ig) * 1e52)
            if lower is not None and upper is not None:
                set_bounds("E_iso", 10.0**float(lower)*1e52, 10.0**float(upper)*1e52)

        elif mname == "nt":  # log10_n_t
            if ig is not None:
                set_param("log10_n_t", float(ig))
            if lower is not None and upper is not None:
                set_bounds("log10_n_t", float(lower), float(upper))

        elif mname == "rt":  # 10^rt cm
            if ig is not None:
                set_param("R_t", 10.0**float(ig))
            if lower is not None and upper is not None:
                set_bounds("R_t", 10.0**float(lower), 10.0**float(upper))

        elif mname == "eps_e":
            if ig is not None:
                val = 10.0**float(ig) if scale == "log" else float(ig)
                set_param("eps_e", val)
            if lower is not None and upper is not None:
                if scale == "log":
                    set_bounds("eps_e", 10.0**float(lower), 10.0**float(upper))
                else:
                    set_bounds("eps_e", float(lower), float(upper))

        elif mname == "eps_b":
            if ig is not None:
                val = 10.0**float(ig) if scale == "log" else float(ig)
                set_param("eps_B", val)
            if lower is not None and upper is not None:
                if scale == "log":
                    set_bounds("eps_B", 10.0**float(lower), 10.0**float(upper))
                else:
                    set_bounds("eps_B", float(lower), float(upper))

        elif mname == "p":
            if ig is not None:
                set_param("p", float(ig))
            if lower is not None and upper is not None:
                set_bounds("p", float(lower), float(upper))

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
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default="/Users/jkeohane/GRBs/GRB_Wind_Bubbles/Data/080413B/080413B.csv"
    )
    ap.add_argument(
        "--priors_in",
        default=None,
        help="TOML file with bounds/priors/init to read (default: parameters.toml next to --data)"
    )
    ap.add_argument(
        "--priors_out",
        default="fit_results.toml",
        help="TOML file to write best+priors"
    )
    args = ap.parse_args()

    # ----- NEW: auto-detect parameters.toml next to the data file -----
    if args.priors_in is None:
        data_dir = os.path.dirname(args.data)
        auto_toml = os.path.join(data_dir, "parameters.toml")
        if os.path.exists(auto_toml):
            args.priors_in = auto_toml
            print(f"Auto-detected priors file: {args.priors_in}")
        else:
            print("No parameters.toml found next to data file.  Using built-in defaults.")
    # ------------------------------------------------------------------

    print(f"Reading data: {args.data}")
    t, nu, f, e = load_080413_schema(args.data)
    tb, nub, fb, eb = bin_data(t, nu, f, e)
    print(f"Parsed rows: {t.size}  |  Binned rows: {tb.size}")

    # Defaults
    # PARAM_NAMES = ["E_iso","Gamma0","theta_c","eps_e","eps_B","p","R_t","log10_n_t","log10_n_ism"]
    theta0 = np.array([2.0e52, 300.0, 0.10, 1.0e-1, 1.0e-3, 2.30, 1.0e17, 0.0, -2.0], float)
    print_parameters(theta0, names=PARAM_NAMES, header="Default Parameters")
    cfg = FitConfig()
    bounds_map = dict(DEFAULT_BOUNDS)
    normal_map: Dict[str, Tuple[float, float]] = {}

    # Load priors if provided (now possibly auto-detected)
    if args.priors_in and os.path.exists(args.priors_in):
        print(f"Reading priors from: {args.priors_in}")
        data = load_priors_toml(args.priors_in)

        # For *our* own fit_results.toml format (with [bounds], [priors], [init]/[best])
        bounds_map = extract_bounds_from_toml(data, defaults=bounds_map)
        normal_map = extract_normals_from_toml(data)
        theta0 = extract_init_from_toml(data, theta0)

        # For Dylan's JetFit-style format
        if "model" in data:
            theta0, bounds_map = apply_dylan_priors(data, theta0, bounds_map)

        print_parameters(theta0, names=PARAM_NAMES, header="Prior Parameters")
    else:
        print(f"Could not read priors from: {args.priors_in}.  Using default values.")


    best, chain, lnps = run_emcee(tb, nub, fb, eb, theta0, cfg, bounds_map, normal_map)

    print("\\nBest-fit parameters (max posterior):")
    for name,val in zip(PARAM_NAMES, best[:9]): print(f"  {name:12s} = {val:.6g}")

    model=build_model(best,cfg)
    outdir=os.path.join(os.path.dirname(__file__),"assets"); os.makedirs(outdir,exist_ok=True)
    plot_lightcurves(model, tb, nub, fb, eb, os.path.join(outdir,"lightcurves_speed.png"))
    plot_spectra(model, tb, nub, fb, eb, os.path.join(outdir,"spectra_speed.png"))

    # Save priors/results for the next run
    if args.priors_out:
        out_path = os.path.abspath(args.priors_out)
        save_results_toml(out_path, best, chain, bounds_map, normal_map, cfg)
        print(f"Saved priors/results to: {out_path}")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: sys.exit(130)
