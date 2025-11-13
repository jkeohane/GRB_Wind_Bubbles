
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fit_GRB_user_medium_emcee_FIXED_v7.py

Tailored CSV loader for schema with columns:
  Time, TimeUnits, Value, ValueLower, ValueUpper, ValueUnits, Wave, WaveUnits, Filter, ...

Converts units (mJy/Jy/uJy → cgs), builds (t, nu, f, e), fits with emcee,
and saves light curves + spectra overlaid with data.
"""

import sys, os, math, re
import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict

from VegasAfterglow import TophatJet, Observer, Radiation, Model, Medium

# ---------- constants ----------
mp = 1.67262192369e-24  # g
h  = 6.62607015e-27     # erg s
c  = 2.99792458e10      # cm s^-1

UNIT_SCALE = {
    'jy':  1e-23,
    'mjy': 1e-26,
    'ujy': 1e-29,
    'µjy': 1e-29,
}

FREQ_SCALE = {
    'hz': 1.0,
    'khz': 1e3,
    'mhz': 1e6,
    'ghz': 1e9,
    'thz': 1e12,
}

# ---------- config ----------
@dataclass
class FitConfig:
    z: float = 0.1
    lumi_dist: float = 1e26     # cm
    theta_obs: float = 0.0
    mu: float = 1.3             # mean molecular mass / mp
    num_nu_grid: int = 64
    nwalkers: int = 36
    nsteps: int = 600
    burn: int = 200
    seed: int = 1234

# ---------- CSV loader for the provided schema ----------
def load_080413_schema(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    # Required columns
    req = ['Time', 'Value', 'ValueUnits']
    for r in req:
        if r not in df.columns:
            raise ValueError(f"CSV missing required column: {r}")
    # Optional columns
    have_lower = 'ValueLower' in df.columns
    have_upper = 'ValueUpper' in df.columns
    have_wave  = 'Wave' in df.columns
    have_wunits = 'WaveUnits' in df.columns
    have_filter = 'Filter' in df.columns

    t = df['Time'].astype(float).to_numpy()

    # Frequency ν
    if have_wave and have_wunits:
        wave = df['Wave'].to_numpy()
        wunits = df['WaveUnits'].astype(str).str.lower().str.strip().to_numpy()
        nu = np.empty_like(t, dtype=float)
        nu.fill(np.nan)
        for i, (w, u) in enumerate(zip(wave, wunits)):
            if not (isinstance(w, (int, float, np.floating)) or (isinstance(w, str) and w.strip() != "")):
                continue
            try:
                val = float(w)
            except Exception:
                continue
            scale = FREQ_SCALE.get(u, None)
            if scale is None:
                # if units appear as 'hz' already, use 1.0 else skip
                scale = 1.0 if 'hz' in u else None
            if scale is not None and val > 0:
                nu[i] = val * scale / scale  # val already in Hz when u=='hz'
        # If some rows still NaN, try infer from filter (g,r,i,z)
        if have_filter and np.any(~np.isfinite(nu)):
            filt = df['Filter'].astype(str).str.lower().str.strip().to_numpy()
            opt = {'u': c/(365e-7), 'b': c/(445e-7), 'v': c/(551e-7), 'r': c/(658e-7), 'i': c/(806e-7), 'g': c/(477e-7), 'z': c/(900e-7)}
            for i in range(nu.size):
                if not np.isfinite(nu[i]):
                    band = filt[i][0] if len(filt[i])>0 else ''
                    if band in opt:
                        nu[i] = float(opt[band])
    else:
        raise ValueError("CSV needs Wave and WaveUnits (or add logic to infer from Filter).")

    # Flux density f
    val = df['Value'].to_numpy()
    vunits = df['ValueUnits'].astype(str).str.lower().str.strip().to_numpy()
    f = np.empty_like(t, dtype=float)
    f.fill(np.nan)
    for i, (x, u) in enumerate(zip(val, vunits)):
        try:
            xv = float(x)
        except Exception:
            continue
        scale = UNIT_SCALE.get(u, None)
        if scale is None:
            # Assume cgs if units missing/unknown
            scale = 1.0
        f[i] = xv * scale

    # Errors e
    e = np.full_like(f, np.nan, dtype=float)
    if have_lower or have_upper:
        lo = df['ValueLower'].to_numpy() if have_lower else np.full_like(f, np.nan)
        hi = df['ValueUpper'].to_numpy() if have_upper else np.full_like(f, np.nan)
        for i, u in enumerate(vunits):
            scale = UNIT_SCALE.get(u, 1.0)
            lo_i = lo[i] if isinstance(lo[i], (int,float,np.floating)) else np.nan
            hi_i = hi[i] if isinstance(hi[i], (int,float,np.floating)) else np.nan
            sigs = [s for s in [lo_i, hi_i] if np.isfinite(s) and s>0]
            if sigs:
                e[i] = max(sigs) * scale
    # Impute missing errors
    mask_base = np.isfinite(t) & np.isfinite(nu) & np.isfinite(f) & (nu>0)
    if np.any(~np.isfinite(e) & mask_base):
        # per-band median or 0.1*|f|
        idx = np.where(mask_base)[0]
        # group rows by frequency closeness
        order = idx[np.argsort(nu[idx])]
        groups = []
        group = [order[0]]
        for j in order[1:]:
            if abs(nu[j]-nu[group[-1]]) <= 1e-3*nu[group[-1]]:
                group.append(j)
            else:
                groups.append(group); group=[j]
        groups.append(group)
        e2 = e.copy()
        for g in groups:
            vals = [e[k] for k in g if np.isfinite(e[k]) and e[k]>0]
            med = float(np.median(vals)) if len(vals)>0 else None
            for k in g:
                if not (np.isfinite(e2[k]) and e2[k]>0):
                    e2[k] = med if (med is not None and med>0) else max(1e-6, 0.1*abs(f[k]))
        e = e2

    # Final mask
    mask = np.isfinite(t) & np.isfinite(nu) & np.isfinite(f) & np.isfinite(e) & (nu>0) & (e>0)
    if not np.any(mask):
        raise ValueError("After parsing, no valid rows with finite t,nu,f,e.")
    return t[mask], nu[mask], f[mask], e[mask]

# ---------- helpers & model ----------
def all_finite(x: np.ndarray) -> bool:
    return np.all(np.isfinite(x))

def nearest_index(arr: np.ndarray, x: float) -> int:
    return int(np.abs(np.asarray(arr,float) - x).argmin())

def predict_from_model(model: Model, times: np.ndarray, freqs: np.ndarray, cfg: FitConfig) -> np.ndarray:
    t_grid = np.unique(times.astype(float))
    nu_min = float(np.nanmin(freqs))
    nu_max = float(np.nanmax(freqs))
    grid = model.flux(t=t_grid, nu_min=nu_min, nu_max=nu_max, num_nu=max(cfg.num_nu_grid,4))
    preds = np.empty_like(times, float)
    for k in range(times.size):
        it = nearest_index(grid.t,  float(times[k]))
        inu= nearest_index(grid.nu, float(freqs[k]))
        preds[k] = float(grid.total[inu, it])
    return preds

def log_prior(theta: np.ndarray) -> float:
    E_iso, Gamma0, theta_c, eps_e, eps_B, p, R_t, log10_n_t, log10_n_ism = theta[:9]
    if not (1e50 <= E_iso <= 1e54): return -np.inf
    if not (10.0 <= Gamma0 <= 600.0): return -np.inf
    if not (1.0e-3 <= theta_c <= 0.5): return -np.inf
    if not (1.0e-5 <= eps_e <= 0.5): return -np.inf
    if not (1.0e-6 <= eps_B <= 0.5): return -np.inf
    if not (2.0 <= p <= 2.8): return -np.inf
    if not (1.0e14 <= R_t <= 1.0e20): return -np.inf
    if not (-6.0 <= theta[7] <= 6.0): return -np.inf
    if not (-6.0 <= theta[8] <= 6.0): return -np.inf
    return 0.0

def build_model_from_theta(theta, cfg: FitConfig) -> Model:
    E_iso, Gamma0, theta_c, eps_e, eps_B, p, R_t, log10_n_t, log10_n_ism = theta[:9]
    n_t   = 10.0 ** float(log10_n_t)
    n_ism = 10.0 ** float(log10_n_ism)
    m_mol = cfg.mu * mp
    def medium_fn(phi, th, r):
        r = max(1.0, float(r))
        n_sh = max(4.0*n_t, 4.0*n_ism)
        denom = (n_sh - n_ism)
        if denom <= 0.0: return np.nan
        R2 = float(R_t) * ((n_sh + 3.0*n_t) / denom) ** (1.0/3.0)
        rho_t  = n_t   * m_mol
        rho_sh = n_sh  * m_mol
        rho_0  = n_ism * m_mol
        if r < R_t:   return rho_t * (R_t/r)**2
        elif r < R2:  return rho_sh
        else:         return rho_0
    jet = TophatJet(theta_c=float(theta_c), E_iso=float(E_iso), Gamma0=float(Gamma0))
    obs = Observer(lumi_dist=float(cfg.lumi_dist), z=float(cfg.z), theta_obs=float(cfg.theta_obs))
    rad = Radiation(eps_e=float(eps_e), eps_B=float(eps_B), p=float(p))
    return Model(jet=jet, medium=Medium(rho=medium_fn), observer=obs, fwd_rad=rad)

def log_likelihood(theta, t, nu, f, e, cfg: FitConfig) -> float:
    try:
        model = build_model_from_theta(theta, cfg)
        m = predict_from_model(model, t, nu, cfg)
        if not all_finite(m) or np.any(e<=0): return -np.inf
        chi2 = np.sum(((f - m) / e) ** 2.0)
        return -0.5 * chi2
    except Exception:
        return -np.inf

def log_prob(theta, t, nu, f, e, cfg: FitConfig) -> float:
    lp = log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    ll = log_likelihood(theta, t, nu, f, e, cfg)
    return lp + ll if np.isfinite(ll) else -np.inf

def init_walkers_full_rank(theta0: np.ndarray, nwalkers: int, rng: np.random.Generator) -> np.ndarray:
    ndim = theta0.size
    rel = np.array([0.02,0.05,0.02,0.05,0.05,0.01,0.05,0.05,0.05])
    step = np.maximum(np.abs(theta0),1.0)*rel
    step[2]=max(0.02,step[2]); step[5]=max(0.01,step[5]); step[7]=max(0.05,step[7]); step[8]=max(0.05,step[8])
    Q,_ = np.linalg.qr(rng.standard_normal((ndim,ndim)))
    p0 = np.zeros((nwalkers,ndim),float)
    for i in range(nwalkers):
        det = 0.5*((i+1)/nwalkers)*step*Q[:,i%ndim]
        rnd = rng.standard_normal(ndim)*step*0.5
        p0[i,:] = theta0 + det + rnd
    return p0

def run_emcee(t, nu, f, e, theta0, cfg: FitConfig):
    rng = np.random.default_rng(cfg.seed)
    ndim = theta0.size
    if cfg.nwalkers < 2*ndim:
        raise ValueError(f"nwalkers must be >= {2*ndim}.")
    p0 = init_walkers_full_rank(theta0, cfg.nwalkers, rng)
    sampler = emcee.EnsembleSampler(cfg.nwalkers, ndim, log_prob, args=(t, nu, f, e, cfg),
                                    moves=emcee.moves.StretchMove(a=1.8))
    state = sampler.run_mcmc(p0, cfg.burn, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, cfg.nsteps, progress=True)
    flat_lnprob = sampler.get_log_prob(flat=True)
    flat_chain  = sampler.get_chain(flat=True)
    ibest = int(np.argmax(flat_lnprob))
    best  = flat_chain[ibest,:]
    return best, sampler.get_chain(), sampler.get_log_prob()

# ---------- plotting ----------
def group_by_frequency(nu: np.ndarray, rel_tol: float = 1e-3) -> Dict[int, np.ndarray]:
    order = np.argsort(nu)
    groups = {}
    current = [order[0]]
    gid = 0
    for idx in order[1:]:
        if abs(nu[idx] - nu[current[-1]]) <= rel_tol * nu[current[-1]]:
            current.append(idx)
        else:
            groups[gid] = np.array(current, int)
            gid += 1
            current = [idx]
    groups[gid] = np.array(current, int)
    return groups

def plot_lightcurves(model: Model, t: np.ndarray, nu: np.ndarray, f: np.ndarray, e: np.ndarray, outpath: str):
    groups = group_by_frequency(nu)
    tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))
    t_grid = np.logspace(np.log10(max(tmin, 1e-6)), np.log10(tmax*1.2), 300)
    plt.figure()
    for gid, idxs in groups.items():
        nu0 = float(np.median(nu[idxs]))
        grid = model.flux(t=t_grid, nu_min=nu0, nu_max=nu0, num_nu=1)
        y = np.asarray(grid.total).reshape(1,-1)[0]
        plt.loglog(t_grid, y, label=f"{nu0:.3g} Hz")
        plt.errorbar(t[idxs], f[idxs], yerr=e[idxs], fmt='o', ms=4, capsize=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Flux density [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
    plt.legend(loc="best", fontsize=8)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def pick_epochs(t: np.ndarray, n: int = 3) -> List[float]:
    tl = np.sort(np.unique(t))
    if tl.size <= n: return tl.tolist()
    qs = np.quantile(np.log10(tl), [0.05, 0.5, 0.95])
    return (10**qs).tolist()

def plot_spectra(model: Model, t: np.ndarray, nu: np.ndarray, f: np.ndarray, e: np.ndarray, outpath: str):
    epochs = pick_epochs(t, n=3)
    nu_min = max(1e1, float(np.nanmin(nu))*0.8)
    nu_max = float(np.nanmax(nu))*1.2
    nu_grid = np.logspace(np.log10(nu_min), np.log10(nu_max), 300)

    plt.figure()
    for tj in epochs:
        grid = model.flux(t=np.array([tj]), nu_min=nu_min, nu_max=nu_max, num_nu=nu_grid.size)
        y = robust_total_to_curve(grid.total, expect_len=nu_grid.size)
        plt.loglog(nu_grid, y, label=f"t = {tj:.3g} s")

        mask = (t > tj/1.2) & (t < tj*1.2)
        if np.any(mask):
            plt.errorbar(nu[mask], f[mask], yerr=e[mask], fmt='o', ms=4, capsize=2)

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Flux density [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
    plt.legend(loc="best", fontsize=8)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ---------- main ----------
def main():
    data_path = "/Users/jkeohane/GRBs/GRB_Wind_Bubbles/Data/080413B/080413B.csv"
    print(f"Reading: {data_path}")
    t, nu, f, e = load_080413_schema(data_path)
    print(f"Parsed rows: {t.size}")

    theta0 = np.array([1.0e52, 300.0, 0.10, 1.0e-1, 1.0e-3, 2.30, 3.0e17, 0.0, -2.0], float)
    cfg = FitConfig()

    best, chain, lnps = run_emcee(t, nu, f, e, theta0, cfg)

    print("\nBest-fit parameters (max posterior):")
    names = ["E_iso", "Gamma0", "theta_c", "eps_e", "eps_B", "p", "R_t", "log10_n_t", "log10_n_ism"]
    for name, val in zip(names, best[:9]):
        print(f"  {name:12s} = {val:.6g}")

    model_best = build_model_from_theta(best, cfg)

    assets = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(assets, exist_ok=True)
    lc_path = os.path.join(assets, "lightcurves.png")
    sp_path = os.path.join(assets, "spectra.png")

    plot_lightcurves(model_best, t, nu, f, e, lc_path)
    plot_spectra(model_best, t, nu, f, e, sp_path)

    print(f"\nSaved plots:\n  {lc_path}\n  {sp_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
