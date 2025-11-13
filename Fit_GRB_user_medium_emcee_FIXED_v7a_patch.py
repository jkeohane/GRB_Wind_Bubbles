
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is a minimal "v7a" drop-in replacing only the plotting robustness.
# If you already have v7 running, you can just replace the plot_spectra function below.

import sys, os, math, re
import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict
from VegasAfterglow import TophatJet, Observer, Radiation, Model, Medium

# ---- constants ----
mp = 1.67262192369e-24
h  = 6.62607015e-27
c  = 2.99792458e10

UNIT_SCALE = {'jy':1e-23,'mjy':1e-26,'ujy':1e-29,'µjy':1e-29}
FREQ_SCALE = {'hz':1.0,'khz':1e3,'mhz':1e6,'ghz':1e9,'thz':1e12}

@dataclass
class FitConfig:
    z: float = 0.1
    lumi_dist: float = 1e26
    theta_obs: float = 0.0
    mu: float = 1.3
    num_nu_grid: int = 64
    nwalkers: int = 36
    nsteps: int = 600
    burn: int = 200
    seed: int = 1234

# ---- (same loader from v7, condensed) ----
def load_080413_schema(csv_path: str):
    df = pd.read_csv(csv_path)
    t = df['Time'].astype(float).to_numpy()
    wave = df['Wave']; wunits = df['WaveUnits'].astype(str).str.lower().str.strip().to_numpy()
    nu = np.full_like(t, np.nan, float)
    for i,(w,u) in enumerate(zip(wave,wunits)):
        try: val=float(w)
        except: continue
        if u in FREQ_SCALE: nu[i]=val*FREQ_SCALE[u]
        elif 'hz' in u: nu[i]=val
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
            if abs(nu[j]-nu[g[-1]])<=1e-3*nu[g[-1]]: g.append(j)
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

# ---- helpers ----
def pick_epochs(t, n=3):
    tl=np.sort(np.unique(t)); 
    if tl.size<=n: return tl.tolist()
    qs=np.quantile(np.log10(tl), [0.05,0.5,0.95]); return (10**qs).tolist()

def robust_total_to_curve(grid_total, expect_len):
    """Return a 1D array of length expect_len from FluxDict.total."""
    A = np.asarray(grid_total)
    A = np.squeeze(A)
    if A.ndim == 1:
        if A.size == expect_len:
            return A
        else:
            # Try transpose-like interpretation if mismatch
            return np.ravel(A)
    elif A.ndim == 2:
        # Prefer (n_nu, n_t) with n_t==1
        if A.shape[0] == expect_len and A.shape[1] == 1:
            return A[:,0]
        if A.shape[1] == expect_len and A.shape[0] == 1:
            return A[0,:]
        # If neither matches, flatten and trim/pad
        B = A.ravel()
        if B.size >= expect_len:
            return B[:expect_len]
        out = np.full(expect_len, np.nan, float)
        out[:B.size] = B
        return out
    else:
        return np.ravel(A)[:expect_len]

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

# Demo main: load CSV, build a trivial model to just reach plot_spectra (no fitting here)
def main_demo():
    data_path = "/Users/jkeohane/GRBs/GRB_Wind_Bubbles/Data/080413B/080413B.csv"
    t, nu, f, e = load_080413_schema(data_path)
    # Build any model (parameters arbitrary here); in your v7 script, reuse your best-fit model
    cfg = FitConfig()
    jet = TophatJet(theta_c=0.1, E_iso=1e52, Gamma0=300.0)
    obs = Observer(lumi_dist=cfg.lumi_dist, z=cfg.z, theta_obs=cfg.theta_obs)
    rad = Radiation(eps_e=0.1, eps_B=1e-3, p=2.3)
    def rho_const(phi, th, r): return 1.0e-24
    model = Model(jet=jet, medium=Medium(rho=rho_const), observer=obs, fwd_rad=rad)

    assets = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(assets, exist_ok=True)
    sp_path = os.path.join(assets, "spectra_v7a_demo.png")
    plot_spectra(model, t, nu, f, e, sp_path)
    print(f"Saved demo spectra (dummy model) to: {sp_path}")

if __name__ == "__main__":
    # This file is intended as a patch: copy plot_spectra (and helper) into your v7 script.
    # Running it directly will just exercise the plotting on a dummy model.
    try:
        main_demo()
    except KeyboardInterrupt:
        sys.exit(130)
