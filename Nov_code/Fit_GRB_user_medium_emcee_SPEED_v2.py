
# Patched SPEED v2 with robust plotting helpers
import sys, os, math, re
import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict
from VegasAfterglow import TophatJet, Observer, Radiation, Model, Medium

mp = 1.67262192369e-24
h  = 6.62607015e-27
c  = 2.99792458e10

UNIT_SCALE = {'jy':1e-23,'mjy':1e-26,'ujy':1e-29,'µjy':1e-29}
FREQ_SCALE = {'hz':1.0,'khz':1e3,'mhz':1e6,'ghz':1e9,'thz':1e12}

TIME_BINS_PER_DECADE = 8
REL_FREQ_GROUP = 1e-3

@dataclass
class FitConfig:
    z: float = 0.1
    lumi_dist: float = 1e26
    theta_obs: float = 0.0
    mu: float = 1.3
    num_nu_grid: int = 32
    nwalkers: int = 36
    nsteps: int = 300
    burn: int = 100
    seed: int = 1234

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

def log_prior(theta):
    E_iso,Gamma0,theta_c,eps_e,eps_B,p,R_t,lnt,lnism = theta[:9]
    if not (1e50<=E_iso<=1e54): return -np.inf
    if not (10.0<=Gamma0<=600.0): return -np.inf
    if not (1e-3<=theta_c<=0.5): return -np.inf
    if not (1e-5<=eps_e<=0.5): return -np.inf
    if not (1e-6<=eps_B<=0.5): return -np.inf
    if not (2.0<=p<=2.8): return -np.inf
    if not (1e14<=R_t<=1e20): return -np.inf
    if not (-6<=lnt<=6 and -6<=lnism<=6): return -np.inf
    return 0.0

def log_like(theta, t, nu, f, e, cfg: FitConfig):
    try:
        m = predict(build_model(theta,cfg), t, nu, cfg)
        if not all_finite(m) or np.any(e<=0): return -np.inf
        return -0.5*np.sum(((f-m)/e)**2)
    except Exception:
        return -np.inf

def log_prob(theta, t, nu, f, e, cfg: FitConfig):
    lp=log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    ll=log_like(theta,t,nu,f,e,cfg)
    return lp+ll if np.isfinite(ll) else -np.inf

def init_walkers(theta0, nwalkers, rng):
    ndim=theta0.size
    rel=np.array([0.02,0.05,0.02,0.05,0.05,0.01,0.05,0.05,0.05])
    step=np.maximum(np.abs(theta0),1.0)*rel
    step[2]=max(0.02,step[2]); step[5]=max(0.01,step[5]); step[7]=max(0.05,step[7]); step[8]=max(0.05,step[8])
    Q,_=np.linalg.qr(rng.standard_normal((ndim,ndim)))
    p0=np.zeros((nwalkers,ndim),float)
    for i in range(nwalkers):
        p0[i,:]=theta0 + 0.5*((i+1)/nwalkers)*step*Q[:,i%ndim] + rng.standard_normal(ndim)*step*0.5
    return p0

def run_emcee(t, nu, f, e, theta0, cfg: FitConfig):
    rng=np.random.default_rng(cfg.seed); ndim=theta0.size
    if cfg.nwalkers<2*ndim: raise ValueError("nwalkers too small")
    p0=init_walkers(theta0,cfg.nwalkers,rng)
    sampler=emcee.EnsembleSampler(cfg.nwalkers, ndim, log_prob, args=(t,nu,f,e,cfg), moves=emcee.moves.StretchMove(a=1.8))
    state=sampler.run_mcmc(p0, cfg.burn, progress=True)
    sampler.reset(); sampler.run_mcmc(state, cfg.nsteps, progress=True)
    flat_lnprob=sampler.get_log_prob(flat=True); flat_chain=sampler.get_chain(flat=True)
    ibest=int(np.argmax(flat_lnprob)); best=flat_chain[ibest,:]
    return best, sampler.get_chain(), sampler.get_log_prob()

# ---- robust plotting helpers ----

def robust_total_to_curve_vs_nu(grid_total, expect_len):
    """
    Normalize FluxDict.total into a 1D array vs frequency of length `expect_len`.
    Handles shapes: (n_nu, 1), (1, n_nu), (n_nu,), unexpected 2D by flattening.
    """
    A = np.asarray(grid_total)
    A = np.squeeze(A)
    # After squeeze, if it's 1D, just align length
    if A.ndim == 1:
        if A.size >= expect_len:
            return A[:expect_len]
        out = np.full(expect_len, np.nan, float)
        out[:A.size] = A
        return out
    # 2D fallback
    if A.ndim == 2:
        # Prefer (n_nu, n_t=1)
        if A.shape[1] == 1 and A.shape[0] == expect_len:
            return A[:, 0]
        # Or (n_t=1, n_nu)
        if A.shape[0] == 1 and A.shape[1] == expect_len:
            return A[0, :]
        # Otherwise, flatten and trim/pad
        B = A.ravel()
        if B.size >= expect_len:
            return B[:expect_len]
        out = np.full(expect_len, np.nan, float)
        out[:B.size] = B
        return out
    # Higher-D: flatten
    B = A.ravel()
    if B.size >= expect_len:
        return B[:expect_len]
    out = np.full(expect_len, np.nan, float)
    out[:B.size] = B
    return out

def robust_total_to_curve_vs_t(grid_total, expect_len):
    """
    Normalize FluxDict.total into a 1D array vs time of length `expect_len`.
    When we requested num_nu=1, shapes are typically (1, n_t) or (n_t, 1).
    """
    A = np.asarray(grid_total)
    A = np.squeeze(A)
    if A.ndim == 1:
        if A.size >= expect_len:
            return A[:expect_len]
        out = np.full(expect_len, np.nan, float)
        out[:A.size] = A
        return out
    if A.ndim == 2:
        # Prefer (n_nu=1, n_t)
        if A.shape[0] == 1 and A.shape[1] == expect_len:
            return A[0, :]
        # Or (n_t, n_nu=1)
        if A.shape[1] == 1 and A.shape[0] == expect_len:
            return A[:, 0]
        # Otherwise flatten
        B = A.ravel()
        if B.size >= expect_len:
            return B[:expect_len]
        out = np.full(expect_len, np.nan, float)
        out[:B.size] = B
        return out
    B = A.ravel()
    if B.size >= expect_len:
        return B[:expect_len]
    out = np.full(expect_len, np.nan, float)
    out[:B.size] = B
    return out

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
        y = robust_total_to_curve_vs_t(grid.total, expect_len=t_grid.size)

        # Draw model line and grab its color
        (line,) = plt.loglog(t_grid, y, lw=1.8, label=f"{nu0:.3g} Hz")
        col = line.get_color()

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
    plt.close()


def pick_epochs(t, n=3):
    tl=np.sort(np.unique(t)); 
    if tl.size<=n: return tl.tolist()
    qs=np.quantile(np.log10(tl), [0.05,0.5,0.95]); return (10**qs).tolist()

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
        y = robust_total_to_curve_vs_nu(grid.total, expect_len=nu_grid.size)

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
    plt.close()


def main():
    data_path="/Users/jkeohane/GRBs/GRB_Wind_Bubbles/Data/080413B/080413B.csv"
    print(f"Reading: {data_path}")
    t,nu,f,e=load_080413_schema(data_path)
    print(f"Parsed rows: {t.size}")
    tb, nub, fb, eb = bin_data(t, nu, f, e)
    print(f"Binned rows: {tb.size} (from {t.size})")

    theta0=np.array([1.0e52,300.0,0.10,1.0e-1,1.0e-3,2.30,3.0e17,0.0,-2.0],float)
    cfg=FitConfig()

    best, chain, lnps = run_emcee(tb, nub, fb, eb, theta0, cfg)

    print("\\nBest-fit parameters (max posterior):")
    names=["E_iso","Gamma0","theta_c","eps_e","eps_B","p","R_t","log10_n_t","log10_n_ism"]
    for name,val in zip(names,best[:9]): print(f"  {name:12s} = {val:.6g}")

    model=build_model(best,cfg)
    assets=os.path.join(os.path.dirname(__file__),"assets"); os.makedirs(assets,exist_ok=True)
    lc_path=os.path.join(assets,"lightcurves_speed.png"); sp_path=os.path.join(assets,"spectra_speed.png")
    plot_lightcurves(model, tb, nub, fb, eb, lc_path); plot_spectra(model, tb, nub, fb, eb, sp_path)
    print(f"\\nSaved plots:\\n  {lc_path}\\n  {sp_path}")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: sys.exit(130)
