# =============================
# Silence noisy deprecation warnings
# =============================
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)

warnings.filterwarnings(
    "ignore",
    message=".*unable to import Axes3D.*",
    category=UserWarning,
)

# =============================
# Standard imports
# =============================

import sys
from pathlib import Path
from functools import lru_cache
from joblib import Parallel, delayed, parallel
from contextlib import contextmanager


# Make this module importable as "synthetic_bd_candidates" in spawned workers.
_THIS_FILE = Path(__file__).resolve()
_THIS_DIR = str(_THIS_FILE.parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_CURRENT_MODULE = sys.modules.get(__name__)
if _CURRENT_MODULE is not None:
    sys.modules.setdefault(_THIS_FILE.stem, _CURRENT_MODULE)


from utils.setup import add_project_root

add_project_root(__file__)

from utils.dependencies import *

asp_import("GaiamockWrapper", alias="gw")


# =============================
# joblib + tqdm integration
# =============================

@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar.
    """
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

# =============================
# Functions for randomly setting orbital angles
# =============================

def random_angle(n=1):
    x = np.random.rand(n) * 2 * np.pi
    return float(x[0]) if n == 1 else x

def random_inc(n=1):
    x = np.arccos(2 * np.random.rand(n) - 1)
    return float(x[0]) if n == 1 else x

def random_Tp(n=1):
    x = np.random.rand(n) - 0.5
    return float(x[0]) if n == 1 else x

# =============================
# period, mass ratio, and eccentricity distributions
# =============================

# eccentricities
es = np.linspace(0, 1, 100)

e_pdf = np.zeros_like(es)
e_pdf[1:] = pexp(es[1:], 1)
e_cdf = np.cumsum(e_pdf / np.sum(e_pdf))

p_therm = pexp(es, 1, val_range=(es[0], es[-1]))
p_therm /= np.sum(p_therm)

rayleigh_params = (0.38, 0.2)
p_gaus = gaussian(es, *rayleigh_params)
p_gaus /= np.sum(p_gaus)

periods_grid = np.linspace(1, 8, 100)
turnover_params = (3.5, 1)
turnover_pdf = gaussian(periods_grid, *turnover_params)
turnover_weight = np.cumsum(turnover_pdf / np.sum(turnover_pdf))

def circular_e(*args):
    return 0.0

def thermal_e(*args):
    return np.interp(np.random.rand(), e_cdf, es)

def turnover_e(logP):
    w = turnover_weight[np.argmin(np.abs(periods_grid - logP))]
    dist = (1 - w) * p_gaus + w * p_therm
    cdf = np.cumsum(dist / np.sum(dist))
    return np.interp(np.random.rand(), cdf, es)

def choose_value(cdf, grid, size):
    u = np.random.uniform(cdf.min(), cdf.max(), size)
    return np.interp(u, cdf, grid)

# =============================
# Cache the scanning law for all the object
# Round the RA/Dec to a certain number of decimal places to reduce the number of unique positions
# Significantly speeds things up
# =============================

c_funcs = gw.generate_cfuncs()

_GOST_NDP = 3  # RA/Dec rounding

@lru_cache(maxsize=None)
def _get_gost_cached(ra_r, dec_r, data_release="dr3"):
    return gw.gaiamock.get_gost_one_position(
        ra_r, dec_r, data_release=data_release
    )

def get_gost(ra, dec, data_release="dr3"):
    return _get_gost_cached(
        round(float(ra), _GOST_NDP),
        round(float(dec), _GOST_NDP),
        data_release=data_release
    )

def _get_gost_cached_mod(ra_r, dec_r, data_release="dr3"):
    return gw.gaiamock_mod.get_gost_one_position(
        ra_r, dec_r, data_release=data_release
    )

def get_gost_mod(ra, dec, data_release="dr3"):
    return _get_gost_cached_mod(
        round(float(ra), _GOST_NDP),
        round(float(dec), _GOST_NDP),
        data_release=data_release
    )

# =============================
# Function for fetching the scanning law and getting the solution type for a source
# =============================

def solve_binary(period, q, ecc, inc, w, omega, Tp, f,
                ra, dec, pmra, pmdec, plx, mass, gmag, return_ruwe=False, return_fits=False):
    t = get_gost(ra, dec)
    t_mod = get_gost_mod(ra, dec)
    return gw.rapid_solution_type(period, q, plx, mass,
                                    gmag, f, ecc,
                                    inc, w, omega, Tp,
                                    ra, dec, pmra, pmdec,
                                    t, t_mod, c_funcs, return_ruwe=return_ruwe, return_fits=return_fits)
    
def solve_binary_dr4(period, q, ecc, inc, w, omega, Tp, f,
                ra, dec, pmra, pmdec, plx, mass, gmag):
    t = get_gost(ra, dec, data_release="dr4")
    return gw.dr4_mode_solution_type(period, q, plx, mass,
                                    gmag, f, ecc,
                                    inc, w, omega, Tp,
                                    ra, dec, pmra, pmdec,
                                    t, c_funcs)
    
# =============================
# Main generator
# =============================

def create_synthetic_data(object_count, catalogue, fm_max=1e-3, f_gamma=None, data_release="dr3",
                        mass_model=None, period_model=None, ecc_type="circular",
                        m_lim=(0.01, 0.08), p_lim=(2, 3), p_resolution=100, return_ruwe=False, return_fits=False,
                        save_bprp=True, verbose=True, n_jobs=-1):
    
    '''
    Create a synthetic dataset of Gaia-like objects, with a specified fraction of binaries and various models for the binary parameters.
    Parameters:
    
    
        THIS IS ALL MODIFIED TO SAMPLE IN COMPANION MASS SPACE, AND ALL OBJECTS ARE RELEVANT BINARIES
        
        - object_count: total number of objects to generate
        - catalogue: Ideally an astropy table containing the properties of the objects to sample from (must include ra, dec, pmra, pmdec, parallax, mass_single, phot_g_mean_mag, and optionally bp_rp)
            Should be something that can be indexed like `catalogue["ra"]` to extract all right ascensions
        - binary_fraction: if not None, the fixed fraction of objects that are binaries (overrides binarity_model)
        - binarity_model: a function that takes an array of masses and returns an array of probabilities of being binary (if binary_fraction is None)  
            Use this if you want some kind of mass-dependent binary fraction or whatever.  
        - mass_model: a parameter for the mass ratio distribution (power law exponent). If None, you get a flat distribution
        - period_model: a tuple (mu, sigma) for a Gaussian distribution of log(period). if None, you get a flat distribution
        - ecc_type: one of "circular", "thermal", or "turnover" to specify the eccentricity distribution
        - m_lim: tuple (m_min, m_max) specifying the allowed range of secondary masses (in solar masses)
        - q_lim: tuple (q_min, q_max) specifying the allowed range of mass ratios (m2/m1)
        - p_lim: tuple (p_min, p_max) specifying the allowed range of log(period) (in days)
        - p_resolution: number of points to use in the period grid if period_model is specified. You don't need to change this.
        - save_bprp: whether to include the bp_rp color in the output (if available in the catalogue). Defaults to True.
        - verbose: whether to show a progress bar during the binary solving step
        - n_jobs: number of parallel jobs to use for solving

    Returns:
        A numpy array of length object_count, where each element is a dictionary containing the properties of the object, including the binary parameters if it is a binary.
        Each object gets the field "is_binary" which is True for binaries and False for singles, and "solution_type" which is 0,5,7,9, or 12.
    '''
        
    # choose which of the three eccentricity functions is called for
    ecc_func = {"circular": circular_e, "thermal": thermal_e, "turnover": turnover_e,}.get(ecc_type, circular_e)

    # randomly select objects from the catalogue
    idx = np.random.choice(len(catalogue), object_count, replace=True)
    ra = catalogue["ra"][idx].astype(float)
    dec = catalogue["dec"][idx].astype(float)
    # for now, try with no proper motions. It's fine.
    pmra = np.zeros(object_count) #catalogue["pmra"][idx].astype(float)
    pmdec = np.zeros(object_count) #catalogue["pmdec"][idx].astype(float)
    mass = catalogue["mass_single"][idx].astype(float)
    plx = catalogue["parallax"][idx].astype(float)
    gmag = catalogue["phot_g_mean_mag"][idx].astype(float)
    #cutoffs = catalogue["q_max"][idx].astype(float)
    
    if save_bprp:
        bprp = catalogue["bp_rp"][idx].astype(float)
    
    # --- periods ---
    if period_model is not None:
        mu, si = period_model
        ps = np.linspace(*p_lim, p_resolution)
        pdf = gaussian(ps, mu, si)
        cdf = np.cumsum(pdf / pdf.sum())
        logP = choose_value(cdf, ps, object_count)
    else:
        logP = np.random.uniform(p_lim[0], p_lim[1], object_count)

    period = 10 ** logP

    # --- mass ratios ---
    def flat_q(count):
        return np.random.uniform(*m_lim, count)
    q_func = flat_q
    # if a mass ratio distribution is called for (power law), set up the function
    if mass_model is not None:
        qs = np.linspace(*m_lim, 1000)
        q_pdf = pexp(qs, mass_model)
        q_cdf = np.cumsum(q_pdf / np.sum(q_pdf))
        def exponential_q(count):
            return np.array([np.interp(np.random.rand(), q_cdf, qs) for _ in range(count)])   
        q_func = exponential_q
        
    # randomly select mass ratios, and then keep resampling
    # until they fall into the restricted range 
    m2 = q_func(object_count)
    if f_gamma is not None:
        fs = (m2/mass)**f_gamma # mass-luminosity relationship
    else:
        fs = np.ones_like(m2) * (1e-10) # basically no light
        
    fms = mass_function_explicit(period, mass, m2, f=fs)
    bad = fms > fm_max
    # if the mass function is too high, we need to reduce the companion mass 
    while np.any(bad):
        m2[bad] = q_func(bad.sum())
        fs[bad] = (m2[bad]/mass[bad])**f_gamma if f_gamma is not None else 1e-10
        fms[bad] = mass_function_explicit(period[bad], mass[bad], m2[bad], f=fs[bad])
        bad = fms > fm_max

    # --- eccentricities ---
    ecc = np.array([ecc_func(lp) for lp in logP])
    
    # --- orbital angles ---
    inc = random_inc(object_count)
    w = random_angle(object_count)
    omega = random_angle(object_count)
    Tp = random_Tp(object_count)

    # =============================
    # Parallel solve with progress bar
    # =============================

    if verbose:
        pbar = tqdm(total=object_count, desc="Computing Binaries")

    if data_release == "dr4":
        with tqdm_joblib(pbar if verbose else tqdm(disable=True)):
            results = Parallel(
                n_jobs=n_jobs,
                backend="loky"
            )(
                delayed(solve_binary_dr4)(period[i], m2[i]/mass[i], ecc[i], inc[i], w[i], omega[i], Tp[i], fs[i],
                                    ra[i], dec[i], pmra[i], pmdec[i], plx[i], mass[i], gmag[i])
                for i in range(object_count)
            )
    else:
        with tqdm_joblib(pbar if verbose else tqdm(disable=True)):
            results = Parallel(
                n_jobs=n_jobs,
                backend="loky"
            )(
                delayed(solve_binary)(period[i], m2[i]/mass[i], ecc[i], inc[i], w[i], omega[i], Tp[i], fs[i],
                                    ra[i], dec[i], pmra[i], pmdec[i],
                                    plx[i], mass[i], gmag[i], return_ruwe=return_ruwe, return_fits=return_fits)
                for i in range(object_count)
            )

    # =============================
    # Assemble output
    # =============================

    outdata = []
    for i in range(object_count):
        out = {
            "ra": ra[i],
            "dec": dec[i],
            "pmra": pmra[i],
            "pmdec": pmdec[i],
            "parallax": plx[i],
            "mass": mass[i],
            "phot_g_mean_mag": gmag[i],
            "solution_type": 0,
        }
        if save_bprp:
            out["bp_rp"] = bprp[i]
            
        # add binary information
        out.update({
            "period": period[i],
            "m2": m2[i],
            "ecc": ecc[i],
            "inc": inc[i],
            "w": w[i],
            "omega": omega[i],
            "Tp": Tp[i],
        })
        if data_release == "dr4":
            out["solution_type"] = results[i]
        elif return_fits:
            out["solution_type"] = results[i][0]
            out["ruwe"] = results[i][1]
            out["p0"] = results[i][2]
            out["s0"] = results[i][3]
        elif return_ruwe:
            out["solution_type"] = results[i][0]
            out["ruwe"] = results[i][1]
        else:
            out["solution_type"] = results[i]
            
        outdata.append(out)

    return np.array(outdata)
