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
from scipy.stats import norm
from scipy.special import ndtr

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

def choose_value(cdf, grid, size):
        u = np.random.uniform(cdf.min(), cdf.max(), size)
        return np.interp(u, cdf, grid)

### ECCENTRICITIES ###
def generate_loc_and_width(ebar, sbar, alpha, beta, gamma, delta, eta, zeta, mass_ratio, period_ratio, q_ratio):
    loc = ebar + alpha*mass_ratio + beta*period_ratio + gamma*q_ratio
    width = sbar + delta*mass_ratio + eta*period_ratio + zeta*q_ratio
    return loc, width

def generate_gaussian_pdf(mu, sigma, eccentricity_bins):
    return np.diff(ndtr((eccentricity_bins - mu) / sigma)) / (ndtr((1.0 - mu) / sigma) - ndtr((0.0 - mu) / sigma))

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
                ra, dec, pmra, pmdec, plx, mass, gmag, return_ruwe=False, return_fits=False, data_release="dr3"):
    t = get_gost(ra, dec, data_release=data_release)
    t_mod = get_gost_mod(ra, dec, data_release=data_release)
    return gw.rapid_solution_type(period, q, plx, mass,
                                    gmag, f, ecc,
                                    inc, w, omega, Tp,
                                    ra, dec, pmra, pmdec,
                                    t, t_mod, c_funcs, return_ruwe=return_ruwe, return_fits=return_fits, data_release=data_release)

def solve_binary_dr4(period, q, ecc, inc, w, omega, Tp, f,
                ra, dec, pmra, pmdec, plx, mass, gmag, return_statistics, data_release="dr4"):
    t = get_gost(ra, dec, data_release=data_release)
    return gw.dr4_mode_solution_type(period, q, plx, mass,
                                    gmag, f, ecc,
                                    inc, w, omega, Tp,
                                    ra, dec, pmra, pmdec,
                                    t, c_funcs, data_release=data_release, return_statistics=return_statistics)

# =============================
# Main generator
# =============================

def create_synthetic_data(object_count, catalogue, f_gamma=None, dr4_mode=False, data_release="dr3",
                        mass_model=None, period_model=None, ecc_type="circular", e_params=(0.38,0.2), turnover_params=(3.5,1), dependant_params=None,
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

    # randomly select objects from the catalogue
    # if -1 is supplied, use the whole catalogue. This is so we can do 1-1 comparisons across different models
    if object_count == -1:
        idx = np.arange(len(catalogue))
        object_count = len(catalogue)
    else:
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

    # --- eccentricities ---
    es = np.linspace(0, 1, 100)

    e_pdf = np.zeros_like(es)
    e_pdf[1:] = pexp(es[1:], 1)
    e_cdf = np.cumsum(e_pdf / np.sum(e_pdf))

    p_therm = pexp(es, 1, val_range=(es[0], es[-1]))
    p_therm /= np.sum(p_therm)

    p_gaus = gaussian(es, *e_params)
    p_gaus /= np.sum(p_gaus)

    periods_grid = np.linspace(1, 8, 100)
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
    
    def dependant_eccentricity(ebar, sbar, alpha, beta, gamma, delta, eta, zeta, mass, period, mass_ratio, reference_mass, reference_period, reference_mass_ratio):
        loc, width = generate_loc_and_width(ebar, sbar, alpha, beta, gamma, delta, eta, zeta, 
                                            np.log(mass)-np.log(reference_mass), np.log(period)-np.log(reference_period), np.log(mass_ratio)-np.log(reference_mass_ratio))
        pdf = generate_gaussian_pdf(loc, width, np.linspace(0, 1, len(es)+1))
        cdf = np.cumsum(pdf / np.sum(pdf))
        return np.interp(np.random.rand(), cdf, es)
    
    # choose which of the three eccentricity functions is called for
    ecc_func = {"circular": circular_e, "thermal": thermal_e, "turnover": turnover_e, "dependant": dependant_eccentricity}.get(ecc_type, circular_e)
        
    # randomly select mass ratios, and then keep resampling
    # until they fall into the restricted range 
    m2 = q_func(object_count)    
    if f_gamma is not None:
        fs = (m2/mass)**f_gamma # mass-luminosity relationship
    else:
        fs = np.ones_like(m2) * (1e-10) # basically no light
    fms = mass_function_reduced(mass, m2, f=fs)
    
    # bad = fms > fm_max
    # # if the mass function is too high, we need to reduce the companion mass 
    # while np.any(bad):
    #     m2[bad] = q_func(bad.sum())
    #     fs[bad] = (m2[bad]/mass[bad])**f_gamma if f_gamma is not None else 1e-10
    #     fms[bad] = mass_function_explicit(period[bad], mass[bad], m2[bad], f=fs[bad])
    #     bad = fms > fm_max

    # --- eccentricities ---
    if ecc_type == "dependant":
        if dependant_params is None:
            raise ValueError("dependant_params must be provided for dependant eccentricity type")
        ebar, sbar, alpha, beta, gamma, delta, eta, zeta, reference_mass, reference_period, reference_mass_ratio = dependant_params
        ecc = np.array([dependant_eccentricity(ebar, sbar, alpha, beta, gamma, delta, eta, zeta, m2[i], period[i], m2[i]/mass[i], reference_mass, reference_period, reference_mass_ratio) for i in range(object_count)])
    else:
        ecc = np.array([ecc_func(lp) for lp in logP])
    
    ecc = np.clip(ecc, 0, 0.95)  # ensure eccentricities are in [0, 1)
    
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

    if dr4_mode:
        with tqdm_joblib(pbar if verbose else tqdm(disable=True)):
            results = Parallel(
                n_jobs=n_jobs,
                backend="loky"
            )(
                delayed(solve_binary_dr4)(period[i], m2[i]/mass[i], ecc[i], inc[i], w[i], omega[i], Tp[i], fs[i],
                                    ra[i], dec[i], pmra[i], pmdec[i], plx[i], mass[i], gmag[i], data_release=data_release, return_statistics=True)
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
                                    plx[i], mass[i], gmag[i], data_release=data_release, return_ruwe=return_ruwe, return_fits=return_fits)
                for i in range(object_count)
            )

    # =============================
    # Assemble output
    # =============================

    output_table = Table()
    
    # Base columns
    output_table["ra"] = ra
    output_table["dec"] = dec
    output_table["pmra"] = pmra
    output_table["pmdec"] = pmdec
    output_table["parallax"] = plx
    output_table["mass"] = mass
    output_table["phot_g_mean_mag"] = gmag
    
    if save_bprp:
        output_table["bp_rp"] = bprp
        
    # Binary info columns
    output_table["period"] = period
    output_table["m2"] = m2
    output_table["ecc"] = ecc
    output_table["inc"] = inc
    output_table["w"] = w
    output_table["omega"] = omega
    output_table["Tp"] = Tp
    output_table["fm"] = fms
    output_table["f"] = fs
    
    if dr4_mode:
        # store ruwe and the orbit solution statistics used for making cuts so that we can make cuts ourselves
        # it seems like it'll just be easiest to store them seperately
        return output_table, results
    
    # Solution info
    if return_fits:
        p0 = []
        s0 = []
    
    if return_ruwe or return_fits:
        solution_type = np.zeros(object_count, dtype=int)
        ruwe = np.empty(object_count, dtype=float)
        for i, r in enumerate(results):
            solution_type[i] = r[0]
            ruwe[i] = r[1]

            if return_fits:
                p0.append(r[2])
                s0.append(r[3])
        output_table["solution_type"] = solution_type
        output_table["ruwe"] = ruwe
    else:
        solution_types = np.array(results, dtype=int)
        output_table["solution_type"] = solution_types
    
    if return_fits:
        return output_table, p0, s0
    return output_table
