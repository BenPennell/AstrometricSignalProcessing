'''
    This is the same as ComputeCube.py, except without separate bins for parallax. Since we right now don't care about acceleration solutions, this is fine.
    
    Also, instead of sampling from the dataset, we will just choose a set mass/parallax and use lambda to compute q for the centre of each bin.
    Especially when the grid is fine, this will be an adequate approximation while also being faster and with less lines of code for troubleshooting purposes.
    
    When setting up the lambda grid, make sure to set the boundaries equal to the lowest possible lambda (lowest mass/parallax, highest q) and the highest possible lambda (highest mass/parallax, lowest q)
    
    The expected structure of the setup file is:
        all grid points should be the "centres". There are P parallax bins, and L lambda bins
        grid: two arrays corresponding to the "edges" of the period and lambda grids
        shape: (period_bins, period_bins, lambda_bins, sample_count)
        1xP period_grid: the centre periods (this is independant of mass/plx/lambda). in days, NOT logdays.
        Lx1 lambda_grid: the centre lambdas for each mass bin. This is used to compute q for each bin, which is then used for computing the cube.
        reference_mass, reference_parallax, reference_magnitude: the fixed values to use for computing the cube. Ideally, use the median from your sample.
        eccentricity_type: "circular", "thermal", or "mixed" (the "turnover" model)
        trial_count: number of trials to run for each grid point. 
        After the set of trial runs are done, if no orbit solutions were computed, the rest of the (sample_count-trial_count) samples will skip computing orbit solutions to speed things up
        
        reference_data: dataset to sample from. It only needs to have ra/dec/pmra/pmdec/phot_g_mean_mag but you can also add mass/parallax for completeness.
        temp_path: path for the log/temp files
        data_path: path for the final results to be stored in
        input_file: the name of the input file where the objects were pulled from
        
        For the reference data (that we marginalise over), you need to create a folder and put each mass bin's corresponding data into it
        it should look like: /[catalogue_path]/[reference_data_locations[mass_idx]].pkl
    
        There are 9 parameters that we marginalise over:
        ra, dec, pmra, pmdec, four angles, eccentricity
        if you imagine (ra,dec) and (pmra,pmdec) really only have a degree of freedom each, this is six parameters
        To ensure you get a sample in every "quadrant" of 6D-space, some factor of 2^6 trials is good. 150 should be the minimum.
'''

from mpi4py import MPI
import gc
from functools import lru_cache
import healpy as hp

from utils.dependencies import *;

asp_import("GaiamockWrapper", alias="gw")
C_FUNCS = gw.generate_cfuncs()

# ================= MPI =================
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# ================= CONFIG =================
with open('./config.json') as f:
    cfg_json = json.load(f)

DATA_PATH = cfg_json["data_path"]
TEMP_PATH = cfg_json["temp_path"]

if len(sys.argv) == 2:
    CONFIG_FILE = sys.argv[1]
    NAME = f"{date.today()}-{CONFIG_FILE}"
elif len(sys.argv) == 3:
    CONFIG_FILE, NAME = sys.argv[1], sys.argv[2]
else:
    raise RuntimeError("Usage: mpirun -n N python script.py CONFIG_FILE OUTPUT")

# ================= LOG =================
LOG_FILE = f"{TEMP_PATH}/{NAME}_log.txt"

def log(msg, root_only=True):
    if (not root_only) or rank == 0:
        with open(LOG_FILE, "a") as f:
            f.write(f"{datetime.now()} | {msg}\n")

if rank == 0:
    open(LOG_FILE, "w").close()
    log(NAME)

# ================= LOAD CONFIG =================
cfg = pickle.load(open(f"{DATA_PATH}/{CONFIG_FILE}.pkl", "rb"))

trial_count = cfg["trial_count"]
ecc_type = cfg["eccentricity_type"]

period_count, lambda_count, sample_count = cfg["shape"]

period_grid = 10**cfg["grid"][0] # convert from log period, represents edges (T+1)
mass_ratio_grid = cfg["mass_ratio_grid"]

reference_mass = cfg["reference_mass"]
reference_parallax = cfg["reference_parallax"]
reference_magnitude = cfg["reference_magnitude"]

median_pm = cfg["median_pm"]

# ================= SCANNING CACHE =================
SCAN_NSIDE = 8
NPIX_SCAN = hp.nside2npix(SCAN_NSIDE)

# precompute pixel -> (ra, dec) lookup
_SCAN_RA = np.empty(NPIX_SCAN, dtype=np.float32)
_SCAN_DEC = np.empty(NPIX_SCAN, dtype=np.float32)
for _pix in range(NPIX_SCAN):
    _theta, _phi = hp.pix2ang(SCAN_NSIDE, _pix)
    _SCAN_RA[_pix] = np.degrees(_phi)
    _SCAN_DEC[_pix] = 90.0 - np.degrees(_theta)

_scan_cache = {}
def cached_scanning_times(pix):
    if pix not in _scan_cache:
        _scan_cache[pix] = gw.generate_scanning_times(_SCAN_RA[pix], _SCAN_DEC[pix])
    return _scan_cache[pix]

def random_radec(rng, count):
    pix = rng.integers(0, NPIX_SCAN, count)
    return _SCAN_RA[pix], _SCAN_DEC[pix], pix

# ================= SAMPLING_FUNCTIONS =================

def q_from_lambda(l, m1, plx):
    roots = np.roots([m1*(plx/l)**3, -1, -2, -1])
    roots = roots[np.isreal(roots)].real
    roots = roots[roots > 0]
    return roots[0] if len(roots) else -1.

def random_pms(maxv, rng, count):
    vels = rng.uniform(0, maxv, count)
    angles = rng.uniform(0, 2*np.pi, count)
    return vels*np.cos(angles), vels*np.sin(angles)

## rest is for eccentricity
e_grid = np.linspace(0, 1, 100)

def gaussian(x, mu, s):
    return np.exp(-(mu - x)**2 / (2*s*s)) / np.sqrt(2*np.pi*s*s)

def powerlaw(x, e, vr=(0,1)):
    a = (e+1)/(vr[1]**(e+1)-vr[0]**(e+1))
    return a*(x**e)

thermal_pdf = np.zeros_like(e_grid)
thermal_pdf[1:] = powerlaw(e_grid[1:],1)
thermal_cdf = np.cumsum(thermal_pdf/np.sum(thermal_pdf))

e_period_grid = np.linspace(1,8,100)

turnover = np.cumsum(gaussian(e_period_grid,3.5,1))
turnover /= turnover[-1]

p_thermal = powerlaw(e_grid,1,(e_grid[0],e_grid[-1]))
p_thermal /= np.sum(p_thermal)

p_gauss = gaussian(e_grid,0.38,0.2)
p_gauss /= np.sum(p_gauss)

def draw_eccentricity(rng, log_periods):
    u = rng.random(len(log_periods))
    if ecc_type == "circular":
        return np.zeros_like(log_periods)
    if ecc_type == "thermal":
        return np.interp(u, thermal_cdf, e_grid)

    idx = np.clip(np.searchsorted(e_period_grid, log_periods), 0, len(e_period_grid)-1)
    mix = turnover[idx]
    out = np.zeros_like(log_periods)
    for i in range(len(log_periods)):
        dist = (1-mix[i])*p_gauss + mix[i]*p_thermal
        cdf = np.cumsum(dist/np.sum(dist))
        out[i] = np.interp(u[i], cdf, e_grid)
    return out

# ================= CORE FUNCTION =================
def compute_column(period_grid, q, plx, mass, g, seeds, n_trial, n_total):
    soltypes = [0,5,7,9,12]
    column = np.full((len(period_grid) - 1, 5), 0, np.int32)
    q, plx, mass, g = float(q), float(plx), float(mass), float(g)

    for i in range(len(period_grid)-1):
        values = np.zeros(5)
        rng = np.random.default_rng(seeds[i])
        periods = rng.random(n_total)*(period_grid[i+1]-period_grid[i]) + period_grid[i] # sample period randomly within the bin

        inc = np.arccos(2*rng.random(n_total)-1)
        w = rng.random(n_total)*2*np.pi
        omega = rng.random(n_total)*2*np.pi
        Tp = rng.random(n_total)-0.5

        ecc = draw_eccentricity(rng, np.log10(periods))

        ra, dec, pix = random_radec(rng, n_total)
        pmra, pmdec = random_pms(median_pm, rng, n_total)

        #for k, star in enumerate(trial_set):
        for k in range(n_trial):
            t, t_mod = cached_scanning_times(int(pix[k]))
            values[soltypes.index(gw.rapid_solution_type(period=periods[k], q=q, parallax=plx, m1=mass, f=1e-10,
                            ecc=ecc[k], inc=inc[k], w=w[k], omega=omega[k], Tp=Tp[k],
                            phot_g_mean_mag=g, ra=ra[k], dec=dec[k],
                            pmra=pmra[k], pmdec=pmdec[k], t=t, t_mod=t_mod,
                            c_funcs=C_FUNCS))] += 1

        skip_full = (values[-1] == 0)

        #for k, star in enumerate(main_set):
        for j in range(n_trial, n_total):
            t, t_mod = cached_scanning_times(int(pix[j]))
            values[soltypes.index(gw.rapid_solution_type(period=periods[j], q=q, parallax=plx, m1=mass, f=1e-10,
                            ecc=ecc[j], inc=inc[j], w=w[j], omega=omega[j], Tp=Tp[j],
                            phot_g_mean_mag=g, ra=ra[j], dec=dec[j],
                            pmra=pmra[j], pmdec=pmdec[j], t=t, t_mod=t_mod,
                            c_funcs=C_FUNCS, skip_full=skip_full))] += 1

        column[i] = values

    return column

# ================= MAIN LOOP =================
# higher lambda -> more compute time
# weave high/low lambda collums together to better divide the compute
high_to_low = np.argsort(-np.arange(lambda_count))
low_to_high = np.arange(lambda_count)
half = lambda_count // 2

indices = np.ravel(np.column_stack((
    high_to_low[:half],
    low_to_high[:half]
)))

if lambda_count % 2 == 1:
    indices = np.append(indices, high_to_low[half])

chunks = np.array_split(indices, size)
lambda_indices = chunks[rank]

local_columns = []
base_seed = int(1000 + rank*1000)

if rank == 0:
    log("COMPUTING GRID...")

for lambda_idx in lambda_indices:
    
    seeds = base_seed + lambda_idx*period_count + np.arange(period_count)

    column = compute_column(period_grid, mass_ratio_grid[lambda_idx],
                reference_parallax, reference_mass, reference_magnitude,
                seeds, trial_count, sample_count)

    local_columns.append((lambda_idx, column))

    log(f"LAMBDA INDEX {lambda_idx} | COMPLETED", root_only=False)

gathered_hist = comm.gather(local_columns, root=0)

# ====== GATHER AND SAVE ======
if rank == 0:
    # flatten across ranks
    flat = [item for rank_list in gathered_hist for item in rank_list]
    
    # now sort globally by lambda index
    flat.sort(key=lambda x: x[0])

    # extract only columns in correct order
    output_cube = np.stack([col for _, col in flat], axis=1)

    results = {
        "meta": {
            "date": str(date.today()),
            "input_file": cfg["input_file"],
            "config_file": CONFIG_FILE,
            "data_path": DATA_PATH,
            "grid": cfg["grid"],
            "period_grid": cfg["period_grid"],
            "lambda_grid": cfg["lambda_grid"],
            "mass_ratio_grid": cfg["mass_ratio_grid"],
            "reference_mass": cfg["reference_mass"],
            "reference_parallax": cfg["reference_parallax"],
            "reference_magnitude": cfg["reference_magnitude"],
            "median_pm": cfg["median_pm"],
            "shape": cfg["shape"],
            "trial_count": cfg["trial_count"],
            "eccentricity_type": cfg["eccentricity_type"],
            "note": cfg["note"]
        },
        "cube": output_cube
    }

    output = f"{DATA_PATH}/{NAME}.pkl"

    with open(output, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    log(f"ENTIRE CUBE COMPLETED | SAVED TO {output}")

comm.Barrier()
gc.collect()