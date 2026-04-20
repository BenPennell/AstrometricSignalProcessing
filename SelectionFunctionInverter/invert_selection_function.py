from scipy.stats import truncnorm
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.setup import add_project_root

add_project_root(__file__)

from utils.dependencies import *

asp_import("GaiamockWrapper", alias="gw")


def sample_normal(row, label, count=1):
    mean = row[label]
    scale = row[label + "_error"]
    
    scale = np.maximum(scale, 1e-5)

    # -1, +1 indicate truncating to 1 sigma
    samples = truncnorm.rvs(-1, 1, loc=mean, scale=scale, size=count)

    return samples[0] if count == 1 else samples

def adjust_magnitude(mag, plx, new_plx):
    """Adjust the magnitude of a star to reflect a change in parallax."""
    # Calculate the absolute magnitude
    abs_mag = mag - 5 * np.log10(1000/plx) + 5
    
    # Calculate the new apparent magnitude
    new_mag = abs_mag + 5 * np.log10(1000/new_plx) - 5
    return new_mag

def generate_parallaxes(dist_range=(50,200), count=1, resolution=1000):
    # sample a distance from a distribution that goes as d^2 between 100 and 200
    # from inverse cdf sampling
    dists = np.linspace(*dist_range, resolution)
    d_pdf = np.zeros_like(dists)
    d_pdf[1:] = pexp(dists[1:], 2) # d^2
    d_cdf = np.cumsum(d_pdf / np.sum(d_pdf))
    d = np.interp(np.random.rand(count), d_cdf, dists)
    parallax = 1000 / d 
    if count == 1:
        return parallax[0]
    return parallax

def select_parallaxes(ref_plx, ref_mag, count=1, dist_range=(50,200), resolution=1000):
    '''
        Select parallaxes for a star, ensuring that the resulting magnitudes are below the XP limit of 17.65.
    '''
    parallaxes = generate_parallaxes(dist_range=dist_range, count=count, resolution=resolution)
    magnitudes = adjust_magnitude(ref_mag, ref_plx, parallaxes)
    # reselect parallaxes until all magnitudes are below the limit
    bad = magnitudes > 17.65
    while np.any(bad):
        parallaxes[bad] = generate_parallaxes(dist_range=dist_range, count=np.sum(bad), resolution=resolution)
        magnitudes[bad] = adjust_magnitude(ref_mag, ref_plx, parallaxes[bad])
        bad = magnitudes > 17.65
    return parallaxes, magnitudes

def simulate_many_realisations(input_row, nss_row, sample_count, q, m1, f, ra, dec, pmra, pmdec, t, t_mod, c_funcs):
    # choose period, eccentricity from uncertainty of the solution within 1sigma
    # clipping just in case (but it should never actually come up)
    periods = np.clip(sample_normal(nss_row, "period", sample_count), 100, 10000)
    eccentricities = np.clip(sample_normal(nss_row, "eccentricity", sample_count), 0, 0.98)
    
    # choose parallax isotropically over the volume of space, limiting to G=17.65 
    ref_plx = input_row["parallax"]
    ref_mag = input_row["phot_g_mean_mag"]
    parallaxes, magnitudes = select_parallaxes(ref_plx, ref_mag, count=sample_count)

    Tps = np.random.uniform(-0.5, 0.5, sample_count)
    ws = np.random.uniform(0, 2*np.pi, sample_count)
    omegas = np.random.uniform(0, 2*np.pi, sample_count)
    incs = np.arccos(np.random.uniform(-1, 1, sample_count))

    counts = 0

    for j in range(sample_count):
        soltype = gw.rapid_solution_type(
            periods[j], q, parallaxes[j], m1, magnitudes[j], f,
            eccentricities[j], incs[j], ws[j], omegas[j], Tps[j],
            ra, dec, pmra, pmdec, t, t_mod, c_funcs,
            skip_full=False, return_ruwe=False, return_fits=False
        )

        if soltype == 12:
            counts += 1
    
    return counts
            
def compute_rate(i, input_table, nss_catalogue, sample_count=50, extra_count=250, f=1e-10):
    try:
        # Initialize per-process (safe for multiprocessing)
        c_funcs = gw.generate_cfuncs()

        input_row = input_table[i]
        nss_row_filtered = nss_catalogue[nss_catalogue["source_id"] == int(input_row["source_id"])]

        if len(nss_row_filtered) == 0:
            raise ValueError(f"No matching NSS entry for source_id {input_row['source_id']}")
        
        # Extract the first row to get Row object instead of Table
        nss_row = nss_row_filtered[0]

        # Extract values
        ra = nss_row["ra"]
        dec = nss_row["dec"]
        pmra = nss_row["pmra"]
        pmdec = nss_row["pmdec"]

        m1 = input_row["mass_single"]
        q = input_row["m2"] / m1

        # Generate scanning law
        t, t_mod = gw.generate_scanning_times(ra, dec)

        # simulate a bunch of binaries! For the binaries that often get orbit solutions, we only need a few samples
        counts = simulate_many_realisations(input_row, nss_row, sample_count, q, m1, f, ra, dec, pmra, pmdec, t, t_mod, c_funcs)
        rate = counts / sample_count

        # if none get an orbit solution, sample even more to get a better resolution for just the rare ones
        if counts == 0 and extra_count > 0:
            counts = simulate_many_realisations(input_row, nss_row, extra_count, q, m1, f, ra, dec, pmra, pmdec, t, t_mod, c_funcs)

            rate = counts / (sample_count + extra_count)

        return rate

    except Exception as e:
        print(f"[ERROR] index {i}: {e}")
        return np.nan

# Parallel execution
def compute_all_rates(input_table, nss_catalogue, **kwargs):
    rates = np.empty(len(input_table))

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(compute_rate, i, input_table, nss_catalogue, **kwargs): i
            for i in range(len(input_table))
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[future]
            try:
                rates[idx] = future.result()
            except Exception as e:
                print(f"[FUTURE ERROR] index {idx}: {e}")
                rates[idx] = np.nan

    return rates