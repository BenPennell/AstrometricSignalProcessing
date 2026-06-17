import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from IPython.display import display


G = 6.6743e-11 # m^3 kg^-1 s^-2
msun_to_kg = 1.988416e30 # kg
au_to_m = 1.495978707e11 # m
day_to_s = 60*60*24 # s

def show_result(number, do_print=False, nice=False, units=None, decimals=2):
    if not do_print:
        return number
    output = str(number)
    if decimals is not None:
        output = f"{number:.{decimals}f}"
    if nice:
        output = output + " " + units if units is not None else output
    return output

def period(mass, separation, **kwargs):
    '''
        mass: in solar masses
        separation: in AU

        returns: logperiod in LogDays
    '''
    return show_result(np.log10(float(np.sqrt(4*np.pi*np.pi*(separation*au_to_m)**3/G/(mass*msun_to_kg))/day_to_s)), units="LogDays", **kwargs)

def separation(mass, logperiod, **kwargs):
    '''
        mass: in solar masses
        logperiod: in LogDays
        
        returns: separation in AU
    '''
    return show_result(((mass*msun_to_kg)*G*((10**logperiod)*day_to_s)**2/4/np.pi/np.pi)**(1/3)/au_to_m, units="AU", **kwargs)

def print_table(table, label=None, dictionary=False):
    if dictionary:
        rates = []
        for soltype in [0,5,7,9,12]:
            rates.append(len(table[[t["solution_type"] == soltype for t in table]])/len(table)*100)
    else:
        rates = [len(table[[t["solution_type"] == soltype for t in table]])/len(table)*100 for soltype in [0,5,7,9,12]]
    grid = [
        [100] + rates,
        [len(table)] + [int(grp/100*len(table)) for grp in rates]
    ]
    row_labels = ["Rate (%)", "Counts"]
    col_labels = ["All", "low RUWE", "high RUWE", "Acceleration", "Jerk", "Full Orbit"]
    df = pd.DataFrame(grid, index=row_labels, columns=col_labels)
    displaydf = df.style.format(
        "{:.0f}",
        subset=pd.IndexSlice["Counts", :]
    ).format(
        "{:.2f}",
        subset=pd.IndexSlice["Rate (%)", :]
    )
    
    if label is not None:
        displaydf.set_caption(label)
    
    display(displaydf)

### --- ###
def aap(m, q, w):
    """
        This is lambda
    """
    return q*w*m**(1/3)*(1 + q)**(-2/3)

def astrometric_amplitude_parameter(m, q, w):
    return aap(m, q, w)

### --- ###
def al_uncertainty_per_ccd_interp(G):
    '''
    (FROM GAIAMOCK)
    This gives the uncertainty *per CCD* (not per FOV transit), taken from Fig 3 of https://arxiv.org/abs/2206.05439
    This is the "EDR3 adjusted" line from that Figure, which is already inflated compared to the formal uncertainties.
    '''
    G_vals =    [ 4,    5,   6,     7,   8.2,  8.4, 10,    11,    12,  13,    14,   15,   16,   17,   18,   19,  20]
    sigma_eta = [0.4, 0.35, 0.15, 0.17, 0.23, 0.13,0.13, 0.135, 0.125, 0.13, 0.15, 0.23, 0.36, 0.63, 1.05, 2.05, 4.1]
    return np.interp(G, G_vals, sigma_eta)

### --- ###
def w_from_l(m, q, l):
    """
        invert lambda to get w
    """
    return l / (q * m**(1/3) * (1 + q)**(-2/3))

### --- ###
def q_from_l(l, m, w):
    """
        sort of a nightmare to disentangle the nonlinear q dependence
        in lambda, this function solves it numerically if that's ever
        needed
    """
    z = m * (w / l)**3
    # Coefficients of z q^3 - q^2 - 2q - 1 = 0
    coeff = [z, -1.0, -2.0, -1.0]
    roots = np.roots(coeff)
    # real roots only
    real_roots = roots[np.isreal(roots)].real
    # choose the physically valid one: q > 0
    valid = real_roots[real_roots > 0]
    if len(valid) == 0:
        return -1.0
    # Usually only one positive root exists
    return valid[0]

### --- ###
def q_from_l_vectorized(l_array, m, w):
    """
        vectorised version of q_from_l()
    """
    z = m * (w / l_array)**3

    # coefficients for all cubics
    coeffs = np.column_stack([z, -np.ones_like(z), -2*np.ones_like(z), -1*np.ones_like(z)])
    roots = np.array([np.roots(c) for c in coeffs])  # shape (N, 3)

    # real roots mask
    real_roots = roots.real * np.isreal(roots)  # imaginary parts removed

    # positive roots mask
    positive_mask = real_roots > 0

    # pick the first positive root (there should be exactly one)
    q_vals = np.where(positive_mask.any(axis=1),
                      real_roots[np.arange(len(real_roots)), positive_mask.argmax(axis=1)],
                      -1.0)
    return q_vals

def scale_resolution(arr, scale=2, axis=0, even=False):
    """
        Upscale a numpy array along a given axis by repeating values.
        If even=True, evenly divide repeated values by the scale to preserve total sum.
    """
    arr = np.asarray(arr)
    expanded = np.expand_dims(arr, axis + 1)
    repeated = np.repeat(expanded, scale, axis=axis + 1)
    if even:
        repeated = repeated / scale
    new_shape = list(arr.shape)
    new_shape[axis] *= scale
    return repeated.reshape(new_shape)

### --- ###
def gaussian(x, mu, sigma):
    """
        Normalised gaussian at x, defined by two
        parameters: peak (mu) and width (sigma)
    """
    return np.exp(-(mu - x)**2/(2*sigma**2)) / np.sqrt(2 * np.pi * sigma**2)

### --- ###
def area_in_range(target_range, mu, sigma, resolution=100):
    xs = np.linspace(*target_range, resolution)
    ys = gaussian(xs, mu, sigma)
    return np.trapezoid(y=ys, x=xs)

### --- ###
def pexp(val, index, val_range=(0, 1), ignore_a=False):
    """
        normalised power law probability
    """
    a = 1
    if not ignore_a:
        a = (index + 1) / (val_range[1] ** (index + 1) - val_range[0] ** (index + 1))
    return a * (val ** index)

### -- ###
def area_in_range_powerlaw(target_range, index, resolution=100):
    xs = np.linspace(*target_range, resolution)
    ys = pexp(xs, index, ignore_a=True)
    return np.trapezoid(y=ys, x=xs)

### --- ###
def cutoff_to_fraction(p_model, pcut, resolution=100):
    p_mu, p_si = p_model
    total_area = area_in_range((1,pcut), p_mu, p_si, resolution)
    observable_area = area_in_range((2,3), p_mu, p_si, resolution)
    return observable_area / total_area

### --- ###
def fraction_to_cutoff(p_model, fraction, resolution=100):
    p_mu, p_si = p_model
    observable_area = area_in_range((2,3), p_mu, p_si, resolution=resolution)
    target_area = observable_area / fraction
    # search for cutoff
    pcut_vals = np.linspace(3,8,1000)
    for pcut in pcut_vals:
        total_area = area_in_range((1,pcut), p_mu, p_si, resolution=resolution)
        if total_area >= target_area:
            return pcut
    return 8.0

### --- ###
def convert_binarity(fb, a):
    """
        convert from binary fraction within some range to a total binary fraction
        where fb is the fraction of binaries in the range of interest,
        a is the fraction of all binaries that fall within that range (i.e. the area under the distribution in that range)
    """
    return a / (a + 1/fb - 1)

### --- ###
def convert_to_fb(f, p_model, pcut=8, resolution=100):
    """
        convert from a total binary fraction to the binary fraction within some range of interest, 
        where f is the total binary fraction, 
        p_model is the period distribution model, 
        and pcut is the upper limit of the period
    """
    a = cutoff_to_fraction(p_model, pcut, resolution=resolution)
    return convert_binarity(f, a)

### --- ###
def adjust_magnitude(mag, plx, new_plx):
    """Adjust the magnitude of a star to reflect a change in parallax."""
    # Calculate the absolute magnitude
    abs_mag = mag - 5 * np.log10(1000/plx) + 5
    
    # Calculate the new apparent magnitude
    new_mag = abs_mag + 5 * np.log10(1000/new_plx) - 5
    return new_mag

### --- ###
def adjust_parallax(plx, mag, new_mag):
    """Adjust the parallax of a star to reflect a change in magnitude."""
    # Calculate the absolute magnitude
    abs_mag = mag - 5 * np.log10(1000/plx) + 5
    
    # Calculate the new parallax
    new_plx = 10 * 10**((abs_mag - new_mag + 5) / 5)
    return new_plx

### --- ###
def generate_parallax(dist_range=(100,200), resolution=1000):
    # sample a distance from a distribution that goes as d^2 between 100 and 200
    # from inverse cdf sampling
    dists = np.linspace(*dist_range, resolution)
    d_pdf = np.zeros_like(dists)
    d_pdf[1:] = pexp(dists[1:], 2) # d^2
    d_cdf = np.cumsum(d_pdf / np.sum(d_pdf))
    d = np.interp(np.random.rand(), d_cdf, dists)
    parallax = 1000 / d 
    return parallax

def relative_volume(mag, plx, dist_range=(100,200), resolution=1000):
    new_plx = adjust_parallax(plx, mag, 17.65) # see where the XP cutoff is
    if new_plx < 1000/dist_range[1]:
        return 1 # if we don't cut it off, the effective volume is 1
    
    covered_area = area_in_range_powerlaw((dist_range[0],1000/new_plx), 2, resolution=resolution)
    total_area = area_in_range_powerlaw(dist_range, 2, resolution=resolution)
    
    return covered_area/total_area # otherwise, return the fractional area where we can see the object

def generate_rolling_average(memory_reduced_catalogue, roll=10000):
    effective_volumes = np.array([
        relative_volume(memory_reduced_catalogue[i]["phot_g_mean_mag"], memory_reduced_catalogue[i]["parallax"]) for i in range(len(memory_reduced_catalogue))])
    masses = np.array([m["mass"] for m in memory_reduced_catalogue])

    # sort volumes according to increasing mass
    sorted_indices = np.argsort(masses)
    sorted_volumes = effective_volumes[sorted_indices]
    
    # get rolling average
    rolling_average = np.convolve(sorted_volumes, np.ones(roll)/roll, mode='same')
    
    # "smooth" the rolling average by not letting it go down
    new_rolling_average = np.zeros(len(rolling_average))
    for i in range(len(rolling_average)):
        keep = True
        if i > 0:
            keep = new_rolling_average[i-1] < rolling_average[i]
            
        if keep:
            new_rolling_average[i] = rolling_average[i]
        else:
            new_rolling_average[i] = new_rolling_average[i-1]
    
    # return to be indexed to correspond to the input catalogue      
    resorted_averages = np.zeros(len(new_rolling_average))
    for i in range(len(new_rolling_average)):
        resorted_averages[sorted_indices[i]] = new_rolling_average[i]
    
    return resorted_averages

### --- ###
def generate_q_cutoffs(catalogue, df):
    # import the csv as a pandas dataframe
    masses = df.iloc[:, 0].values
    # Column headers (metallicities)
    metallicities = np.array([float(c) for c in df.columns[1:]])

    # Grid values
    z_grid = df.iloc[:, 1:].values    
    Z_filled = z_grid.copy()

    # Interpolate along columns to fill up the empty low-mass, high-metallicity region
    for j in range(z_grid.shape[1]):
        col = Z_filled[:, j]
        mask = ~np.isnan(col)
        
        if np.sum(mask) > 2:
            f = interp1d(masses[mask], col[mask], kind='linear', fill_value='extrapolate')
            Z_filled[:, j] = f(masses)
    
    # create interpolator
    interp_func = RegularGridInterpolator((masses, metallicities),
                                        Z_filled,
                                        method='linear',
                                        bounds_error=False,
                                        fill_value=None)
    
    # apply to the catalogue
    M, Z = np.broadcast_arrays(catalogue["mass_single"], catalogue["mh_single"])
    points = np.stack([M, Z], axis=-1)
    catalogue["q_max"] = interp_func(points)

### --- ###
def generate_q_cutoffs_simple(catalogue, q_cutoff_csv, mh="-0.0"):
    # rewrite to interpolate metallicity and mass somehow!!
    ms, qs = q_cutoff_csv["Mini"], q_cutoff_csv[mh]
    catalogue["q_max"] = np.interp(catalogue["mass_single"], ms, qs)

### --- ###
def resample_histogram(data, bins, sample_size, parameter):
    resample = np.random.choice(data, size=sample_size, replace=True)
    resample_nss = resample[[s["solution_type"] == 12 for s in resample]]
    param_array = np.array([s[parameter] for s in resample_nss])
    if parameter == "period": # period ought to be in log space
        param_array = np.log10(param_array)
    vals, _ = np.histogram(param_array, bins=bins)
    return vals/np.sum(vals)*100

### --- ###
def bootstrap_histogram(data, bins, sample_size, parameter, n_bootstraps=1000, max_fm=None):
    bin_values = np.zeros((len(bins)-1, n_bootstraps))
    for i in range(n_bootstraps):
        resample = np.random.choice(data, size=sample_size, replace=True)
        if max_fm is not None:
            resample_nss = resample[[(s["solution_type"] == 12) & (s["fm"] < max_fm) for s in resample]]
        else:
            resample_nss = resample[[(s["solution_type"] == 12) for s in resample]]
        if parameter == "q":
            param_array = np.array([s["m2"] / s["mass"] for s in resample_nss])
        else:
            param_array = np.array([s[parameter] for s in resample_nss])
        if parameter == "period": # period ought to be in log space
            param_array = np.log10(param_array)
        vals, _ = np.histogram(param_array, bins=bins)
        bin_values[:, i] = vals/np.sum(vals)*100
    return bin_values

### --- ###
def bootstrap_histogram_table(data, bins, sample_size, parameter, n_bootstraps=1000, max_fm=None):
    bin_values = np.zeros((len(bins)-1, n_bootstraps))
    for i in range(n_bootstraps):
        resample = np.random.choice(data, size=sample_size, replace=True)
        if max_fm is not None:
            resample_nss = resample[(resample["solution_type"] == 12) & (resample["fm"] < max_fm)]
        else:
            resample_nss = resample[resample["solution_type"] == 12]
        if parameter == "q":
            param_array = np.array(resample_nss["m2"] / resample_nss["mass"])
        else:
            param_array = resample_nss[parameter]
        if parameter == "period": # period ought to be in log space
            param_array = np.log10(param_array)
        vals, _ = np.histogram(param_array, bins=bins)
        bin_values[:, i] = vals/np.sum(vals)*100
    return bin_values

### --- ###
def bootstrap_uncertainties(bin_values, lower_percentile=16, upper_percentile=84):
    bootstrap_means = np.mean(bin_values, axis=1)
    yup_err = np.percentile(bin_values, upper_percentile, axis=1) - bootstrap_means
    ydown_err = bootstrap_means - np.percentile(bin_values, lower_percentile, axis=1)
    return yup_err, ydown_err

### --- ###
def histogram_likelihood(bin1_means, bin1_errors, bin2_means, bin2_errors, minimums=(1e-3,1e-3), cutoff=-30):
    '''
       this is a little bit nonsense
       I have two histograms, each with a bin location and a yup_error and ydown_error
       I want to compute a likelihood by constructing gaussians from these.
       
       The way we're going to do this is take each bin and compute the chi2 essentially
    '''
    
    likelihood = 0
    for i in range(len(bin1_means))[1:]: # skip the first bin because the histogram is noramlised so there's only 6 D.O.F
        # take the corresponding up or down error depending on the relative location of the means
        if bin2_means[i] > bin1_means[i]: # if bin 2's mean is greater than bin 1's mean, we want to use the upper error for bin 1
            sigma1 = bin1_errors[1][i]
        else:
            sigma1 = bin1_errors[0][i]
        if bin1_means[i] > bin2_means[i]: # if bin 1's mean is greater than bin 2's mean, we want to use the upper error for bin 2
            sigma2 = bin2_errors[1][i]
        else:
            sigma2 = bin2_errors[0][i]
        
        # the input minimum sigma should correspond to 1/#objects in the sample, which is the minimum error you can get from a histogram
        sigma1 = max(sigma1, minimums[0]) # set a minimum sigma to avoid numerical issues
        sigma2 = max(sigma2, minimums[1]) # set a minimum sigma to avoid numerical issues
        
        # geometric average probability (not a real likelihood)
        #prob = np.sqrt(gaussian(bin2_means[i], bin1_means[i], sigma1) * gaussian(bin1_means[i], bin2_means[i], sigma2))
        # chi2
        chi2_val = ((bin2_means[i] - bin1_means[i])**2) / (sigma1**2 + sigma2**2)
        # cutoff gives a minimum probability to remain numerically stable
        likelihood += np.maximum(-0.5*chi2_val, cutoff)
    return likelihood