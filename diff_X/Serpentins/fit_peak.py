import numpy as np
from scipy.optimize import curve_fit


def peak_Gauss(x, position, fwhm, amplitude,
               base_height, base_slope):
    amplitude = amplitude * 2/fwhm * np.sqrt(np.log(2)/np.pi)
    gauss = np.exp(-4*np.log(2)/fwhm**2 * (x-position)**2 )
    base_line = base_height + base_slope * x
    return amplitude*gauss + base_line

def peak_Lorentzien(x, position, fwhm, amplitude, base_height, base_slope):
    base_line = base_height + base_slope * x
    return 2*amplitude/np.pi/fwhm/(1 + 4*(np.sqrt(2)-1)/fwhm**2*(x-position)**2) + base_line

def peak_pseudoVoigt(x, position, fwhm, amplitude, base_height, base_slope, eta):
    return (1-eta)*peak_Gauss(x, position, fwhm, amplitude, base_height, base_slope) + \
           eta*peak_Lorentzien(x, position, fwhm, amplitude, 0, 0)


def fit_peak(x, y, function='Lorentz'):
    ''' Fit a peak function on the given x, y data
        function 'Lorentz', 'pseudoVoigt' or 'Gauss'
        
        Returns:
            center position
            FWHM
            peak(x): function of the fitted peak 
    '''
    x, y = np.asarray(x), np.asarray(y)

    p0 = [x[np.argmax(y)], # center
          x[y>(y.min()+y.ptp()/2)].ptp(), # estimated width
          y.ptp(), # amplitude, peak to peak, i.e. max - min
          y.min(), # base height
          0]       # base slope

    if function == 'Lorentz':
        popt, pcov = curve_fit(peak_Lorentzien, x, y, p0=p0)
        #plt.plot(x, peak_Lorentzien(x, *popt), 'r');
        fitted_function = lambda u: peak_Lorentzien(u, *popt)
        FWHM = 2*popt[1]
        
    elif function == 'pseudoVoigt':
        p0.append(0.5) # eta
        popt, pcov = curve_fit(peak_pseudoVoigt, x, y, p0=p0)
        fitted_function = lambda u: peak_pseudoVoigt(u, *popt)
        FWHM = 2*popt[1] # no!!
        
    else: # Gauss
        popt, pcov = curve_fit(peak_Gauss, x, y, p0=p0)
        #plt.plot(x, peak_Gauss(x, *popt), 'r');
        fitted_function = lambda u: peak_Gauss(u, *popt)
        FWHM = 2*np.sqrt(2*np.log(2))*popt[1]
    
    
    return popt[0], np.abs(FWHM), fitted_function


# Monte-Carlo based estimation of the peak position x0 and FWHM 
# using Poisson distribution for noise
def fit_random_peak(x, y, function='pseudoVoigt'):
    """Generate random data using Poison distribution
        Fit the data and return corresponding (x0, fwhm)
    """
    try:
        I = np.random.poisson(lam=y)
        r = fit_peak(x, I, function=function)[:2]
    except RuntimeError:
        r = (np.nan, np.nan)
    return r

def estimate_std(x, y_fit, N=100, function='pseudoVoigt'):
    """ Monte-Carlo based estimation of the peak position x0 and FWHM
        using Poisson distribution for noise
        Generate N times random peak data and fit each
        
        x, y_fit: 1D arrays of fitted data
        N: number of draws
        function: model to use
        
        returns std_x0, std_fwhm
    """
    res = [fit_random_peak(x, y_fit, function=function) for k in range(N)]

    x0 = np.array([u[0] for u in res if np.isfinite(u[0])])
    fwhm = np.array([u[1] for u in res if np.isfinite(u[1])])

    return x0.std(), fwhm.std()



def fit_many_peaks(x, y, x0_list, x_range=2, **kargs):
    ''' Fit all peaks in the data (x, y)
        using the interval [-range + x0_i, +range + x0_i]
        around each positions 'deuxtheta_theo'

        Returns a list of fit results (see fct fit_peak)

        # TODO: x_range is a list?
    '''
    x, y = np.asarray(x), np.asarray(y)

    results = []
    for x0 in x0_list:

        mask = (x < (x0 + x_range)) & (x > (x0 - x_range))

        try:
            results.append(fit_peak(x[mask],
                                    y[mask],
                                    **kargs))
        except:
            print('error during fit for x0=', x0)
            results.append(None)

    return results




# draft to save image as data
#import io
#import urllib, base64
## https://stackoverflow.com/questions/5314707/matplotlib-store-image-in-variable

#plt.figure(1, figsize=(4, 3))
#plt.title(peak['hkl'])
#plt.plot(x-deuxTheta_fit, y, '.k')
#plt.plot(x-deuxTheta_fit, fit_function(x), 'r', linewidth=3, alpha=0.7);

#fig = plt.gcf()

#buf = io.BytesIO()
#fig.savefig(buf, format='png')
#buf.seek(0)
#string = base64.b64encode(buf.read())

#uri = 'data:image/png;base64,' + urllib.parse.quote(string)
#html = '<img src = "%s"/>' % uri