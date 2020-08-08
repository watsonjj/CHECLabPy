import numpy as np
from numba import njit, vectorize, float64, int64
from math import exp, pow, sqrt, lgamma, log
from CHECLabPy.stats.pdf import binom, poisson, normal_pdf, poisson_logpmf
from CHECLabPy.core.spectrum_fitter import SpectrumFitter, SpectrumParameter, \
    SpectrumParameterCollection
from iminuit.cost import _sum_log_x


class SiPMGeneralizedPoissonFitter(SpectrumFitter):
    def __init__(self, n_illuminations):
        """
        SpectrumFitter which uses the SiPM fitting formula from Gentile 2010
        http://adsabs.harvard.edu/abs/2010arXiv1006.3263G
        """
        super().__init__(n_illuminations)

        self.parameters = SpectrumParameterCollection([
            SpectrumParameter("eped", 0, (-10, 10)),
            SpectrumParameter("eped_sigma", 9, (0, 20)),
            SpectrumParameter("spe", 25, (0, 40)),
            SpectrumParameter("spe_sigma", 2, (0, 20)),
            SpectrumParameter("opct", 0.4, (0, 1)),
            SpectrumParameter("lambda_", 0.7, (0, 3), multi=True),
        ], n_illuminations)
        self.n_bins = 100
        self.range = (-30, 200)

    @staticmethod
    @njit(fastmath=True)
    def _get_spectra(n_illuminations, data_x, lookup, *parameter_values):
        spectra = []
        for i in range(n_illuminations):
            spectrum = calculate_spectrum(data_x, lookup[i], *parameter_values)
            spectra.append(spectrum)
        return spectra

    @staticmethod
    @njit(fastmath=True)
    def _get_likelihood(n_illuminations, data, lookup, *parameter_values):
        likelihood = 0
        for i in range(n_illuminations):
            spectrum = calculate_spectrum(data[i], lookup[i], *parameter_values)
            likelihood += -_sum_log_x(spectrum)
        return likelihood


@njit(fastmath=True)
def calculate_spectrum(data_x, lookup, *parameter_values):
    return sipm_gen_poisson_spe(
        x=data_x,
        eped=parameter_values[lookup["eped"]],
        eped_sigma=parameter_values[lookup["eped_sigma"]],
        spe=parameter_values[lookup["spe"]],
        spe_sigma=parameter_values[lookup["spe_sigma"]],
        opct=parameter_values[lookup["opct"]],
        lambda_=parameter_values[lookup["lambda_"]],
    )


@vectorize([float64(int64, float64, float64)], fastmath=True)
def generalized_poisson(k, mu, xtalk):
    """
    Generalized Poisson probabilities for a given mean number per event
    and per xtalk event.

    Parameters
    ----------
    k : int
    mu : float
        The mean number per event
    xtalk : float
        The mean number per xtalk event

    Returns
    -------
    probability : float
    """
    mu_dash = (mu + k * xtalk)
    return mu * exp((k-1) * log(mu_dash) - mu_dash - lgamma(k+1))


@njit(fastmath=True)
def sipm_gen_poisson_spe(x, eped, eped_sigma, spe, spe_sigma, opct, lambda_):
    """
    Fit for the SPE spectrum of a SiPM

    Parameters
    ----------
    x : ndarray
        The x values to evaluate at
    eped : float
        Distance of the zeroth peak from the origin
    eped_sigma : float
        Sigma of the zeroth peak, represents electronic noise of the system
    spe : float
        Signal produced by 1 photo-electron
    spe_sigma : float
        Spread in the number of photo-electrons incident on the MAPMT
    opct : float
        Optical crosstalk probability
    lambda_ : float
        Poisson mean (average illumination in p.e.)


    Returns
    -------
    spectrum : ndarray
        The y values of the total spectrum.
    """
    spectrum = np.zeros_like(x)
    p_max = 0
    for k in range(100):
        p = generalized_poisson(k, lambda_, opct)

        # Skip insignificant probabilities
        if p > p_max:
            p_max = p
        elif p < 1e-4:
            break

        # Combine spread of pedestal and pe peaks
        pe_sigma = sqrt(k * spe_sigma ** 2 + eped_sigma ** 2)

        # Evaluate probability at each value of x
        spectrum += p * normal_pdf(x, eped + k * spe, pe_sigma)

    return spectrum
