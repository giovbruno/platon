import matplotlib.pyplot as plt
import numpy as np
import corner
from .constants import METRES_TO_UM, BAR_TO_PASCALS, R_jup
from .retrieval_result import RetrievalResult
from . TP_profile import Profile
from . import _cupy_numpy as xp
import dynesty
import rebin

from pdb import set_trace

default_style = ['default',
    {   'font.size': 14,
        'xtick.top': True,
        'xtick.direction': 'out',
        'ytick.right': True,
        'ytick.direction': 'out',
        }]
plt.style.use(default_style)
plt.ion()


class Plotter():
    def __init__(self):
        pass


    def plot_retrieval_TP_profiles(self, retrieval_result, plot_samples=False, plot_1sigma_bounds=True, plot_median=True, num_samples=100, prefix=None):
        """
        Input a RetrievalResult object to make a plot of the best fit temperature profile
        and 1 sigma bounds for the profile and/or plot samples of the temperature profile.
        """
        assert(isinstance(retrieval_result, RetrievalResult))
        if retrieval_result.retrieval_type == "dynesty":
            equal_samples = dynesty.utils.resample_equal(retrieval_result.samples, retrieval_result.weights)
            np.random.shuffle(equal_samples)
        elif retrieval_result.retrieval_type == "pymultinest":
            equal_samples = retrieval_result.equal_samples
        elif retrieval_result.retrieval_type == "emcee":
            equal_samples = np.copy(retrieval_result.flatchain)
        else:
            assert(False)

        indices = np.random.choice(len(equal_samples), num_samples)
        profile_type = retrieval_result.fit_info.all_params['profile_type'].best_guess
        t_p_profile = Profile()
        profile_pressures = xp.cpu(t_p_profile.pressures)

        temperature_arr = []
        for index in indices:
            params = equal_samples[index]
            params_dict = retrieval_result.fit_info._interpret_param_array(params)
            t_p_profile.set_from_params_dict(profile_type, params_dict)
            temperature_arr.append(xp.cpu(t_p_profile.temperatures))

        plt.figure()
        if plot_samples:
            plt.plot(np.array(temperature_arr).T, profile_pressures / BAR_TO_PASCALS, color='b', alpha=0.25, zorder=2, label='samples')
        if plot_1sigma_bounds:
            plt.fill_betweenx(profile_pressures / BAR_TO_PASCALS, np.percentile(temperature_arr, 16, axis=0),
                            np.percentile(temperature_arr, 84, axis=0), color='0.1', alpha=0.25, zorder=1, label='1$\\sigma$ bounds')
        if plot_median:
            plt.plot(np.percentile(temperature_arr, 50., axis=0), profile_pressures / BAR_TO_PASCALS,
                            color='r', zorder=3, label='Median')

        params_dict = retrieval_result.fit_info._interpret_param_array(retrieval_result.best_fit_params)
        t_p_profile.set_from_params_dict(profile_type, params_dict)
        #plt.plot(xp.cpu(t_p_profile.temperatures), profile_pressures / BAR_TO_PASCALS, zorder=3, color='r', label='best fit')

        plt.yscale('log')
        plt.ylim(min(profile_pressures / BAR_TO_PASCALS), max(profile_pressures / BAR_TO_PASCALS))
        plt.gca().invert_yaxis()
        plt.xlabel("Temperature (K)")
        plt.ylabel("Pressure/bars")
        plt.legend()
        plt.tight_layout()
        if prefix is not None:
            plt.savefig(prefix + "_retrieved_temp_profiles.pdf")


    def plot_retrieval_corner(self, retrieval_result, filename=None, **args):
        """
        Input a RetrievalResult object to make a corner plot for the
        posteriors of the fitted parameters.
        """

        import matplotlib as mpl
        mpl.rcParams['axes.labelsize'] = 16
        mpl.rcParams['axes.titlelocation'] = 'right'

        # Divide Rp by RJ
        labels = np.array(retrieval_result.fit_info.fit_param_names)
        lab = np.where(labels == 'Rp')[0][0]
        # If results were obtained with pymultinest
        if 'equal_samples' in dir(retrieval_result):
            retrieval_result.equal_samples[:, lab] /= R_jup
        labels[lab] = r'Rp/Rj'

        # Get nicer labels
        lab = np.where(labels == 'cloudtop_pressure')[0][0]
        labels[lab] = 'log Pc'

        lab = np.where(labels == 'CO_ratio')[0][0]
        labels[lab] = 'C/O'

        lab = np.where(labels == 'scatt_factor')[0][0]
        labels[lab] = 'scatt. factor'

        lab = np.where(labels == 'scatt_slope')[0][0]
        labels[lab] = 'scatt. slope'

        newlabels = [s.replace('_', ' ') for s in labels]

        assert(isinstance(retrieval_result, RetrievalResult))
        if retrieval_result.retrieval_type == "dynesty":
            fig = corner.corner(retrieval_result.samples, weights=retrieval_result.weights,
                                range=[0.99] * retrieval_result.samples.shape[1],
                                show_titles=True, title_kwargs={'fontsize':20,
                                'loc':'left'}, quantiles=[0.16, 0.50, 0.84],
                                smooth1d=3, plot_contours=False,
                                labels=newlabels, **args)
        elif retrieval_result.retrieval_type == "pymultinest":
            fig = corner.corner(retrieval_result.equal_samples,
                                range=[0.99] * retrieval_result.equal_samples.shape[1],
                                show_titles=True, title_kwargs={'fontsize':20,
                                'loc':'left'}, quantiles=[0.16, 0.50, 0.84],
                                smooth1d=3, plot_contours=False,
                                labels=newlabels, **args)
        elif retrieval_result.retrieval_type == "emcee":
            fig = corner.corner(retrieval_result.flatchain,
                                range=[0.99] * retrieval_result.flatchain.shape[1],
                                labels=newlabels, **args)
        else:
            assert(False)

        if filename is not None:
            fig.savefig(filename)


    def plot_retrieval_transit_spectrum(self, retrieval_result, prefix=None,
            plot_best_fit=False, bin_spectrum=100):
        """
        Input a RetrievalResult object to make a plot of the data,
        best fit transit model both at native resolution and data's resolution,
        and a 1 sigma range for models.
        """
        assert(isinstance(retrieval_result, RetrievalResult))
        assert(retrieval_result.transit_bins is not None)

        lower_spectrum = np.percentile(retrieval_result.random_transit_depths, 16, axis=0)
        upper_spectrum = np.percentile(retrieval_result.random_transit_depths, 84, axis=0)
        median_spectrum = np.percentile(retrieval_result.random_transit_depths, 50, axis=0)

        if bin_spectrum is not None:
            wlbin = rebin.rebin(retrieval_result.best_fit_transit_dict["unbinned_wavelengths"], bin_spectrum)
            lower_spectrum = rebin.rebin(lower_spectrum, bin_spectrum)
            upper_spectrum = rebin.rebin(upper_spectrum, bin_spectrum)
            median_spectrum = rebin.rebin(median_spectrum, bin_spectrum)
        else:
            wlbin = retrieval_result.best_fit_transit_dict["unbinned_wavelengths"]

        fig, ax = plt.subplots()
        ax.fill_between(METRES_TO_UM * wlbin, lower_spectrum*1e6, upper_spectrum*1e6,
                            color="#f2c8c4", zorder=2)
        if plot_best_fit:
            ax.plot(METRES_TO_UM * retrieval_result.best_fit_transit_dict["unbinned_wavelengths"],
                    retrieval_result.best_fit_transit_dict["unbinned_depths"] *
                    retrieval_result.best_fit_transit_dict['unbinned_correction_factors']*1e6,
                    color='r', label="Calculated", zorder=3)
        else:
            ax.plot(METRES_TO_UM * wlbin,
                    median_spectrum*1e6, color='r', label="Calculated (binned)",
                    zorder=3)
        ax.errorbar(METRES_TO_UM * retrieval_result.transit_wavelengths,
                        retrieval_result.transit_depths*1e6,
                        yerr = retrieval_result.transit_errors*1e6,
                        fmt='.', color='k', label="Observed", zorder=5)
        if bin_spectrum is None:
            ax.scatter(METRES_TO_UM * retrieval_result.transit_wavelengths,
                    retrieval_result.best_fit_transit_depths*1e6,
                    color='b', label="Calculated (binned)", zorder=4)

        ax.set_xlabel("Wavelength ($\mu m$)")
        ax.set_ylabel("Transit depth [ppm]")
        ax.set_xscale('log')
        ax.xaxis.set_ticks([0.5, 1, 2, 3, 4, 5], \
                labels=['0.5', '1', '2', '3', '4', '5'])
        plt.tight_layout()
        plt.legend()
        if prefix is not None:
            plt.savefig(prefix + "_best_fit.pdf")


    def plot_retrieval_eclipse_spectrum(self, retrieval_result, prefix=None,
            plot_best_fit=False, bin_spectrum=100):
        """
        Input a RetrievalResult object to make a plot of the data,
        best fit eclipse model both at native resolution and data's resolution,
        and a 1 sigma range for models.
        """
        assert(isinstance(retrieval_result, RetrievalResult))
        assert(retrieval_result.eclipse_bins is not None)

        lower_spectrum = np.percentile(retrieval_result.random_eclipse_depths, 16, axis=0)
        upper_spectrum = np.percentile(retrieval_result.random_eclipse_depths, 84, axis=0)
        median_spectrum = np.percentile(retrieval_result.random_eclipse_depths, 50, axis=0)

        if bin_spectrum is not None:
            wlbin = rebin.rebin(retrieval_result.best_fit_eclipse_dict["unbinned_wavelengths"], bin_spectrum)
            lower_spectrum = rebin.rebin(lower_spectrum, bin_spectrum)
            upper_spectrum = rebin.rebin(upper_spectrum, bin_spectrum)
            median_spectrum = rebin.rebin(median_spectrum, bin_spectrum)
        else:
            wlbin = retrieval_result.best_fit_eclipse_dict["unbinned_wavelengths"]

        fig, ax = plt.subplots()
        ax.fill_between(METRES_TO_UM * wlbin, lower_spectrum*1e6, upper_spectrum*1e6,
                            color="#f2c8c4")

        if plot_best_fit:
            ax.plot(METRES_TO_UM * retrieval_result.best_fit_eclipse_dict["unbinned_wavelengths"]*1e6,
                    retrieval_result.best_fit_eclipse_dict["unbinned_eclipse_depths"]*1e6,
                    alpha=0.4, color='r', label="Calculated (unbinned)")
        else:
            ax.plot(METRES_TO_UM * wlbin,
                    median_spectrum*1e6, color='r', label="Calculated",
                    zorder=3)
        ax.errorbar(METRES_TO_UM * retrieval_result.eclipse_wavelengths,
                        retrieval_result.eclipse_depths*1e6,
                        yerr=retrieval_result.eclipse_errors*1e6,
                        fmt='.', color='k', label="Observed")
        if bin_spectrum is None:
            ax.scatter(METRES_TO_UM * retrieval_result.eclipse_wavelengths,
                    retrieval_result.best_fit_eclipse_depths*1e6,
                    color='r', label="Calculated (binned)")
        plt.legend()
        ax.set_xlabel("Wavelength ($\mu m$)")
        ax.set_ylabel("Eclipse depth [ppm]")
        ax.set_xscale('log')
        ax.xaxis.set_ticks([1, 2, 3, 4, 5], \
                labels=['1', '2', '3', '4', '5'])
        plt.tight_layout()
        plt.legend()
        if prefix is not None:
            plt.savefig(prefix + "_best_fit.pdf")


    def plot_optical_depth(self, depth_dict, prefix=None):
        """
        Input a depth dictionary created by the TransitDepthCalculator or EclipseDepthCalculator
        to plot optical depth as a function of wavelength and pressure.
        """
        plt.figure(figsize=(6,4))

        if 'tau_los' in depth_dict.keys():
            plt.contourf(depth_dict['unbinned_wavelengths'] * METRES_TO_UM,
                         np.log10(0.5 * (depth_dict['P_profile'][1:] + depth_dict['P_profile'][:-1]) / BAR_TO_PASCALS), np.log10(depth_dict['tau_los'].T), cmap='magma_r')
            fname = '_transit'
        elif 'taus' in depth_dict.keys():
            plt.contourf(depth_dict['unbinned_wavelengths'] * METRES_TO_UM,
                         np.log10(0.5 * (depth_dict['P_profile'][1:] + depth_dict['P_profile'][:-1]) / BAR_TO_PASCALS), np.log10(depth_dict['taus'].T), cmap='magma_r')
            fname = '_eclipse'
        else:
            print("Depth dictionary does not contain optical depth information.")
            assert(False)

        cbar = plt.colorbar(location='right')
        cbar.set_label('log (Optical depth)')
        plt.gca().invert_yaxis()
        plt.xlabel('Wavelength ($\\mu$m)')
        plt.ylabel('log (Pressure/bars)')
        plt.tight_layout()
        if prefix is not None:
            plt.savefig(prefix + fname + "_optical_depth.png")


    def plot_eclipse_contrib_func(self, eclipse_depth_dict, log_scale=False, prefix=None):
        """
        Input an eclipse depth dictionary created by the EclipseDepthCalculator
        to plot emission contribution function as a function of wavelength and pressure.
        The log_scale parameter allows the user to toggle between plotting of the contribution
        function in log or linear scale.
        """
        assert('contrib' in eclipse_depth_dict.keys())

        if log_scale:
            contrib_func = np.log10(eclipse_depth_dict['contrib'].T)
            contrib_func[np.logical_or(np.isinf(contrib_func), contrib_func < -9.)] = np.nan
        else:
            contrib_func = eclipse_depth_dict['contrib'].T

        plt.figure(figsize=(6,4))
        plt.contourf(eclipse_depth_dict['unbinned_wavelengths'] * METRES_TO_UM,
                         np.log10(0.5 * (eclipse_depth_dict['P_profile'][1:] + eclipse_depth_dict['P_profile'][:-1]) / BAR_TO_PASCALS), contrib_func, cmap='magma_r', vmin=np.nanmin(contrib_func), vmax=np.nanmax(contrib_func))

        cbar = plt.colorbar(location='right')
        if log_scale:cbar.set_label('log (Contribution function)')
        else:cbar.set_label('Contribution function')
        plt.gca().invert_yaxis()
        plt.xlabel('Wavelength ($\\mu$m)')
        plt.ylabel('log (Pressure/bars)')
        plt.tight_layout()
        if prefix is not None:
            plt.savefig(prefix + "_eclipse_contrib_func.png")


    def plot_atm_abundances(self, atm_info, min_abund=1e-9, prefix=None):
        """
        Input a depth dictionary created by the TransitDepthCalculator or EclipseDepthCalculator
        or a dictionary outputed by AtmsophereSolver
        to plot abundance of different species (calculated for a given TP profile) as a function of pressure.
        """
        assert('atm_abundances' in atm_info.keys())
        abundances = atm_info['atm_abundances']

        plt.figure()
        for k in abundances.keys():
            if k == 'He' or k == 'H2' or k == 'H':
                continue
            if np.any(abundances[k] > min_abund):
                plt.loglog(abundances[k], atm_info['P_profile'] / BAR_TO_PASCALS, label=k)

        plt.gca().invert_yaxis()
        plt.xlim(min_abund,)
        plt.xlabel('Abundance ($n/n_{\\rm tot}$)')
        plt.ylabel('Pressure (bars)')
        plt.legend()
        plt.tight_layout()
        if prefix is not None:
            plt.savefig(prefix + "_atm_abundances.png")
