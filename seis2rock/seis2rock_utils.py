import numpy as np
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Tuple
from scipy.ndimage.interpolation import shift

from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_Volve_seismic_data(wellname, offset_num, angle, 
                            zwell, z_seismic_prestack_fence, 
                            window_min, window_max, offset, seismicd_prestack_fence,
                            fontsize, title_on=False):
    """
    This function plots the different types of the prestack data.

    Parameters:
    well_name (str): Name of the well.
    offset_num (int): Offset number to plot.
    angle (int): Angle of the prestack to plot.
    zwell (1darray): Array of well depth.
    z_seismic_prestack_fence (3darray): Array of seismic prestack fence.
    window_min (int): Minimum index for the window depth for the prestack data.
    window_max (int): Maximum index for the window depth for the prestack data.
    offset (3darrat): Array of offset data (x-axis, offset, depth/time-axis).
    seismicd_prestack_fence (3darray): Array of seismic prestack fence data (x-axis, angles, depth/time-axis).

    Returns:
    fig (object): figure of the plot
    """
    plt.rcParams.update({'font.size': fontsize})
    
    # Calculate the indices for the closest values in zwell array
    z_min_index = np.abs(zwell - z_seismic_prestack_fence[window_min]).argmin()
    z_max_index = np.abs(zwell - z_seismic_prestack_fence[window_max]).argmin()

    # Copy zwell array and set values outside the window to NaN
    zwell_plot = np.copy(zwell)
    zwell_plot[:z_min_index] = np.nan
    zwell_plot[z_max_index:] = np.nan

    # Create subplots
    fig, ax = plt.subplots(1,3, figsize=(22, 6), gridspec_kw={'width_ratios': [3, 3, 3]})
    if title_on==True:
        plt.subplots_adjust(top=0.85)
        fig.suptitle(f'Fence along well: {wellname}', fontsize=fontsize+5)
    
    ax = ax.ravel()

    # Plot offset data
    ax[0].imshow(offset[:,offset_num,:].T, cmap='gray', vmin=-1, vmax=1,
           extent=[0, seismicd_prestack_fence.shape[0], 
                   z_seismic_prestack_fence[-1], 
                   z_seismic_prestack_fence[0]])
    ax[0].plot(np.arange(len(zwell)), zwell_plot, 'r')
    ax[0].set_title(f'Prestack Offset (off={offset_num})')
    ax[0].axis('tight')
    ax[0].set_ylabel('Depth (m)')
    ax[0].set_xlabel('# Trace')

    # Plot prestack data depth
    ax[1].imshow(seismicd_prestack_fence[:,angle,:].T, cmap='gray', vmin=-1, vmax=1,
           extent=[0, seismicd_prestack_fence.shape[0], 
                   z_seismic_prestack_fence[-1], 
                   z_seismic_prestack_fence[0]])
    ax[1].plot(np.arange(len(zwell)), zwell_plot, 'r')
    ax[1].set_title(f'Prestack angles ($\\theta=$ {angle})')
    ax[1].axis('tight')
    ax[1].set_xlabel('# Trace')

    # Plot prestack for inversion
    ax[2].imshow(seismicd_prestack_fence[:,angle,window_min:window_max].T, cmap='gray', vmin=-1, vmax=1,
           extent=[0, seismicd_prestack_fence.shape[0], 
                   z_seismic_prestack_fence[window_max], 
                   z_seismic_prestack_fence[window_min]])
    ax[2].plot(np.arange(len(zwell_plot)), zwell_plot, 'r')
    ax[2].set_title(f'Prestack angles for inversion ($\\theta=$ {angle})')
    ax[2].axis('tight')
    ax[2].set_ylim((z_seismic_prestack_fence[window_max],z_seismic_prestack_fence[window_min]))
    ax[2].set_xlabel('# Trace')

    # Return
    fig.tight_layout()
    
    return fig

def plot_background_models_fence(vp_fence_bg, vs_fence_bg, rho_fence_bg, 
                                 phi_fence_bg, vsh_fence_bg, sw_fence_bg, 
                                 seismicd_prestack_fence, z_seismic_prestack_fence):
    """
    This function plots the background models using matplotlib.

    Parameters:
    vp_fence_bg (numpy array): Array of vp background model data.
    vs_fence_bg (numpy array): Array of vs background model data.
    rho_fence_bg (numpy array): Array of rho background model data.
    phi_fence_bg (numpy array): Array of phi background model data.
    vsh_fence_bg (numpy array): Array of vsh background model data.
    sw_fence_bg (numpy array): Array of sw background model data.
    seismicd_prestack_fence (numpy array): Array of seismic prestack fence data.
    z_seismic_prestack_fence (numpy array): Array of seismic prestack fence depths.

    Returns:
    fig (object): figure of the plot
    """
    # Create subplots
    fig, ax = plt.subplots(2,3, figsize=(20, 10))
    fig.suptitle("Background models Fence")
    ax = ax.ravel()

    # Plot vp background model
    img0 = ax[0].imshow(vp_fence_bg.T, extent=[0, seismicd_prestack_fence.shape[0], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]],
                        cmap='terrain')
    ax[0].set_title('$\mathrm{V_p}$')
    ax[0].axis('tight')
    ax[0].set_ylabel('Depth (m)')
    plt.colorbar(img0, ax=ax[0], shrink=0.9)

    # Plot vs background model
    img1 = ax[1].imshow(vs_fence_bg.T, extent=[0, seismicd_prestack_fence.shape[0], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]],
                        cmap='terrain')
    ax[1].set_title('$\mathrm{V_s}$')
    ax[1].axis('tight')
    plt.colorbar(img1, ax=ax[1], shrink=0.9)

    # Plot rho background model
    img2 = ax[2].imshow(rho_fence_bg.T, extent=[0, seismicd_prestack_fence.shape[0], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]],
                        cmap='terrain')
    ax[2].set_title('$\mathrm{\\rho}$')
    ax[2].axis('tight')
    plt.colorbar(img2, ax=ax[2], shrink=0.9)

    # Plot phi background model
    img3 = ax[3].imshow(phi_fence_bg.T, extent=[0, seismicd_prestack_fence.shape[0], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]],
                        cmap='jet')
    ax[3].set_title('$\mathrm{\phi}$')
    ax[3].axis('tight')
    ax[3].set_ylabel('Depth (m)')
    ax[3].set_ylabel('# Traces')
    plt.colorbar(img3, ax=ax[3], shrink=0.9)

    # Plot vsh background model
    img4 = ax[4].imshow(vsh_fence_bg.T, extent=[0, seismicd_prestack_fence.shape[0], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]],
                        cmap='summer_r')
    ax[4].set_title('$\mathrm{V_{sh}}$')
    ax[4].axis('tight')
    ax[4].set_ylabel('# Traces')
    plt.colorbar(img4, ax=ax[4], shrink=0.9)

    # Plot sw background model
    img5 = ax[5].imshow(sw_fence_bg.T, extent=[0, seismicd_prestack_fence.shape[0], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]],
                        cmap='winter_r')
    ax[5].set_title('$\mathrm{S_w}$')
    ax[5].axis('tight')
    ax[5].set_ylabel('# Traces')
    plt.colorbar(img5, ax=ax[5], shrink=0.9)
    
    fig.tight_layout()

    return fig
    





def display_wavelet(wav_est, fwest, wav_est_fft, t_wav, nfft, fontsize = 14):
    """
    Function to display the wavelet estimate.
    
    Parameters:
    wav_est : 1darray
        The wavelet estimate.
    fwest : 1darray
        The FFT frequency.
    wav_est_fft : 1darray
        The FFT of the estimated wavelet.
    t_wav : 1darray
        The wavelet time/depth axis.
    nfft : int
        Number of samples of FFT.
    fontsize : int
        fontsize of the plot. Default is 14.
    """

    plt.rcParams.update({'font.size': fontsize})

    # Display wavelet
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    fig.suptitle('Statistical wavelet estimate')
    axs[0].plot(fwest[:nfft//2], wav_est_fft[:nfft//2], 'k')
    axs[0].set_title('Frequency')
    axs[1].plot(t_wav, wav_est, 'k')
    axs[1].set_title('Time')
    plt.show()
    



def plot_set_logs(well_name=None, well_prestack=None, extent_prestack=None, well_depth=None, 
              vp=None, vs=None, rho=None, phi=None, vsh=None, sw=None,
              vp_back=None, vs_back=None, rho_back=None,
              phi_back=None, vsh_back=None, sw_back=None, figsize=(16,8)):
    """
    Function to plot well log data and prestack data.

    Parameters:
    well_name: str, optional
        The name of the well. Default is None, in which case 'Well-logs information' will be used as the title.
    
    well_prestack: 2darray, optional
        Array containing prestack data along the well. If not provided, prestack plot will not be shown (depth/time-axis,angles).

    extent_prestack: tuple, optional
        Extent of the prestack plot in the format (thetamin, thetamax, depth_well_end, depth_well_start)). If not provided, prestack plot will be shown without extent.
    
    well_depth: 1darray
        Array containing depth data of the well.

    vp, vs, rho, phi, vsh, sw: 1darray
        Arrays containing original log data.

    vp_back, vs_back, rho_back, phi_back, vsh_back, sw_back: 1darray
        Arrays containing the background models for respective logs.

    figsize: tuple, optional
        Size of the figure to be plotted. Default is (16,8).
    """
    fig, ax = plt.subplots(1,7, figsize=figsize, sharey=False)
    fig.suptitle(well_name if well_name else 'Well-logs information')
    ax = ax.ravel()

    if well_prestack is not None:
        if extent_prestack is not None:
            ax0=ax[0].imshow(well_prestack, extent=extent_prestack, cmap='gray', vmin=-4, vmax=4)
        else:
            ax0=ax[0].imshow(well_prestack, cmap='gray', vmin=-4, vmax=4)
        ax[0].axis('tight')
        ax[0].set_title('Pre-stack along the well')
        plt.colorbar(ax0, ax=ax[0], shrink=0.9)

    ax[1].plot(vp, well_depth, 'k', lw=3, label='VP log')
    ax[1].plot(vp_back, well_depth, '--r', lw=3, label='bg VP log')
    ax[1].legend()
    ax[1].set_title('VP')
    ax[1].invert_yaxis()

    ax[2].plot(vs, well_depth,'k', lw=3, label='VS log')
    ax[2].plot(vs_back,well_depth,'--r', lw=3, label='bg VS log')
    ax[2].legend()
    ax[2].set_title('VS')
    ax[2].invert_yaxis()

    ax[3].plot(rho, well_depth,'k', lw=3, label='Rho log')
    ax[3].plot(rho_back,well_depth,'--r', lw=3, label='bg Rho log')
    ax[3].legend()
    ax[3].set_title('Rho')
    ax[3].invert_yaxis()

    ax[4].plot(phi, well_depth,'k', lw=3, label='Phi log')
    ax[4].plot(phi_back,well_depth,'--r', lw=3, label='bg Phi log')
    ax[4].legend()
    ax[4].set_title('Phi')
    ax[4].invert_yaxis()

    ax[5].plot(vsh, well_depth,'k', lw=3, label='Vsh log')
    ax[5].plot(vsh_back,well_depth,'--r', lw=3, label='bg Vsh log')
    ax[5].legend()
    ax[5].set_title('Vsh')
    ax[5].invert_yaxis()

    ax[6].plot(sw, well_depth,'k', lw=3, label='Sw log')
    ax[6].plot(sw_back,well_depth,'--r', lw=3, label='bg Sw log')
    ax[6].legend()
    ax[6].set_title('Sw')
    ax[6].invert_yaxis()

    fig.tight_layout()


def plot_inversion_results_fence(wellname, prestack_fence, zwell, z_seismic_prestack_fence, 
                           phi_inv_dense_reg, vsh_inv_dense_reg, sw_inv_dense_reg, b_optAVO,
                           phi_fence_bg, vsh_fence_bg, sw_fence_bg, fontsize=12, title_on=True):
    
    """
    Function that plots the inversion results of the petrophysical properties of the fence, 
    the prestack data, the B optimal coefficient matrix, and the respective background models
    used in the inversion.

    Parameters:
        wellname (str): Name of the well.
        prestack_fence (ndarray): Preprocessed seismic data for the fence.
        zwell (1darray): Depth values for the well log.
        z_seismic_prestack_fence (3darray): Depth values for the seismic data (x-axis, angles, depth/time-axis).
        phi_inv_dense_reg (2darray): Inverted porosity values (depth/time-axis, x-axis).
        vsh_inv_dense_reg (2darray): Inverted volume of shale values (depth/time-axis, x-axis).
        sw_inv_dense_reg (2darray): Inverted water saturation values (depth/time-axis, x-axis).
        b_optAVO (3darray): Matrix of petrophysical coefficients B "new data term" (property, depth/time-axis, x-axis).
        phi_fence_bg (2darray): Background model for porosity (x-axis, depth/time-axis).
        vsh_fence_bg (2darray): Background model for volume of shale (x-axis, depth/time-axis).
        sw_fence_bg (2darray): Background model for water saturation (x-axis, depth/time-axis).
        fontsize (int, optional): Font size for the plot. Default is 12.
        

    Returns:
        fig (Figure): The matplotlib figure object containing the plot.
    """
    
    plt.rcParams.update({'font.size': fontsize})
    
    fig = plt.figure(figsize=(22,15))
    gs = gridspec.GridSpec(5, 3, height_ratios=[1,1,1,0.1,0.1])
    
    
    # Calculate x-values and z-values
    xfence = np.arange(prestack_fence.shape[0])
    xfence_int = np.arange(0, phi_inv_dense_reg.shape[1], 0.2)
    xwell_int = np.arange(0, len(zwell), 0.2)

    zwell_int = np.interp(xwell_int, np.arange(len(zwell))[~np.isnan(zwell)],
                          zwell[~np.isnan(zwell)], left=np.nan, right=np.nan)
    
    
    # Subplots 1-2 share a colorbar
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    
    # vmin_b0, vmax_b0 = -1.5,1.5
    vmin_b0, vmax_b0 = np.percentile(b_optAVO[0], [1,99])
    im0 = ax0.imshow(b_optAVO[0],cmap='gray', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_b0, vmax=vmax_b0)
    ax0.set_title('$\mathrm{B_{\phi}}$')
    ax0.axis('tight')
    # ax0.set_xlabel('# Traces')
    # ax0.set_ylabel('a) \n Depth', fontsize=fontsize+3)
    ax0.set_ylabel('Depth (m)', fontsize=fontsize+3)
    ax0.set_xticklabels([])
    ax0.plot(xfence_int, zwell_int,  'r', linewidth=2)
    
    vmin_b1, vmax_b1 = np.percentile(b_optAVO[1], [1,99])
    im1 = ax1.imshow(b_optAVO[1],cmap='gray', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_b1, vmax=vmax_b1)
    ax1.set_title('$\mathrm{B_{V_{sh}}}$')
    ax1.axis('tight')
    # ax1.set_xlabel('# Traces')
    # ax1.set_ylabel('Depth (m)')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.plot(xfence_int, zwell_int,  'r', linewidth=2)
    
    # vmin_b2, vmax_b2 = -1.5,1.5
    vmin_b2, vmax_b2 = np.percentile(b_optAVO[2], [1,99])
    im2 = ax2.imshow(b_optAVO[2],cmap='gray', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_b2, vmax=vmax_b2)
    ax2.set_title('$\mathrm{B_{S_{w}}}$')
    ax2.axis('tight')
    # ax2.set_xlabel('# Traces')
    # ax2.set_ylabel('Depth (m)')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.plot(xfence_int, zwell_int,  'r', linewidth=2)

    # Subplots for phi, Vsh, and Sw with their colorbars
    vmin_phi, vmax_phi = np.percentile(phi_inv_dense_reg, [1,99])
    ax3 = fig.add_subplot(gs[1, 0])
    ax6 = fig.add_subplot(gs[2, 0])
    im3 = ax3.imshow(phi_inv_dense_reg, cmap='jet',  extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_phi, vmax=vmax_phi)
    ax3.set_title('$\phi$')
    ax3.axis('tight')
    # ax3.set_xlabel('# Traces')
    ax3.set_xticklabels([])
    # ax3.set_yticklabels([])
    ax3.set_ylabel('Depth (m)', fontsize=fontsize+3)
    # ax3.set_ylabel('b) \nDepth (m)', fontsize=fontsize+3)

    im6 = ax6.imshow(phi_fence_bg.T, cmap='jet', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_phi, vmax=vmax_phi)
    ax6.set_title('${\mathrm{Background}}~\phi$')
    ax6.axis('tight')
    ax6.set_xlabel('# Traces', fontsize=fontsize+3)
    ax6.set_ylabel('Depth (m)', fontsize=fontsize+3)
    # ax6.set_ylabel('c) \nDepth (m)', fontsize=fontsize+3)
    # ax6.set_xticklabels([])
    # ax6.set_yticklabels([])

    
    vmin_vsh, vmax_vsh = np.percentile(vsh_inv_dense_reg, [1,99]) 
    # vmin_vsh = 0
    ax4 = fig.add_subplot(gs[1, 1])
    ax7 = fig.add_subplot(gs[2, 1])
    im4 = ax4.imshow(vsh_inv_dense_reg, cmap='summer_r', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_vsh, vmax=vmax_vsh)
    ax4.set_title('$\mathrm{V_{sh}}$')
    ax4.axis('tight')
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    # ax2.set_xlabel('# Traces')
    # ax2.set_ylabel('Depth (m)')

    im7 = ax7.imshow(vsh_fence_bg.T, cmap='summer_r', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_vsh, vmax=vmax_vsh)
    ax7.set_title('$\mathrm{Background~V_{sh}}$')
    ax7.axis('tight')
    ax7.set_xlabel('# Traces', fontsize=fontsize+3)
    # ax7.set_ylabel('Depth (m)')
    # ax7.set_xticklabels([])
    ax7.set_yticklabels([])


    vmin_sw, vmax_sw = np.percentile(sw_inv_dense_reg, [1,99])
    vmin_sw = 0
    ax5 = fig.add_subplot(gs[1, 2])
    ax8 = fig.add_subplot(gs[2, 2])
    im5 = ax5.imshow(sw_inv_dense_reg, cmap='winter_r', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_sw, vmax=vmax_sw)
    ax5.set_title('$\mathrm{S_{w}}$')
    ax5.axis('tight')
    # ax5.set_xlabel('# Traces')
    # ax5.set_ylabel('Depth (m)')
    ax5.set_xticklabels([])
    ax5.set_yticklabels([])

    im8 = ax8.imshow(sw_fence_bg.T, cmap='winter_r', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_sw, vmax=vmax_sw)
    ax8.set_title('$\mathrm{Background~S_{w}}$')
    ax8.axis('tight')
    ax8.set_xlabel('# Traces', fontsize=fontsize+3)
    # ax8.set_ylabel('Depth (m)')
    # ax8.set_xticklabels([])
    ax8.set_yticklabels([])
    
    
    # Get the position and size of the subplots
    box0 = ax0.get_position()
    box1 = ax1.get_position()
    box2 = ax2.get_position()

    # Create the colorbar axes
    cbaxes0 = fig.add_axes([box0.x0, 0.08, box0.width, 0.02]) 
    cbaxes1 = fig.add_axes([box1.x0, 0.08, box1.width, 0.02]) 
    cbaxes2 = fig.add_axes([box2.x0, 0.08, box2.width, 0.02]) 
    cbaxes3 = fig.add_axes([box0.x0, 0.01, box0.width, 0.02]) 
    cbaxes4 = fig.add_axes([box1.x0, 0.01, box1.width, 0.02])
    cbaxes5 = fig.add_axes([box2.x0, 0.01, box2.width, 0.02]) 
    

    # Add the colorbars to the colorbar axes
    cbar0 = fig.colorbar(im3, cax=cbaxes0, orientation='horizontal', cmap='jet')
    cbar1 = fig.colorbar(im4, cax=cbaxes1, orientation='horizontal', cmap='summer_r')
    cbar2 = fig.colorbar(im5, cax=cbaxes2, orientation='horizontal', cmap='winter_r')
    
    cbar3 = fig.colorbar(im0, cax=cbaxes3, orientation='horizontal', cmap= 'gray')
    cbar4 = fig.colorbar(im1, cax=cbaxes4, orientation='horizontal', cmap= 'gray')
    cbar5 = fig.colorbar(im2, cax=cbaxes5, orientation='horizontal', cmap= 'gray')

    # Add a title to each colorbar
    cbar0.set_label('$\phi$', fontsize=fontsize)
    cbar1.set_label('$\mathrm{V_{sh}}$', fontsize=fontsize)
    cbar2.set_label('$\mathrm{S_{w}}$', fontsize=fontsize)
    
    cbar3.set_label('$\mathrm{B_{\phi}}$', fontsize=fontsize)
    cbar4.set_label('$\mathrm{B_{V_{sh}}}$', fontsize=fontsize)
    cbar5.set_label('$\mathrm{B_{S_{w}}}$', fontsize=fontsize)

    # Adjust the main title to prevent overlap with the top colorbar
    if title_on==True:
        plt.subplots_adjust(top=0.85)
        fig.suptitle(f'Inversion results for well: {wellname}', fontsize=fontsize+2)
        
    fig.tight_layout()
    
    return fig
 
 
 
def plot_well_results_from_fence(wellname, zwell_seismic, seismicd_prestack_fence, zwell,
                                 z_seismic_prestack_fence,
                                 phi, vsh, sw,
                                 well_start_data, well_end_data, 
                                 phi_inv_dense_reg, vsh_inv_dense_reg, sw_inv_dense_reg,
                                 phi_fence_bg, vsh_fence_bg, sw_fence_bg,
                                 shift=30, fontsize=12, title_on=True):
    """
    Function that plots the results along the well by extracting the data from the fence 
    for the petrophysical properties: porosity, volume of shale, and water saturation.

    Parameters:
        wellname (str): Name of the well.
        zwell_seismic (1darray): Depth values for the well in the seismic data sampling.
        seismicd_prestack_fence (3darray): Seismic data along the fence (x-axis, angles, depth/time-axis).
        zwell (1darray): Depth values for the well at the seismic sampling.
        z_seismic_prestack_fence (1darray): Depth values for the seismic data at the seismic sampling.
        phi (1darray): Porosity values from the well log.
        vsh (1darray): Volume of shale values from the well log.
        sw (1darray): Water saturation values from the well log.
        well_start_data (int): Start index of the well data.
        well_end_data (int): End index of the well data.
        phi_inv_dense_reg (2darray): Inverted porosity values from the fence (depth/time-axis, x-axis).
        vsh_inv_dense_reg (2darray): Inverted volume of shale values from the fence (depth/time-axis, x-axis).
        sw_inv_dense_reg (2darray): Inverted water saturation values from the fence (depth/time-axis, x-axis).
        phi_fence_bg (2darray): Background model for porosity from the fence (x-axis, depth/time-axis).
        vsh_fence_bg (2darray): Background model for volume of shale from the fence (x-axis, depth/time-axis)..
        sw_fence_bg (2darray): Background model for water saturation from the fence (x-axis, depth/time-axis)..
        shift (float): Shift parameter for calibration. Default is 30 and is the calibration in index.
        fontsize (int): Font size for the plot. Default is 12.
        title_on (bool): If True, the plot will have a title. Default is True.

    Returns:
        fig (Figure): The matplotlib figure object containing the plot.
    """
    
    plt.rcParams.update({'font.size': fontsize})
    
    #Plotting well
    xfence = np.arange(seismicd_prestack_fence.shape[0])
    xfence_int = np.arange(0, phi_inv_dense_reg.shape[1], 0.2)
    xwell_int = np.arange(0, len(zwell), 0.2)

    zwell_int = np.interp(xwell_int, np.arange(len(zwell))[~np.isnan(zwell)],
                        zwell[~np.isnan(zwell)], left=np.nan, right=np.nan)

    #porosity
    phiextr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            phi_inv_dense_reg.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)

    phi_bg_extr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            phi_fence_bg, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)


    #shale content
    vshextr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            vsh_inv_dense_reg.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)

    vsh_bg_extr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            vsh_fence_bg, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)

    #water saturation
    swextr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            sw_inv_dense_reg.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)

    sw_bg_extr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            sw_fence_bg, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)
    
    fig, ax = plt.subplots(1, 3, figsize=(7,8), sharey=True)
    

    ax[0].plot(phi, zwell_seismic[well_start_data:well_end_data]+shift, label='Well log', color='k', linewidth=1.5, alpha=0.7)
    ax[0].plot(phiextr1_t_int, zwell_int, label='Inversion',  color='r', linewidth=2)
    ax[0].plot(phi_bg_extr1_t_int, zwell_int, label='Background', color='k', linestyle='--',linewidth=0.8)
    ax[0].set_title('$\phi$')
    ax[0].set_ylabel('Depth (m)')
    ax[0].invert_yaxis()
    ax[0].legend(loc="lower right", fontsize=9)
    

    ax[1].plot(vsh, zwell_seismic[well_start_data:well_end_data]+shift, label='Well log', color='k', linewidth=1.5, alpha=0.7)
    ax[1].plot(vshextr1_t_int, zwell_int, label='Inversion',  color='g', linewidth=2)
    ax[1].plot(vsh_bg_extr1_t_int, zwell_int, label='Background', color='k', linestyle='--',linewidth=0.8)
    ax[1].set_title('$\mathrm{V_{sh}}$')
    ax[1].invert_yaxis()
    ax[1].legend(loc="lower right", fontsize=9)

    ax[2].plot(sw, zwell_seismic[well_start_data:well_end_data]+shift, label='Well log', color='k', linewidth=1.5, alpha=0.7)
    ax[2].plot(swextr1_t_int, zwell_int,  label='Inversion',  color='b', linewidth=2)
    ax[2].plot(sw_bg_extr1_t_int, zwell_int, label='Background', color='k', linestyle='--',linewidth=0.8)
    ax[2].set_title('$\mathrm{S_{w}}$')
    ax[2].invert_yaxis()
    ax[2].legend(loc="lower right", fontsize=fontsize-2)
    
     # Adjust the main title to prevent overlap with the top colorbar
    if title_on==True:
        plt.subplots_adjust(top=0.85)
        fig.suptitle(f'Inversion results for well: {wellname}', fontsize=fontsize+2)
    fig.tight_layout()

    return fig


def plot_comparison_gathers_wavelet(well_prestack, synthetic_gather, vp, vp_back, thetamin, 
                                    thetamax, zwell_seismic, z_seismic_prestack_fence, well_start_data,
                                    well_end_data, window_min, window_max):
    """
    This function plots a comparison of the prestack gather along the well (real and synthetic) and also one well log.

    Parameters:
    well_prestack: The real well prestack data.
    synthetic_gather: The synthetic gather data.
    vp, vp_back: The VP log data and background VP log data.
    thetamin, thetamax: The minimum and maximum theta values.
    zwell_seismic: The seismic depth data for the well.
    z_seismic_prestack_fence: The seismic depth data for the prestack fence.
    well_start_data, well_end_data: The start and end indices for the well data.
    window_min, window_max: The minimum and maximum window indices for the prestack fence.

    Returns:
    fig (object): A plot comparing the real and synthetic prestack gather along the well and the VP log.
    """

    # Create a figure with 4 subplots
    fig, ax = plt.subplots(1,4, figsize=(12, 8))

    # Plot the real well prestack data
    max_abs_val = np.max(np.abs(well_prestack))
    ax0=ax[0].imshow(well_prestack, extent=(thetamin, thetamax, zwell_seismic[-1], zwell_seismic[0]), cmap='gray',
                     vmin=-max_abs_val, vmax=max_abs_val)
    ax[0].set_title('Real')
    ax[0].axis('tight')
    # ax[0].axhline(y=2950, color='red')
    # ax[0].axhline(y=2625, color='red')
    plt.colorbar(ax0, ax=ax[0], shrink=0.7)

    # Plot the real well prestack data with well data
    ax1=ax[1].imshow(well_prestack, extent=(thetamin, thetamax, zwell_seismic[-1], zwell_seismic[0]), cmap='gray', 
                      vmin=-max_abs_val, vmax=max_abs_val)
    ax[1].set_title('Real (well data)')
    ax[1].axis('tight')
    # ax[1].axhline(y=2950, color='red')
    # ax[1].axhline(y=2625, color='red')
    ax[1].set_ylim((zwell_seismic[well_end_data],zwell_seismic[well_start_data]))
    plt.colorbar(ax1, ax=ax[1], shrink=0.7)

    # Plot the synthetic gather data
    max_abs_val = np.max(np.abs(synthetic_gather))
    ax2=ax[2].imshow(synthetic_gather, extent=(thetamin, thetamax, z_seismic_prestack_fence[window_max], z_seismic_prestack_fence[window_min]), cmap='gray',
                     vmin=-max_abs_val, vmax=max_abs_val)
    ax[2].set_title('Synthetic')
    ax[2].axis('tight')
    # ax[2].axhline(y=2950, color='red')
    # ax[2].axhline(y=2625, color='red')
    ax[2].set_ylim((zwell_seismic[well_end_data],zwell_seismic[well_start_data]))
    plt.colorbar(ax2, ax=ax[2], shrink=0.7)

    # Plot the VP log data
    ax3=ax[3].plot(vp, zwell_seismic[well_start_data:well_end_data], 'k', lw=3, label='VP log')
    ax[3].plot(vp_back, zwell_seismic[well_start_data:well_end_data], '--b', lw=3, label='bg VP log')
    ax[3].set_title('VP')
    ax[3].axis('tight')
    # ax[3].axhline(y=2950, color='red')
    # ax[3].axhline(y=2625, color='red')
    ax[3].legend()
    ax[3].invert_yaxis()
    ax[3].set_ylim((zwell_seismic[well_end_data],zwell_seismic[well_start_data]))

    # Adjust the layout of the figure
    fig.tight_layout()

    # Show the plot
    return fig




#For Synthetic Data
def plot_petrophysical_2D_sections(phi_2D, vsh_2D, sw_2D, x_axis, depth, fontsize=16):
    """
    This function plots the petrophysical 2D sections of a synthetic dataset.

    Parameters:
    phi_2D (np.array): 2D array representing porosity.
    vsh_2D (np.array): 2D array representing shale volume.
    sw_2D (np.array): 2D array representing water saturation.
    x_axis (np.array): 1D array representing the x-axis values.
    depth (np.array): 1D array representing the depth values.
    fontsize (int): Font size for the plot.

    Returns:
    fig (Figure): Matplotlib Figure object.
    """
    plt.rcParams.update({'font.size': fontsize})
    
    # Create subplots
    fig, ax = plt.subplots(1,3, figsize=(22, 6))

    # Plot porosity
    vmin_phi, vmax_phi = np.percentile(phi_2D, [1,99])
    im0 = ax[0].imshow(phi_2D, cmap='jet', vmin=vmin_phi, vmax=vmax_phi, extent=(x_axis[0], x_axis[-1], depth[-1], depth[0]))
    ax[0].set_title('$\phi$')
    ax[0].set_ylabel('Depth (m)')
    ax[0].set_xlabel('x (m)')
    fig.colorbar(im0, ax=ax[0], shrink=0.7)

    # Plot shale volume
    vmin_vsh, vmax_vsh = np.percentile(vsh_2D, [1,99])
    im1 = ax[1].imshow(vsh_2D, cmap='summer_r', vmin=vmin_vsh, vmax=vmax_vsh, extent=(x_axis[0], x_axis[-1], depth[-1], depth[0]))
    ax[1].set_title('$\mathrm{V_{sh}}$')
    ax[1].set_xlabel('x (m)')
    fig.colorbar(im1, ax=ax[1], shrink=0.7)

    # Plot water saturation
    vmin_sw, vmax_sw = np.percentile(sw_2D, [1,99])
    im2 = ax[2].imshow(sw_2D, cmap='winter_r', vmin=vmin_sw, vmax=vmax_sw, extent=(x_axis[0], x_axis[-1], depth[-1], depth[0]))
    ax[2].set_title('$\mathrm{S_{w}}$')
    ax[2].set_xlabel('x (m)')
    fig.colorbar(im2, ax=ax[2], shrink=0.7)

    # Adjust layout
    fig.tight_layout()

    # Return the figure
    return fig

def plot_elastic_2D_sections(vp_2D, vs_2D, rho_2D, x_axis, depth, fontsize=16):
    """
    This function plots the petrophysical 2D sections of a synthetic dataset.

    Parameters:
    vp_2D (np.array): 2D array representing Vp.
    vs_2D (np.array): 2D array representing Vs.
    rho_2D (np.array): 2D array representing density.
    x_axis (np.array): 1D array representing the x-axis values.
    depth (np.array): 1D array representing the depth values.
    fontsize (int): Font size for the plot.

    Returns:
    fig (Figure): Matplotlib Figure object.
    """
    plt.rcParams.update({'font.size': fontsize})
    
    # Create subplots
    fig, ax = plt.subplots(1,3, figsize=(22, 6))

    # Plot porosity
    vmin_vp, vmax_vp = np.percentile(vp_2D, [1,99])
    im0 = ax[0].imshow(vp_2D, cmap='terrain', vmin=vmin_vp, vmax=vmax_vp, extent=(x_axis[0], x_axis[-1], depth[-1], depth[0]))
    ax[0].set_title('Vp')
    ax[0].set_ylabel('Depth (m)')
    ax[0].set_xlabel('x (m)')
    fig.colorbar(im0, ax=ax[0], shrink=0.7)

    # Plot shale volume
    vmin_vs, vmax_vs = np.percentile(vs_2D, [1,99])
    im1 = ax[1].imshow(vs_2D, cmap='terrain', vmin=vmin_vs, vmax=vmax_vs, extent=(x_axis[0], x_axis[-1], depth[-1], depth[0]))
    ax[1].set_title('Vs')
    ax[1].set_xlabel('x (m)')
    fig.colorbar(im1, ax=ax[1], shrink=0.7)

    # Plot water saturation
    vmin_rho, vmax_rho = np.percentile(rho_2D, [1,99])
    im2 = ax[2].imshow(rho_2D, cmap='terrain', vmin=vmin_rho, vmax=vmax_rho, extent=(x_axis[0], x_axis[-1], depth[-1], depth[0]))
    ax[2].set_title('$\\rho$')
    ax[2].set_xlabel('x (m)')
    fig.colorbar(im2, ax=ax[2], shrink=0.7)

    # Adjust layout
    fig.tight_layout()

    # Return the figure
    return fig




def plot_inversion_results_2D(wellname,  depth, x_axis,
                           phi_inv_dense_reg, vsh_inv_dense_reg, sw_inv_dense_reg, b_optAVO,
                           phi_fence_bg, vsh_fence_bg, sw_fence_bg, fontsize=12, title_on=True):
    
    """
    Function that plots the inversion results of the petrophysical properties of the fence, 
    the prestack data, the B optimal coefficient matrix, and the respective background models
    used in the inversion.

    Parameters:
        wellname (str): Name of the well.
        depth (1darray): Depth/time-axis.
        x_axis (1darray): x-axis.
        phi_inv_dense_reg (2darray): Inverted porosity values (depth/time-axis, x-axis).
        vsh_inv_dense_reg (2darray): Inverted volume of shale values (depth/time-axis, x-axis).
        sw_inv_dense_reg (2darray): Inverted water saturation values (depth/time-axis, x-axis).
        b_optAVO (3darray): Matrix of petrophysical coefficients B "new data term" (property, depth/time-axis, x-axis).
        phi_fence_bg (2darray): Background model for porosity (x-axis, depth/time-axis).
        vsh_fence_bg (2darray): Background model for volume of shale (x-axis, depth/time-axis).
        sw_fence_bg (2darray): Background model for water saturation (x-axis, depth/time-axis).
        fontsize (int, optional): Font size for the plot. Default is 12.
        

    Returns:
        fig (Figure): The matplotlib figure object containing the plot.
    """
    
    plt.rcParams.update({'font.size': fontsize})
    
    fig = plt.figure(figsize=(22,15))
    gs = gridspec.GridSpec(5, 3, height_ratios=[1,1,1,0.1,0.1])
    
    
    # Subplots 1-2 share a colorbar
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    
    # vmin_b, vmax_b = -3,3
    vmin_b0, vmax_b0 = np.percentile(b_optAVO[0], [1,99])
    im0 = ax0.imshow(b_optAVO[0],cmap='gray', extent=(x_axis[0], x_axis[-1], depth[-1], depth[0]), vmin=vmin_b0, vmax=vmax_b0)
    ax0.set_title('$\mathrm{B_{\phi}}$')
    ax0.axis('tight')
    # ax0.set_xlabel('# Traces')
    ax0.set_ylabel('a) \n Depth', fontsize=fontsize+3)
    
    vmin_b1, vmax_b1 = np.percentile(b_optAVO[1], [1,99])
    im1 = ax1.imshow(b_optAVO[1],cmap='gray', extent=(x_axis[0], x_axis[-1], depth[-1], depth[0]), vmin=vmin_b1, vmax=vmax_b1)
    ax1.set_title('$\mathrm{B_{V_{sh}}}$')
    ax1.axis('tight')
    # ax1.set_xlabel('# Traces')
    # ax1.set_ylabel('Depth (m)')
    
    vmin_b2, vmax_b2 = np.percentile(b_optAVO[2], [1,99])
    im2 = ax2.imshow(b_optAVO[2],cmap='gray', extent=(x_axis[0], x_axis[-1], depth[-1], depth[0]), vmin=vmin_b2, vmax=vmax_b2)
    ax2.set_title('$\mathrm{B_{S_{w}}}$')
    ax2.axis('tight')
    # ax2.set_xlabel('# Traces')
    # ax2.set_ylabel('Depth (m)')

    # Subplots for phi, Vsh, and Sw with their colorbars
    vmin_phi, vmax_phi = np.percentile(phi_inv_dense_reg, [1,99])
    ax3 = fig.add_subplot(gs[1, 0])
    ax6 = fig.add_subplot(gs[2, 0])
    im3 = ax3.imshow(phi_inv_dense_reg, cmap='jet',  extent=(x_axis[0], x_axis[-1], depth[-1], depth[0]), vmin=vmin_phi, vmax=vmax_phi)
    ax3.set_title('$\phi$')
    ax3.axis('tight')
    # ax3.set_xlabel('# Traces')
    ax3.set_ylabel('b) \nDepth (m)', fontsize=fontsize+3)

    im6 = ax6.imshow(phi_fence_bg.T, cmap='jet', extent=(x_axis[0], x_axis[-1], depth[-1], depth[0]), vmin=vmin_phi, vmax=vmax_phi)
    ax6.set_title('${\mathrm{Background}}~\phi$')
    ax6.axis('tight')
    ax6.set_xlabel('# Traces', fontsize=fontsize+3)
    ax6.set_ylabel('c) \nDepth (m)', fontsize=fontsize+3)

    
    vmin_vsh, vmax_vsh = np.percentile(vsh_inv_dense_reg, [1,99]) 
    # vmin_vsh = 0
    ax4 = fig.add_subplot(gs[1, 1])
    ax7 = fig.add_subplot(gs[2, 1])
    im4 = ax4.imshow(vsh_inv_dense_reg, cmap='summer_r', extent=(x_axis[0], x_axis[-1], depth[-1], depth[0]), vmin=vmin_vsh, vmax=vmax_vsh)
    ax4.set_title('$\mathrm{V_{sh}}$')
    ax4.axis('tight')
    # ax2.set_xlabel('# Traces')
    # ax2.set_ylabel('Depth (m)')

    im7 = ax7.imshow(vsh_fence_bg.T, cmap='summer_r', extent=(x_axis[0], x_axis[-1], depth[-1], depth[0]), vmin=vmin_vsh, vmax=vmax_vsh)
    ax7.set_title('$\mathrm{Background~V_{sh}}$')
    ax7.axis('tight')
    ax7.set_xlabel('# Traces', fontsize=fontsize+3)
    # ax7.set_ylabel('Depth (m)')


    vmin_sw, vmax_sw = np.percentile(sw_inv_dense_reg, [1,99])
    vmin_sw = 0
    ax5 = fig.add_subplot(gs[1, 2])
    ax8 = fig.add_subplot(gs[2, 2])
    im5 = ax5.imshow(sw_inv_dense_reg, cmap='winter_r', extent=(x_axis[0], x_axis[-1], depth[-1], depth[0]), vmin=vmin_sw, vmax=vmax_sw)
    ax5.set_title('$\mathrm{S_{w}}$')
    ax5.axis('tight')
    # ax5.set_xlabel('# Traces')
    # ax5.set_ylabel('Depth (m)')

    im8 = ax8.imshow(sw_fence_bg.T, cmap='winter_r', extent=(x_axis[0], x_axis[-1], depth[-1], depth[0]), vmin=vmin_sw, vmax=vmax_sw)
    ax8.set_title('$\mathrm{Background~S_{w}}$')
    ax8.axis('tight')
    ax8.set_xlabel('# Traces', fontsize=fontsize+3)
    # ax8.set_ylabel('Depth (m)')
    
    
    # Get the position and size of the subplots
    box0 = ax0.get_position()
    box1 = ax1.get_position()
    box2 = ax2.get_position()

    # Create the colorbar axes
    cbaxes0 = fig.add_axes([box0.x0, 0.08, box0.width, 0.02]) 
    cbaxes1 = fig.add_axes([box1.x0, 0.08, box1.width, 0.02]) 
    cbaxes2 = fig.add_axes([box2.x0, 0.08, box2.width, 0.02]) 
    cbaxes3 = fig.add_axes([box0.x0, 0.01, box0.width, 0.02]) 
    cbaxes4 = fig.add_axes([box1.x0, 0.01, box1.width, 0.02])
    cbaxes5 = fig.add_axes([box2.x0, 0.01, box2.width, 0.02]) 
    

    # Add the colorbars to the colorbar axes
    cbar0 = fig.colorbar(im3, cax=cbaxes0, orientation='horizontal', cmap='jet')
    cbar1 = fig.colorbar(im4, cax=cbaxes1, orientation='horizontal', cmap='summer_r')
    cbar2 = fig.colorbar(im5, cax=cbaxes2, orientation='horizontal', cmap='winter_r')
    
    cbar3 = fig.colorbar(im0, cax=cbaxes3, orientation='horizontal', cmap= 'gray')
    cbar4 = fig.colorbar(im1, cax=cbaxes4, orientation='horizontal', cmap= 'gray')
    cbar5 = fig.colorbar(im2, cax=cbaxes5, orientation='horizontal', cmap= 'gray')

    # Add a title to each colorbar
    cbar0.set_label('$\phi$', fontsize=fontsize)
    cbar1.set_label('$\mathrm{V_{sh}}$', fontsize=fontsize)
    cbar2.set_label('$\mathrm{S_{w}}$', fontsize=fontsize)
    
    cbar3.set_label('$\mathrm{B_{\phi}}$', fontsize=fontsize)
    cbar4.set_label('$\mathrm{B_{V_{sh}}}$', fontsize=fontsize)
    cbar5.set_label('$\mathrm{B_{S_{w}}}$', fontsize=fontsize)

    # Adjust the main title to prevent overlap with the top colorbar
    if title_on==True:
        plt.subplots_adjust(top=0.85)
        fig.suptitle(f'Inversion results for well: {wellname}', fontsize=fontsize+2)
        
    fig.tight_layout()
    
    return fig


def plot_comparison_training_stacking(depth, x_axis, x_locs, phi_inv_dense_regs, vsh_inv_dense_regs, sw_inv_dense_regs, fontsize=12):
    """
    This function generates a 3x3 grid of plots to compare the training of different numbers of well logs.

    Parameters:
    ----------
    depth : array-like
        An array representing the depth of the well. Should be in the format of [start, end]
    x_axis : array-like
        An array representing the x-axis of the plot. Should be in the format of [start, end]
    x_locs : list of int
        A list of the locations on the x-axis where well logs are known
    phi_inv_dense_regs : list of 2D array-like
        A list of 2D arrays, where each array contains porosity data for a well log
    vsh_inv_dense_regs : list of 2D array-like
        A list of 2D arrays, where each array contains shale volume data for a well log
    sw_inv_dense_regs : list of 2D array-like
        A list of 2D arrays, where each array contains water saturation data for a well log
    fontsize : int
        The size of the font for the plot labels

    Returns:
    -------
    fig:
        This function does not return an object. It produces a 3x3 grid of plots displaying the porosity,
        shale volume, and water saturation data for different numbers of known well logs.
    """
    
    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(3, 3, sharey=True, figsize=(10,6), constrained_layout=True)

    for i in range(3):
        axs[i,0].imshow((phi_inv_dense_regs[i]), cmap='jet', vmin=0, vmax=0.40, extent=[x_axis[0], x_axis[-1], depth[-1], depth[0]])
        axs[i,1].imshow((vsh_inv_dense_regs[i]), cmap='summer_r', vmin=0, vmax=1, extent=[x_axis[0], x_axis[-1], depth[-1], depth[0]])
        axs[i,2].imshow((sw_inv_dense_regs[i]), cmap='winter_r', vmin=0, vmax=1, extent=[x_axis[0], x_axis[-1], depth[-1], depth[0]])

        for loc in x_locs[:i+1]:
            axs[i,0].axvline(x=x_axis[loc], color='white', lw=1)
            axs[i,1].axvline(x=x_axis[loc], color='white', lw=1)
            axs[i,2].axvline(x=x_axis[loc], color='white', lw=1)

        axs[i,0].axis('tight')
        axs[i,1].axis('tight')
        axs[i,2].axis('tight')

        if i == 2:
            axs[i,0].set_xlabel(r'x')
            axs[i,1].set_xlabel(r'x')
            axs[i,2].set_xlabel(r'x')

        axs[i,0].set_ylabel(f'Known Profiles ={i+1} \nDepth (m)')
        
        fig.colorbar(axs[i,0].get_images()[0], ax=axs[i,0], shrink=0.9)
        fig.colorbar(axs[i,1].get_images()[0], ax=axs[i,1], shrink=0.9)
        fig.colorbar(axs[i,2].get_images()[0], ax=axs[i,2], shrink=0.9)

    return fig





def plot_comparison_error_stacking(depth, x_axis, x_locs, phi_inv_dense_regs, vsh_inv_dense_regs, sw_inv_dense_regs, fontsize):
    """
    This function generates a 3x3 grid of plots to compare the training of different numbers of well logs. 
    A common colorbar is used for all plots.

    Parameters:
    ----------
    depth : array-like
        An array representing the depth of the well. Should be in the format of [start, end]
    x_axis : array-like
        An array representing the x-axis of the plot. Should be in the format of [start, end]
    x_locs : list of int
        A list of the locations on the x-axis where well logs are known
    phi_inv_dense_regs : list of 2D array-like
        A list of 2D arrays, where each array contains the difference of the true porosity and the inverted.
    vsh_inv_dense_regs : list of 2D array-like
        A list of 2D arrays, where each array contains the difference of the true shale content  and the inverted.
    sw_inv_dense_regs : list of 2D array-like
        A list of 2D arrays, where each array contains the difference of the true water saturation  and the inverted.
    fontsize : int
        The size of the font for the plot labels

    Returns:
    -------
    None
        This function does not return a value. It produces a 3x3 grid of plots displaying the porosity,
        shale volume, and water saturation data for different numbers of known well logs.
    """
    
    plt.rcParams.update({'font.size': fontsize})
    
    fig, axs = plt.subplots(3, 3, figsize=(12,8), sharey=True)

    images = []
    for i in range(3):
        img0 = axs[i,0].imshow(phi_inv_dense_regs[i], cmap='bwr', extent=[x_axis[0], x_axis[-1], depth[-1], depth[0]], vmin=-1, vmax=1)
        img1 = axs[i,1].imshow(vsh_inv_dense_regs[i], cmap='bwr', extent=[x_axis[0], x_axis[-1], depth[-1], depth[0]], vmin=-1, vmax=1)
        img2 = axs[i,2].imshow(sw_inv_dense_regs[i], cmap='bwr', extent=[x_axis[0], x_axis[-1], depth[-1], depth[0]], vmin=-1, vmax=1)

        # # Adjust the colorbar range
        # img0.set_clim(-1, 1)
        # img1.set_clim(-1, 1)
        # img2.set_clim(-1, 1)
        
        images.extend([img0, img1, img2])
        
        for loc in x_locs[:i+1]:
            axs[i,0].axvline(x=x_axis[loc], color='black', lw=1, linestyle='--')
            axs[i,1].axvline(x=x_axis[loc], color='black', lw=1, linestyle='--')
            axs[i,2].axvline(x=x_axis[loc], color='black', lw=1, linestyle='--')

        axs[i,0].axis('tight')
        axs[i,1].axis('tight')
        axs[i,2].axis('tight')

        if i == 2:
            axs[i,0].set_xlabel(r'x')
            axs[i,1].set_xlabel(r'x')
            axs[i,2].set_xlabel(r'x')

        axs[i,0].set_ylabel(f'Known Profiles ={i+1} \nDepth (m)')

    fig.subplots_adjust(right=0.85)
    # Add a colorbar to the right side of the entire figure
    cbar = fig.colorbar(images[0], ax=axs.ravel().tolist(), pad=0.01, shrink=0.3)
    return fig



def plot_compare_b_reflectivities(b_true, b_optAVO, fontsize=10):
    """
    This function plots the comparison between the true and optimized AVO reflectivities.

    Parameters:
    -----------
    b_true : numpy array
        True reflectivities. Each column corresponds to a different attribute.
    b_optAVO : numpy array
        Optimized reflectivities computed by the AVO algorithm.
        Each column corresponds to a different attribute.
    fontsize : int
        Fontsize of the labels in the plot.

    Returns:
    --------
    fig : matplotlib figure
    """
    plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(1, 3, figsize=(5,7), sharey=True)

    attributes = ['$b_{\phi}$', '$b_{\mathrm{V_{sh}}}$', '$b_{\mathrm{S_{w}}}$']
    colors = ['r', 'g', 'b']

    for i in range(3):
        ax[i].plot(b_true[:,i], np.arange(len(b_true[:,i])), label='b true', color='k', linewidth=1.5)
        ax[i].plot(b_optAVO[:,i], np.arange(len(b_optAVO[:,i])), label='b Seis2Rock', color=colors[i], linewidth=2, alpha=0.8, linestyle='--')
        ax[i].set_title(attributes[i])
        ax[i].invert_yaxis()
        ax[i].legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    
    return fig


def plot_well_results_2Dsynthethic(depth, well_logs, inv_dense_reg, backgrounds, x_loc=100, fontsize=12):
    """
    This function plots the petrophysical properties along a well, given its x location in a 2D array.

    Parameters:
    -----------
    depth : numpy array
        Depth values for the well log.
    well_logs : list of numpy arrays
        List of well log measurements. Order should be [phi, vsh, sw].
    inv_dense_reg : list of 2D numpy arrays
        List of inverted density regularized measurements. Order should be [phi_inv_dense_reg, vsh_inv_dense_reg, sw_inv_dense_reg].
    backgrounds : list of numpy arrays
        List of background measurements. Order should be [phi_back, vsh_back, sw_back].
    x_loc : int, optional
        X location of the well in the 2D array. Defaults to 100.
    fontsize : int, optional
        Font size for the plot. Defaults to 12.

    Returns:
    --------
    fig : matplotlib figure
        Figure with the well log, inversion, and background plots.
    """
    plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(1, 3, figsize=(7,8), sharey=True)

    attributes = ['$\phi$', '$\mathrm{V_{sh}}$', '$\mathrm{S_{w}}$']
    colors = ['r', 'g', 'b']

    for i in range(3):
        ax[i].plot(well_logs[i], depth, label='Well log', color='k', linewidth=1.5, alpha=0.7)
        ax[i].plot(inv_dense_reg[i][:,x_loc], depth, label='Inversion', color=colors[i], linewidth=2)
        ax[i].plot(backgrounds[i], depth, label='Background', color='k', linestyle='--',linewidth=0.8)
        ax[i].set_title(attributes[i])
        ax[i].invert_yaxis()
        ax[i].legend(loc="lower right", fontsize=9)

    ax[0].set_ylabel('Depth (m)')
    fig.tight_layout()
    
    return fig



def plot_Volve_data_as_2d(data, title, line_index=0, property_index=0, cmap='gray', fontsize=11):
    """
    This function plots a 3D or 4D data as a series of 2D plots along the different dimensions.

    Args:
        data (3d or 4darray): 3D or 4D array to plot.
        title (str): Title for the plot.
        line_index (int): Index along the last dimension to plot.
        property_index (int): Index along the first dimension for 4D data.
        cmap (str): Colormap to use for the plots. Default is 'gray'.
        fontsize (int): Font size for the plots. Default is 11.

    Returns:
        None. The function directly plots the data.
    """
    plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(1,3, figsize=(10,4))
    
    # Check if the data is 3D or 4D
    if data.ndim == 3:
        # First image
        im0 = ax[0].imshow(data[0,:,:], cmap=cmap)
        ax[0].set_title(f'{title} depth direction')
        ax[0].axis('tight')
        fig.colorbar(im0, ax=ax[0])  # Colorbar for the first image
        # Second image
        im1 = ax[1].imshow(data[:,line_index,:], cmap=cmap)
        ax[1].set_title(f'{title} xline direction')
        ax[1].axis('tight')
        fig.colorbar(im1, ax=ax[1])  # Colorbar for the second image
        # Third image
        im2 = ax[2].imshow(data[:,:,line_index], cmap=cmap)
        ax[2].set_title(f'{title} iline direction')
        ax[2].axis('tight')
        fig.colorbar(im2, ax=ax[2])  # Colorbar for the third image
    elif data.ndim == 4:
        # First image
        im0 = ax[0].imshow(data[property_index,line_index,:,:], cmap=cmap)
        
        ax[0].set_title(f'{title} depth direction')
        ax[0].axis('tight')
        fig.colorbar(im0, ax=ax[0])  # Colorbar for the first image
        # Second image
        im1 = ax[1].imshow(data[property_index,:,line_index,:], cmap=cmap)
        ax[1].set_title(f'{title} xline direction')
        ax[1].axis('tight')
        fig.colorbar(im1, ax=ax[1])  # Colorbar for the second image
        # Third image
        im2 = ax[2].imshow(data[property_index,:,:,line_index], cmap=cmap)
        ax[2].set_title(f'{title} iline direction')
        ax[2].axis('tight')
        fig.colorbar(im2, ax=ax[2])  # Colorbar for the third image
    else:
        raise ValueError("Data should be 3D or 4D.")
        
    fig.tight_layout()
    return fig




def plot_well_results_from_fence_comparison3D(wellname, zwell_seismic, seismicd_prestack_fence, zwell,
                                 z_seismic_prestack_fence,
                                 phi, vsh, sw,
                                 well_start_data, well_end_data, 
                                 phi_inv_dense_reg, vsh_inv_dense_reg, sw_inv_dense_reg,
                                 phi_fence_3d, vsh_fence_3d, sw_fence_3d,
                                 phi_fence_bg, vsh_fence_bg, sw_fence_bg,
                                 shift=30, fontsize=12, title_on=True):
    """
    Function that plots the results along the well by extracting the data from the fence 
    for the petrophysical properties: porosity, volume of shale, and water saturation.

    Parameters:
        wellname (str): Name of the well.
        zwell_seismic (1darray): Depth values for the well in the seismic data sampling.
        seismicd_prestack_fence (3darray): Seismic data along the fence (x-axis, angles, depth/time-axis).
        zwell (1darray): Depth values for the well at the seismic sampling.
        z_seismic_prestack_fence (1darray): Depth values for the seismic data at the seismic sampling.
        phi (1darray): Porosity values from the well log.
        vsh (1darray): Volume of shale values from the well log.
        sw (1darray): Water saturation values from the well log.
        well_start_data (int): Start index of the well data.
        well_end_data (int): End index of the well data.
        phi_inv_dense_reg (2darray): Inverted porosity values from the fence (depth/time-axis, x-axis).
        vsh_inv_dense_reg (2darray): Inverted volume of shale values from the fence (depth/time-axis, x-axis).
        sw_inv_dense_reg (2darray): Inverted water saturation values from the fence (depth/time-axis, x-axis).
        phi_fence_3d (2darray): Background model for porosity from the fence (x-axis, depth/time-axis).
        vsh_fence_3d (2darray): Background model for volume of shale from the fence (x-axis, depth/time-axis)..
        sw_fence_3d (2darray): Background model for water saturation from the fence (x-axis, depth/time-axis)..
        shift (float): Shift parameter for calibration. Default is 30 and is the calibration in index.
        fontsize (int): Font size for the plot. Default is 12.
        title_on (bool): If True, the plot will have a title. Default is True.

    Returns:
        fig (Figure): The matplotlib figure object containing the plot.
    """
    
    plt.rcParams.update({'font.size': fontsize})
    
    #Plotting well
    xfence = np.arange(seismicd_prestack_fence.shape[0])
    xfence_int = np.arange(0, phi_inv_dense_reg.shape[1], 0.2)
    xwell_int = np.arange(0, len(zwell), 0.2)

    zwell_int = np.interp(xwell_int, np.arange(len(zwell))[~np.isnan(zwell)],
                        zwell[~np.isnan(zwell)], left=np.nan, right=np.nan)

    #porosity
    phiextr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            phi_inv_dense_reg.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)
    
    phi_3d_extr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            phi_fence_3d, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)

    phi_bg_extr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            phi_fence_bg, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)


    #shale content
    vshextr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            vsh_inv_dense_reg.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)
    
    vsh_3d_extr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            vsh_fence_3d, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)

    vsh_bg_extr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            vsh_fence_bg, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)

    #water saturation
    swextr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            sw_inv_dense_reg.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)
    
    sw_3d_extr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            sw_fence_3d, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)

    sw_bg_extr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            sw_fence_bg, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)
    
    fig, ax = plt.subplots(1, 3, figsize=(7,8), sharey=True)
    

    ax[0].plot(phi, zwell_seismic[well_start_data:well_end_data]+shift, label='Well log', color='k', linewidth=1.5, alpha=0.7)
    ax[0].plot(phiextr1_t_int, zwell_int, label='Inversion 2D',  color='r', linewidth=2)
    ax[0].plot(phi_3d_extr1_t_int, zwell_int, label='Inversion 3D', color='r', linestyle='--',linewidth=1.2)
    ax[0].plot(phi_bg_extr1_t_int, zwell_int, label='Background', color='k', linestyle='--',linewidth=0.8)
    ax[0].set_title('$\phi$')
    ax[0].set_ylabel('Depth (m)')
    ax[0].invert_yaxis()
    ax[0].legend(loc="lower right", fontsize=9)
    

    ax[1].plot(vsh, zwell_seismic[well_start_data:well_end_data]+shift, label='Well log', color='k', linewidth=1.5, alpha=0.7)
    ax[1].plot(vshextr1_t_int, zwell_int, label='Inversion 2D',  color='g', linewidth=2)
    ax[1].plot(vsh_3d_extr1_t_int, zwell_int, label='Inversion 3D', color='g', linestyle='--',linewidth=1.2)
    ax[1].plot(vsh_bg_extr1_t_int, zwell_int, label='Background', color='k', linestyle='--',linewidth=0.8)
    ax[1].set_title('$\mathrm{V_{sh}}$')
    ax[1].invert_yaxis()
    ax[1].legend(loc="lower right", fontsize=9)

    ax[2].plot(sw, zwell_seismic[well_start_data:well_end_data]+shift, label='Well log', color='k', linewidth=1.5, alpha=0.7)
    ax[2].plot(swextr1_t_int, zwell_int,  label='Inversion 2D',  color='b', linewidth=2)
    ax[2].plot(sw_3d_extr1_t_int, zwell_int, label='Inversion 3D', color='b', linestyle='--',linewidth=1.2)
    ax[2].plot(sw_bg_extr1_t_int, zwell_int, label='Background', color='k', linestyle='--',linewidth=0.8)
    ax[2].set_title('$\mathrm{S_{w}}$')
    ax[2].invert_yaxis()
    ax[2].legend(loc="lower right", fontsize=fontsize-2)
    
     # Adjust the main title to prevent overlap with the top colorbar
    if title_on==True:
        plt.subplots_adjust(top=0.85)
        fig.suptitle(f'Inversion results for well: {wellname}', fontsize=fontsize+2)
    fig.tight_layout()

    return fig



def cross_sections_volume(volume: np.ndarray, cross_section_idx: tuple, colorbar_range: tuple = (None, None),
                          figsize: tuple = (8, 8), cmap: str = 'bone') -> plt.figure:
    """
    Visualizes cross-sectional views of a 3D volume array.

    This function creates a figure with three subplots: a central plot showing
    the cross-section along the y-axis, a top plot showing the cross-section
    along the x-axis, and a right plot showing the cross-section along the z-axis.
    The fourth subplot (top-right) is unused and hidden.

    Parameters:
    -----------
    volume : np.ndarray
        A 3D numpy array representing the volumetric data.
    
    cross_section_idx : tuple
        A tuple of three integers (t, x, y) representing the indices at which 
        the cross-sections are to be taken. 't' is for the z-axis (depth), 
        'x' and 'y' for the horizontal axes in the volume.
    
    colorbar_range : tuple, optional
        A tuple (vmin, vmax) to set the color scaling for the imshow plots. 
        Defaults to (None, None), which autoscales to the volume data range.
    
    figsize : tuple, optional
        Size of the figure (width, height) in inches. Defaults to (8, 8).
    
    cmap : str, optional
        Colormap used for the imshow plots. Defaults to 'bone'.

    Returns:
    --------
    plt.figure
        A matplotlib figure object containing the generated subplots.
    
    Examples:
    ---------
    >>> fig = cross_sections_volume(my_volume_data, (30, 40, 50))
    >>> fig.show()
    
    This will display cross-sections of 'my_volume_data' at indices 30, 40, 50
    along the z, x, and y axes respectively.
    """
    
    t, x, y = cross_section_idx
    vmin, vmax = colorbar_range
    nt, nx, ny = volume.shape

    # Instantiate plots
    fig, axes = plt.subplots(2, 2, figsize=figsize, 
                             gridspec_kw={'height_ratios': [1, 2], 'width_ratios': [2, 1]})
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    ax = axes[1][0]
    ax_top = axes[0][0]
    ax_right = axes[1][1]

    # Hide the unused subplot (top right)
    axes[0][1].axis('off')

    # Options for imshow
    opts = dict(cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    # Central plot
    ax.imshow(volume[:, :, y], extent=[0, nx, nt, 0], **opts)
    ax.axvline(x, color='gray', ls='-', lw=1)
    ax.axhline(t, color='gray', ls='-', lw=1)

    # Top plot
    ax_top.imshow(volume[t].T, extent=[0, nx, ny, 0], **opts)
    ax_top.axvline(x, color='gray', ls='-', lw=1)
    ax_top.axhline(y, color='gray', ls='-', lw=1)
    ax_top.invert_yaxis()

    # Right plot
    ax_right.imshow(volume[:, x], extent=[0, ny, nt, 0], **opts)
    ax_right.axvline(y, color='gray', ls='-', lw=1)
    ax_right.axhline(t, color='gray', ls='-', lw=1)

    # Hide tick labels for the y-axis of the right plot
    ax_right.set_yticklabels([])

    # Hide tick labels for the x-axis of the top plot
    ax_top.set_xticklabels([])

    # Set labels
    ax.set_xlabel("x samples")
    ax.set_ylabel("Depth samples")
    ax_right.set_xlabel("y samples")
    ax_top.set_ylabel("y samples")

    return fig





def plot_inversion_results_2D(wellname, idx_well, z_seismic_prestack_fence, xlines,
                           phi_inv_dense_reg, vsh_inv_dense_reg, sw_inv_dense_reg, b_optAVO,
                           phi_fence_bg, vsh_fence_bg, sw_fence_bg, fontsize=12, title_on=True):
    
    """
    Function that plots the inversion results of the petrophysical properties of the fence, 
    the prestack data, the B optimal coefficient matrix, and the respective background models
    used in the inversion.

    Parameters:
        wellname (str): Name of the well.
        prestack_fence (ndarray): Preprocessed seismic data for the fence.
        zwell (1darray): Depth values for the well log.
        z_seismic_prestack_fence (3darray): Depth values for the seismic data (x-axis, angles, depth/time-axis).
        xlines (1darray): xlines array
        phi_inv_dense_reg (2darray): Inverted porosity values (depth/time-axis, x-axis).
        vsh_inv_dense_reg (2darray): Inverted volume of shale values (depth/time-axis, x-axis).
        sw_inv_dense_reg (2darray): Inverted water saturation values (depth/time-axis, x-axis).
        b_optAVO (3darray): Matrix of petrophysical coefficients B "new data term" (property, depth/time-axis, x-axis).
        phi_fence_bg (2darray): Background model for porosity (x-axis, depth/time-axis).
        vsh_fence_bg (2darray): Background model for volume of shale (x-axis, depth/time-axis).
        sw_fence_bg (2darray): Background model for water saturation (x-axis, depth/time-axis).
        fontsize (int, optional): Font size for the plot. Default is 12.
        

    Returns:
        fig (Figure): The matplotlib figure object containing the plot.
    """
    
    plt.rcParams.update({'font.size': fontsize})
    
    fig = plt.figure(figsize=(22,15))
    gs = gridspec.GridSpec(5, 3, height_ratios=[1,1,1,0.1,0.1])
    
    
    # Subplots 1-2 share a colorbar
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    
    # vmin_b, vmax_b = -3,3
    vmin_b0, vmax_b0 = np.percentile(b_optAVO[0], [1,99])
    im0 = ax0.imshow(b_optAVO[0],cmap='gray', extent=(xlines[0], xlines[-1], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_b0, vmax=vmax_b0)
    ax0.set_title('$\mathrm{B_{\phi}}$')
    ax0.axis('tight')
    # ax0.set_xlabel('# Traces')
    ax0.set_ylabel('a) \n Depth', fontsize=fontsize+3)
    ax0.axvline(idx_well)
    
    vmin_b1, vmax_b1 = np.percentile(b_optAVO[1], [1,99])
    im1 = ax1.imshow(b_optAVO[1],cmap='gray', extent=(xlines[0], xlines[-1], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_b1, vmax=vmax_b1)
    ax1.set_title('$\mathrm{B_{V_{sh}}}$')
    ax1.axis('tight')
    # ax1.set_xlabel('# Traces')
    # ax1.set_ylabel('Depth (m)')
    ax1.axvline(idx_well)
    
    vmin_b2, vmax_b2 = np.percentile(b_optAVO[2], [1,99])
    im2 = ax2.imshow(b_optAVO[2],cmap='gray', extent=(xlines[0], xlines[-1], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_b2, vmax=vmax_b2)
    ax2.set_title('$\mathrm{B_{S_{w}}}$')
    ax2.axis('tight')
    # ax2.set_xlabel('# Traces')
    # ax2.set_ylabel('Depth (m)')
    ax2.axvline(idx_well)

    # Subplots for phi, Vsh, and Sw with their colorbars
    vmin_phi, vmax_phi = np.percentile(phi_inv_dense_reg, [1,99])
    ax3 = fig.add_subplot(gs[1, 0])
    ax6 = fig.add_subplot(gs[2, 0])
    im3 = ax3.imshow(phi_inv_dense_reg, cmap='jet',  extent=(xlines[0], xlines[-1], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_phi, vmax=vmax_phi)
    ax3.set_title('$\phi$')
    ax3.axis('tight')
    # ax3.set_xlabel('# Traces')
    ax3.set_ylabel('b) \nDepth (m)', fontsize=fontsize+3)

    im6 = ax6.imshow(phi_fence_bg, cmap='jet', extent=(xlines[0], xlines[-1], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_phi, vmax=vmax_phi)
    ax6.set_title('${\mathrm{Background}}~\phi$')
    ax6.axis('tight')
    ax6.set_xlabel('# Traces', fontsize=fontsize+3)
    ax6.set_ylabel('c) \nDepth (m)', fontsize=fontsize+3)

    
    vmin_vsh, vmax_vsh = np.percentile(vsh_inv_dense_reg, [1,99]) 
    # vmin_vsh = 0
    ax4 = fig.add_subplot(gs[1, 1])
    ax7 = fig.add_subplot(gs[2, 1])
    im4 = ax4.imshow(vsh_inv_dense_reg, cmap='summer_r', extent=(xlines[0], xlines[-1], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_vsh, vmax=vmax_vsh)
    ax4.set_title('$\mathrm{V_{sh}}$')
    ax4.axis('tight')
    # ax2.set_xlabel('# Traces')
    # ax2.set_ylabel('Depth (m)')

    im7 = ax7.imshow(vsh_fence_bg, cmap='summer_r', extent=(xlines[0], xlines[-1], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_vsh, vmax=vmax_vsh)
    ax7.set_title('$\mathrm{Background~V_{sh}}$')
    ax7.axis('tight')
    ax7.set_xlabel('# Traces', fontsize=fontsize+3)
    # ax7.set_ylabel('Depth (m)')


    vmin_sw, vmax_sw = np.percentile(sw_inv_dense_reg, [1,99])
    vmin_sw = 0
    ax5 = fig.add_subplot(gs[1, 2])
    ax8 = fig.add_subplot(gs[2, 2])
    im5 = ax5.imshow(sw_inv_dense_reg, cmap='winter_r', extent=(xlines[0], xlines[-1], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_sw, vmax=vmax_sw)
    ax5.set_title('$\mathrm{S_{w}}$')
    ax5.axis('tight')
    # ax5.set_xlabel('# Traces')
    # ax5.set_ylabel('Depth (m)')

    im8 = ax8.imshow(sw_fence_bg, cmap='winter_r', extent=(xlines[0], xlines[-1], z_seismic_prestack_fence[-1], z_seismic_prestack_fence[0]), vmin=vmin_sw, vmax=vmax_sw)
    ax8.set_title('$\mathrm{Background~S_{w}}$')
    ax8.axis('tight')
    ax8.set_xlabel('# Traces', fontsize=fontsize+3)
    # ax8.set_ylabel('Depth (m)')
    
    
    # Get the position and size of the subplots
    box0 = ax0.get_position()
    box1 = ax1.get_position()
    box2 = ax2.get_position()

    # Create the colorbar axes
    cbaxes0 = fig.add_axes([box0.x0, 0.08, box0.width, 0.02]) 
    cbaxes1 = fig.add_axes([box1.x0, 0.08, box1.width, 0.02]) 
    cbaxes2 = fig.add_axes([box2.x0, 0.08, box2.width, 0.02]) 
    cbaxes3 = fig.add_axes([box0.x0, 0.01, box0.width, 0.02]) 
    cbaxes4 = fig.add_axes([box1.x0, 0.01, box1.width, 0.02])
    cbaxes5 = fig.add_axes([box2.x0, 0.01, box2.width, 0.02]) 
    

    # Add the colorbars to the colorbar axes
    cbar0 = fig.colorbar(im3, cax=cbaxes0, orientation='horizontal', cmap='jet')
    cbar1 = fig.colorbar(im4, cax=cbaxes1, orientation='horizontal', cmap='summer_r')
    cbar2 = fig.colorbar(im5, cax=cbaxes2, orientation='horizontal', cmap='winter_r')
    
    cbar3 = fig.colorbar(im0, cax=cbaxes3, orientation='horizontal', cmap= 'gray')
    cbar4 = fig.colorbar(im1, cax=cbaxes4, orientation='horizontal', cmap= 'gray')
    cbar5 = fig.colorbar(im2, cax=cbaxes5, orientation='horizontal', cmap= 'gray')

    # Add a title to each colorbar
    cbar0.set_label('$\phi$', fontsize=fontsize)
    cbar1.set_label('$\mathrm{V_{sh}}$', fontsize=fontsize)
    cbar2.set_label('$\mathrm{S_{w}}$', fontsize=fontsize)
    
    cbar3.set_label('$\mathrm{B_{\phi}}$', fontsize=fontsize)
    cbar4.set_label('$\mathrm{B_{V_{sh}}}$', fontsize=fontsize)
    cbar5.set_label('$\mathrm{B_{S_{w}}}$', fontsize=fontsize)

    # Adjust the main title to prevent overlap with the top colorbar
    if title_on==True:
        plt.subplots_adjust(top=0.85)
        fig.suptitle(f'Inversion results for well: {wellname}', fontsize=fontsize+2)
        
    fig.tight_layout()
    
    return fig



def plot_comparison_Reg_fence(wellname, prestack_fence, zwell, z_seismic_prestack_fence, b_optAVO,
                           phi_inv_dense_reg, vsh_inv_dense_reg, sw_inv_dense_reg, 
                           minv_phi_lap, minv_vsh_lap, minv_sw_lap, 
                           minv_phi_tv, minv_vsh_tv, minv_sw_tv, 
                           minv_phi_pnp, minv_vsh_pnp, minv_sw_pnp,
                           phi_fence_bg, vsh_fence_bg, sw_fence_bg, fontsize=12, title_on=True):
    
    """
    Comparison Function that plots the inversion results of the petrophysical properties of the fence, 
    the prestack data, the B optimal coefficient matrix, and the respective background models
    used in the inversion.

    Parameters:
        wellname (str): Name of the well.
        prestack_fence (ndarray): Preprocessed seismic data for the fence.
        zwell (1darray): Depth values for the well log.
        z_seismic_prestack_fence (3darray): Depth values for the seismic data (x-axis, angles, depth/time-axis).
        phi_inv_dense_reg (2darray): Inverted porosity values (depth/time-axis, x-axis).
        vsh_inv_dense_reg (2darray): Inverted volume of shale values (depth/time-axis, x-axis).
        sw_inv_dense_reg (2darray): Inverted water saturation values (depth/time-axis, x-axis).
        minv_phi_lap (2darray): Inverted porosity values (depth/time-axis, x-axis), Laplacian w/ bounds.
        minv_vsh_lap (2darray): Inverted volume of shale values (depth/time-axis, x-axis), Laplacian w/ bounds.
        minv_sw_lap (2darray): Inverted water saturation values (depth/time-axis, x-axis), Laplacian w/ bounds.
        minv_phi_tv (2darray): Inverted porosity values (depth/time-axis, x-axis), Laplacian w/ bounds.
        minv_vsh_tv (2darray): Inverted volume of shale values (depth/time-axis, x-axis), TV w/ bounds.
        minv_sw_tv (2darray): Inverted water saturation values (depth/time-axis, x-axis), TV w/ bounds.
        minv_phi_pnp (2darray): Inverted porosity values (depth/time-axis, x-axis), Laplacian w/ bounds.
        minv_vsh_pnp (2darray): Inverted volume of shale values (depth/time-axis, x-axis), PnP w/ bounds.
        minv_sw_pno (2darray): Inverted water saturation values (depth/time-axis, x-axis), PnP w/ bounds.
        b_optAVO (3darray): Matrix of petrophysical coefficients B "new data term" (property, depth/time-axis, x-axis).
        phi_fence_bg (2darray): Background model for porosity (x-axis, depth/time-axis).
        vsh_fence_bg (2darray): Background model for volume of shale (x-axis, depth/time-axis).
        sw_fence_bg (2darray): Background model for water saturation (x-axis, depth/time-axis).
        fontsize (int, optional): Font size for the plot. Default is 12.
        

    Returns:
        fig (Figure): The matplotlib figure object containing the plot.
    """
    
    plt.rcParams.update({'font.size': fontsize})
    
    fig = plt.figure(figsize=(35,12))
    gs = gridspec.GridSpec(5, 6, height_ratios=[1,1,1,0.1, 0.1])
    
    
    # Calculate x-values and z-values
    xfence = np.arange(prestack_fence.shape[0])
    xfence_int = np.arange(0, phi_inv_dense_reg.shape[1], 0.2)
    xwell_int = np.arange(0, len(zwell), 0.2)

    zwell_int = np.interp(xwell_int, np.arange(len(zwell))[~np.isnan(zwell)],
                          zwell[~np.isnan(zwell)], left=np.nan, right=np.nan)
    
    #Cropping to avoid ploting the zero data at the right: 
    idx_crop_r = -100
    
    phi_inv_dense_reg = phi_inv_dense_reg[:, :idx_crop_r]
    vsh_inv_dense_reg = vsh_inv_dense_reg[:, :idx_crop_r]
    sw_inv_dense_reg = sw_inv_dense_reg[:, :idx_crop_r]
    minv_phi_lap = minv_phi_lap[:, :idx_crop_r]
    minv_vsh_lap = minv_vsh_lap[:, :idx_crop_r]
    minv_sw_lap = minv_sw_lap[:, :idx_crop_r]
    minv_phi_tv = minv_phi_tv[:, :idx_crop_r]
    minv_vsh_tv = minv_vsh_tv[:, :idx_crop_r]
    minv_sw_tv = minv_sw_tv[:, :idx_crop_r]
    minv_phi_pnp = minv_phi_pnp[:, :idx_crop_r]
    minv_vsh_pnp = minv_vsh_pnp[:, :idx_crop_r]
    minv_sw_pnp = minv_sw_pnp[:, :idx_crop_r]
    phi_fence_bg = phi_fence_bg[:, :idx_crop_r]
    vsh_fence_bg = vsh_fence_bg[:, :idx_crop_r]
    sw_fence_bg = sw_fence_bg[:, :idx_crop_r]

    prestack_fence = prestack_fence[:idx_crop_r]
    
    # Plots for data term

    ax0 = fig.add_subplot(gs[0, 0])
    ax6 = fig.add_subplot(gs[1, 0])
    ax12 = fig.add_subplot(gs[2, 0])
    
    # vmin_b, vmax_b = -3,3
    vmin_b0, vmax_b0 = np.percentile(b_optAVO[0], [1,99])
    im0 = ax0.imshow(b_optAVO[0][:, :idx_crop_r],cmap='gray', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                      z_seismic_prestack_fence[0]), vmin=vmin_b0, vmax=vmax_b0)
    ax0.set_title('a) Petrophysical \n reflectivities')
    ax0.axis('tight')
    # ax0.set_xlabel('# Traces')
    # ax0.set_ylabel('$\mathbf{\phi}$ \n Depth [m]', fontsize=fontsize+3)
    ax0.set_ylabel('Porosity \n Depth [m]', fontsize=fontsize+3)
    ax0.plot(xfence_int, zwell_int,  'r', linewidth=2)
    ax0.set_xticklabels([])
    
    vmin_b1, vmax_b1 = np.percentile(b_optAVO[1], [1,99])
    im6 = ax6.imshow(b_optAVO[1][:, :idx_crop_r],cmap='gray', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                      z_seismic_prestack_fence[0]), vmin=vmin_b1, vmax=vmax_b1)
    # ax6.set_title('$\mathrm{B_{V_{sh}}}$')
    ax6.axis('tight')
    # ax6.set_xlabel('# Traces')
    # ax6.set_ylabel('$\mathbf{V_{sh}}$ \n Depth [m]', fontsize=fontsize+3)
    ax6.set_ylabel('Shale content \n Depth [m]', fontsize=fontsize+3)
    ax6.plot(xfence_int, zwell_int,  'r', linewidth=2)
    ax6.set_xticklabels([])
    
    vmin_b2, vmax_b2 = np.percentile(b_optAVO[2], [1,99])
    im12 = ax12.imshow(b_optAVO[2][:, :idx_crop_r],cmap='gray', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                        z_seismic_prestack_fence[0]), vmin=vmin_b2, vmax=vmax_b2)
    # ax12.set_title('$\mathrm{B_{S_{w}}}$')
    ax12.axis('tight')
    ax12.set_xlabel('# Traces')
    # ax12.set_ylabel('$\mathbf{S_w}$ \n Depth [m]', fontsize=fontsize+3)
    ax12.set_ylabel('Water saturation \n Depth [m]', fontsize=fontsize+3)
    ax12.plot(xfence_int, zwell_int,  'r', linewidth=2)

    #Plots for Laplacian Regularization: 

    ax1 = fig.add_subplot(gs[0, 1])
    ax7 = fig.add_subplot(gs[1, 1])
    ax13 = fig.add_subplot(gs[2, 1])

    vmin_phi, vmax_phi = np.percentile(phi_inv_dense_reg, [1,99])
    im1 = ax1.imshow(phi_inv_dense_reg, cmap='jet',  extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                             z_seismic_prestack_fence[0]), vmin=vmin_phi, vmax=vmax_phi)
    ax1.set_title('b) Laplacian \n w/o bounds')
    ax1.axis('tight')
    # ax1.set_xlabel('# Traces')
    # ax1.set_ylabel('Depth (m)', fontsize=fontsize+3)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])


    # vmin_vsh, vmax_vsh = np.percentile(vsh_inv_dense_reg, [1,99]) 
    vmin_vsh, vmax_vsh = 0, 1
    im7 = ax7.imshow(vsh_inv_dense_reg, cmap='summer_r', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                                 z_seismic_prestack_fence[0]), vmin=vmin_vsh, vmax=vmax_vsh)
    # ax7.set_title('$\mathrm{V_{sh}}$')
    ax7.axis('tight')
    # ax7.set_xlabel('# Traces')
    # ax7.set_ylabel('Depth (m)')
    ax7.set_xticklabels([])
    ax7.set_yticklabels([])


    # vmin_sw, vmax_sw = np.percentile(sw_inv_dense_reg, [1,99])
    vmin_sw, vmax_sw = 0.4, 1
    im13 = ax13.imshow(sw_inv_dense_reg, cmap='winter_r', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                                  z_seismic_prestack_fence[0]), vmin=vmin_sw, vmax=vmax_sw)
    # ax13.set_title('$\mathrm{S_{w}}$')
    ax13.axis('tight')
    ax13.set_xlabel('# Traces')
    # ax13.set_ylabel('Depth (m)')
    ax13.set_yticklabels([])


    #Plots for Laplacian Regularization with bounds:

    ax2 = fig.add_subplot(gs[0, 2])
    ax8 = fig.add_subplot(gs[1, 2])
    ax14 = fig.add_subplot(gs[2, 2])

    im2 = ax2.imshow(minv_phi_lap, cmap='jet',  extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                             z_seismic_prestack_fence[0]), vmin=vmin_phi, vmax=vmax_phi)
    ax2.set_title('c) Laplacian \n w/ bounds')
    ax2.axis('tight')
    # ax2.set_xlabel('# Traces')
    # ax2.set_ylabel('Depth (m)', fontsize=fontsize+3)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])


    im8 = ax8.imshow(minv_vsh_lap, cmap='summer_r', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                                 z_seismic_prestack_fence[0]), vmin=vmin_vsh, vmax=vmax_vsh)
    # ax8.set_title('$\mathrm{V_{sh}}$')
    ax8.axis('tight')
    # ax8.set_xlabel('# Traces')
    # ax8.set_ylabel('Depth (m)')
    ax8.set_xticklabels([])
    ax8.set_yticklabels([])

    im14 = ax14.imshow(minv_sw_lap, cmap='winter_r', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                                  z_seismic_prestack_fence[0]), vmin=vmin_sw, vmax=vmax_sw)
    # ax14.set_title('$\mathrm{S_{w}}$')
    ax14.axis('tight')
    ax14.set_xlabel('# Traces')
    # ax14.set_ylabel('Depth (m)')
    ax14.set_yticklabels([])


   #Plots for TV Regularization with bounds:

    ax3 = fig.add_subplot(gs[0, 3])
    ax9 = fig.add_subplot(gs[1, 3])
    ax15 = fig.add_subplot(gs[2, 3])

    im3 = ax3.imshow(minv_phi_tv, cmap='jet',  extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                             z_seismic_prestack_fence[0]), vmin=vmin_phi, vmax=vmax_phi)
    ax3.set_title('d) TV (PD) \n w/ bounds')
    ax3.axis('tight')
    # ax3.set_xlabel('# Traces')
    # ax3.set_ylabel('Depth (m)', fontsize=fontsize+3)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])


    im9 = ax9.imshow(minv_vsh_tv, cmap='summer_r', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                                 z_seismic_prestack_fence[0]), vmin=vmin_vsh, vmax=vmax_vsh)
    # ax9.set_title('$\mathrm{V_{sh}}$')
    ax9.axis('tight')
    # ax9.set_xlabel('# Traces')
    # ax9.set_ylabel('Depth (m)')
    ax9.set_xticklabels([])
    ax9.set_yticklabels([])

    im15 = ax15.imshow(minv_sw_tv, cmap='winter_r', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                                  z_seismic_prestack_fence[0]), vmin=vmin_sw, vmax=vmax_sw)
    # ax15.set_title('$\mathrm{S_{w}}$')
    ax15.axis('tight')
    ax15.set_xlabel('# Traces')
    # ax15.set_ylabel('Depth (m)')
    ax15.set_yticklabels([])

   #Plots for PnP Regularization with bounds:

    ax4 = fig.add_subplot(gs[0, 4])
    ax10 = fig.add_subplot(gs[1, 4])
    ax16 = fig.add_subplot(gs[2, 4])

    im4 = ax4.imshow(minv_phi_pnp, cmap='jet',  extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                             z_seismic_prestack_fence[0]), vmin=vmin_phi, vmax=vmax_phi)
    ax4.set_title('e) PnP (PD) \n w/ bounds')
    ax4.axis('tight')
    # ax4.set_xlabel('# Traces')
    # ax4.set_ylabel('Depth (m)', fontsize=fontsize+3)
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])

    im10 = ax10.imshow(minv_vsh_pnp, cmap='summer_r', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                                 z_seismic_prestack_fence[0]), vmin=vmin_vsh, vmax=vmax_vsh)
    # ax10.set_title('$\mathrm{V_{sh}}$')
    ax10.axis('tight')
    # ax10.set_xlabel('# Traces')
    # ax10.set_ylabel('Depth (m)')
    ax10.set_xticklabels([])
    ax10.set_yticklabels([])

    im16 = ax16.imshow(minv_sw_pnp, cmap='winter_r', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                                  z_seismic_prestack_fence[0]), vmin=vmin_sw, vmax=vmax_sw)
    # ax16.set_title('$\mathrm{S_{w}}$')
    ax16.axis('tight')
    ax16.set_xlabel('# Traces')
    # ax16.set_ylabel('Depth (m)')
    ax16.set_yticklabels([])


   #Plots for background models

    ax5 = fig.add_subplot(gs[0, 5])
    ax11 = fig.add_subplot(gs[1, 5])
    ax17 = fig.add_subplot(gs[2, 5])

    im5 = ax5.imshow(phi_fence_bg.T, cmap='jet', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                         z_seismic_prestack_fence[0]), vmin=vmin_phi, vmax=vmax_phi)
    ax5.set_title('Background models')
    ax5.axis('tight')
    # ax5.set_xlabel('# Traces', fontsize=fontsize+3)
    # ax5.set_ylabel('c) \nDepth (m)', fontsize=fontsize+3)
    ax5.set_xticklabels([])
    ax5.set_yticklabels([])



    im11 = ax11.imshow(vsh_fence_bg.T, cmap='summer_r', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                              z_seismic_prestack_fence[0]), vmin=vmin_vsh, vmax=vmax_vsh)
    # ax11.set_title('$\mathrm{Background~V_{sh}}$')
    ax11.axis('tight')
    # ax11.set_xlabel('# Traces', fontsize=fontsize+3)
    # ax11.set_ylabel('Depth (m)')
    ax11.set_xticklabels([])
    ax11.set_yticklabels([])


    im17 = ax17.imshow(sw_fence_bg.T, cmap='winter_r', extent=(0, prestack_fence.shape[0], z_seismic_prestack_fence[-1], 
                                                             z_seismic_prestack_fence[0]), vmin=vmin_sw, vmax=vmax_sw)
    # ax17.set_title('$\mathrm{Background~S_{w}}$')
    ax17.axis('tight')
    ax17.set_xlabel('# Traces', fontsize=fontsize+3)
    # ax17.set_ylabel('Depth (m)')
    ax17.set_yticklabels([])
    
    #Colorbar
    box0 = ax12.get_position()
    box1 = ax13.get_position()
    box2 = ax14.get_position()
    box3 = ax15.get_position()
    box4 = ax16.get_position()
    box5 = ax17.get_position()

    # Create the colorbar axes
    cbaxes0 = fig.add_axes([box0.x0, 0.08, box0.width, 0.02]) 
    cbaxes1 = fig.add_axes([box1.x0, 0.08, box1.width, 0.02]) 
    cbaxes2 = fig.add_axes([box2.x0, 0.08, box2.width, 0.02]) 
    
    cbaxes3 = fig.add_axes([box3.x0, 0.08, box3.width, 0.02]) 
    cbaxes4 = fig.add_axes([box4.x0, 0.08, box4.width, 0.02])
    cbaxes5 = fig.add_axes([box5.x0, 0.08, box2.width, 0.02]) 


    # Add the colorbars to the colorbar axes
    cbar0 = fig.colorbar(im1, cax=cbaxes0, orientation='horizontal', cmap='jet')
    cbar1 = fig.colorbar(im7, cax=cbaxes1, orientation='horizontal', cmap='summer_r')
    cbar2 = fig.colorbar(im13, cax=cbaxes2, orientation='horizontal', cmap='winter_r')
    
    cbar3 = fig.colorbar(im0, cax=cbaxes3, orientation='horizontal', cmap= 'gray')
    cbar4 = fig.colorbar(im6, cax=cbaxes4, orientation='horizontal', cmap= 'gray')
    cbar5 = fig.colorbar(im12, cax=cbaxes5, orientation='horizontal', cmap= 'gray')

    # Add a title to each colorbar
    cbar0.set_label('$\phi$', fontsize=fontsize+4)
    cbar1.set_label('$\mathrm{V_{sh}}$', fontsize=fontsize+4)
    cbar2.set_label('$\mathrm{S_{w}}$', fontsize=fontsize+4)
    
    cbar3.set_label('$\mathrm{B_{\phi}}$', fontsize=fontsize+4)
    cbar4.set_label('$\mathrm{B_{V_{sh}}}$', fontsize=fontsize+4)
    cbar5.set_label('$\mathrm{B_{S_{w}}}$', fontsize=fontsize+4)

    # Adjust the main title to prevent overlap with the top colorbar
    if title_on==True:
        plt.subplots_adjust(top=0.85)
        fig.suptitle(f'Inversion results for well: {wellname}', fontsize=fontsize+2)
        
    # fig.tight_layout()
    
    return fig


def plot_well_results_from_fence_Regularization(wellname, zwell_seismic, seismicd_prestack_fence, zwell,
                                 z_seismic_prestack_fence,
                                 phi, vsh, sw,
                                 well_start_data, well_end_data, 
                                 phi_inv_dense_reg, vsh_inv_dense_reg, sw_inv_dense_reg,
                                 minv_phi_lap, minv_vsh_lap, minv_sw_lap, 
                                 minv_phi_tv, minv_vsh_tv, minv_sw_tv, 
                                 minv_phi_pnp, minv_vsh_pnp, minv_sw_pnp,
                                 phi_fence_bg, vsh_fence_bg, sw_fence_bg,
                                 shift=30, fontsize=12, title_on=True):
    """
    Function that plots the results along the well by extracting the data from the fence 
    for the petrophysical properties: porosity, volume of shale, and water saturation.

    Parameters:
        wellname (str): Name of the well.
        zwell_seismic (1darray): Depth values for the well in the seismic data sampling.
        seismicd_prestack_fence (3darray): Seismic data along the fence (x-axis, angles, depth/time-axis).
        zwell (1darray): Depth values for the well at the seismic sampling.
        z_seismic_prestack_fence (1darray): Depth values for the seismic data at the seismic sampling.
        phi (1darray): Porosity values from the well log.
        vsh (1darray): Volume of shale values from the well log.
        sw (1darray): Water saturation values from the well log.
        well_start_data (int): Start index of the well data.
        well_end_data (int): End index of the well data.
        phi_inv_dense_reg (2darray): Inverted porosity values from the fence (depth/time-axis, x-axis).
        vsh_inv_dense_reg (2darray): Inverted volume of shale values from the fence (depth/time-axis, x-axis).
        sw_inv_dense_reg (2darray): Inverted water saturation values from the fence (depth/time-axis, x-axis).
        minv_phi_lap (2darray): Inverted porosity values (depth/time-axis, x-axis), Laplacian w/ bounds.
        minv_vsh_lap (2darray): Inverted volume of shale values (depth/time-axis, x-axis), Laplacian w/ bounds.
        minv_sw_lap (2darray): Inverted water saturation values (depth/time-axis, x-axis), Laplacian w/ bounds.
        minv_phi_tv (2darray): Inverted porosity values (depth/time-axis, x-axis), Laplacian w/ bounds.
        minv_vsh_tv (2darray): Inverted volume of shale values (depth/time-axis, x-axis), TV w/ bounds.
        minv_sw_tv (2darray): Inverted water saturation values (depth/time-axis, x-axis), TV w/ bounds.
        minv_phi_pnp (2darray): Inverted porosity values (depth/time-axis, x-axis), Laplacian w/ bounds.
        minv_vsh_pnp (2darray): Inverted volume of shale values (depth/time-axis, x-axis), PnP w/ bounds.
        minv_sw_pno (2darray): Inverted water saturation values (depth/time-axis, x-axis), PnP w/ bounds.
        phi_fence_bg (2darray): Background model for porosity from the fence (x-axis, depth/time-axis).
        vsh_fence_bg (2darray): Background model for volume of shale from the fence (x-axis, depth/time-axis)..
        sw_fence_bg (2darray): Background model for water saturation from the fence (x-axis, depth/time-axis)..
        shift (float): Shift parameter for calibration. Default is 30 and is the calibration in index.
        fontsize (int): Font size for the plot. Default is 12.
        title_on (bool): If True, the plot will have a title. Default is True.

    Returns:
        fig (Figure): The matplotlib figure object containing the plot.
    """
    
    plt.rcParams.update({'font.size': fontsize})
    
    #Plotting well
    xfence = np.arange(seismicd_prestack_fence.shape[0])
    xfence_int = np.arange(0, phi_inv_dense_reg.shape[1], 0.2)
    xwell_int = np.arange(0, len(zwell), 0.2)

    zwell_int = np.interp(xwell_int, np.arange(len(zwell))[~np.isnan(zwell)],
                        zwell[~np.isnan(zwell)], left=np.nan, right=np.nan)

    # Laplacian without bounds
    phiextr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            phi_inv_dense_reg.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)

    phi_bg_extr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            phi_fence_bg, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)

    vshextr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            vsh_inv_dense_reg.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)

    vsh_bg_extr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            vsh_fence_bg, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)

    swextr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            sw_inv_dense_reg.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)

    sw_bg_extr1_t_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            sw_fence_bg, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)

    #Laplacian with bounds
    phi_lap_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            minv_phi_lap.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)  
    
    vsh_lap_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            minv_vsh_lap.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)  

    sw_lap_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            minv_sw_lap.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)
    #TV with bounds 

    phi_tv_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            minv_phi_tv.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)  
    vsh_tv_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            minv_vsh_tv.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)  

    sw_tv_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            minv_sw_tv.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)  

    #PnP with bounds 

    phi_pnp_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            minv_phi_pnp.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)  
    vsh_pnp_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            minv_vsh_pnp.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T)  

    sw_pnp_int = RegularGridInterpolator((xfence, z_seismic_prestack_fence),
                                            minv_sw_pnp.T, 
                                            bounds_error=False,
                                            fill_value=np.nan)(np.vstack((xfence_int, zwell_int)).T) 


    
    fig, ax = plt.subplots(1, 3, figsize=(7,8), sharey=True)
    phi_lap_int
    lw_plot = 1.8
    
    ax[0].plot(phi, zwell_seismic[well_start_data:well_end_data]+shift, label='Well log', color='k', linewidth=lw_plot)
    ax[0].plot(phi_bg_extr1_t_int, zwell_int, label='Background', color='k', linestyle='--',linewidth=lw_plot)
    ax[0].plot(phiextr1_t_int, zwell_int, label='Laplacian',linewidth=lw_plot, color='blue', alpha =0.8)
    ax[0].plot(phi_lap_int, zwell_int, label='Laplacian w/ bounds', linewidth=lw_plot, color='y', alpha =0.8)
    ax[0].plot(phi_tv_int, zwell_int, label='TV w/ bounds', linewidth=lw_plot, color='green', alpha =0.8) 
    ax[0].plot(phi_pnp_int, zwell_int, label='PnP w/ bounds', linewidth=lw_plot, color='red', alpha =0.8) 
    ax[0].set_title('$\phi$')
    ax[0].set_ylabel('Depth [m]')
    ax[0].invert_yaxis()
    # ax[0].legend(loc="lower right", fontsize=fontsize-5)
    ax[0].grid()
    

    ax[1].plot(vsh, zwell_seismic[well_start_data:well_end_data]+shift, label='Well log', color='k', linewidth=lw_plot)
    ax[1].plot(vsh_bg_extr1_t_int, zwell_int, label='Background', color='k', linestyle='--',linewidth=lw_plot)
    ax[1].plot(vshextr1_t_int, zwell_int, label='Laplacian',linewidth=lw_plot, color='blue',alpha =0.8)
    ax[1].plot(vsh_lap_int, zwell_int, label='Laplacian w/ bounds', linewidth=lw_plot, color='y', alpha =0.8)
    ax[1].plot(vsh_tv_int, zwell_int, label='TV w/ bounds', linewidth=lw_plot, color='green', alpha =0.8) 
    ax[1].plot(vsh_pnp_int, zwell_int, label='PnP w/ bounds', linewidth=lw_plot, color='red', alpha =0.8) 
    ax[1].set_title('$\mathrm{V_{sh}}$')
    ax[1].invert_yaxis()
    # ax[1].legend(loc="best", fontsize=fontsize-5)
    ax[1].grid()

    ax[2].plot(sw, zwell_seismic[well_start_data:well_end_data]+shift, label='Well log', color='k', linewidth=lw_plot)
    ax[2].plot(sw_bg_extr1_t_int, zwell_int, label='Background', color='k', linestyle='--',linewidth=lw_plot)
    ax[2].plot(swextr1_t_int, zwell_int,  label='Laplacian', linewidth=lw_plot, color='blue',alpha =0.8)
    ax[2].plot(sw_lap_int, zwell_int, label='Laplacian w/ bounds', linewidth=lw_plot, color='y', alpha =0.8)
    ax[2].plot(sw_tv_int, zwell_int, label='TV w/ bounds', linewidth=lw_plot, color='green', alpha =0.8) 
    ax[2].plot(sw_pnp_int, zwell_int, label='PnP w/ bounds', linewidth=lw_plot, color='red', alpha =0.8) 
    ax[2].set_title('$\mathrm{S_{w}}$')
    ax[2].invert_yaxis()
    # ax[2].legend(loc="best", fontsize=fontsize)
    ax[2].grid()

    #  # Adjust the main title to prevent overlap with the top colorbar
    # if title_on==True:
    #     plt.subplots_adjust(top=0.85)
    #     fig.suptitle(f'Inversion results for well: {wellname}', fontsize=fontsize+2)
    # fig.tight_layout()

    # return fig

    # Create a single legend for all subplots at the bottom of the figure
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
    
    # Adjust layout to prevent overlap with the legend
    plt.subplots_adjust(bottom=0.15, top=0.85)
    
    if title_on:
        fig.suptitle(f'Inversion results for well: {wellname}', fontsize=fontsize+2)
    
    fig.tight_layout()
    
    return fig
    
