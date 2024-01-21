#Seis2Rock functions

import pylops
from pylops.utils.wavelets             import *
from pylops.utils.seismicevents        import *
from pylops.basicoperators             import *
from pylops.signalprocessing           import *
from pylops.waveeqprocessing.mdd       import *
from pylops.avo.avo                    import *
from pylops.avo.poststack              import *
from pylops.avo.prestack               import *


import numpy as np
import scipy as sp
from scipy.signal import filtfilt



def get_depth_window_fence_Volve(well_vp_prestack_sampling, zwell_seismic, z_seismic_prestack_fence):
    """
    This function calculates the depth window where the well log information starts 
    on the trajectory of the well in the Volve dataset. Aso it provides the indexes
    of the well logs where the data starts and ends.

    Parameters:
    well_vp_prestack_sampling (1darray): The well log data of one property i.e vp sampled at the sesimic sampling.
    zwell_seismic (1darray): The depth of the well log in the seismic (seismic sampling)
    z_seismic_prestack_fence (numpy array): The seismic data depth (seismic sampling).

    Returns:
    window_min: The minimum index depth window.
    window_max: The maximum index depth window.
    well_start_data: The index of the well log where the data starts.
    well_end_data: The index of the well log where the data ends.
    
    """
    # Copy the well log data
    vp = np.copy(well_vp_prestack_sampling)
    
    # Remove nans from array with the last non-nan value
    vp[np.isnan(vp)] = np.interp(np.flatnonzero(np.isnan(vp)), np.flatnonzero(~np.isnan(vp)), vp[~np.isnan(vp)])

    # Find the start and end of the well data
    well_start_data = np.where(vp != vp[0])[0][0]
    well_end_data = np.where(vp != vp[-1])[0][-1]

    # Calculate the depth window for the wavelet
    window_min = np.where(zwell_seismic[well_start_data]==z_seismic_prestack_fence)[0][0]
    window_max = np.where(zwell_seismic[well_end_data]==z_seismic_prestack_fence)[0][0]

    return window_min, window_max, well_start_data, well_end_data



def get_wavelet_estimate(nt_wav=41, nfft=512, wav_scaling=18, prestack_data=None, dt=4):
    """
    Function to calculate the statistical wavelet estimate.
    
    Parameters:
    nt_wav : float, optional
        Number of samples of statistical wavelet. Default is 41.0
    nfft : float, optional
        Number of samples of FFT. Default is 512.0
    wav_scaling : float, optional
        Scaling factor for the wavelet. Default is 18.0
    prestack_data : 3darray, optional
        Prestack Seismic data [x- axis, angles, depth/time - axis]. Default is None. 
        This must be provided.
    dt : float, optional
        Sampling interval. Default is 4. 

    
    Returns:
    wav_est : 1darray
        The wavelet estimate.
    t_wav : 1darray
        The wavelet time axis.
    fwest : 1darray
        The FFT frequency.
    wav_est_fft : 1darray
        The FFT of the estimated wavelet.

    """

    # Make sure the necessary parameters are provided
    if prestack_data is None or dt is None:
        raise ValueError("prestack_data, dt  must be provided")

    # Wavelet time axis
    t_wav = np.arange(nt_wav) * (dt/1000)
    t_wav = np.concatenate((np.flipud(-t_wav[1:]), t_wav), axis=0)

    # Estimate wavelet spectrum
    wav_est_fft = np.mean(np.abs(np.fft.fft(prestack_data, 
                                             nfft, axis=-1)), axis=(0, 1))
    fwest = np.fft.fftfreq(nfft, d=dt/1000)

    # Create wavelet in time
    wav_est = np.real(np.fft.ifft(wav_est_fft)[:nt_wav])
    wav_est = np.concatenate((np.flipud(wav_est[1:]), wav_est), axis=0)
    wav_est = sp.signal.hamming(nt_wav*2-1) * wav_est

    # Normalization and scaling
    wav_est = wav_est / wav_est.max()
    wav_est *= wav_scaling

    return wav_est, t_wav ,fwest, wav_est_fft





def create_background_model(a, b, vel):
    """
    Function to create a background model from a linear relationship ax + b,
    where a and b are coefficients and x is the velocity.

    Parameters:
    a : float
        The coefficient a in the linear equation (gradient).
    b : float
        The coefficient b in the linear equation (intercept).
    vel : ndarray
        The velocities for which to calculate the background model.

    Returns:
    model : ndarray
        The background model.
    """
    # Ensure that velocities is a numpy array
    vel = np.array(vel)

    # Calculate the model using the linear relationship
    model = a * vel + b

    return model





def process_well_logs(log_1, log_2, log_3, depth_min=None, depth_max=None, nsmooth=40):
    """
    This function performs sanity checks (removing nan values),
    and computes background models for well logs.

    Parameters:
    log_1, log_2, log_3: well log data as 1D numpy arrays.
    depth_min, depth_max: indices indicating start and end of data window in well log. If None, full log is processed.
    nsmooth (optional): smoothing factor for background model calculation. Default is 40.
    
    Returns:
    log_1, log_2, log_3: cleaned well log data.
    log_1_back, log_2_back, log_3_back: background models of well log data.
    """
    # Check for nan values and interpolate if present
    for array in [log_1, log_2, log_3]:
        array[np.isnan(array)] = np.interp(np.flatnonzero(np.isnan(array)), 
                                           np.flatnonzero(~np.isnan(array)), 
                                           array[~np.isnan(array)])

    # Compute background models for entire well log
    log_1_back = filtfilt(np.ones(nsmooth)/nsmooth, 1, log_1)
    log_2_back = filtfilt(np.ones(nsmooth)/nsmooth, 1, log_2)
    log_3_back = filtfilt(np.ones(nsmooth)/nsmooth, 1, log_3)

    # Window the well log data and corresponding background models
    if depth_min is not None and depth_max is not None:
        log_1 = log_1[depth_min:depth_max]
        log_2 = log_2[depth_min:depth_max]
        log_3 = log_3[depth_min:depth_max]
        log_1_back = log_1_back[depth_min:depth_max]
        log_2_back = log_2_back[depth_min:depth_max]
        log_3_back = log_3_back[depth_min:depth_max]

    return log_1, log_2, log_3, log_1_back, log_2_back, log_3_back




def avo_synthetic_gather(vp, vs, rho, wav_est, nt_wav, thetamin=0, thetamax=25, ntheta=25):
    """
    This function computes the AVO synthetic gather given well log information and using the Zoeppritz equation, 
    which is also convolved with the wavelet.

    Args:
        vp (1darray): P-wave velocity.
        vs (1darray): S-wave velocity.
        rho (1darray): Density.
        wav_est (1darray): Estimated wavelet to convolve with.
        nt_wav (int): Number of samples of statistical wavelet
        thetamin (int): Minimum angle for reflectivity computation. Default is 0.
        thetamax (int): Maximum angle for reflectivity computation. Default is 25.
        ntheta (int): Number of angles. Default is 25.

    Returns:
        r_zoeppritz (2darray) : Array (depth/time axis, angles) containing the synthetic gather computed from the well logs
    """

    Logsize = len(vp)  # Size of the logs
    theta = np.linspace(thetamin, thetamax, ntheta)  # Define the angles to obtain reflectivity

    # Apply Full Zoeppritz equation
    rpp_zoep = np.zeros(shape=(ntheta, Logsize))

    for i in range(Logsize-1):
        rpp_zoep[:,i] += zoeppritz_pp(vp[i], vs[i], rho[i], vp[i+1], vs[i+1], rho[i+1], theta)

    # Replace last column with second-to-last column <---------- Modfying here
    rpp_zoep[:, -1] = rpp_zoep[:, -2]
    
    # Wavelet Operator W
    Cop = Convolve1D(Logsize, h=wav_est, offset=nt_wav)    
     
    # Convolve with wavelet    
    r_zoeppritz = Cop*rpp_zoep.T

    return r_zoeppritz




def avo_synthetic_gather_2D(vp, vs, rho, wav_est, nt_wav, thetamin=0, thetamax=25, ntheta=25):
    """
    This function computes the AVO synthetic gather given 2D  models information and using the Zoeppritz equation, 
    which is also convolved with the wavelet.

    Args:
        vp (2darray): P-wave velocity.
        vs (2darray): S-wave velocity.
        rho (2darray): Density .
        wav_est (1darray): Estimated wavelet to convolve with.
        nt_wav (int): Number of samples of statistical wavelet
        thetamin (int): Minimum angle for reflectivity computation. Default is 0.
        thetamax (int): Maximum angle for reflectivity computation. Default is 25.
        ntheta (int): Number of angles. Default is 25.

    Returns:
        r_zoeppritz (3darray) : Array (x -axis, depth/time axis, angles) containing the synthetic gather computed from the background models.
    """

    n_sections = vp.shape[0]
    Logsize = vp.shape[1]  # Size of the logs
    theta = np.linspace(thetamin, thetamax, ntheta)  # Define the angles to obtain reflectivity

    # Apply Full Zoeppritz equation
    rpp_zoep = np.zeros(shape=(n_sections, ntheta, Logsize))

    for j in range(n_sections):
        for i in range(Logsize-1):
            rpp_zoep[j, :, i] += zoeppritz_pp(vp[j, i], vs[j, i], rho[j, i], vp[j, i+1], vs[j, i+1], rho[j, i+1], theta)

    # Replace last column with second-to-last column <---------- To get the same dims
    rpp_zoep[:, :, -1] = rpp_zoep[:, :, -2]
    
    # Wavelet Operator W
    Cop = Convolve1D(Logsize, h=wav_est, offset=nt_wav)    
    
    # Convolve with wavelet    
    r_zoeppritz = np.zeros(shape=(n_sections, Logsize, ntheta))
    for j in range(n_sections):
        r_zoeppritz[j] = Cop * rpp_zoep[j].T

    return r_zoeppritz





# def Seis2Rock_training(vp, vs, rho, 
#                      wav_est, nt_wav, 
#                      vp_back, vs_back, rho_back, 
#                      p, 
#                      thetamin=0, thetamax=25, ntheta=25):
#     """
#     This function perfoms the Seis2Rock training Routine,it has as a goal obtain the optimal basis functions Fp.
#     Here, AVO synthetic gathers are created given well log information and using the Zoeppritz equation, 
#     which is also convolved with the wavelet. The function also calculates synthetic gathers for the background models. 
#     The function then computes the difference between the synthetic gathers (d-db) and performs Singular Value Decomposition (SVD).
#     Subsets of each decomposed matrix are extracted based on input p (optimal number of singular values for the reconstruction).

#     Args:
#         vp (1darray): P-wave velocity from well logs.
#         vs (1darray): S-wave velocity from well logs.
#         rho (1darray): Density from well logs.
#         wav_est (1darray): Estimated wavelet to convolve with.
#         nt_wav (int): Number of samples of statistical wavelet
#         vp_back (1darray): Background P-wave velocity lfrom well logs..
#         vs_back (1darray): Background S-wave velocity from well logs..
#         rho_back (1darray): Background density from well logs..
#         p (int): optimal number of singular values for the reconstruction
#         thetamin (int): Minimum angle for reflectivity computation. Default is 0.
#         thetamax (int): Maximum angle for reflectivity computation. Default is 25.
#         ntheta (int): Number of angles. Default is 25.

#     Returns:
#         r_zoeppritz, r_zoeppritz_back: Synthetic gathers computed from well logs and background models.
#         F, L, V : Matrices obtained from SVD decomposition.
#         Fp, Lp, Vp : Extracted subsets of each matrix, based on input p.
#     """

#     # Compute AVO synthetic gather for well logs and background models
#     print('Computing AVO synthetic gathers from the well logs...')
#     r_zoeppritz = avo_synthetic_gather(vp, vs, rho, wav_est, nt_wav, thetamin, thetamax, ntheta)
#     r_zoeppritz_back = avo_synthetic_gather(vp_back, vs_back, rho_back, wav_est, nt_wav, thetamin, thetamax, ntheta)

#     # Compute difference between synthetic gathers
#     d = r_zoeppritz.T - r_zoeppritz_back.T

#     # Perform Singular Value Decomposition
#     print('Performing SVD...')
#     F, L, V = sp.linalg.svd(d, full_matrices=False)
#     V = V.T
#     L = np.diag(L)

#     # Change sign to have positive values of the functions at first offset/angle 
#     # this is due to the fact that SVD is not normalized... similar to:
#     # https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/utils/extmath.py#L504)
    
#     sign_vect = np.sign(F[0])
#     F = F * np.outer(np.ones((ntheta,1)), sign_vect)
#     L = L * np.outer(sign_vect.T, np.ones((1, ntheta)))

#     # Extract decomposed matrix
#     print('Extracting Optimal basis functions Fp..')
#     Fp = F[:,0:p]
#     Lp = L[0:p,0:p]
#     Vp = V[:, 0:p]
    
#     print('Done! xD')

#     return  Fp, Lp, Vp, F, L, V, r_zoeppritz, r_zoeppritz_back

def Seis2Rock_training(vp, vs, rho, 
                     wav_est, nt_wav, 
                     vp_back, vs_back, rho_back, 
                     p, 
                     thetamin=0, thetamax=25, ntheta=25):
    """
    This function perfoms the Seis2Rock training Routine,it has as a goal obtain the optimal basis functions Fp.
    Here, AVO synthetic gathers are created given well log information and using the Zoeppritz equation, 
    which is also convolved with the wavelet. The function also calculates synthetic gathers for the background models. 
    The function then computes the difference between the synthetic gathers (d-db) and performs Singular Value Decomposition (SVD).
    Subsets of each decomposed matrix are extracted based on input p (optimal number of singular values for the reconstruction).

    Args:
        vp (1darray): P-wave velocity from well logs.
        vs (1darray): S-wave velocity from well logs.
        rho (1darray): Density from well logs.
        wav_est (1darray): Estimated wavelet to convolve with.
        nt_wav (int): Number of samples of statistical wavelet
        vp_back (1darray): Background P-wave velocity lfrom well logs..
        vs_back (1darray): Background S-wave velocity from well logs..
        rho_back (1darray): Background density from well logs..
        p (int): optimal number of singular values for the reconstruction
        thetamin (int): Minimum angle for reflectivity computation. Default is 0.
        thetamax (int): Maximum angle for reflectivity computation. Default is 25.
        ntheta (int): Number of angles. Default is 25.

    Returns:
        r_zoeppritz, r_zoeppritz_back: Synthetic gathers computed from well logs and background models.
        F, L, V : Matrices obtained from SVD decomposition.
        Fp, Lp, Vp : Extracted subsets of each matrix, based on input p.
    """

    # Compute AVO synthetic gather for well logs and background models
    print('Computing AVO synthetic gathers from the well logs...')
    r_zoeppritz = avo_synthetic_gather(vp, vs, rho, wav_est, nt_wav, thetamin, thetamax, ntheta)
    r_zoeppritz_back = avo_synthetic_gather(vp_back, vs_back, rho_back, wav_est, nt_wav, thetamin, thetamax, ntheta)

    # Compute difference between synthetic gathers
    d = r_zoeppritz.T - r_zoeppritz_back.T

    # Perform Singular Value Decomposition
    print('Performing SVD...')
    F, L, V = sp.linalg.svd(d, full_matrices=False)
    V = V.T
    L = np.diag(L)

    # Change sign to have positive values of the functions at first offset/angle 
    # this is due to the fact that SVD is not normalized... similar to:
    # https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/utils/extmath.py#L504)
    
    sign_vect = np.sign(F[0])
    F = F * np.outer(np.ones((ntheta,1)), sign_vect)
    L = L * np.outer(sign_vect.T, np.ones((1, ntheta)))

    # Extract decomposed matrix
    print('Extracting Optimal basis functions Fp..')
    Fp = F[:,0:p]
    Lp = L[0:p,0:p]
    Vp = V[:, 0:p]
    
    print('Done! xD')
    print(p)

    return  Fp, Lp, Vp, F, L, V, r_zoeppritz, r_zoeppritz_back, d



# def Seis2Rock_inference(vp, vs, rho, 
#                       wav_est, nt_wav, 
#                       dtheta, 
#                       Fp, Lp, Vp, 
#                       phi, vsh, sw, 
#                       phi_back, vsh_back, sw_back, 
#                       thetamin=0, thetamax=25, ntheta=25):
#     """
#     This function computes the AVO synthetic gather given 2D background models information and using the Zoeppritz equation, 
#     which is also convolved with the wavelet. The function then uses the resulting synthetic gathers to compute the new data 
#     term called Petrophysical coefficients B.

#     Args:
#         vp (2darray): P-wave velocity background model (x-axis, depth/time axis).
#         vs (2darray): S-wave velocity background model (x-axis, depth/time axis).
#         rho (2darray): Density background model (x-axis, depth/time axis).
#         wav_est (1darray): Estimated wavelet to convolve with.
#         nt_wav (int): Number of samples of statistical wavelet
#         dtheta (3darray): Model data matrix, the dimensions are (x-axis, angles, depth/time axis).
#         Fp (2darray): Subset of matrix F obtained from SVD decomposition (basis functions ,p). Being p the optimal number of singular values for the reconstruction.
#         Lp (2darray): Subset of matrix L obtained from SVD decomposition (p,p).
#         Vp (2darray): Subset of matrix V obtained from SVD decomposition (vsize,p).
#         phi (1darray): Petrophysical property (porosity) from well logs.
#         vsh (1darray): Petrophysical property (shale volume) well logs.
#         sw (1darray): Petrophysical property (water saturation) well logs.
#         phi_back (1darray): Background porosity from well logs.
#         vsh_back (1darray): Background shale volume from well logs.
#         sw_back (1darray): Background water saturation from well logs.
#         thetamin (int): Minimum angle for reflectivity computation. Default is 0.
#         thetamax (int): Maximum angle for reflectivity computation. Default is 25.
#         ntheta (int): Number of angles. Default is 25.

#     Returns:
#         b_optAVO (3darray): New data term (Petrophysical coefficients B) The output shape 
#                                     (#properties, depth/time axis, x-axis).
#         r_zoeppritz_back (3darray): Synthetic gather computed from the background models.
#         Cp (3darray): Matrix obtained from the process (x-axis, p, depth/time-axis).
#     """

    
#     # Create petrophysical models
#     m_full = np.stack((phi, vsh, sw), axis=0)
#     m_back = np.stack((phi_back, vsh_back, sw_back), axis=0)

#     # Compute AVO synthetic gather for background models
#     print('Computing AVO synthetic gather for background models...')
#     r_zoeppritz_back = avo_synthetic_gather_2D(vp, vs, rho, wav_est, nt_wav, thetamin, thetamax, ntheta)

#     # Compute difference between model data and synthetic gathers
#     d_testing = (dtheta - r_zoeppritz_back.transpose(0, 2, 1))

#     # Calculate Cp matrix
#     print('Calculating matrix of optimal coefficients Cp...')
#     Cp = Fp.T @ d_testing

#     # Calculate Hp matrix
#     print('Creating the new data term (Petrophysical coefficeints B)...')
#     Hp = Vp @ np.diag(1. / np.diag(Lp))

#     Logsize = len(phi)
#     kind = 'centered' # 'forward' is better once available in PyLops

#     # PoststackLinearModelling operator
#     D = pylops.avo.poststack.PoststackLinearModelling(wav_est, nt0=Logsize, spatdims=Logsize, explicit=True)
#     D_A = D.A.copy()

#     # Compute b_full_prof and b_back_prof
#     b_full_prof = D_A @ m_full.T
#     b_back_prof = D_A @ m_back.T

#     # Calculate the new data term (Petrophysical coefficients B)
#     b_optAVO = Cp.transpose(0, 2, 1) @ Hp.T @ (b_full_prof - b_back_prof)
#     b_optAVO = b_optAVO.transpose(2, 1, 0)
#     print('Done xD !')

#     return b_optAVO, r_zoeppritz_back, Cp

def Seis2Rock_inference(vp, vs, rho, 
                      wav_est, nt_wav, 
                      dtheta, 
                      Fp, Lp, Vp, 
                      phi, vsh, sw, 
                      phi_back, vsh_back, sw_back, 
                      d,
                      thetamin=0, thetamax=25, ntheta=25):
    """
    This function computes the AVO synthetic gather given 2D background models information and using the Zoeppritz equation, 
    which is also convolved with the wavelet. The function then uses the resulting synthetic gathers to compute the new data 
    term called Petrophysical coefficients B.

    Args:
        vp (2darray): P-wave velocity background model (x-axis, depth/time axis).
        vs (2darray): S-wave velocity background model (x-axis, depth/time axis).
        rho (2darray): Density background model (x-axis, depth/time axis).
        wav_est (1darray): Estimated wavelet to convolve with.
        nt_wav (int): Number of samples of statistical wavelet
        dtheta (3darray): Model data matrix, the dimensions are (x-axis, angles, depth/time axis).
        Fp (2darray): Subset of matrix F obtained from SVD decomposition (basis functions ,p). Being p the optimal number of singular values for the reconstruction.
        Lp (2darray): Subset of matrix L obtained from SVD decomposition (p,p).
        Vp (2darray): Subset of matrix V obtained from SVD decomposition (vsize,p).
        phi (1darray): Petrophysical property (porosity) from well logs.
        vsh (1darray): Petrophysical property (shale volume) well logs.
        sw (1darray): Petrophysical property (water saturation) well logs.
        phi_back (1darray): Background porosity from well logs.
        vsh_back (1darray): Background shale volume from well logs.
        sw_back (1darray): Background water saturation from well logs.
        d (2darray): AVO gather from the training well log (x-axis, depth/time axis).
        thetamin (int): Minimum angle for reflectivity computation. Default is 0.
        thetamax (int): Maximum angle for reflectivity computation. Default is 25.
        ntheta (int): Number of angles. Default is 25.

    Returns:
        b_optAVO (3darray): New data term (Petrophysical coefficients B) The output shape 
                                    (#properties, depth/time axis, x-axis).
        r_zoeppritz_back (3darray): Synthetic gather computed from the background models.
        Cp (3darray): Matrix obtained from the process (x-axis, p, depth/time-axis).
    """

    
    # Create petrophysical models
    m_full = np.stack((phi, vsh, sw), axis=0)
    m_back = np.stack((phi_back, vsh_back, sw_back), axis=0)

    # Compute AVO synthetic gather for background models
    print('Computing AVO synthetic gather for background models...')
    r_zoeppritz_back = avo_synthetic_gather_2D(vp, vs, rho, wav_est, nt_wav, thetamin, thetamax, ntheta)
    
    # Cp to compare the boptAVO later
    Cp_estimated= Fp.T @ d
    


    # Compute difference between model data and synthetic gathers
    d_testing = (dtheta - r_zoeppritz_back.transpose(0, 2, 1))

    # Calculate Cp matrix
    print('Calculating matrix of optimal coefficients Cp...')
    Cp = Fp.T @ d_testing

    # Calculate Hp matrix
    print('Creating the new data term (Petrophysical coefficeints B)...')
    Hp = Vp @ np.diag(1. / np.diag(Lp))

    Logsize = len(phi)
    # kind = 'centered' # 'forward' is better once available in PyLops
    kind = 'forward'

    # PoststackLinearModelling operator
    D = pylops.avo.poststack.PoststackLinearModelling(wav_est, nt0=Logsize, spatdims=Logsize, explicit=True, kind=kind)
    D_A = D.A.copy()

    # Compute b_full_prof and b_back_prof
    b_full_prof = D_A @ m_full.T
    b_back_prof = D_A @ m_back.T

    # Calculate the new data term (Petrophysical coefficients B)
    b_optAVO = Cp.transpose(0, 2, 1) @ Hp.T @ (b_full_prof - b_back_prof)
    b_optAVO = b_optAVO.transpose(2, 1, 0)
    print('Done xD !')

    return b_optAVO, r_zoeppritz_back, Cp, Hp, Cp_estimated






def create_background_models_synthetic(phi, vsh, sw, nsmooth=15):
    """
    This function creates the background models of the petrophysical properties.

    Parameters:
    phi (np.array): 2D array representing porosity.
    vsh (np.array): 2D array representing shale volume.
    sw (np.array): 2D array representing water saturation.
    nsmooth (int): The smoothing factor.

    Returns:
    phi_back (np.array): 2D array representing the background model of porosity.
    vsh_back (np.array): 2D array representing the background model of shale volume.
    sw_back (np.array): 2D array representing the background model of water saturation.
    """
    # Create smoothing filter
    smooth_filter = np.ones(nsmooth) / float(nsmooth)

    # Create background model for porosity
    phi_back = filtfilt(smooth_filter, 1, phi, axis=0)
    phi_back = filtfilt(smooth_filter, 1, phi_back, axis=1)

    # Create background model for shale volume
    vsh_back = filtfilt(smooth_filter, 1, vsh, axis=0)
    vsh_back = filtfilt(smooth_filter, 1, vsh_back, axis=1)

    # Create background model for water saturation
    sw_back = filtfilt(smooth_filter, 1, sw, axis=0)
    sw_back = filtfilt(smooth_filter, 1, sw_back, axis=1)

    return phi_back, vsh_back, sw_back



def extract_well_logs_from_2D(vp_2D, vs_2D, rho_2D,
                              phi_2D, vsh_2D, sw_2D, 
                              x_loc):
    """
    Extracts three well logs from a 2D array based on the x location.

    Args:
        array_2d (ndarray): Input 2D array with shape (ntheta, Logsize).
        x_location (int): The x location or index of the vertical well.

    Returns:
        tuple: A tuple containing three extracted well logs as 1D arrays.

    """
    vp = vp_2D[:, x_loc] 
    vs = vs_2D[:, x_loc]  
    rho = rho_2D[:, x_loc]  
    phi = phi_2D[:, x_loc] 
    vsh = vsh_2D[:, x_loc] 
    sw = sw_2D[:, x_loc] 

    return vp, vs, rho, phi, vsh, sw


## Trying to solve the issue of Zoeppritz in the last dimension

# def avo_synthetic_gather(vp, vs, rho, wav_est, nt_wav, thetamin=0, thetamax=25, ntheta=25):
#     """
#     This function computes the AVO synthetic gather given well log information and using the Zoeppritz equation, 
#     which is also convolved with the wavelet.

#     Args:
#         vp (1darray): P-wave velocity.
#         vs (1darray): S-wave velocity.
#         rho (1darray): Density.
#         wav_est (1darray): Estimated wavelet to convolve with.
#         nt_wav (int): Number of samples of statistical wavelet
#         thetamin (int): Minimum angle for reflectivity computation. Default is 0.
#         thetamax (int): Maximum angle for reflectivity computation. Default is 25.
#         ntheta (int): Number of angles. Default is 25.

#     Returns:
#         r_zoeppritz (2darray) : Array (depth/time axis, angles) containing the synthetic gather computed from the well logs
#     """

#     Logsize = len(vp)  # Size of the logs
#     theta = np.linspace(thetamin, thetamax, ntheta)  # Define the angles to obtain reflectivity

#     # Apply Full Zoeppritz equation
#     rpp_zoep = np.zeros(shape=(ntheta, Logsize-1))

#     for i in range(Logsize-1):
#         rpp_zoep[:,i] += zoeppritz_pp(vp[i], vs[i], rho[i], vp[i+1], vs[i+1], rho[i+1], theta)

#     # Wavelet Operator W
#     Cop = Convolve1D(Logsize-1, h=wav_est, offset=nt_wav)    
     
#     # Convolve with wavelet    
#     r_zoeppritz = Cop*rpp_zoep.T

#     return r_zoeppritz


# def avo_synthetic_gather_2D(vp, vs, rho, wav_est, nt_wav, thetamin=0, thetamax=25, ntheta=25):
#     """
#     This function computes the AVO synthetic gather given 2D  models information and using the Zoeppritz equation, 
#     which is also convolved with the wavelet.

#     Args:
#         vp (2darray): P-wave velocity.
#         vs (2darray): S-wave velocity.
#         rho (2darray): Density .
#         wav_est (1darray): Estimated wavelet to convolve with.
#         nt_wav (int): Number of samples of statistical wavelet
#         thetamin (int): Minimum angle for reflectivity computation. Default is 0.
#         thetamax (int): Maximum angle for reflectivity computation. Default is 25.
#         ntheta (int): Number of angles. Default is 25.

#     Returns:
#         r_zoeppritz (3darray) : Array (x -axis, depth/time axis, angles) containing the synthetic gather computed from the background models.
#     """

#     n_sections = vp.shape[0]
#     Logsize = vp.shape[1]  # Size of the logs
#     theta = np.linspace(thetamin, thetamax, ntheta)  # Define the angles to obtain reflectivity

#     # Apply Full Zoeppritz equation
#     rpp_zoep = np.zeros(shape=(n_sections, ntheta, Logsize-1))

#     for j in range(n_sections):
#         for i in range(Logsize-1):
#             rpp_zoep[j, :, i] += zoeppritz_pp(vp[j, i], vs[j, i], rho[j, i], vp[j, i+1], vs[j, i+1], rho[j, i+1], theta)

#     # Wavelet Operator W
#     Cop = Convolve1D(Logsize-1, h=wav_est, offset=nt_wav)    
    
#     # Convolve with wavelet    
#     r_zoeppritz = np.zeros(shape=(n_sections, Logsize-1, ntheta))
#     for j in range(n_sections):
#         r_zoeppritz[j] = Cop * rpp_zoep[j].T

#     return r_zoeppritz



# def Seis2Rock_inference(vp, vs, rho, 
#                       wav_est, nt_wav, 
#                       dtheta, 
#                       Fp, Lp, Vp, 
#                       phi, vsh, sw, 
#                       phi_back, vsh_back, sw_back, 
#                       thetamin=0, thetamax=25, ntheta=25):
#     """
#     This function computes the AVO synthetic gather given 2D background models information and using the Zoeppritz equation, 
#     which is also convolved with the wavelet. The function then uses the resulting synthetic gathers to compute the new data 
#     term called Petrophysical coefficients B.

#     Args:
#         vp (2darray): P-wave velocity background model (x-axis, depth/time axis).
#         vs (2darray): S-wave velocity background model (x-axis, depth/time axis).
#         rho (2darray): Density background model (x-axis, depth/time axis).
#         wav_est (1darray): Estimated wavelet to convolve with.
#         nt_wav (int): Number of samples of statistical wavelet
#         dtheta (3darray): Model data matrix, the dimensions are (x-axis, angles, depth/time axis).
#         Fp (2darray): Subset of matrix F obtained from SVD decomposition (basis functions ,p). Being p the optimal number of singular values for the reconstruction.
#         Lp (2darray): Subset of matrix L obtained from SVD decomposition (p,p).
#         Vp (2darray): Subset of matrix V obtained from SVD decomposition (vsize,p).
#         phi (1darray): Petrophysical property (porosity) from well logs.
#         vsh (1darray): Petrophysical property (shale volume) well logs.
#         sw (1darray): Petrophysical property (water saturation) well logs.
#         phi_back (1darray): Background porosity from well logs.
#         vsh_back (1darray): Background shale volume from well logs.
#         sw_back (1darray): Background water saturation from well logs.
#         thetamin (int): Minimum angle for reflectivity computation. Default is 0.
#         thetamax (int): Maximum angle for reflectivity computation. Default is 25.
#         ntheta (int): Number of angles. Default is 25.

#     Returns:
#         b_optAVO (3darray): New data term (Petrophysical coefficients B) The output shape 
#                                     (#properties, depth/time axis, x-axis).
#         r_zoeppritz_back (3darray): Synthetic gather computed from the background models.
#         Cp (3darray): Matrix obtained from the process (x-axis, p, depth/time-axis).
#     """

    
#     # Create petrophysical models
#     m_full = np.stack((phi, vsh, sw), axis=0)
#     m_back = np.stack((phi_back, vsh_back, sw_back), axis=0)

#     # Compute AVO synthetic gather for background models
#     print('Computing AVO synthetic gather for background models...')
#     r_zoeppritz_back = avo_synthetic_gather_2D(vp, vs, rho, wav_est, nt_wav, thetamin, thetamax, ntheta)

#     # Compute difference between model data and synthetic gathers
#     d_testing = (dtheta - r_zoeppritz_back.transpose(0, 2, 1))

#     # Calculate Cp matrix
#     print('Calculating matrix of optimal coefficients Cp...')
#     Cp = Fp.T @ d_testing

#     # Calculate Hp matrix
#     print('Creating the new data term (Petrophysical coefficeints B)...')
#     Hp = Vp @ np.diag(1. / np.diag(Lp))

#     Logsize = len(phi)
#     kind = 'centered' # 'forward' is better once available in PyLops

#     # PoststackLinearModelling operator
#     D = pylops.avo.poststack.PoststackLinearModelling(wav_est, nt0=Logsize-1, spatdims=Logsize-1, explicit=True)
#     D_A = D.A.copy()

#     print(m_full.shape)
#     # Compute b_full_prof and b_back_prof
#     b_full_prof = D_A @ m_full[:,:-1].T
#     b_back_prof = D_A @ m_back[:,:-1].T

#     # Calculate the new data term (Petrophysical coefficients B)
#     b_optAVO = Cp.transpose(0, 2, 1) @ Hp.T @ (b_full_prof - b_back_prof)
#     b_optAVO = b_optAVO.transpose(2, 1, 0)
#     print('Done xD !')

#     return b_optAVO, r_zoeppritz_back, Cp






##### Correcting functions Dec 2023

def avo_synthetic_gather_dev(vp, vs, rho, wav, offset, thetamin=0, thetamax=45, ntheta=46):
    """
    Apply the Full Zoeppritz equation to the given input data and then convolve with a wavelet.
    This function handles 1D, 2D, and 3D inputs for P-wave velocity (vp), S-wave velocity (vs), 
    and density (rho). The inputs are reshaped to ensure they have three dimensions in the format 
    (t/depth, x, y). It also computes the incident angles based on the provided thetamin, 
    thetamax, and ntheta values.

    Parameters:
    vp (numpy.ndarray): Array of P-wave velocities. Should be 1D, 2D, or 3D.
    vs (numpy.ndarray): Array of S-wave velocities. Should be 1D, 2D, or 3D.
    rho (numpy.ndarray): Array of densities. Should be 1D, 2D, or 3D.
    thetamin (float): Minimum theta value for incident angles.
    thetamax (float): Maximum theta value for incident angles.
    ntheta (int): Number of theta values to generate between thetamin and thetamax.
    wav (numpy.1darray): Wavelet for convolution.
    offset (int): Center of the wavelet.

    Returns:
    numpy.ndarray: The convolved Zoeppritz response.
    """

    # Reshape the inputs to 3D if they are not
    def reshape_to_3d(arr):
        if arr.ndim == 1:  # For 1D input
            return arr.reshape(-1, 1, 1)
        elif arr.ndim == 2:  # For 2D input
            return np.expand_dims(arr, axis=2)
        return arr  # For 3D input

    vp = reshape_to_3d(vp)
    vs = reshape_to_3d(vs)
    rho = reshape_to_3d(rho)

    # Generate theta array
    theta = np.linspace(thetamin, thetamax, ntheta)
    
    # Apply Full Zoeppritz equation
    rpp_zoep = np.zeros(shape=(vp.shape[0], vp.shape[1], vp.shape[2], ntheta))
    for k in range(vp.shape[2]):
        for j in range(vp.shape[1]):
            for i in range(vp.shape[0]-1):
                rpp_zoep[i, j, k, :] += zoeppritz_pp(vp[i, j, k], vs[i, j, k], rho[i, j, k],
                                                     vp[i+1, j, k], vs[i+1, j, k], rho[i+1, j, k], theta)

    
    # Replace the last row of rpp_zoep with the previous row
    rpp_zoep[-1, :, :, :] = rpp_zoep[-2, :, :, :]

    
    # Wavelet Operator W
    Cop = Convolve1D(vp.shape[0], h=wav, offset=offset)

    # Convolve with wavelet    
    r_zoeppritz = np.zeros(shape=(vp.shape[0], vp.shape[1], vp.shape[2], ntheta))
    for k in range(vp.shape[2]):
        for j in range(vp.shape[1]):
            r_zoeppritz[:, j, k, :] = Cop * rpp_zoep[:, j, k, :]
            
    # Remove dimensions of size 1 from the output array to be in agreenment for 1d or 2d input
    # r_zoeppritz_squeezed = np.squeeze(r_zoeppritz)

    return r_zoeppritz


def Seis2Rock_training_dev(vp, vs, rho, 
                     wav, offset, 
                     vp_back, vs_back, rho_back, 
                     p, 
                     thetamin=0, thetamax=25, ntheta=25):
    """
    This function perfoms the Seis2Rock training Routine,it has as a goal obtain the optimal basis functions Fp.
    Here, AVO synthetic gathers are created given well log information and using the Zoeppritz equation, 
    which is also convolved with the wavelet. The function also calculates synthetic gathers for the background models. 
    The function then computes the difference between the synthetic gathers (d-db) and performs Singular Value Decomposition (SVD).
    Subsets of each decomposed matrix are extracted based on input p (optimal number of singular values for the reconstruction).

    Args:
        vp (1darray): P-wave velocity from well logs.
        vs (1darray): S-wave velocity from well logs.
        rho (1darray): Density from well logs.
        wav (1darray): Estimated wavelet to convolve with.
        offset (int): Number of samples of statistical wavelet
        vp_back (1darray): Background P-wave velocity lfrom well logs..
        vs_back (1darray): Background S-wave velocity from well logs..
        rho_back (1darray): Background density from well logs..
        p (int): optimal number of singular values for the reconstruction
        thetamin (int): Minimum angle for reflectivity computation. Default is 0.
        thetamax (int): Maximum angle for reflectivity computation. Default is 25.
        ntheta (int): Number of angles. Default is 25.

    Returns:
        r_zoeppritz, r_zoeppritz_back: Synthetic gathers computed from well logs and background models.
        F, L, V : Matrices obtained from SVD decomposition.
        Fp, Lp, Vp : Extracted subsets of each matrix, based on input p.
    """

    # Compute AVO synthetic gather for well logs and background models
    print('Computing AVO synthetic gathers from the well logs...')
    rz_well = avo_synthetic_gather_dev(vp=vp, vs=vs, rho=rho, wav=wav, offset=offset, 
                              thetamin=thetamin, thetamax=thetamax, ntheta=ntheta)

    rz_well_bg = avo_synthetic_gather_dev(vp=vp_back, vs=vs_back, rho=rho_back, wav=wav, offset=offset, 
                              thetamin=thetamin, thetamax=thetamax, ntheta=ntheta)

    # Compute difference between synthetic gathers
    d = rz_well - rz_well_bg

    # Perform Singular Value Decomposition
    print('Performing SVD...')
    F, L, V = sp.linalg.svd(np.squeeze(d.T), full_matrices=False)
    # V = V.T
    L = np.diag(L)

    # Change sign to have positive values of the functions at first offset/angle 
    # this is due to the fact that SVD is not normalized... similar to:
    # https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/utils/extmath.py#L504)
    
    sign_vect = np.sign(F[0])
    F = F * np.outer(np.ones((ntheta,1)), sign_vect)
    L = L * np.outer(sign_vect.T, np.ones((1, ntheta)))

    # Extract decomposed matrix
    print('Extracting Optimal basis functions Fp..')
    Fp = F[:, :p]
    Lp = L[:p, :p]
    Vp = V[:p, :]   
    
    print('Done!')

    return  Fp, Lp, Vp, F, L, V, rz_well, rz_well_bg, d


def Seis2Rock_inference_dev(vp, vs, rho, 
                      wav, offset, 
                      dtheta, 
                      Fp, Lp, Vp, 
                      phi, vsh, sw, 
                      phi_back, vsh_back, sw_back, 
                      thetamin=0, thetamax=25, ntheta=25):
    """
    This function computes the AVO synthetic gather given 2D background models information and using the Zoeppritz equation, 
    which is also convolved with the wavelet. The function then uses the resulting synthetic gathers to compute the new data 
    term called Petrophysical coefficients B.
    
    Only when well log info is from 1 well

    Args:
        vp (2darray): P-wave velocity background model (depth/time axis, x-axis).
        vs (2darray): S-wave velocity background model (depth/time axis, x-axis).
        rho (2darray): Density background model (depth/time axis, x-axis).
        wav_est (1darray): Estimated wavelet to convolve with.
        nt_wav (int): Number of samples of statistical wavelet
        dtheta (4darray): Model data matrix, the dimensions are (depth/time axis, x-axis, y-axis,angles).
        Fp (2darray): Subset of matrix F obtained from SVD decomposition (basis functions ,p). Being p the optimal number of singular values for the reconstruction.
        Lp (2darray): Subset of matrix L obtained from SVD decomposition (p,p).
        Vp (2darray): Subset of matrix V obtained from SVD decomposition (vsize,p).
        phi (1darray): Petrophysical property (porosity) from well logs.
        vsh (1darray): Petrophysical property (shale volume) well logs.
        sw (1darray): Petrophysical property (water saturation) well logs.
        phi_back (1darray): Background porosity from well logs.
        vsh_back (1darray): Background shale volume from well logs.
        sw_back (1darray): Background water saturation from well logs.
        thetamin (int): Minimum angle for reflectivity computation. Default is 0.
        thetamax (int): Maximum angle for reflectivity computation. Default is 25.
        ntheta (int): Number of angles. Default is 25.

    Returns:
        b_optAVO (3darray): New data term (Petrophysical coefficients B) The output shape 
                                    (#properties, depth/time axis, x-axis).
        Cp (3darray): Matrix obtained from the process (x-axis, p, depth/time-axis).
        Hp 
    """

    
    r_zoeppritz_back = avo_synthetic_gather_dev(vp=vp, vs=vs, rho=rho, wav=wav, offset=offset, 
                                thetamin=thetamin, thetamax=thetamax, ntheta=ntheta)


    # Compute difference between model data and synthetic gathers
    d_testing = (dtheta - r_zoeppritz_back)

    # Calculate Cp matrix
    print('Calculating matrix of optimal coefficients Cp...')
    # Cp = Fp.T @ d_testing.transpose(0, 2, 1) # it only works for 2D arrays of m
    Cp = np.expand_dims(Fp, axis=2).T @ d_testing.transpose(0, 1, 3, 2) # it works for 3d arrays of m

    # Calculate Hp matrix
    print('Creating the new data term (Petrophysical coefficeints B)...')
    Hp = Vp.T @ np.diag(1. / np.diag(Lp))

    # Create petrophysical models
    m_full = np.stack((phi, vsh, sw), axis=0)
    m_back = np.stack((phi_back, vsh_back, sw_back), axis=0)

    Logsize = len(phi)
    # kind = 'centered' # 'forward' is better once available in PyLops
    kind = 'forward'

    # PoststackLinearModelling operator
    D = pylops.avo.poststack.PoststackLinearModelling(wav, nt0=Logsize, spatdims=Logsize, explicit=True, kind=kind)
    D_A = D.A.copy()

    # Compute b_full_prof and b_back_prof
    b_full_prof = D_A @ m_full.T
    b_back_prof = D_A @ m_back.T

    # Calculate the new data term (Petrophysical coefficients B) (only handles 2D)
    # b_optAVO = Cp.transpose(2,0,1) @ (Hp.T @ (b_full_prof - b_back_prof))
    # b_optAVO = b_optAVO.transpose(2,1,0)
    
     # Calculate the new data term (Petrophysical coefficients B) (handles 3D)
    b_optAVO = Cp.transpose(0,1,3,2) @ np.expand_dims((Hp.T @ (b_full_prof - b_back_prof)), axis=0)
    b_optAVO = b_optAVO.transpose(3,0,1,2)
    print('Done!')

    return b_optAVO, Cp, Hp



def Seis2Rock_training_stacks_dev(vp, vs, rho, 
                     wav, offset, 
                     vp_back, vs_back, rho_back, 
                     p, 
                     thetamin=0, thetamax=25, ntheta=25):
    """
    This function perfoms the Seis2Rock training Routine,it has as a goal obtain the optimal basis functions Fp.
    Here, AVO synthetic gathers are created given well log information and using the Zoeppritz equation, 
    which is also convolved with the wavelet. The function also calculates synthetic gathers for the background models. 
    The function then computes the difference between the synthetic gathers (d-db) and performs Singular Value Decomposition (SVD).
    Subsets of each decomposed matrix are extracted based on input p (optimal number of singular values for the reconstruction).

    Args:
        vp (1darray): P-wave velocity from well logs.
        vs (1darray): S-wave velocity from well logs.
        rho (1darray): Density from well logs.
        wav (1darray): Estimated wavelet to convolve with.
        offset (int): Number of samples of statistical wavelet
        vp_back (1darray): Background P-wave velocity lfrom well logs..
        vs_back (1darray): Background S-wave velocity from well logs..
        rho_back (1darray): Background density from well logs..
        p (int): optimal number of singular values for the reconstruction
        thetamin (int): Minimum angle for reflectivity computation. Default is 0.
        thetamax (int): Maximum angle for reflectivity computation. Default is 25.
        ntheta (int): Number of angles. Default is 25.

    Returns:
        r_zoeppritz, r_zoeppritz_back: Synthetic gathers computed from well logs and background models.
        F, L, V : Matrices obtained from SVD decomposition.
        Fp, Lp, Vp : Extracted subsets of each matrix, based on input p.
    """

    # Compute AVO synthetic gather for well logs and background models
    print('Computing AVO synthetic gathers from the well logs...')
    rz_well = avo_synthetic_gather_dev(vp=vp, vs=vs, rho=rho, wav=wav, offset=offset, 
                              thetamin=thetamin, thetamax=thetamax, ntheta=ntheta)

    rz_well_bg = avo_synthetic_gather_dev(vp=vp_back, vs=vs_back, rho=rho_back, wav=wav, offset=offset, 
                              thetamin=thetamin, thetamax=thetamax, ntheta=ntheta)

    # Compute difference between synthetic gathers
    d_full = rz_well - rz_well_bg
    
    d_near = np.mean(d_full[:,:,:,6:18], axis =3)
    d_mid = np.mean(d_full[:,:,:,18:30], axis =3)
    d_far = np.nanmean(d_full[:,:,:,30:42], axis =3)
    
    d = np.stack((d_near, d_mid, d_far), axis=-1)
    
    nstacks = 3
    # Perform Singular Value Decomposition
    print('Performing SVD...')
    F, L, V = sp.linalg.svd(np.squeeze(d.T), full_matrices=False)
    # V = V.T
    L = np.diag(L)

    # Change sign to have positive values of the functions at first offset/angle 
    # this is due to the fact that SVD is not normalized... similar to:
    # https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/utils/extmath.py#L504)
    
    sign_vect = np.sign(F[0])
    F = F * np.outer(np.ones((nstacks,1)), sign_vect)
    L = L * np.outer(sign_vect.T, np.ones((1, nstacks)))

    # Extract decomposed matrix
    print('Extracting Optimal basis functions Fp..')
    Fp = F[:, :p]
    Lp = L[:p, :p]
    Vp = V[:p, :]   
    
    print('Done!')

    return  Fp, Lp, Vp, F, L, V, rz_well, rz_well_bg, d



def Seis2Rock_inference_stacks_dev(vp, vs, rho, 
                      wav, offset, 
                      dtheta, 
                      Fp, Lp, Vp, 
                      phi, vsh, sw, 
                      phi_back, vsh_back, sw_back, 
                      thetamin=0, thetamax=25, ntheta=25):
    """
    This function computes the AVO synthetic gather given 2D background models information and using the Zoeppritz equation, 
    which is also convolved with the wavelet. The function then uses the resulting synthetic gathers to compute the new data 
    term called Petrophysical coefficients B.
    
    Only when well log info is from 1 well

    Args:
        vp (2darray): P-wave velocity background model (depth/time axis, x-axis).
        vs (2darray): S-wave velocity background model (depth/time axis, x-axis).
        rho (2darray): Density background model (depth/time axis, x-axis).
        wav_est (1darray): Estimated wavelet to convolve with.
        nt_wav (int): Number of samples of statistical wavelet
        dtheta (4darray): Model data matrix, the dimensions are (depth/time axis, x-axis, y-axis,angles).
        Fp (2darray): Subset of matrix F obtained from SVD decomposition (basis functions ,p). Being p the optimal number of singular values for the reconstruction.
        Lp (2darray): Subset of matrix L obtained from SVD decomposition (p,p).
        Vp (2darray): Subset of matrix V obtained from SVD decomposition (vsize,p).
        phi (1darray): Petrophysical property (porosity) from well logs.
        vsh (1darray): Petrophysical property (shale volume) well logs.
        sw (1darray): Petrophysical property (water saturation) well logs.
        phi_back (1darray): Background porosity from well logs.
        vsh_back (1darray): Background shale volume from well logs.
        sw_back (1darray): Background water saturation from well logs.
        thetamin (int): Minimum angle for reflectivity computation. Default is 0.
        thetamax (int): Maximum angle for reflectivity computation. Default is 25.
        ntheta (int): Number of angles. Default is 25.

    Returns:
        b_optAVO (3darray): New data term (Petrophysical coefficients B) The output shape 
                                    (#properties, depth/time axis, x-axis).
        Cp (3darray): Matrix obtained from the process (x-axis, p, depth/time-axis).
        Hp 
    """

    print('Computing 2D background models')
    r_zoeppritz_back = avo_synthetic_gather_dev(vp=vp, vs=vs, rho=rho, wav=wav, offset=offset, 
                                thetamin=thetamin, thetamax=thetamax, ntheta=ntheta)
    
    
    r_near = np.nanmean(r_zoeppritz_back[:,:,:,6:18], axis =3)
    r_mid = np.nanmean(r_zoeppritz_back[:,:,:,18:30], axis =3)
    r_far = np.nanmean(r_zoeppritz_back[:,:,:,30:42], axis =3)
    r_z = np.stack((r_near, r_mid, r_far), axis=-1)


    # Compute difference between model data and synthetic gathers
    d_testing = (dtheta - r_z)

    # Calculate Cp matrix
    print('Calculating matrix of optimal coefficients Cp...')
    # Cp = Fp.T @ d_testing.transpose(0, 2, 1) # it only works for 2D arrays of m
    Cp = np.expand_dims(Fp, axis=2).T @ d_testing.transpose(0, 1, 3, 2) # it works for 3d arrays of m

    # Calculate Hp matrix
    print('Creating the new data term (Petrophysical coefficeints B)...')
    Hp = Vp.T @ np.diag(1. / np.diag(Lp))

    # Create petrophysical models
    m_full = np.stack((phi, vsh, sw), axis=0)
    m_back = np.stack((phi_back, vsh_back, sw_back), axis=0)

    Logsize = len(phi)
    # kind = 'centered' # 'forward' is better once available in PyLops
    kind = 'forward'

    # PoststackLinearModelling operator
    D = pylops.avo.poststack.PoststackLinearModelling(wav, nt0=Logsize, spatdims=Logsize, explicit=True, kind=kind)
    D_A = D.A.copy()

    # Compute b_full_prof and b_back_prof
    b_full_prof = D_A @ m_full.T
    b_back_prof = D_A @ m_back.T

    # Calculate the new data term (Petrophysical coefficients B) (only handles 2D)
    # b_optAVO = Cp.transpose(2,0,1) @ (Hp.T @ (b_full_prof - b_back_prof))
    # b_optAVO = b_optAVO.transpose(2,1,0)
    
     # Calculate the new data term (Petrophysical coefficients B) (handles 3D)
    b_optAVO = Cp.transpose(0,1,3,2) @ np.expand_dims((Hp.T @ (b_full_prof - b_back_prof)), axis=0)
    b_optAVO = b_optAVO.transpose(3,0,1,2)
    print('Done!')

    return b_optAVO, Cp, Hp