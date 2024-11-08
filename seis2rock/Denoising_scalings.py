import torch
from math import sqrt
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Denoiser_scale_iter(model, img, tau, verb=False):
    """
    Uses the min and max of the image at every iteration to scale it.
    Scales the image between 0 and 1 and after the denoising step it 
    scales back to the original range.
    """
    if verb: print(f'img, min{img.min()}, max{img.max()}')

    #Denoising the image with the DnCNN
    img_min = img.min()
    img_max = img.max()
    scaled_img = (img - img_min) / (img_max-img_min)
    if  verb: print(f'img, min{scaled_img.min()}, max{scaled_img.max()}')

    model.eval()
    Denoise_img = model(scaled_img.unsqueeze(0).unsqueeze(0).cuda()).squeeze().detach().cpu().numpy()
    
    Denoise_img = Denoise_img*(img_max-img_min).detach().numpy() + img_min.detach().numpy()
    if verb: print(f'Denoise_img, min{Denoise_img.min()}, max{Denoise_img.max()}')
    return Denoise_img


def scaling_DRUNET(model, img, mu, sigma, verb=False):

    #Shifting and Scaling    
    if verb: print(f'img, min{img.min()}, max{img.max()}')
    
    if img.min() == 0 and img.min() == 0:
        # Input is all zeros, simply return it as is
        return img
        
    img_min = img.min()
    numerator = img - img_min
    scaled_img = numerator/numerator.max()
    if verb: print(f'scaled_img, min{scaled_img.min()}, max{scaled_img.max()}')
        
    extra_left, extra_right = 0, 1024-img.shape[1]
    extra_top, extra_bottom = 0, 1024-img.shape[0]
    
    scaled_img = np.pad(scaled_img, ((extra_top, extra_bottom), (extra_left, extra_right)),
       mode='constant', constant_values=0) 
        
    # Create sigma map
    sigma = sqrt(sigma * mu)
#     print('sigma level used: %.3f' % sigma)
    sigmamap = torch.tensor(sigma, dtype=torch.float).repeat(1, 1, scaled_img.shape[0], scaled_img.shape[1])
    scaled_img = torch.cat((torch.from_numpy(scaled_img).unsqueeze(0).unsqueeze(0), sigmamap), dim=1)

    Denoise_img = model(scaled_img.to(device))
    Denoise_img = Denoise_img[0,0,:img.shape[0], :img.shape[1]]

    Denoise_img = Denoise_img*(numerator.max()) + img_min
    if verb: print(f'Denoise_img, min{Denoise_img.min()}, max{Denoise_img.max()}')
        
    return Denoise_img.cpu().detach().numpy()