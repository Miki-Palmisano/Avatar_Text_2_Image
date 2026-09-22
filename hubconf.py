'''
Module for loading a pretrained UNet model for the Carvana dataset.
Provides the function unet_carvana to return a UNet model instance,
optionally initializing it with pretrained weights.
'''
import torch
from unet import UNet as _UNet


def unet_carvana(pretrained=False, scale=0.5):
    """
    Returns a UNet model trained on the Carvana dataset.
    
    Args:
        pretrained (bool): If True, the model will load pretrained weights.
        scale (float): Scale factor for the model's weights. Only 0.5 and 1.0 are supported.
    
    Returns:
        net: An instance of the UNet model.
    
    Raises:
        RuntimeError: If the provided scale is not 0.5 or 1.0 when pretrained is True.
    """
    
    # Instantiate the UNet model with specified parameters.
    net = _UNet(n_channels=3, n_classes=2, bilinear=False)
    
    # Check if the model should be loaded with pretrained weights.
    if pretrained:
        
        # Choose checkpoint URL based on scale value.
        if scale == 0.5:
            checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth'
        elif scale == 1.0:
            checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth'
        else:
            raise RuntimeError('Only 0.5 and 1.0 scales are available')
        
        # Load the state dictionary from the URL.
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=True)
        
        # Remove 'mask_values' key if it exists to avoid incompatibility issues.
        if 'mask_values' in state_dict:
            state_dict.pop('mask_values')
        
        # Load the loaded state dictionary into the UNet model.
        net.load_state_dict(state_dict)

    # Return the configured UNet model.
    return net
