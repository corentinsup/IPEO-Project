import torch, math
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class GlacierEncoder(nn.Module):
    
    def __init__(self, dino_name="facebook/dinov3-vitl16-pretrain-sat493m", device="auto"):
        """Uses DinoV3 as an encoder to produce a rich feature map that will be used by the decoder.

        Args:
            dino_name (str): The name (see Hugging Face Models) of the DINO model to use. Defaults to "facebook/dinov3-vitl16-pretrain-sat493m".
            device (str, optional): The device to map the model to. Defaults to "auto".
        """
    
        # Initialize the parent class
        super().__init__()
        
        # Hugging Face DINO model 
        try: # If the account isn't logged in, the access is restricted
            self.dino_model = AutoModel.from_pretrained(dino_name, device_map=device)
        except:
            raise ValueError(f"Could not load the DINO model '{dino_name}'. Make sure you have access to it on Hugging Face. (and you are logged in in your environment).")

        self.patch_size = self.dino_model.config.patch_size
        self.hidden_size = self.dino_model.config.hidden_size
        
    def forward(self, x):
        """Takes as input an image (processed) and does the forward pass with the Dino model. It keeps the last hidden state and not the pooled features
        This function also removes the unwanted token (like CLS) if present in the feature map.
        
        Args:
            x : Preprocessed image batch of shape (B, 3, H, W)
        """
        
        # Forward pass through DINO model
        outputs = self.dino_model(x)
        B, C, H, W = x.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        num_spatial_tokens = h_patches * w_patches
        
        # Get the last hidden state (feature map)
        # Hpatches ​= ImageHeight​/PatchSize​
        # Wpatches​ = ImageWidth​/PatchSize​
        # Ntokens ​= (Hpatches ​* Wpatches​) + 1CLS​.
        feature_map = outputs.last_hidden_state # (BatchSize, Ntokens​, HiddenDim)
        
        # Skip the CLS token (we don't care about it here) + some random other register tokens if present   
        feature_map = feature_map[:, -num_spatial_tokens:, :]  # (BatchSize, Ntokens​-1, HiddenDim)
        
        # We need to reshape as (Batch, Channels, Height, Width)
        B, N, C = feature_map.shape
        h = w = int(math.sqrt(N)) # -> this assumes square images
        
        spatial_features = feature_map.reshape(B, h, w, C).permute(0, 3, 1, 2)
        
        return spatial_features
    
class GlacierDecoder(nn.Module):
    
    def __init__(self, encoder_channels, num_classes, patch_size):
        """A simple decoder that upsamples the feature map from the encoder to the original image size.
        
        Args:
            encoder_channels (int): Number of channels in the encoder output feature map.
            num_classes (int): Number of output classes for segmentation. (out channels)
            patch_size (int): The patch size used in the encoder (DINO).
        """
        super().__init__()
        
        # Calculate number of upsampling layers needed to reach original size based on the patch size
        num_upsamples = int(math.log2(patch_size))
        if 2 ** num_upsamples != patch_size:
            print(f"Warning: Patch size {patch_size} is not a power of 2.")
            print("The decoder might not align perfectly. Consider resizing the final output.")
        
        # Create dynamic number of upsampling blocks
        self.blocks = nn.ModuleList()
        current_channels = encoder_channels
        for _ in range(num_upsamples):
            out_channels = max(current_channels // 2, 64)
            block = UpBlock(current_channels, out_channels)
            self.blocks.append(block)
            current_channels = out_channels           
        
        # Final convolution to get the desired number of classes
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        """Forward pass of the decoder.
        
        Args:
            x : Feature map from the encoder of shape (B, Channels, H', W')
        """
        
        for block in self.blocks:
            x = block(x)
            
        x = self.final_conv(x)
        return x
    
class GlacierSegmenter(nn.Module):
    def __init__(self, num_classes=2):
        """Full semantic segmentation model combining the encoder and the decoder

        Args:
            num_classes (int, optional): Number of classes to do the segmentation with. Defaults to 2.
        """
        super().__init__()
        
        self.encoder = GlacierEncoder()
        self.encoder.eval()

        # We use the spatial information of the encoder model (patch + hidden size)
        self.decoder = GlacierDecoder(
            encoder_channels=self.encoder.hidden_size, 
            num_classes=num_classes, 
            patch_size=self.encoder.patch_size
        )
        
    def forward(self, x):
        # freeze the encoder
        with torch.no_grad():
            features = self.encoder(x)
            
        segmentation_map = self.decoder(features)
        return segmentation_map
    
    
class UpBlock(nn.Module):
    """
    A helper block that performs:
    Deconvolution -> Convolution -> Batch Norm -> ReLU
    Go up by a factor of 2 in spatial dimensions (how we recover the initial size)
    """
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        # kernel_size=2, stride=2 is the cleanest way to deconvolve 
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        
        # Regular Conv to smooth out artifacts and learn features
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x