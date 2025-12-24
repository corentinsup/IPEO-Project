import segmentation_models_pytorch as smp

if __name__ == "__main__":

    """Script to download and cache pre-trained weights for segmentation_models_pytorch."""

    _ = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
    )
    print("Weights downloaded and cached.")
