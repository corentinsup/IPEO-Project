import segmentation_models_pytorch as smp

if __name__ == "__main__":

    # This forces download and caching
    _ = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
    )
    print("Weights downloaded and cached.")
