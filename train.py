#
# Once hyperparaters have been selected, we can use this script to train the model.
#

# Lib Imports
import gc
import random
import torch, os, wandb
from tqdm.auto import tqdm
from datetime import datetime
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from huggingface_hub import login, whoami
import segmentation_models_pytorch as smp

# Local Imports
from models.utils.inference import run_eval
from models.utils.training import train_one_epoch
from models.DinoV3.SemanDino import GlacierSegmenter
from models.DinoV3.GlacierDataset import GlacierDataset
from models.utils.metrics import get_combined_loss, get_iou_metric

# Constants/Configuration
IGNORE_INDEX = 255
NUM_CLASS = 2
BATCH_SIZE = 8
NUM_WORKERS = 4 # DataLoader workers

# Paths
TRAIN_IMAGE_DIR = "dataset/clean/images/"
TRAIN_MASK_DIR = "dataset/clean/masks/"
TEST_IMAGE_DIR = "dataset/test/images/"
TEST_MASK_DIR = "dataset/test/masks/"
CHECKPOINT_DIR = "checkpoints/"
LOGS_DIR = "logs/"


def parse_args():
    """
    Parse command line arguments. You can select the model architecture here and give the parameters.
    The model can be Either 'UNet' or 'DinoV3'.
    """
    parser = ArgumentParser(description="Cross-validation for hyperparameter tuning")
    parser.add_argument(
        "--model",
        type=str,
        choices=["UNet", "DinoV3"],
        default="DinoV3",
        help="Model architecture to use for segmentation.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--ce_weight",
        type=float,
        help="Cross-Entropy loss weight.",
    )
    parser.add_argument(
        "--dice_weight",
        type=float,
        help="Dice loss weight.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs.",
    )
    
    return parser.parse_args()

def is_logged_in_wandb_hf(model="DinoV3"):
    """
    Check if the user is logged in both W&B and Hugging Face.
    
    Args:
        model (str, optional): Model name for logging context. Defaults to "DinoV3".
    
    Returns:
        bool: True if logged in both W&B and Hugging Face, False otherwise.
    """
    try:
        wandb_api = wandb.Api()
        _ = wandb_api.viewer
        # If we aren't using the DinoV3 model, we skip the HF login check
        if model == "DinoV3":
            hf_user = whoami()
            
        return True
    except Exception as e:
        print(f"Not logged in: {e}")
        return False
    
    
def run_training(epochs, lr, ce_weight, dice_weight, train_ds, test_ds, model_name, device='cuda'): 
    """Performs a full training with the given hyperparameters.

    Args:
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        ce_weight (float): Weight for the Cross-Entropy loss component.
        dice_weight (float): Weight for the Dice loss component.
        train_ds (Dataset): Full training dataset.
        model_name (str): Name of the model architecture to use.
        device (str, optional): Device to run the training on. Defaults to 'cuda'.
    Returns:
        dict: Dictionary containing results and statistics from the K-Fold experiment.
    """
    
    # Logging setup
    config_tag = f"full_train_{model_name}_LR{lr}_CE{ce_weight}_Dice{dice_weight}_Epochs{epochs}"
    saved_dir = os.path.join(CHECKPOINT_DIR, f"{model_name}_{config_tag}")
    os.makedirs(saved_dir, exist_ok=True)
    run = wandb.init(
                entity="jeremy-hugentobler-epfl",
                project=f"Glacier_Segmentation_{model_name}_full_train",
                group=f"{config_tag}",
                name=f"{config_tag}",
                config={
                    "lr": lr,
                    "ce_weight": ce_weight,
                    "dice_weight": dice_weight,
                    "batch_size": BATCH_SIZE,
                    "epochs": epochs,
                    "num_workers": NUM_WORKERS,
                    "ignore_index": IGNORE_INDEX,
                    "num_classes": NUM_CLASS,
                },
                reinit=True
            )
    print("-"*50, f"\n CONFIG: lr={lr}, ce={ce_weight}, dice={dice_weight} \n", "-"*50)
    
    # Loss function
    criterion = get_combined_loss(ce_weight, dice_weight, ignore_index=IGNORE_INDEX)
    
    if model_name == "UNet":
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            decoder_use_batchnorm=True,
            in_channels=3,
            classes=NUM_CLASS,
        ).to(device)
        # Freeze encoder
        for param in model.encoder.parameters():
            param.requires_grad = False
    else: # DinoV3
        model = GlacierSegmenter(num_classes=NUM_CLASS).to(device)

    # Optimizer
    params_to_update = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params_to_update, lr=lr)
    
    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,  
        prefetch_factor=4
    )
        
    epoch_pbar = tqdm(range(1, epochs + 1), desc=f"[{config_tag}] Full training", leave=False)      
    for epoch in epoch_pbar:
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        # log the loss
        run.log({"Train Loss": train_loss}, step=epoch)
    
    # Save the final model
    checkpoints_path = os.path.join(saved_dir, f"final_model.pth")
    torch.save(
        model.state_dict(),
        checkpoints_path,
    )   
    print(f"Final model saved at: {checkpoints_path}")
    print(f"Testing the final performance on the test set...")

    # Cleanup to avoid errors with too many open files
    del train_loader
    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # Restore the model for testing
    if model_name == "UNet":
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            decoder_use_batchnorm=True,
            in_channels=3,
            classes=NUM_CLASS,
        ).to(device)
        # Freeze encoder
        for param in model.encoder.parameters():
            param.requires_grad = False
    else: # DinoV3
        model = GlacierSegmenter(num_classes=NUM_CLASS).to(device)
    model.load_state_dict(torch.load(checkpoints_path))
    
    # Test DataLoader
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,  
        prefetch_factor=4
    )
    
    iou_metric = get_iou_metric(ignore_index=IGNORE_INDEX)
    avg_loss, avg_iou, imgs, masks, preds = run_eval(model, test_loader, criterion, iou_metric, return_preds=True, device=device)
    print(f"Test Loss: {avg_loss:.4f}, Test IoU: {avg_iou:.4f}")
    run.log({"Test Loss": avg_loss, "Test IoU": avg_iou})

    # Log 5 random predictions
    random.seed(42)
    random_indices = random.sample(range(len(imgs)), 5)
    for idx in random_indices:
        img = imgs[idx]
        mask = masks[idx]
        pred = preds[idx]
        # Create a W&B table
        wandb.log({
            "Predictions": wandb.Image(
                img,
                masks={
                    "predictions": {
                        "mask_data": pred,
                        "class_labels": {0: "Background", 1: "Glacier"}
                    },
                    "ground_truth": {
                        "mask_data": mask,
                        "class_labels": {0: "Background", 1: "Glacier"}
                    }
                }
            )
        }, step=epochs)
    
    run.finish()
    

        
def main():
    
    args = parse_args()
    model_name = args.model
    lr = args.lr
    ce_weight = args.ce_weight
    dice_weight = args.dice_weight
    epochs = args.epochs
    
    print("Checking W&B and Hugging Face login...")
    is_logged_in_wandb_hf(model_name)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
        
    print(f"Starting K-Fold to see how many epoch to use with: {model_name}")
    
    train_ds = GlacierDataset(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dir=TRAIN_MASK_DIR,
        model=model_name,
        mode="train",
    )
    
    test_ds = GlacierDataset(
        image_dir=TEST_IMAGE_DIR,
        mask_dir=TEST_MASK_DIR,
        model=model_name,
        mode="test",
    )
    
    run_training(epochs, lr, ce_weight, dice_weight, train_ds, test_ds, model_name, device)
            
if __name__ == "__main__":
    main()


