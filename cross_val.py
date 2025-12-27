#
# Script to perform cross-validation with K-Fold with a selected model to find hyper parameters.
#

# Lib Imports
import numpy as np
import pandas as pd
import torch, os, wandb
from tqdm.auto import tqdm
from datetime import datetime
from argparse import ArgumentParser
from huggingface_hub import login, whoami
import segmentation_models_pytorch as smp
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset

# Local Imports
from models.DinoV3.SemanDino import GlacierSegmenter
from models.DinoV3.GlacierDataset import GlacierDataset
from models.utils.metrics import get_combined_loss, get_iou_metric
from models.utils.training import train_one_epoch, validate_one_epoch

# Constants/Configuration
IGNORE_INDEX = 255
NUM_CLASS = 2
BATCH_SIZE = 8
MAX_EPOCHS = 30
PATIENCE = 7
MIN_DELTA = 0.002 # Minimum change to qualify as an improvement in the pateince mechanism
N_SPLITS = 5 # K-Fold splits
NUM_WORKERS = 4 # DataLoader workers

# Paths
TRAIN_IMAGE_DIR = "dataset/clean/images/"
TRAIN_MASK_DIR = "dataset/clean/masks/"
TEST_IMAGE_DIR = "dataset/test/images/"
TEST_MASK_DIR = "dataset/test/masks/"
CHECKPOINT_DIR = "checkpoints/"
LOGS_DIR = "logs/"

# Hyperparameters to tune
LRS = [1e-3, 1e-4, 1e-5]
LOSS_WEIGHTS = [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7), (1, 0), (0, 1)]  # (CE weight, Dice weight)


def parse_args():
    """
    Parse command line arguments. You can select the model architecture here.
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
        _ = wandb_api.viewer()
        # If we aren't using the DinoV3 model, we skip the HF login check
        if model == "DinoV3":
            hf_user = whoami()
            
        return True
    except Exception as e:
        print(f"Not logged in: {e}")
        return False
    
    
def k_fold_experiment(lr, ce_weight, dice_weight, full_train_ds, train_ds_for_val, model_name, device='cuda'): 
    """Performs grouped K-Fold for the given hyperparameters and dataset.

    Args:
        lr (float): Learning rate for the optimizer.
        ce_weight (float): Weight for the Cross-Entropy loss component.
        dice_weight (float): Weight for the Dice loss component.
        full_train_ds (Dataset): Full training dataset.
        train_ds_for_val (Dataset): Dataset used for validation.
        model_name (str): Name of the model architecture to use.
        device (str, optional): Device to run the training on. Defaults to 'cuda'.
    Returns:
        ??
    """
    
    # gkf doesn't support shuffling so we do it manually
    groups = np.array(full_train_ds.groups)
    indices = np.arange(len(full_train_ds))
    np.random.seed(42)
    shuffle_idx = np.random.permutation(len(indices))
    indices = indices[shuffle_idx]
    groups = groups[shuffle_idx]
    gkf = GroupKFold(n_splits=N_SPLITS)
    
    # Logging setup
    config_tag = f"{model_name}_LR{lr}_CE{ce_weight}_Dice{dice_weight}"
    saved_dir = os.path.join(CHECKPOINT_DIR, f"{model_name}_{config_tag}")
    os.makedirs(saved_dir, exist_ok=True)
    run = wandb.init(
                project=f"Glacier_Segmentation_{model_name}",
                group=f"{config_tag}",
                name=f"{config_tag}",
                config={
                    "lr": lr,
                    "ce_weight": ce_weight,
                    "dice_weight": dice_weight,
                    "batch_size": BATCH_SIZE,
                    "max_epochs": MAX_EPOCHS,
                    "patience": PATIENCE,
                    "min_delta": MIN_DELTA,
                    "n_splits": N_SPLITS,
                    "num_workers": NUM_WORKERS,
                    "ignore_index": IGNORE_INDEX,
                    "num_classes": NUM_CLASS,
                },
                reinit=True
            )
    print("-"*50, f"\n CONFIG: lr={lr}, ce={ce_weight}, dice={dice_weight} \n", "-"*50)
    
    # Metrics for each fold
    fold_best_ious = []
    fold_best_epochs = []
    fold_ckpts = []
    
    # K-Fold Cross Validation
    for fold, (train_ids, val_ids) in enumerate(gkf.split(indices, groups=groups), start=1):
        criterion = get_combined_loss(ce_weight, dice_weight, ignore_index=IGNORE_INDEX)
        iou_metric = get_iou_metric(ignore_index=IGNORE_INDEX)
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
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = Subset(full_train_ds, train_ids)
        val_subsampler = Subset(train_ds_for_val, val_ids)
        
        # DataLoaders
        train_loader = DataLoader(
                train_subsampler,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,  
                prefetch_factor=4
            )
            
        val_loader = DataLoader(
            val_subsampler,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,  
            prefetch_factor=4
        )
        
        # Training Loop with Early Stopping
        best_val_iou = -1.0
        best_epoch = -1
        best_ckpt_path = os.path.join(saved_dir, f"best_model_fold_{fold}.pth")
        no_improve_epochs = 0  
        
        epoch_pbar = tqdm(range(1, MAX_EPOCHS + 1), desc=f"[{config_tag}] Fold {fold}", leave=False)      
        for epoch in epoch_pbar:
            
            # Train
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            
            # Validation
            val_loss, val_iou = validate_one_epoch(
                model, val_loader, criterion, iou_metric, device
            )
            
            # Logging
            run.log({
                f"fold_{fold}/train_loss": float(train_loss),
                f"fold_{fold}/val_iou": float(val_iou),
                f"fold_{fold}/val_loss": float(val_loss),
                "epoch": int(epoch)
            })
            # Internal progress bar update
            epoch_pbar.set_postfix({
                "T_Loss": f"{train_loss:.3f}",
                "V_Loss": f"{val_loss:.3f}",
                "V_IoU": f"{val_iou:.3f}",
                "Best": f"{best_val_iou:.3f}" if best_val_iou >= 0 else "NA",
            })
            
            # Save Best Model
            if val_iou > best_val_iou + MIN_DELTA:
                best_val_iou = val_iou
                best_epoch = epoch
                no_improve_epochs = 0  # reset counter
                torch.save(
                    model.state_dict(),
                    best_ckpt_path
                )   
            else:
                no_improve_epochs += 1
            # Early Stopping
            if no_improve_epochs >= PATIENCE:
                run.log({"early_stop/epoch": epoch, "fold": fold})
                break
        # For this fold
        run.log({
            f"fold_{fold}/best_iou": best_val_iou,
            f"fold_{fold}/best_epoch": best_epoch,
        })            
        
        # Local record keeping
        fold_best_ious.append(best_val_iou)
        fold_best_epochs.append(best_epoch)
        fold_ckpts.append(best_ckpt_path)
        
    # After all folds -> we do the stats
    mean_iou = float(np.mean(fold_best_ious)) if fold_best_ious else float("nan")
    std_iou = float(np.std(fold_best_ious)) if fold_best_ious else float("nan")
    median_epoch = int(np.median(fold_best_epochs)) if fold_best_epochs else -1

    # Log split
    run.log({
        "mean_iou" : mean_iou,
        "std_iou" : std_iou,
        "median_epoch" : median_epoch
    })
    
    run.finish()

    return {
        "lr": lr,
        "ce_weight": ce_weight,
        "dice_weight": dice_weight,
        "mean_best_iou": mean_iou,
        "std_best_iou": std_iou,
        "median_best_epoch": median_epoch,
        "fold_best_ious": fold_best_ious,
        "fold_best_epochs": fold_best_epochs,
        "fold_ckpts": fold_ckpts,
        "config_tag": config_tag,
        "saved_dir": saved_dir,
    }
        
def main():
    args = parse_args()
    model_name = args.model
    
    print("Checking W&B and Hugging Face login...")
    is_logged_in_wandb_hf(model_name)
    
    print(f"Starting cross-validation for model: {model_name}")
    
    full_train_ds = GlacierDataset(
        image_dir="dataset/clean/images/",
        mask_dir="dataset/clean/masks/",
        model=model_name,
        mode="train",
    )

    train_ds_for_val = GlacierDataset(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dir=TRAIN_MASK_DIR,
        model=model_name,
        mode="test" # -> this disables data augmentation on the validation set
    )
    
    # Final results
    results = []
    
    for lr in LRS:
        for ce_weight, dice_weight in LOSS_WEIGHTS:
            
            results.append(k_fold_experiment(lr, ce_weight, dice_weight, full_train_ds, train_ds_for_val, model_name))
            
            # Save intermediate results
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"cross_val_results_{model_name}.csv", index=False)
            
    # Final save
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{LOGS_DIR}cross_val_results_{model_name}_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
    

if __name__ == "__main__":
    main()


