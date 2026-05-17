"""
unet_api.py — INIA segmentation API (skeleton).

Superclass defines the common contract; subclasses fill in Keras-specific and
nnU-Net-specific behavior. Team members can pick up individual methods to implement.

Data assumptions (cardiac ultrasound, Dataset101_CardiacUS):
    Raw:
        images.npz["images"]   — (208, 300, 300, 3) uint8   (only channel 0 is used)
        images.npz["filenames"]— (208,) str
        masks.npz["masks"]     — (208, 300, 300)    uint8   already binary {0, 1}
        masks.npz["filenames"] — (208,) str

    After load_data():
        X_train : (180, 320, 320, 1) float32 in [0, 1]
        y_train : (180, 320, 320, 1) float32 in {0, 1}
        X_test  : (28,  320, 320, 1) float32 in [0, 1]
        y_test  : (28,  320, 320, 1) float32 in {0, 1}

Usage:
    from unet_api import load_data, KerasSegModel, NnUNetSegModel

    X_train, y_train, X_test, y_test = load_data()

    # Keras path
    model = KerasSegModel("unet++")
    model.fit(X_train, y_train, epochs=50)
    metrics = model.evaluate(X_test, y_test)
    masks   = model.predict(X_test)
    model.plot_predictions(X_test, y_test, n=3)

    # Keras from checkpoint
    model = KerasSegModel.from_checkpoint("best_unetpp.keras")

    # nnU-Net path
    model = NnUNetSegModel.from_checkpoint(
        model_folder="chimera_results_H200/nnunet/nnUNet/nnUNet_results/"
                     "Dataset101_CardiacUS/nnUNetTrainer__nnUNetPlans__2d",
        dataset_name="Dataset101_CardiacUS",
    )
    masks = model.predict(X_test)


    file from Beckner
    task: filling in the blank for all the methods
""" 
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import torch
import nibabel as nib

# nnUNet imports
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results

# =============================================================================
# Constants
# =============================================================================
INPUT_SIZE = (320, 320, 1)
SMOOTH = 1e-6

# =============================================================================
# Data loading & preprocessing (already provided)
# =============================================================================
def load_data(images_path="images.npz", masks_path="masks.npz", test_split=28, seed=42):
    """Load and preprocess data"""
    images = np.load(images_path)["images"]
    masks = np.load(masks_path)["masks"]
    
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(images))
    images = images[indices]
    masks = masks[indices]
    
    images, masks = preprocess(images, masks)
    
    X_test, y_test = images[:test_split], masks[:test_split]
    X_train, y_train = images[test_split:], masks[test_split:]
    
    print(f"[load_data] Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def preprocess(images, masks): 
    return images, masks

# =============================================================================
# Base Class
# =============================================================================
class SegModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, **kwargs):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def evaluate(self, X, y_true):
        pass


class KerasSegModel(SegModel):
    # ... (Keep all the Keras code your teammate already wrote) ...
    pass   # ← Replace this with your teammate's full Keras class

# My implementation 
class NnUNetSegModel(SegModel):
    def __init__(self):
        self.predictor = None
        self.model_folder = None

    def _load_predictor(self):
        """Load the trained nnUNet model"""
        model_folder = Path("chimera_results_H200/nnunet/nnUNet/nnUNet_results/Dataset101_CardiacUS/nnUNetTrainernnUNetPlans2d")
        
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            verbose=False,
            verbose_preprocessing=False,
        )
        
        self.predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=(0, 1, 2, 3, 4),
            checkpoint_name='checkpoint_final.pth'
        )
        print("✅ nnUNet predictor loaded successfully from checkpoint.")

    def from_checkpoint(self, model_folder: str, dataset_name: str = "Dataset101_CardiacUS"):
        """Initialize from trained model folder"""
        self.model_folder = model_folder
        self._load_predictor()
        return self

    def _arrays_to_nnunet_layout(self, images: np.ndarray):
        """Convert to nnUNet expected format"""
        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]  # (N, 1, H, W)
        elif images.ndim == 4 and images.shape[-1] == 1:
            images = images.transpose(0, 3, 1, 2)  # (N, 1, H, W)
        return images.astype(np.float32)

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict binary masks"""
        if self.predictor is None:
            self._load_predictor()
        
        images = self._arrays_to_nnunet_layout(images)
        predictions = []
        
        for i, img in enumerate(images):
            pred = self.predictor.predict_single_npy_array(
                img[np.newaxis],
                image_properties=None,
                segmentation_export_mode='binary'
            )
            predictions.append(pred)
        
        return np.array(predictions).squeeze()

    def evaluate(self, images: np.ndarray, ground_truth: np.ndarray):
        """Evaluate using Dice and IoU"""
        pred_masks = self.predict(images)
        pred_masks = (pred_masks > 0.5).astype(np.uint8)
        gt = (ground_truth > 0).astype(np.uint8)
        
        intersection = np.logical_and(pred_masks, gt).sum(axis=(1, 2))
        union = np.logical_or(pred_masks, gt).sum(axis=(1, 2))
        
        dice = (2 * intersection) / (union + intersection + SMOOTH)
        iou = intersection / (union + SMOOTH)
        
        return {
            'dice': float(dice.mean()),
            'iou': float(iou.mean()),
            'dice_per_case': dice.tolist(),
            'iou_per_case': iou.tolist()
        }

    def fit(self, X_train, y_train, **kwargs):
        """Full training (placeholder - better to use nnUNet command line)"""
        print("⚠️ nnUNet full training is best done via command line (nnUNetv2_train).")
        print("This method is not fully implemented yet.")
        return self

    def plot_predictions(self, X, y_true, n=3):
        # Optional: reuse your teammate's plotting code
        pass
