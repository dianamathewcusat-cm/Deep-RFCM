import os
import warnings
warnings.filterwarnings("ignore")

# ==================== STANDARD LIBRARIES ====================
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ==================== SCIKIT-LEARN ====================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==================== SCIPY ====================
from scipy.ndimage import gaussian_filter, median_filter

# ==================== TORCH ====================
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== SHAP ====================
import shap

# ==================== NIBABEL (optional) ====================
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

# ============================================================
# CONFIGURATION
# ============================================================

# -------------------- PATHS --------------------
IMAGE_PATH = r"D:\Rahul\implementation\Diana\dataset\dataset lung\images\LNG001.png"
MODEL_PATH = None  # Set to trained classifier .pth for real Grad-CAM results

# SHAP dataset paths
SHAP_PATHS = {
    "Brain Cancer": r"D:\Rahul\implementation\Diana\dataset\dataset brain\brain_genomic_clinical_gene_clusters.csv",
    "Breast Cancer": r"D:\Rahul\implementation\Diana\dataset\dataset breast\breast_genomic_clinical_gene_clusters.csv",
    "Lung Cancer": r"D:\Rahul\implementation\Diana\dataset\dataset lung\lung_cancer_genomic_clinical_gene_clusters.csv"
}

# -------------------- OUTPUT DIRECTORIES --------------------
OUTPUT_DIR_PO_HYVA = "./image_results"
OUTPUT_DIR_GRADCAM = "./gradcam_results"
OUTPUT_DIR_SHAP = "./shap_results"

# -------------------- IMAGE PROCESSING --------------------
TARGET_SIZE = (256, 256)
NOISE_REDUCTION = "gaussian"
SIGMA = 1.0

# -------------------- PO-HyVA PARAMETERS --------------------
LATENT_DIM = 128
PATCH_SIZE = 8
FIT_STEPS = 1500

# -------------------- GRAD-CAM PARAMETERS --------------------
CLASS_NAMES = ["benign", "malignant"]
TARGET_CLASS = None  # None = explain top prediction

# -------------------- SHAP PARAMETERS --------------------
N_ESTIMATORS = 300
TEST_SIZE = 0.20
TOP_FEATURES_DISPLAY = 10

# -------------------- RANDOM SEED --------------------
RANDOM_SEED = 42


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories():
    os.makedirs(OUTPUT_DIR_PO_HYVA, exist_ok=True)
    os.makedirs(OUTPUT_DIR_GRADCAM, exist_ok=True)
    os.makedirs(OUTPUT_DIR_SHAP, exist_ok=True)

# ============================================================
# PART 1: PREPROCESSING (Shared between PO-HyVA and Grad-CAM)
# ============================================================

class RadiomicData:
    
    def __init__(self, target_size=(256, 256), noise_reduction='gaussian', sigma=1.0):
        self.target_size = target_size
        self.noise_reduction = noise_reduction
        self.sigma = sigma
    
    def load_2d_image(self, file_path):
        """Load image from various formats"""
        if file_path.endswith(('.nii.gz', '.nii')):
            if not HAS_NIBABEL:
                raise RuntimeError(
                    "nibabel is not installed but a .nii/.nii.gz file was "
                    "provided. Install with: pip install nibabel"
                )
            img = nib.load(file_path)
            data = img.get_fdata()
            if data.ndim == 3:
                data = data[:, :, data.shape[2] // 2]  # middle slice
            return data
        else:
            img = Image.open(file_path).convert('L')
            return np.array(img).astype(np.float32)
    
    def voxel_normalization(self, image):
        image = image.astype(np.float32)
        non_zero_mask = image > 0
        if np.any(non_zero_mask):
            mean = np.mean(image[non_zero_mask])
            std = np.std(image[non_zero_mask])
            if std > 0:
                image[non_zero_mask] = (image[non_zero_mask] - mean) / std
        return image
    
    def apply_noise_reduction(self, image):
        if self.noise_reduction == 'gaussian':
            return gaussian_filter(image, sigma=self.sigma)
        elif self.noise_reduction == 'median':
            return median_filter(image, size=3)
        return image
    
    def resize_image(self, image, target_size):
        if image.shape != target_size:
            image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image_resized = cv2.resize(image_uint8, target_size, interpolation=cv2.INTER_AREA)
            return image_resized.astype(np.float32)
        return image
    
    def extract_2d_descriptors(self, image):

        image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Intensity features
        intensity_features = image.copy()
        
        # Gradient magnitude
        gradient_x = cv2.Sobel(image_uint8, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image_uint8, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        
        # Laplacian
        laplacian = cv2.Laplacian(image_uint8, cv2.CV_64F)
        
        # Normalize to [0, 1]
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        laplacian = cv2.normalize(laplacian, None, 0, 1, cv2.NORM_MINMAX)
        
        return np.stack([intensity_features, gradient_magnitude, laplacian], axis=-1)
    
    def preprocess_image(self, file_path):
        raw = self.load_2d_image(file_path)
        normed = self.voxel_normalization(raw)
        denoised = self.apply_noise_reduction(normed)
        resized = self.resize_image(denoised, self.target_size)
        features = self.extract_2d_descriptors(resized)
        return raw, resized, features


# ============================================================
# PART 2: PO-HyVA (Chaotic Puma Optimized Hypergraph VAE)
# ============================================================

class PumaOptimizer:

    def __init__(self, population_size=10, max_iter=100):
        self.population_size = population_size
        self.max_iter = max_iter
    
    def _logistic_chaotic_map(self, x, r=3.9):
        return r * x * (1 - x)
    
    def optimize_hyperparameters(self, latent_dim, input_dim, seed=42):
        population = []
        chaotic_seed = 0.1
        
        for _ in range(self.population_size):
            chaotic_seq = [chaotic_seed]
            for _ in range(9):
                chaotic_seq.append(self._logistic_chaotic_map(chaotic_seq[-1]))
            
            beta = 0.1 + 0.9 * chaotic_seq[0]
            lr = 1e-4 + (1e-3 - 1e-4) * chaotic_seq[1]
            latent_factor = 0.5 + 2.0 * chaotic_seq[2]
            
            population.append({
                'beta': beta,
                'learning_rate': lr,
                'latent_dim': max(4, int(latent_dim * latent_factor)),
                'fitness': 0.0
            })
            chaotic_seed = self._logistic_chaotic_map(chaotic_seed)
        
        best_solution = None
        for _ in range(self.max_iter):
            for solution in population:
                fitness = self._evaluate_fitness(solution, input_dim)
                solution['fitness'] = fitness
                if best_solution is None or fitness > best_solution['fitness']:
                    best_solution = solution.copy()
            
            for solution in population:
                chaotic_val = self._logistic_chaotic_map(solution['beta'])
                solution['beta'] = np.clip(solution['beta'] + 0.1 * (2 * chaotic_val - 1), 0.1, 1.0)
                solution['learning_rate'] = np.clip(
                    solution['learning_rate'] * (1 + 0.1 * (2 * chaotic_val - 1)), 1e-4, 1e-3
                )
        
        return best_solution
    
    def _evaluate_fitness(self, solution, input_dim):
        """Evaluate fitness based on latent ratio and beta balance"""
        latent_ratio = solution['latent_dim'] / input_dim
        beta_balance = 1.0 - abs(solution['beta'] - 0.5)
        return latent_ratio * 0.6 + beta_balance * 0.4


class HypergraphVAE(nn.Module):
    """
    Hypergraph Variational Autoencoder
    Represents voxel-level descriptors as hyperedges (multi-relational groups)
    spanning several voxels at once, then learns a latent embedding via VAE.
    """
    
    def __init__(self, input_dim, latent_dim, hyperedge_size=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.hyperedge_size = hyperedge_size
        self.input_dim = input_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * hyperedge_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim * hyperedge_size),
            nn.Sigmoid()
        )
    
    def create_hyperedges(self, x):
        """Create hyperedges from image patches"""
        batch_size, channels, height, width = x.shape
        x_flat = x.reshape(batch_size, channels, -1).permute(0, 2, 1)
        
        hyperedges = []
        for i in range(batch_size):
            num_voxels = x_flat.size(1)
            step = max(1, num_voxels // self.hyperedge_size)
            indices = torch.arange(0, num_voxels, step)[:self.hyperedge_size]
            if len(indices) < self.hyperedge_size:
                indices = torch.cat([indices, indices[:self.hyperedge_size - len(indices)]])
            hyperedges.append(x_flat[i, indices].flatten())
        
        return torch.stack(hyperedges)
    
    def create_patch_hyperedges(self, x, patch_size, stride=None):
        """
        Tile image into patches, each treated as one hyperedge.
        Supports overlapping patches for smooth reconstruction.
        """
        if stride is None:
            stride = patch_size
        
        _, channels, height, width = x.shape
        unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
        patches = unfold(x)
        patches = patches.squeeze(0).permute(1, 0)
        
        return patches, stride
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        hyperedges = self.create_hyperedges(x)
        mu, logvar = self.encode(hyperedges)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar, hyperedges


class PO_HyVA:
    """
    Chaotic Puma Optimized HyVA
    CPuO tunes (beta, lr, latent_dim), then HyVA is fit on the given image
    to learn hyperedge embeddings and reconstruct the full image.
    """
    
    def __init__(self, input_channels=3, latent_dim=128, patch_size=8, image_size=(256, 256)):
        self.patch_size = patch_size
        self.channels = input_channels
        self.image_size = image_size
        
        patch_dim = input_channels * patch_size * patch_size
        
        # Optimize hyperparameters
        self.puma = PumaOptimizer(population_size=10, max_iter=50)
        params = self.puma.optimize_hyperparameters(latent_dim, patch_dim)
        
        # Keep latent_dim sane
        params['latent_dim'] = max(4, min(params['latent_dim'], patch_dim // 2))
        
        # Initialize HyVA
        self.hyva = HypergraphVAE(
            input_dim=patch_dim,
            latent_dim=params['latent_dim'],
            hyperedge_size=1
        )
        self.optimizer = torch.optim.Adam(self.hyva.parameters(), lr=params['learning_rate'])
        self.beta = min(params['beta'], 0.01)
        self.params = params
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.hyva.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def fit_and_reconstruct(self, image_chw, steps=200):
        """
        Fit HyVA on overlapping patches and reconstruct the full image.
        Overlap-add averaging removes hard tile seams.
        """
        x = image_chw.unsqueeze(0)
        
        # Determine optimal stride for smooth overlap
        span = self.image_size[0] - self.patch_size
        stride = max(1, self.patch_size // 2)
        candidate = stride
        found = False
        
        while candidate >= 1:
            if span % candidate == 0:
                stride = candidate
                found = True
                break
            candidate -= 1
        
        if not found:
            stride = max(1, self.patch_size // 2)
            while stride < self.patch_size and span % stride != 0:
                stride += 1
        
        patches, stride = self.hyva.create_patch_hyperedges(x, self.patch_size, stride=stride)
        
        # Training
        self.hyva.train()
        for _ in range(steps):
            mu, logvar = self.hyva.encode(patches)
            logvar = torch.clamp(logvar, min=-10, max=10)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
            
            recon = self.hyva.decode(z)
            recon_loss = F.mse_loss(recon, patches)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + self.beta * kl_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.hyva.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # Reconstruction
        self.hyva.eval()
        with torch.no_grad():
            mu_final, logvar_final = self.hyva.encode(patches)
            recon = self.hyva.decode(mu_final)
        
        # Overlap-add reassembly
        fold = nn.Fold(output_size=self.image_size, kernel_size=self.patch_size, stride=stride)
        recon_patches = recon.permute(1, 0).unsqueeze(0)
        recon_sum = fold(recon_patches).squeeze(0)
        
        ones = torch.ones_like(patches).permute(1, 0).unsqueeze(0)
        overlap_count = fold(ones).squeeze(0)
        recon_image = recon_sum / overlap_count.clamp(min=1e-6)
        
        return recon_image, mu_final.detach()


def run_po_hyva():
    """Execute PO-HyVA pipeline on a single image"""
    
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] Image not found: {IMAGE_PATH}")
        return None, None
    
    image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    
    print("\n" + "="*70)
    print("PO-HyVA: Radiomic Feature Extraction")
    print("="*70)
    
    # Preprocess
    print("Preprocessing image...")
    preprocessor = RadiomicData(
        target_size=TARGET_SIZE,
        noise_reduction=NOISE_REDUCTION,
        sigma=SIGMA
    )
    raw, resized, features = preprocessor.preprocess_image(IMAGE_PATH)
    
    # Normalize features to [0, 1]
    features_norm = features.copy()
    for c in range(features_norm.shape[-1]):
        chan = features_norm[..., c]
        chan_min, chan_max = chan.min(), chan.max()
        if chan_max > chan_min:
            features_norm[..., c] = (chan - chan_min) / (chan_max - chan_min)
    
    features_tensor = torch.FloatTensor(features_norm).permute(2, 0, 1)
    
    # Run PO-HyVA
    print("Running CPuO chaotic search for HyVA hyperparameters...")
    po_hyva = PO_HyVA(
        input_channels=3,
        latent_dim=LATENT_DIM,
        patch_size=PATCH_SIZE,
        image_size=TARGET_SIZE
    )
    
    print(f" Selected params -> beta: {po_hyva.beta:.3f}, "
          f"lr: {po_hyva.params['learning_rate']:.6f}, "
          f"latent_dim: {po_hyva.params['latent_dim']}")
    
    print(f"Fitting HyVA on overlapping patches for {FIT_STEPS} steps...")
    recon_image_chw, mu = po_hyva.fit_and_reconstruct(features_tensor, steps=FIT_STEPS)
    
    # Extract intensity channel for display
    recon_img = recon_image_chw[0].numpy()
    recon_img = cv2.normalize(recon_img, None, 0, 1, cv2.NORM_MINMAX)
    
    # Create output panel
    print("Rendering result panel...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    axes[0].imshow(raw, cmap='gray')
    axes[0].set_title("Raw Image", fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(resized, cmap='gray')
    axes[1].set_title("Preprocessed", fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(recon_img, cmap='gray')
    axes[2].set_title("PO-HyVA Reconstruction", fontsize=16, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR_PO_HYVA, f"{image_name}_po_hyva_result.jpg")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved {out_path}")
    print("PO-HyVA complete.\n")
    
    return raw, resized


# ============================================================
# PART 3: GRAD-CAM (Explainable AI for Classifier)
# ============================================================

class SimpleCNNClassifier(nn.Module):
    """
    Lightweight CNN for radiomic descriptor classification.
    conv4 is the target layer for Grad-CAM.
    """
    
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        pooled = self.pool(x).flatten(1)
        return self.fc(pooled)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    Visualizes which regions of an image are most important for
    a classifier's prediction.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for the target class
        """
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().numpy()
        
        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min) if cam_max > cam_min else np.zeros_like(cam)
        
        probs = F.softmax(output, dim=1).detach().numpy()[0]
        
        return cam, target_class, probs


def overlay_heatmap(base_image_uint8, cam, target_size, alpha=0.45):
    """Overlay Grad-CAM heatmap on the original image"""
    cam_resized = cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    base_rgb = cv2.cvtColor(base_image_uint8, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(base_rgb, 1 - alpha, heatmap, alpha, 0)
    
    return overlay, cam_resized


def run_gradcam():
    """Execute Grad-CAM pipeline on a single image"""
    
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] Image not found: {IMAGE_PATH}")
        return
    
    image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    
    print("\n" + "="*70)
    print("Grad-CAM: Explainable AI for Classifier")
    print("="*70)
    
    # Preprocess
    print("Preprocessing image...")
    preprocessor = RadiomicData(
        target_size=TARGET_SIZE,
        noise_reduction=NOISE_REDUCTION,
        sigma=SIGMA
    )
    raw, resized, features = preprocessor.preprocess_image(IMAGE_PATH)
    
    # Normalize features
    features_norm = features.copy()
    for c in range(features_norm.shape[-1]):
        chan = features_norm[..., c]
        chan_min, chan_max = chan.min(), chan.max()
        if chan_max > chan_min:
            features_norm[..., c] = (chan - chan_min) / (chan_max - chan_min)
    
    features_tensor = torch.FloatTensor(features_norm).permute(2, 0, 1).unsqueeze(0)
    
    # Load or initialize classifier
    model = SimpleCNNClassifier(num_classes=len(CLASS_NAMES))
    
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print(f"Loaded trained classifier weights from {MODEL_PATH}")
    else:
        print("[WARNING] No trained MODEL_PATH found - using randomly initialized classifier.")
        print("Set MODEL_PATH to your trained checkpoint for real explanations.")
    
    # Generate Grad-CAM
    print("Running Grad-CAM...")
    gradcam = GradCAM(model, model.conv4)
    cam, pred_class_idx, probs = gradcam.generate(features_tensor, target_class=TARGET_CLASS)
    
    pred_class_name = CLASS_NAMES[pred_class_idx] if pred_class_idx < len(CLASS_NAMES) else str(pred_class_idx)
    
    # Create overlay
    resized_uint8 = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    overlay, cam_resized = overlay_heatmap(resized_uint8, cam, TARGET_SIZE)
    
    # Render output panel
    print("Rendering result panel...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(raw, cmap='gray')
    axes[0].set_title("Raw Image", fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(resized, cmap='gray')
    axes[1].set_title("Preprocessed", fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    im = axes[2].imshow(cam_resized, cmap='jet')
    axes[2].set_title("Grad-CAM Heatmap", fontsize=16, fontweight='bold')
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    axes[3].imshow(overlay)
    axes[3].set_title(f"Overlay (pred: {pred_class_name})", fontsize=16, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR_GRADCAM, f"{image_name}_gradcam_result.jpg")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved {out_path}")
    prob_str = ", ".join(f"{n}={p:.3f}" for n, p in zip(CLASS_NAMES, probs))
    print(f" Predicted: {pred_class_name} ({prob_str})")
    print("Grad-CAM complete.\n")


# ============================================================
# PART 4: SHAP ANALYSIS (Feature Importance)
# ============================================================

def preprocess_shap_data(df):
    """
    Preprocess data for SHAP analysis:
    - Remove Patient ID
    - Handle target column (death01)
    - Encode categorical variables
    - Handle missing values
    """
    df = df.copy()
    
    # Remove Patient ID
    if "Patient" in df.columns:
        df = df.drop(columns=["Patient"])
    
    # Target column
    target = "death01"
    df = df.dropna(subset=[target]).copy()
    
    # Convert target to numeric
    if not pd.api.types.is_numeric_dtype(df[target]):
        target_encoder = LabelEncoder()
        df[target] = target_encoder.fit_transform(df[target].astype(str))
    
    # Separate X and y
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    
    # Handle categorical variables
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for col in categorical_columns:
        X[col] = X[col].fillna("Unknown").astype(str)
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
    
    # Handle numeric missing values
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        X[col] = X[col].fillna(X[col].median())
    
    return X, y


def run_shap_analysis():
    """
    Execute SHAP analysis on Brain, Breast, and Lung cancer datasets.
    Generates feature importance plots and comparison visualization.
    """
    
    print("\n" + "="*70)
    print("SHAP Analysis: Feature Importance Across Cancer Types")
    print("="*70)
    
    all_results = {}
    
    # Process each dataset
    for dataset_name, file_path in SHAP_PATHS.items():
        print(f"\nProcessing {dataset_name}...")
        
        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}")
            continue
        
        # Load and preprocess
        df = pd.read_csv(file_path)
        print(f"  Dataset shape: {df.shape}")
        
        X, y = preprocess_shap_data(df)
        print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"  Target distribution:\n{y.value_counts()}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
        )
        
        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {accuracy * 100:.2f}%")
        
        # SHAP analysis
        print("  Computing SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[1]
        else:
            shap_values_plot = shap_values
            if len(shap_values.shape) == 3:
                shap_values_plot = shap_values[:, :, 1]
        
        # Store results
        all_results[dataset_name] = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "shap_values": shap_values_plot,
            "accuracy": accuracy
        }
        
        # Generate individual SHAP plots
        # Summary dot plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values_plot, X_test, show=False, max_display=20)
        plt.title(f"SHAP Feature Importance - {dataset_name}", fontsize=16, fontweight="bold")
        plt.xlabel("SHAP value (impact on death prediction)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_SHAP, f"shap_summary_{dataset_name.replace(' ', '_')}.jpg"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bar plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values_plot, X_test, plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP Feature Importance - {dataset_name}", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_SHAP, f"shap_bar_{dataset_name.replace(' ', '_')}.jpg"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  SHAP plots saved for {dataset_name}")
    
    # Generate comparison plot
    if all_results:
        create_shap_comparison_plot(all_results)
    else:
        print("[WARNING] No SHAP results to plot.")
    
    return all_results


def create_shap_comparison_plot(all_results):
    """
    Create comparative visualization of top features across cancer types
    """
    
    print("\nCreating SHAP comparison plot...")
    
    # Font settings
    plt.rcParams["font.family"] = "Times New Roman"
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    for ax, (cancer_name, result) in zip(axes, all_results.items()):
        X_test = result["X_test"]
        shap_values = result["shap_values"]
        
        # Mean absolute SHAP value
        importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "Feature": X_test.columns,
            "SHAP": importance
        })
        
        # Select top features
        importance_df = importance_df.sort_values("SHAP", ascending=True).tail(TOP_FEATURES_DISPLAY)
        
        # Horizontal bar plot
        ax.barh(importance_df["Feature"], importance_df["SHAP"], height=0.65)
        
        # Formatting
        ax.set_title(cancer_name, fontsize=16, fontweight="bold", pad=12)
        ax.set_xlabel("Mean |SHAP value|", fontsize=14, fontweight="bold")
        ax.set_ylabel("", fontsize=14, fontweight="bold")
        
        # Tick labels
        for label in ax.get_yticklabels():
            label.set_fontsize(14)
            label.set_fontweight("bold")
            label.set_fontname("Times New Roman")
        for label in ax.get_xticklabels():
            label.set_fontsize(11)
            label.set_fontweight("bold")
            label.set_fontname("Times New Roman")
        
        # Grid
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        
        # Spine formatting
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    # Main title
    plt.suptitle(
        "Top SHAP Features Across Cancer Types",
        fontsize=18,
        fontweight="bold",
        fontname="Times New Roman",
        y=1.02
    )
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR_SHAP, "Top_SHAP_Features_Across_Cancer_Types.jpg")
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f" Saved {out_path}")
    print("SHAP comparison complete.\n")


# ============================================================
# MAIN PIPELINE EXECUTION
# ============================================================

def main():
    """Execute complete Deep-RFCM pipeline"""
    
    print("\n" + "="*70)
    print("DEEP-RFCM COMPLETE PIPELINE")
    print("="*70)
    print("Integrated: PO-HyVA + Grad-CAM + SHAP")
    print("="*70)
    
    # Create output directories
    create_directories()
    
    # Set random seed
    set_seed(RANDOM_SEED)
    
    # ============================================================
    # PART 1: PO-HyVA - Radiomic Feature Extraction
    # ============================================================
    run_po_hyva()
    
    # ============================================================
    # PART 2: Grad-CAM - Explainable AI
    # ============================================================
    run_gradcam()
    
    # ============================================================
    # PART 3: SHAP Analysis - Feature Importance
    # ============================================================
    run_shap_analysis()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Outputs saved to:")
    print(f"  - PO-HyVA: {OUTPUT_DIR_PO_HYVA}")
    print(f"  - Grad-CAM: {OUTPUT_DIR_GRADCAM}")
    print(f"  - SHAP: {OUTPUT_DIR_SHAP}")
    print("="*70)


if __name__ == "__main__":
    main()