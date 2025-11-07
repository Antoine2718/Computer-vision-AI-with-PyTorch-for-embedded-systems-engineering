import os
import math
from typing import Dict, Tuple, Optional
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state: Dict, path: str):
    torch.save(state, path)

def load_checkpoint(path: str, device: str = "cpu"):
    return torch.load(path, map_location=device)

# ---------------------------
# Dataset (example unified)
# ---------------------------
class ImageDataset(Dataset):
    def __init__(self, items: list, task: str = "classification", img_size: int = 224, augment: bool = False):
        """
        items: list of dicts like {'img_path':..., 'label':int} for classification
               or {'img_path':..., 'mask_path':...} for segmentation
        task: 'classification' or 'segmentation'
        """
        self.items = items
        self.task = task
        self.img_size = img_size
        self.augment = augment
        self._build_transforms()

    def _build_transforms(self):
        base = [
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ]
        if self.augment:
            aug = [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.3)
            ]
            self.transform = A.Compose(aug + base)
        else:
            self.transform = A.Compose(base)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = cv2.imread(it['img_path'], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.task == "classification":
            label = int(it['label'])
            data = self.transform(image=img)
            return data['image'], torch.tensor(label, dtype=torch.long)
        elif self.task == "segmentation":
            mask = cv2.imread(it['mask_path'], cv2.IMREAD_GRAYSCALE)
            data = self.transform(image=img, mask=mask)
            # mask might be single channel ints; convert to 0/1 float
            mask_tensor = torch.tensor(data['mask'], dtype=torch.long)
            return data['image'], mask_tensor
        else:
            raise ValueError("Unsupported task")

# ---------------------------
# Model: backbone + modular heads
# ---------------------------
class Backbone(nn.Module):
    def __init__(self, model_name: str = "resnet34", pretrained: bool = True, out_feats: int = 512):
        super().__init__()
        # timm provides many backbones; we'll extract global feature dimension automatically
        self.body = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.feature_info = self.body.feature_info
        # choose highest-level feature channel count
        self.out_ch = self.feature_info[-1]['num_chs']

    def forward(self, x):
        # returns list of feature maps (multi-scale) or final feature
        feats = self.body(x)
        return feats  # list of tensors

class ClassificationHead(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, dropout: float = 0.25):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, max(128, in_ch//4)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(128, in_ch//4), n_classes)
        )

    def forward(self, feats):
        x = feats[-1]  # take last feature map
        x = self.pool(x)
        return self.fc(x)

class SegmentationHeadSimple(nn.Module):
    def __init__(self, in_ch: int, n_classes: int = 1):
        super().__init__()
        # simple decoder: conv + upsampling blocks
        self.conv1 = nn.Conv2d(in_ch, in_ch//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_ch//2, in_ch//4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_ch//4, n_classes, kernel_size=1)
    def forward(self, feats):
        x = feats[-1]
        x = F.relu(self.conv1(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))
        x = F.relu(self.conv2(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))
        x = self.conv3(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))
        return x  # logits at upsampled resolution (may need further resize)

class LRMModel(nn.Module):
    def __init__(self, backbone_name: str = "resnet34", pretrained: bool = True, task: str = "classification", n_classes: int = 2):
        super().__init__()
        self.task = task
        self.backbone = Backbone(model_name=backbone_name, pretrained=pretrained)
        out_ch = self.backbone.out_ch
        if task == "classification":
            self.head = ClassificationHead(in_ch=out_ch, n_classes=n_classes)
        elif task == "segmentation":
            self.head = SegmentationHeadSimple(in_ch=out_ch, n_classes=n_classes)
        else:
            raise ValueError("Unsupported task")

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)
        return out

# ---------------------------
# Losses and metrics
# ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        loss = F.nll_loss(((1 - p) ** self.gamma) * logp, targets)
        return loss

def dice_loss_logits(logits, targets, eps=1e-6):
    # logits: (B, C, H, W) ; targets: (B, H, W) int class labels (for binary convert)
    probs = torch.softmax(logits, dim=1)
    if probs.size(1) == 1:
        probs = torch.sigmoid(logits)
        targets_f = targets.float()
        inter = (probs * targets_f).sum()
        union = probs.sum() + targets_f.sum()
        dice = (2 * inter + eps) / (union + eps)
        return 1 - dice
    else:
        # multi-class dice (mean over classes)
        C = probs.size(1)
        dice = 0
        for c in range(C):
            pc = probs[:, c]
            tc = (targets == c).float()
            inter = (pc * tc).sum()
            union = pc.sum() + tc.sum()
            dice += (2 * inter + eps) / (union + eps)
        return 1 - dice / C

# ---------------------------
# Trainer (simple, extensible)
# ---------------------------
class Trainer:
    def __init__(self, model: nn.Module, optimizer, device: str = "cuda", amp: bool = True):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.amp = amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def train_epoch(self, train_loader: DataLoader, loss_fn, scheduler=None, max_grad_norm: Optional[float]=1.0):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc="train", leave=False)
        for imgs, targets in pbar:
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.amp):
                logits = self.model(imgs)
                loss = loss_fn(logits, targets)
            self.scaler.scale(loss).backward()
            if max_grad_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if scheduler:
                scheduler.step()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({'loss': f"{running_loss/(pbar.n+1e-9):.4f}"})
        return running_loss / len(train_loader.dataset)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, loss_fn, postprocess=None):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        pbar = tqdm(val_loader, desc="val", leave=False)
        for imgs, targets in pbar:
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.amp):
                logits = self.model(imgs)
                loss = loss_fn(logits, targets)
            total_loss += loss.item() * imgs.size(0)
            # collect metrics for classification
            if self.model.task == "classification":
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                all_preds.append(preds)
                all_targets.append(targets.detach().cpu().numpy())
        metrics = {}
        if all_preds:
            preds = np.concatenate(all_preds)
            tars = np.concatenate(all_targets)
            from sklearn.metrics import accuracy_score, f1_score
            metrics['accuracy'] = accuracy_score(tars, preds)
            metrics['f1'] = f1_score(tars, preds, average='macro')
        return total_loss / len(val_loader.dataset), metrics

# ---------------------------
# Simple Grad-CAM utility for explainability (classification)
# ---------------------------
class SimpleGradCAM:
    def __init__(self, model: nn.Module, target_layer_name: str = None):
        self.model = model
        self.model.eval()
        self.activations = None
        self.grads = None
        # find a high-level conv layer if not given
        if target_layer_name is None:
            # heuristics: take last conv in backbone body
            self.target = self._find_last_conv(self.model)
        else:
            self.target = dict(self.model.named_modules())[target_layer_name]
        self.hook_handles = []
        self._register_hooks()

    def _find_last_conv(self, model):
        last = None
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                last = m
        return last

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.grads = grad_out[0].detach()
        self.hook_handles.append(self.target.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target.register_backward_hook(backward_hook))

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None, device='cpu'):
        input_tensor = input_tensor.to(device).unsqueeze(0)
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[0, class_idx]
        score.backward(retain_graph=False)
        weights = self.grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def close(self):
        for h in self.hook_handles:
            h.remove()

# ---------------------------
# Example usage
# ---------------------------
def example_workflow():
    set_seed(42)
    # 1) Fake dataset (replace with real paths)
    items_train = [{'img_path':'/path/to/img1.jpg','label':0}, {'img_path':'/path/to/img2.jpg','label':1}]
    items_val = [{'img_path':'/path/to/img3.jpg','label':0}]
    train_ds = ImageDataset(items_train, task='classification', img_size=224, augment=True)
    val_ds = ImageDataset(items_val, task='classification', img_size=224, augment=False)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # 2) Model
    model = LRMModel(backbone_name='resnet34', pretrained=True, task='classification', n_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    # 3) Trainer
    trainer = Trainer(model, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu', amp=True)
    loss_fn = FocalLoss(gamma=2.0)

    # 4) Train loop (toy)
    epochs = 3
    for ep in range(epochs):
        train_loss = trainer.train_epoch(train_loader, loss_fn, scheduler=None)
        val_loss, metrics = trainer.validate(val_loader, loss_fn)
        print(f"Epoch {ep}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} metrics={metrics}")
        # checkpoint
        save_checkpoint({'epoch': ep, 'model_state': model.state_dict(), 'opt_state': optimizer.state_dict()}, f'checkpoint_ep{ep}.pt')

    # 5) Explainability example (Grad-CAM) on a single image
    # load a sample image path
    sample_img = cv2.imread(items_val[0]['img_path'], cv2.IMREAD_COLOR)
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([A.Resize(224,224), A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])
    t = transform(image=sample_img)['image']
    cam_gen = SimpleGradCAM(model)
    cam = cam_gen.generate(t, class_idx=None, device=trainer.device)
    # overlay cam on image (for visualization)
    img_vis = (t.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    heatmap = (cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1])
    overlay = cv2.addWeighted(img_vis, 0.6, heatmap, 0.4, 0)
    cv2.imwrite('gradcam_overlay.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cam_gen.close()

if __name__ == "__main__":
    # For direct run: example_workflow()
    # In production replace example_workflow by orchestrated data loaders and configuration
    print("LRM CV module loaded. Call example_workflow() to run a demo.")
