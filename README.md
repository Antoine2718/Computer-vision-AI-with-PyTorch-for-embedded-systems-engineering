# Computer vision AI engineering using PyTorch

### ⚠️🖥️⚙️ This model has the advantage of being easy to install and relatively energy efficient, and therefore being particularly suitable for embedded systems.

This model provides a configurable pipeline for classification and segmentation. It illustrates all the blocks: data, model backbone, modular heads, training loop with amp, validation, checkpoint backup, and inference and explainability functions (Simple Grad-CAM).

Lightweight LRM for Computer Vision
- Supports: classification and segmentation (modular heads)
- Dependencies: torch, torchvision, timm, albumentations, opencv-python, numpy, tqdm


Repository purpose
Lightweight PyTorch utilities and model scaffolding for image classification and segmentation, intended for experimentation and deployment on resource-constrained/embedded platforms. Includes dataset wrapper, backbone/head abstractions (timm-compatible), training loop with AMP/gradient scaling, focal/dice losses, a simple Grad-CAM implementation, and helpers to save/load checkpoints.

⸻

Table of contents
	•	Features
	•	Quick start
	•	Installation
	•	Usage (English: full usage notice)
	•	Importing the library
	•	Dataset preparation
	•	Creating a model (classification / segmentation)
	•	Training example
	•	Validation / inference example
	•	Visualizing with Grad-CAM
	•	API reference (brief)
	•	Embedded deployment guidance
	•	Optimizations for embedded systems
	•	Advanced mathematical appendix
	•	Convolutional operator math
	•	Backpropagation for conv layers
	•	Softmax + Cross-entropy gradients
	•	Focal loss derivation and gradient
	•	Dice loss and differentiability
	•	FLOPs, parameter count and memory formulas
	•	Quantization error model & low-rank approximation
	•	Troubleshooting

⸻

Features
	•	ImageDataset — unified dataset for classification & segmentation; uses Albumentations transforms and returns tensors ready for model.
	•	Backbone — wrapper around timm backbones to extract feature maps.
	•	ClassificationHead and SegmentationHeadSimple — small task heads (adaptive pooling + fully connected for classification; lightweight decoder for segmentation).
	•	LRMModel — lightweight model composition (backbone + head) supporting both tasks.
	•	FocalLoss and dice functions — focal loss for class imbalance and Dice metric/loss for segmentation.
	•	Trainer — training/validation loops with AMP (mixed precision), gradient scaling and optional gradient clipping.
	•	SimpleGradCAM — simple Grad-CAM generator for visualization.
	•	Checkpoint helpers: save_checkpoint, load_checkpoint.
	•	example_workflow() — example showing dataset, model, trainer and Grad-CAM usage.

⸻

## Quick start
	1.	Install dependencies (see next section).
	2.	Prepare dataset as list of dicts:
	•	Classification: {'img_path': '/path/to/jpg', 'label': 0}
	•	Segmentation: {'img_path': '/path/to/jpg', 'mask_path': '/path/to/mask.png'}
	3.	Import and run example_workflow() or adapt the example code below.

⸻

Installation

Recommended to use a virtualenv/conda environment.

### with pip
```
python -m pip install -U pip
pip install torch torchvision timm albumentations opencv-python tqdm numpy
```

### If your target uses a specific CUDA version, install the matching torch wheels:
### See https://pytorch.org for correct install command.

If you plan to export to ONNX / TensorRT / TVM, install the corresponding toolchains (onnx, onnxruntime, tensorrt, tvm) in separate environments as needed.

⸻

## Usage

The following usage examples assume the repository main module is named model.py (file provided as model.py.py). Adjust import paths if different.

Importing the library

``` python
# example usage
from model import (
    ImageDataset, Backbone, ClassificationHead,
    SegmentationHeadSimple, LRMModel,
    FocalLoss, dice, Trainer, SimpleGradCAM,
    set_seed, save_checkpoint, load_checkpoint
)
```

Dataset preparation

Prepare lists of items:

``` python
# Classification items:
items_train = [
    {'img_path': '/data/cat1.jpg', 'label': 0},
    {'img_path': '/data/dog1.jpg', 'label': 1},
    # ...
]

# Segmentation items:
items_train_seg = [
    {'img_path': '/data/img1.jpg', 'mask_path': '/data/mask1.png'},
    # ...
]

train_ds = ImageDataset(items_train, task='classification', img_size=224, augment=True)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)

Creating a model (classification & segmentation)

# classification
model = LRMModel(backbone_name='resnet34', pretrained=True, task='classification', n_classes=2)

# segmentation
model_seg = LRMModel(backbone_name='resnet34', pretrained=True, task='segmentation', n_classes=1)
```

LRMModel composes a Backbone (timm compatible) with either a ClassificationHead or SegmentationHeadSimple.

Training example
``` python

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
trainer = Trainer(model, optimizer, device=device, amp=True)
loss_fn = FocalLoss(gamma=2.0)  # helps handle class imbalance

epochs = 10
for ep in range(epochs):
    train_loss = trainer.train_epoch(train_loader, loss_fn, scheduler=None, max_grad_norm=1.0)
    val_loss, metrics = trainer.validate(val_loader, loss_fn)
    print(f"Epoch {ep}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, metrics={metrics}")
    save_checkpoint({'epoch': ep, 'model_state': model.state_dict(), 'opt_state': optimizer.state_dict()}, f'checkpoint_ep{ep}.pt')
```

Notes:
	•	Trainer uses torch.cuda.amp by default when amp=True.
	•	Gradient clipping via max_grad_norm is supported.
	•	For segmentation, use a combination of Dice loss + BCE (or use provided dice metric as loss variant).

Validation / inference example
``` python
model.eval()
with torch.no_grad():
    imgs, targets = next(iter(val_loader))
    imgs = imgs.to(device)
    logits = model(imgs)
    # for classification:
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    # for segmentation:
    # logits shape: (B, C, H, W) -> apply appropriate activation (sigmoid for binary, softmax for multiclass)

Visualizing with Grad-CAM

# given an input tensor `t` of shape (C, H, W)
cam_gen = SimpleGradCAM(model)
cam = cam_gen.generate(t, class_idx=None, device=device)  # returns heatmap HxW normalized [0,1]
cam_gen.close()
```

The example workflow in the source demonstrates reading an image, creating a transform (albumentations), generating a cam and saving overlay image.

⸻

### API reference (brief)

Only listing the primary public components — see source for exact signatures and optional args.

	•	set_seed(seed: int = 42) — deterministic seeds for reproducibility.
	•	save_checkpoint(state: Dict, path: str) — save checkpoint dict.
	•	load_checkpoint(path: str, device: str = "cpu") -> dict — load checkpoint and map to device.
	•	ImageDataset(items: list, task='classification', img_size=224, augment=False) — dataset wrapper.
	•	Backbone(model_name: str='resnet34', pretrained=True, out_feats: int=512) — timm backbone wrapper.
	•	ClassificationHead(in_ch: int, n_classes: int, dropout: float = 0.25) — classification head.
	•	SegmentationHeadSimple(in_ch: int, n_classes: int = 1) — small segmentation decoder.
	•	LRMModel(backbone_name: str, pretrained: bool, task: str, n_classes: int) — top-level model combining backbone and head.
	•	FocalLoss(gamma: float=2.0, alpha: Optional[float] = None) — focal loss module.
	•	dice(preds, targets, eps=1e-7) — dice metric / loss helper.
	•	Trainer(model, optimizer, device='cuda', amp=True) — training/validation loop.
	•	train_epoch(train_loader, loss_fn, scheduler=None, max_grad_norm: Optional[float]=1.0)
	•	validate(val_loader, loss_fn, postprocess=None)
	•	SimpleGradCAM(model, target_layer_name: Optional[str] = None)
	•	generate(input_tensor: torch.Tensor, class_idx: Optional[int] = None, device='cpu') -> np.ndarray
	•	close() to remove hooks.

⸻

## Embedded deployment guidance

This repo is a research/engineering starting point. For embedded deployment:
	1.	Profile: measure latency and memory on target (Raspberry Pi, Jetson, microcontrollers with ML accelerators).
	2.	Model selection: prefer MobileNetV3/EdgeTPU-friendly nets, EfficientNet-lite, or quantization-friendly ResNet variants. Use timm to select smaller backbones (e.g., mobilenetv3_small_100).
	3.	Export:
	•	Convert to TorchScript: torch.jit.trace or torch.jit.script.
	•	OR export to ONNX: torch.onnx.export (ensure dynamic axes as necessary).
	•	For NVIDIA Jetson/TensorRT: export ONNX -> TensorRT.
	•	For microcontrollers / Coral Edge TPU: use TFLite conversion (requires re-export and possibly architecture changes).
	4.	Quantize:
	•	Post-Training Static Quantization (PTQ) or Quantization-Aware Training (QAT) to reach INT8 accuracy.
	5.	Prune / Distill:
	•	Apply structured pruning (filters/channels) for speed and memory gains.
	•	Knowledge distillation: train a small “student” using large “teacher” predictions.

⸻

## Optimizations for embedded systems
	•	Memory:
	•	Reduce batch size to 1 for inference.
	•	Use in-place ops (ReLU inplace=True) but check autograd compatibility.
	•	Use torch.utils.checkpoint (gradient checkpointing) only during training to reduce memory.
	•	Compute:
	•	Replace nn.Conv2d with depthwise-separable convolutions.
	•	Use grouped convolutions if appropriate.
	•	Prefer lower-precision (FP16/INT8) arithmetic on supported hardware.
	•	IO and pre/post processing:
	•	Move image preprocessing into C++/hardware acceleration when possible.
	•	Use fused kernels for activation + batchnorm when supported.
	•	Framework choices:
	•	For Jetson: TensorRT.
	•	For general CPU: ONNX Runtime with MKL/oneDNN.
	•	For microcontrollers: TFLite Micro, or vendor SDKs.

⸻

### Advanced mathematical appendix

This section gives the deep mathematical intuition and derivations used by the model and loss functions. Useful for engineers aiming to tune algorithms tightly for embedded constraints.

Convolutional operator math

Discrete 2D convolution used in CNNs:

For input feature map $$X \in \mathbb{R}^{C_{in}\times H\times W}$$, filter $$W \in \mathbb{R}^{C_{out}\times C_{in}\times K_h \times K_w}$$, output Y \in \mathbb{R}^{C_{out}\times H’\times W’}:$$

$$Y_{o,i,j} = \sum_{c=0}^{C_{in}-1}\sum_{u=0}^{K_h-1}\sum_{v=0}^{K_w-1} W_{o,c,u,v}\; X_{c,\, i\cdot s + u - p_h,\, j\cdot s + v - p_w}$$

where s is stride and p_h,p_w are padding. The operation is linear and can be seen as matrix multiplication after im2col:

$$\operatorname{vec}(Y) = W_{\text{mat}} \cdot \operatorname{im2col}(X)$$

This viewpoint is useful for low-rank approximations and SVD-based compression.

Backpropagation for conv layers

Let the loss be L. The gradient w.r.t. filter W is convolution of input with output-gradients:

$$\frac{\partial L}{\partial W_{o,c,u,v}} = \sum_{i,j} \frac{\partial L}{\partial Y_{o,i,j}} \cdot X_{c,\, i\cdot s + u - p_h,\, j\cdot s + v - p_w}$$

and w.r.t. input:

$$\frac{\partial L}{\partial X_{c,x,y}} = \sum_{o}\sum_{u,v} \frac{\partial L}{\partial Y_{o,i,j}} \cdot W_{o,c,u’,v’}$$

with index arithmetic for correct flipping/reverse.

Softmax + Cross-entropy gradients

Given logits $\mathbf{z}\in\mathbb{R}^C$, softmax

$$\sigma(\mathbf{z})i = \frac{\exp(z_i)}{\sum{j}\exp(z_j)}$$

Cross-entropy for one-hot target y is $$\ell = -\sum_i y_i \log \sigma(\mathbf{z})i = -\log \sigma(\mathbf{z}){t} $$ for target class t. Gradient w.r.t. logits:

$$\frac{\partial \ell}{\partial z_i} = \sigma(\mathbf{z})_i - y_i$$

This compact form is the reason numerically-stable fused softmax-crossentropy kernels are important on embedded devices.

Focal loss derivation and gradient

Focal loss (for classification, binary/multiclass variant) modifies cross-entropy to focus on hard examples:

For binary case with probability p for correct class:

$$\text{FL}(p) = -\alpha (1-p)^\gamma \log p$$

Gradient w.r.t. logit z (where $p = \sigma(z)$):

$$ \frac{d \text{FL}}{dz} = -\alpha \left[ \gamma (1-p)^{\gamma-1}(-\tfrac{dp}{dz}) \log p + (1-p)^\gamma \frac{1}{p}\frac{dp}{dz} \right] $$

Using $\frac{dp}{dz} = p(1-p)$, we simplify to a form used in implementations. The multiplicative term $(1-p)^\gamma$ downweights easy examples (where p\rightarrow 1).

The focal-modulation effectively rescales the gradient magnitude by $(1-p)^\gamma$, changing the learning dynamics to prioritize hard examples — valuable when class imbalance is severe.

Dice loss and differentiability

Dice coefficient for two sets (or soft predictions) for class c:

$$\text{Dice}c = \frac{2 \sum{i} p_{i,c} g_{i,c} + \epsilon}{\sum_{i} p_{i,c} + \sum_{i} g_{i,c} + \epsilon}$$

Dice loss often written as 1 - \text{Dice}. Differentiable if p_{i,c} are soft predictions (sigmoid/softmax outputs). Gradient requires quotient rule:

$$\frac{\partial \text{Dice}}{\partial p_{k}} = \frac{2 g_k(\sum_i p_i + \sum_i g_i) - 2 \sum_i p_i g_i}{(\sum_i p_i + \sum_i g_i)^2}$$

(where indices and class dims are expanded appropriately). Numerically, add \epsilon to denominator.

FLOPs, parameter count and memory formulas
	•	Parameters for a conv layer: \#\text{params} = C_{out}\cdot C_{in}\cdot K_h \cdot K_w \; (+ C_{out}\ \text{bias}).
	•	Multiply-accumulate FLOPs (approx): for convolution output H’\times W’:
$$\text{FLOPs} \approx 2 \cdot H’ \cdot W’ \cdot C_{out} \cdot C_{in} \cdot K_h \cdot K_w$$
(factor 2 counts mul+add; some authors count only multiplications).
	•	Activation memory during inference: activation footprint per layer = B \cdot C \cdot H \cdot W \cdot \text{bytes_per_element}.
	•	For embedded: aim to minimize activation footprint (often dominates), and work to reduce max activation size across layers.

Quantization error model & low-rank approximation

Uniform symmetric quantization mapping real value x to integer q with scale s:

$$ q = \text{round}\left(\frac{x}{s}\right),\quad \hat{x} = s q$$

Quantization error e=x-\hat{x} can be modeled as additive noise with variance \sigma_e^2 \approx s^2/12 (for uniformly-distributed rounding errors). The effect on downstream output can be propagated linearly using the Jacobian J of the network: output covariance approx J \Sigma_e J^\top.

Low-rank approximation: represent large weight matrix W\in\mathbb{R}^{m\times n} by truncated SVD:

$$ W \approx U_k \Sigma_k V_k^\top, \quad k \ll \min(m,n) $$

This reduces parameters and FLOPs: O(k(m+n)) vs O(mn). For conv kernels, flattening spatial and input dims yields matrices amenable to SVD or tensor decompositions (CP/Tucker).

⸻

Troubleshooting
	•	CUDA OOM: reduce batch size, use amp=True, or use smaller image size.
	•	Poor convergence: try learning rate schedule (Cosine / OneCycle), tune weight_decay, use FocalLoss if class imbalance severe.
	•	Grad-CAM no activation: ensure target layer is a conv layer and model is eval() when generating activations; hooks are attached to the last conv found by SimpleGradCAM.
	•	ONNX export fails: switch torch.jit.trace if the model uses non-traceable control flow, or refactor non-tensor ops out of forward.
