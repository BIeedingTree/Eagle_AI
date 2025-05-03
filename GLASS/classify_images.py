import os
import shutil
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from glass import GLASS
from backbones import load as load_backbone

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_WEIGHTS = '/shared/ssd_30T/NO_WZ/DroneModel/GLASS/results/models/backbone_0/mvtec_thermal_image/ckpt_best_9.pth'
INPUT_DIR    = '/shared/ssd_30T/NO_WZ/DroneModel/GLASS/datasets/mvtec/classfications/input_images'
NORMAL_DIR   = '/shared/ssd_30T/NO_WZ/DroneModel/GLASS/datasets/mvtec/classifications/output/normal'
ABNORMAL_DIR = '/shared/ssd_30T/NO_WZ/DroneModel/GLASS/datasets/mvtec/classifications/output/abnormal'
THRESHOLD    = 0.9832839

os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(ABNORMAL_DIR, exist_ok=True)

inference_transform = transforms.Compose([
    transforms.Resize(288),
    transforms.CenterCrop((288,288)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

# ---- load model ----
backbone = load_backbone("wideresnet50")
glass = GLASS(device=DEVICE)

glass.load(
    backbone=backbone,
    layers_to_extract_from=['layer2','layer3'],
    device=DEVICE,
    input_shape=(3,288,288),
    pretrain_embed_dimension=1536,
    target_embed_dimension=1536,
    meta_epochs=0, eval_epochs=1,
    dsc_layers=2, dsc_hidden=1024,
    dsc_margin=0.5, train_backbone=False,
    pre_proj=1, mining=1, noise=0.015,
    radius=0.75, p=0.5, lr=1e-4,
    svd=0, step=20, limit=392
)

ckpt = torch.load(MODEL_WEIGHTS, map_location=DEVICE)

if isinstance(ckpt, dict) and 'discriminator' in ckpt:
    glass.discriminator.load_state_dict(ckpt['discriminator'])
    if hasattr(glass, 'pre_projection'):
        glass.pre_projection.load_state_dict(ckpt['pre_projection'])
else:
    glass.load_state_dict(ckpt)
glass.to(DEVICE).eval()

# ---- Grad-CAM helper ----
def generate_gradcam(x, model, target_layer):
    fmap, grads = [], []

    def fh(m, i, o): fmap.append(o)
    def bh(m, gi, go): grads.append(go[0])
    h1 = target_layer.register_forward_hook(fh)
    h2 = target_layer.register_backward_hook(bh)

    # Enable gradients on input
    x.requires_grad = True

    out_embeds, _ = model._embed(x, evaluation=False)  # disable eval mode to preserve autograd
    emb = out_embeds[0]

    # Enable gradients on embedding explicitly
    emb.requires_grad_(True)

    if hasattr(model, 'pre_projection'):
        emb = model.pre_projection(emb)

    scores = model.discriminator(emb.unsqueeze(0) if emb.dim() == 1 else emb)
    score = scores.max()

    model.zero_grad()
    score.backward(retain_graph=True)

    fmap, grads = fmap[0].detach(), grads[0].detach()
    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = F.relu((weights * fmap).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
    hmap = cam.squeeze().cpu()
    hmap /= hmap.max()

    h1.remove()
    h2.remove()
    return hmap.numpy()

# @torch.no_grad()
def classify_and_move(img_path):
    img = Image.open(img_path).convert('RGB')
    t   = inference_transform(img)
    x   = t.unsqueeze(0).to(DEVICE)

    # global score (can use no_grad)
    with torch.no_grad():
        embeds, _ = glass._embed(x, evaluation=True)
        emb       = embeds[0]
        if hasattr(glass, 'pre_projection'):
            emb = glass.pre_projection(emb)
        gmap = glass.discriminator(emb.unsqueeze(0) if emb.dim() == 1 else emb)
        global_score = gmap.max().item()

    # Grad-CAM heatmap (needs grad!)
    heatmap = generate_gradcam(x, glass, glass.backbone.layer3)
    hm_score = float(heatmap.max())

    final_score = hm_score
    dst = NORMAL_DIR if final_score < THRESHOLD else ABNORMAL_DIR
    shutil.move(img_path, os.path.join(dst, os.path.basename(img_path)))
    print(f"{os.path.basename(img_path)} | global={global_score:.3f} heatmap={final_score:.3f}")

# ---- run through folder ----
for fn in sorted(os.listdir(INPUT_DIR)):
    if fn.lower().endswith(('.png','.jpg','.jpeg')):
        classify_and_move(os.path.join(INPUT_DIR, fn))




# import os
# import shutil
# import torch
# from torchvision import transforms
# from PIL import Image

# from glass import GLASS 
# from backbones import load as load_backbone

# # Configuration
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MODEL_WEIGHTS = '/shared/ssd_30T/NO_WZ/DroneModel/GLASS/results/models/backbone_0/mvtec_thermal_image/ckpt_best_9.pth'
# INPUT_DIR     = '/shared/ssd_30T/NO_WZ/DroneModel/GLASS/datasets/mvtec/classfications/input_images'
# NORMAL_DIR    = '/shared/ssd_30T/NO_WZ/DroneModel/GLASS/datasets/mvtec/classfications/output/normal'
# ABNORMAL_DIR  = '/shared/ssd_30T/NO_WZ/DroneModel/GLASS/datasets/mvtec/classfications/output/abnormal'
# THRESHOLD     = 0.9832839  # anomaly score threshold; adjust based on validation

# # Create output directories if they don't exist
# os.makedirs(NORMAL_DIR, exist_ok=True)
# os.makedirs(ABNORMAL_DIR, exist_ok=True)

# # Inference transform (deterministic parts of training pipeline)
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD  = [0.229, 0.224, 0.225]

# inference_transform = transforms.Compose([
#     transforms.Resize(288),                  # same as --resize
#     transforms.CenterCrop((288, 288)),       # same as --imagesize
#     transforms.ToTensor(),
#     transforms.Normalize(mean=IMAGENET_MEAN,
#                          std=IMAGENET_STD),
# ])

# # Initialize and load model
# backbone = load_backbone("wideresnet50")
# glass_model = GLASS(device=DEVICE)
# glass_model.load(
#     backbone=backbone,
#     layers_to_extract_from=['layer2', 'layer3'],  # same layers used in training
#     device=DEVICE,
#     input_shape=(3, 288, 288),
#     pretrain_embed_dimension=1536,
#     target_embed_dimension=1536,
#     meta_epochs=0, eval_epochs=1,
#     dsc_layers=2, dsc_hidden=1024,
#     dsc_margin=0.5, train_backbone=False,
#     pre_proj=1, mining=1, noise=0.015,
#     radius=0.75, p=0.5, lr=1e-4,
#     svd=0, step=20, limit=392
# )
# # Load the saved weights
# checkpoint = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
# if isinstance(checkpoint, dict) and 'discriminator' in checkpoint:
#     glass_model.discriminator.load_state_dict(checkpoint['discriminator'])
#     if hasattr(glass_model, 'pre_projection'):
#         glass_model.pre_projection.load_state_dict(checkpoint['pre_projection'])
# else:
#     glass_model.load_state_dict(checkpoint)
# glass_model.to(DEVICE)
# glass_model.eval()

# # Utility: predict anomaly score for a single image
# @torch.no_grad()
# def predict_image(img_path):
#     img = Image.open(img_path).convert('RGB')
#     t = inference_transform(img)
#     print(f"[DEBUG] {os.path.basename(img_path)} -> {t.shape}")  # should be (3,288,288)
#     x = t.unsqueeze(0).to(DEVICE)

#     # Get raw embeddings
#     embeds_list, _ = glass_model._embed(x, evaluation=True)
#     embed = embeds_list[0]
#     print(f"[DEBUG] raw embed shape: {embed.shape}")

#     # Apply projection if present
#     if hasattr(glass_model, 'pre_projection'):
#         embed = glass_model.pre_projection(embed)
#         print(f"[DEBUG] after pre_projection shape: {embed.shape}")

#     # Finally, feed into discriminator (ensure 2D input)
#     if embed.dim() == 1:
#         embed_in = embed.unsqueeze(0)
#     else:
#         embed_in = embed
#     try:
#         scores = glass_model.discriminator(embed_in)
#     except Exception as e:
#         print(f"[ERROR] discriminator input shape: {embed_in.shape}")
#         raise

#     # Expect scores shape [batch_size, 1] or [num_patches,1]
#     anomaly_score = scores.max().item()
#     return anomaly_score

# # Process folder
# for fname in os.listdir(INPUT_DIR):
#     if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
#         continue
#     fpath = os.path.join(INPUT_DIR, fname)
#     try:
#         score = predict_image(fpath)
#         dest = NORMAL_DIR if score < THRESHOLD else ABNORMAL_DIR
#         shutil.move(fpath, os.path.join(dest, fname))
#         print(f"{fname}: score={score:.3f} -> {'normal' if score < THRESHOLD else 'abnormal'}")
#     except Exception as e:
#         print(f"Failed to process {fname}: {e}")
