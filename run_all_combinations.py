r"""
Batch pipeline: trains every (encoder, decoder) combo in COMBOS below, back to
back, on the same official 78/16/16 split / augmentation / Dice+Focal loss
recipe used by every other script in this project. One bad or OOM combo does
NOT kill the batch -- each combo is wrapped in try/except and logged.

Everything gets saved under a single, consistent tree so visualize_all_models.py
can find it automatically:

    <out_dir>/
      checkpoints/
        <encoder>_<decoder>/
          best_model.pth      (state_dict + encoder/decoder names + epoch)
          history.json        (per-epoch train/val loss, IoU, Dice)
      summary.json            (test-set metrics for every combo, sorted by Dice)
      failures.json           (any combo that errored out, with the error)

Install:
    pip install -r requirements.txt

Usage:
    python run_all_combinations.py --repo_root "C:\...\DFUTissueSegNet-main"

Edit the COMBOS list below to add/remove/reorder what gets trained. Heavier
combos (anything with DPT / plain ViT / SAM-ViT) default to a smaller batch
size since they run ~120M params -- adjust if you run out of VRAM either way.

---------------------------------------------------------------------------
UPDATE: SAM ViT-B + DPT now trains successfully on the first attempt.
---------------------------------------------------------------------------
Earlier versions of this script let this combo fail with a strict
state_dict shape-mismatch error: the real SAM ViT-B checkpoint
('samvit_base_patch16.sa1b') was pretrained at a fixed 1024x1024 resolution
with non-interpolating position embeddings, and timm's own pretrained-
loading path errors out (rather than resizing) when the model is built at
any other resolution -- which every other combo in this study needs
(256x256), since that's our shared training resolution.

The fix (see load_samvit_pretrained_partial() below): build this one
encoder with encoder_weights=None (skipping timm's own broken pretrained
loading), then manually transplant every tensor from a real SAM ViT-B
checkpoint whose shape matches our 256x256 model -- i.e. everything except
the handful of resolution-dependent tensors (pos_embed, rel_pos_h,
rel_pos_w). This was verified (not just reasoned about) by reproducing the
exact same skip-list against the real checkpoint before landing on this
approach. Those specific tensors are left randomly initialized and adapt
during fine-tuning; every other pretrained weight (~168 of ~177 named
tensors) transfers normally.
"""

import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime

import cv2
import numpy as np
import torch
import albumentations as albu
from torch.utils.data import DataLoader, Dataset as BaseDataset

CLASS_NAMES = ["Background", "Fibrin", "Granulation", "Callus"]

# ============================================================================
# Edit this list to control what gets trained. Each entry:
#   (encoder_name, decoder_name, extra_model_kwargs, batch_size_override)
# extra_model_kwargs is passed straight into the smp decoder constructor
# (e.g. Swin/ViT/SAM need img_size=256 since their timm configs default to 224).
# ============================================================================
COMBOS = [
    # -- CNN baselines --
    ("resnet34",                             "Unet",   {},                 16),  # the literal, classic "U-Net"
    ("efficientnet-b3",                      "MAnet",  {},                 16),  # same encoder as the FPN run, attention decoder instead
    ("efficientnet-b3",                      "FPN",    {},                 16),  # the combo from your very first dedicated run -- now folded into the batch
    ("resnext50_32x4d",                      "FPN",    {},                 16),  # different CNN family
    ("tu-convnext_tiny",                     "MAnet",  {},                 16),  # modern CNN (ConvNeXt), attention decoder
    # -- MiT-B3 ablations: same encoder as the original pscse run, simpler decoders --
    ("mit_b3",                               "FPN",    {},                 16),
    ("mit_b3",                               "Segformer", {},              16),  # this IS the literal "SegFormer" architecture
    # -- hierarchical transformer encoders: broad decoder compatibility --
    ("tu-sam2_hiera_tiny",                   "FPN",    {},                 16),
    ("tu-sam2_hiera_tiny",                   "MAnet",  {},                 16),
    ("tu-swin_tiny_patch4_window7_224",      "MAnet",  {"img_size": 256},  16),
    ("tu-swin_tiny_patch4_window7_224",      "FPN",    {"img_size": 256},  16),
    ("tu-swin_tiny_patch4_window7_224",      "UPerNet",{"img_size": 256},  16),  # literal Swin+UPerNet pairing
    # -- plain (non-hierarchical) ViT-family encoders: DPT decoder only --
    ("tu-vit_base_patch16_224",              "DPT",    {"img_size": 256},  4),
    ("tu-samvit_base_patch16",               "DPT",    {"img_size": 256},  4),  # needs the partial-weight-transfer fix below
]

# Encoders that need the manual partial-pretrained-weight-transfer path
# instead of the standard encoder_weights="imagenet" construction, because
# their real pretrained checkpoint uses a fixed, non-interpolating input
# resolution. Add more names here if you hit the same failure mode with a
# different checkpoint.
MANUAL_PRETRAINED_ENCODERS = {
    "tu-samvit_base_patch16": "samvit_base_patch16.sa1b",
}


def load_samvit_pretrained_partial(smp_model, timm_checkpoint_name):
    """Manually loads a fixed-resolution pretrained checkpoint (e.g. real SAM
    ViT-B, trained natively at 1024x1024) into an encoder built at a
    DIFFERENT resolution (256x256 here), skipping only the tensors whose
    shape depends on input resolution (pos_embed, rel_pos_h, rel_pos_w).

    Needed because timm's own pretrained-loading path does a strict
    shape-checked load and raises a RuntimeError otherwise -- it does not
    attempt to resize/interpolate these tensors for this checkpoint family.
    """
    import timm
    print(f"  Fetching {timm_checkpoint_name} pretrained checkpoint at its "
          f"native resolution (temporary reference model, not used for training)...")
    ref_model = timm.create_model(timm_checkpoint_name, pretrained=True, num_classes=0)
    ref_state = ref_model.state_dict()
    del ref_model

    target = smp_model.encoder.model  # the real timm model inside our encoder wrapper
    target_state = target.state_dict()

    compatible, skipped = {}, []
    for k, v in ref_state.items():
        if k in target_state and target_state[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)

    target.load_state_dict(compatible, strict=False)
    print(f"  Loaded {len(compatible)}/{len(ref_state)} pretrained tensors "
          f"(skipped {len(skipped)} resolution-dependent tensors, left randomly "
          f"initialized: {skipped})")


def build_dataset_classes():
    """Returns (Dataset, get_training_augmentation, get_validation_augmentation,
    get_preprocessing) shared by every combo -- kept as a factory so nothing
    is accidentally shared/mutated across combos."""

    def get_training_augmentation():
        return albu.Compose([
            albu.OneOf([albu.HorizontalFlip(p=0.5), albu.VerticalFlip(p=0.5)], p=0.8),
            albu.OneOf([
                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0, p=0.1, border_mode=0),
                albu.ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0, p=0.1, border_mode=0),
                albu.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=0.6, border_mode=0),
                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=0.2, border_mode=0),
            ], p=0.9),
            albu.OneOf([
                albu.Perspective(p=0.2), albu.GaussNoise(p=0.2), albu.Sharpen(p=0.2),
                albu.Blur(blur_limit=3, p=0.2), albu.MotionBlur(blur_limit=3, p=0.2),
            ], p=0.5),
            albu.OneOf([
                albu.CLAHE(p=0.25),
                albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
                albu.RandomGamma(p=0.25), albu.HueSaturationValue(p=0.25),
            ], p=0.3),
        ], p=0.9)

    def get_validation_augmentation():
        return albu.Compose([])

    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype("float32")

    def get_preprocessing(preprocessing_fn):
        return albu.Compose([
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor),
        ])

    class Dataset(BaseDataset):
        def __init__(self, ids, images_dir, masks_dir, augmentation=None, preprocessing=None):
            self.ids = ids
            self.images_fps = [os.path.join(images_dir, i) for i in ids]
            self.masks_fps = [os.path.join(masks_dir, i) for i in ids]
            self.augmentation = augmentation
            self.preprocessing = preprocessing

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            image = cv2.imread(self.images_fps[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_fps[i], 0)
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]
            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image = sample["image"]
            return image, mask.astype(np.int64)

    return Dataset, get_training_augmentation, get_validation_augmentation, get_preprocessing


def read_names(txt_file, ext=".png"):
    with open(txt_file, "r") as f:
        names = f.readlines()
    names = [n.strip("\n").strip() for n in names]
    return [n + ext for n in names if n]


def train_one_combo(encoder, decoder, extra_kwargs, batch_size, args, smp, paths):
    """Trains a single (encoder, decoder) combo end to end. Returns a dict of
    test-set metrics, or raises -- caller decides how to handle failure."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 4
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Dataset, get_train_aug, get_val_aug, get_preprocessing = build_dataset_classes()

    list_IDs_train = read_names(os.path.join(paths["labeled_root"], "labeled_train_names.txt"))
    list_IDs_val = read_names(os.path.join(paths["labeled_root"], "labeled_val_names.txt"))
    list_IDs_test = read_names(os.path.join(paths["labeled_root"], "test_names.txt"))

    if encoder in MANUAL_PRETRAINED_ENCODERS:
        # Skip smp/timm's own (broken, strict) pretrained loading for this
        # encoder, then patch in every compatible pretrained tensor manually.
        model = getattr(smp, decoder)(
            encoder_name=encoder, encoder_weights=None,
            classes=n_classes, activation=None, **extra_kwargs,
        )
        load_samvit_pretrained_partial(model, MANUAL_PRETRAINED_ENCODERS[encoder])
    else:
        model = getattr(smp, decoder)(
            encoder_name=encoder, encoder_weights="imagenet",
            classes=n_classes, activation=None, **extra_kwargs,
        )
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.1f}M")

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, "imagenet")

    train_dataset = Dataset(list_IDs_train, paths["trainval_img_dir"], paths["trainval_mask_dir"],
                             augmentation=get_train_aug(), preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = Dataset(list_IDs_val, paths["trainval_img_dir"], paths["trainval_mask_dir"],
                             augmentation=get_val_aug(), preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    dice_loss_fn = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    focal_loss_fn = smp.losses.FocalLoss(mode="multiclass")

    def compute_loss(logits, target):
        return dice_loss_fn(logits, target) + focal_loss_fn(logits, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, mode="min", patience=10, min_lr=1e-6)

    model_name = f"{encoder}_{decoder}".replace("/", "-")
    checkpoint_loc = os.path.join(args.
                                  , "checkpoints", model_name)
    os.makedirs(checkpoint_loc, exist_ok=True)

    def run_epoch(loader, train):
        model.train() if train else model.eval()
        total_loss = 0.0
        tp_all, fp_all, fn_all, tn_all = [], [], [], []
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            if train:
                optimizer.zero_grad()
                logits = model(images)
                loss = compute_loss(logits, masks)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    logits = model(images)
                    loss = compute_loss(logits, masks)
            total_loss += float(loss.item()) * images.size(0)
            with torch.no_grad():
                pred_labels = torch.argmax(logits, dim=1)
                tp, fp, fn, tn = smp.metrics.get_stats(pred_labels, masks, mode="multiclass", num_classes=n_classes)
                tp_all.append(tp); fp_all.append(fp); fn_all.append(fn); tn_all.append(tn)
        tp_all, fp_all = torch.cat(tp_all), torch.cat(fp_all)
        fn_all, tn_all = torch.cat(fn_all), torch.cat(tn_all)
        iou = smp.metrics.iou_score(tp_all, fp_all, fn_all, tn_all, reduction="macro")
        dice = smp.metrics.f1_score(tp_all, fp_all, fn_all, tn_all, reduction="macro")
        return total_loss / len(loader.dataset), float(iou.item()), float(dice.item())

    best_vloss, cnt_patience, best_epoch = float("inf"), 0, 0
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": [],
               "train_dice": [], "val_dice": []}

    for epoch in range(args.epochs):
        tr_loss, tr_iou, tr_dice = run_epoch(train_loader, True)
        v_loss, v_iou, v_dice = run_epoch(valid_loader, False)
        print(f"  Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={v_loss:.4f} val_iou={v_iou:.4f}")

        for k, v in [("train_loss", tr_loss), ("val_loss", v_loss), ("train_iou", tr_iou),
                     ("val_iou", v_iou), ("train_dice", tr_dice), ("val_dice", v_dice)]:
            history[k].append(v)
        scheduler.step(v_loss)

        if v_loss < best_vloss:
            best_vloss, best_epoch, cnt_patience = v_loss, epoch, 0
            torch.save({"epoch": epoch + 1, "state_dict": model.state_dict(),
                        "encoder": encoder, "decoder": decoder},
                       os.path.join(checkpoint_loc, "best_model.pth"))
        else:
            cnt_patience += 1
        if cnt_patience >= args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    with open(os.path.join(checkpoint_loc, "history.json"), "w") as f:
        json.dump({"history": history, "best_model_epoch": best_epoch, "best_val_loss": best_vloss,
                    "model_name": model_name, "encoder": encoder, "decoder": decoder,
                    "batch_size": batch_size, "epochs_run": epoch + 1, "n_params": n_params,
                    "checkpoint_path": os.path.join(checkpoint_loc, "best_model.pth")}, f, indent=2)

    # ---- reload best checkpoint before test evaluation (not just the last epoch) ----
    ckpt = torch.load(os.path.join(checkpoint_loc, "best_model.pth"), map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ep = 1e-6
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(1, n_classes)}
    preprocessing = get_preprocessing(preprocessing_fn)
    for name in list_IDs_test:
        image_bgr = cv2.imread(os.path.join(paths["test_img_dir"], name))
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(os.path.join(paths["test_mask_dir"], name), 0)
        sample = preprocessing(image=image)
        tensor = torch.from_numpy(sample["image"]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = torch.argmax(model(tensor), dim=1).squeeze().cpu().numpy()
        for c in range(1, n_classes):
            per_class[c]["tp"] += int(np.sum((pred == c) & (gt_mask == c)))
            per_class[c]["fp"] += int(np.sum((pred == c) & (gt_mask != c)))
            per_class[c]["fn"] += int(np.sum((pred != c) & (gt_mask == c)))

    stp = sum(d["tp"] for d in per_class.values())
    sfp = sum(d["fp"] for d in per_class.values())
    sfn = sum(d["fn"] for d in per_class.values())
    overall_iou = stp / (stp + sfp + sfn + ep) * 100
    overall_dice = 2 * stp / (2 * stp + sfp + sfn + ep) * 100

    per_class_metrics = {}
    for c in range(1, n_classes):
        d = per_class[c]
        iou = d["tp"] / (d["tp"] + d["fp"] + d["fn"] + ep) * 100
        dice = 2 * d["tp"] / (2 * d["tp"] + d["fp"] + d["fn"] + ep) * 100
        per_class_metrics[CLASS_NAMES[c]] = {"iou": iou, "dice": dice}

    return {"model_name": model_name, "encoder": encoder, "decoder": decoder, "n_params": n_params,
            "best_epoch": best_epoch, "overall_iou": overall_iou, "overall_dice": overall_dice,
            "per_class": per_class_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./run_output_all_combos")
    args = parser.parse_args()

    import segmentation_models_pytorch as smp

    os.makedirs(args.out_dir, exist_ok=True)
    labeled_root = os.path.join(args.repo_root, "DFUTissue", "Labeled")
    data_root = os.path.join(labeled_root, "Padded")
    paths = {
        "labeled_root": labeled_root,
        "trainval_img_dir": os.path.join(data_root, "Images", "TrainVal"),
        "trainval_mask_dir": os.path.join(data_root, "Annotations", "TrainVal"),
        "test_img_dir": os.path.join(data_root, "Images", "Test"),
        "test_mask_dir": os.path.join(data_root, "Annotations", "Test"),
    }

    # ---- Merge with any existing summary/failures from a PREVIOUS staged run ----
    # (e.g. you commented out some COMBOS, ran a subset, then later uncommented
    # the rest and ran again -- this keeps both runs' results instead of the
    # second run's summary.json wiping out the first's).
    summary_path = os.path.join(args.out_dir, "summary.json")
    failures_path = os.path.join(args.out_dir, "failures.json")
    existing_results = json.load(open(summary_path)) if os.path.exists(summary_path) else []
    existing_failures = json.load(open(failures_path)) if os.path.exists(failures_path) else []

    results, failures = [], []
    for i, (encoder, decoder, extra_kwargs, batch_size) in enumerate(COMBOS):
        print(f"\n{'='*70}\n[{i+1}/{len(COMBOS)}] {encoder} + {decoder}\n{'='*70}")
        t0 = time.time()
        try:
            metrics = train_one_combo(encoder, decoder, extra_kwargs, batch_size, args, smp, paths)
            metrics["train_time_min"] = (time.time() - t0) / 60
            results.append(metrics)
            print(f"  Done in {metrics['train_time_min']:.1f} min. "
                  f"Overall IoU={metrics['overall_iou']:.2f} Dice={metrics['overall_dice']:.2f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            failures.append({"encoder": encoder, "decoder": decoder, "error": str(e),
                              "traceback": traceback.format_exc()})
        finally:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # New results/failures replace any prior entry for the same combo (in case
    # you re-ran something that previously failed).
    merged_results = {r["model_name"]: r for r in existing_results}
    merged_results.update({r["model_name"]: r for r in results})
    merged_failures = {f"{f_['encoder']}_{f_['decoder']}": f_ for f_ in existing_failures}
    merged_failures.update({f"{f_['encoder']}_{f_['decoder']}": f_ for f_ in failures})
    for r in results:
        merged_failures.pop(f"{r['encoder']}_{r['decoder']}", None)

    results = sorted(merged_results.values(), key=lambda r: r["overall_dice"], reverse=True)
    failures = list(merged_failures.values())

    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(failures_path, "w") as f:
        json.dump(failures, f, indent=2)

    print(f"\n\n{'='*70}\nSUMMARY ({len(results)} succeeded, {len(failures)} failed)\n{'='*70}")
    for r in results:
        print(f"  {r['model_name']:<45} Dice={r['overall_dice']:.2f}  IoU={r['overall_iou']:.2f}  "
              f"({r['n_params']/1e6:.1f}M params, {r['train_time_min']:.1f} min)")
    if failures:
        print("\nFailed combos:")
        for f_ in failures:
            print(f"  {f_['encoder']} + {f_['decoder']}: {f_['error'][:100]}")
    print(f"\nFull results: {summary_path}")


if __name__ == "__main__":
    main()
