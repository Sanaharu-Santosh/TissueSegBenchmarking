r"""
Unified results visualizer -- covers ALL models trained across this project:
  1. MiT-B3 + pscse U-Net       (run_supervised.py,        vendored fork checkpoint)
  2. Mask2Former                (run_mask2former.py,        HF save_pretrained dir)
  3. EfficientNet-B3 + FPN      (run_efficientnetb3_fpn.py, real-smp checkpoint)
  4+. Anything from run_all_combinations.py (real-smp checkpoints, auto-discovered)

For each model found it:
  - plots train/val loss + IoU curves from history.json, IF that file exists
    (the very first MiT-B3+pscse run predates the history-logging fix and its
    history.json was never written -- this is called out explicitly in the
    report rather than silently skipped)
  - runs inference on all 16 held-out test images using the correct
    architecture for that checkpoint's type
  - saves an Original | Ground Truth | Prediction comparison image per test image
  - computes per-class + overall IoU/Dice/Precision/Recall
  - writes ONE combined report.html: a leaderboard table across all models,
    then a per-model section with its curves (if available), metrics, and
    all 16 predictions.

Usage:
    python visualize_all_models.py --repo_root "C:\...\DFUTissueSegNet-main"

By default it auto-discovers checkpoints under these folders (edit SEARCH_DIRS
below if you used different --out_dir values):
    ./run_output                   (MiT-B3 + pscse, run_supervised.py)
    ./run_output_mask2former       (Mask2Former, run_mask2former.py)
    ./run_output_efficientnetb3_fpn (EfficientNet-B3 + FPN)
    ./run_output_smp               (ad-hoc run_smp_model.py runs)
    ./run_output_all_combos        (run_all_combinations.py batch runs)
"""

import os
import sys
import json
import glob
import argparse

import cv2
import numpy as np
import torch
import albumentations as albu

CLASS_NAMES = ["Background", "Fibrin", "Granulation", "Callus"]
PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

SEARCH_DIRS = [
    "./run_output",
    "./run_output_mask2former",
    "./run_output_efficientnetb3_fpn",
    "./run_output_smp",
    "./run_output_all_combos",
]


def colorize(mask, palette=PALETTE):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(palette):
        out[mask == cls_idx] = color
    return out


def read_names(txt_file, ext=".png"):
    with open(txt_file, "r") as f:
        names = f.readlines()
    names = [n.strip("\n").strip() for n in names]
    return [n + ext for n in names if n]


def discover_checkpoints(search_dirs):
    """Returns a list of dicts: {display_name, ckpt_type, path, history_path}.
    ckpt_type is one of: 'vendored_pscse', 'mask2former', 'smp_generic'."""
    found = []
    for base in search_dirs:
        ckpt_root = os.path.join(base, "checkpoints")
        if not os.path.isdir(ckpt_root):
            continue
        for run_dir in sorted(glob.glob(os.path.join(ckpt_root, "*"))):
            if not os.path.isdir(run_dir):
                continue
            run_name = os.path.basename(run_dir)

            # Mask2Former: HF save_pretrained directory (config.json + weights)
            if os.path.exists(os.path.join(run_dir, "config.json")):
                found.append({
                    "display_name": run_name, "ckpt_type": "mask2former",
                    "path": run_dir,
                    "history_path": os.path.join(run_dir, "history.json"),
                })
                continue

            pth_path = os.path.join(run_dir, "best_model.pth")
            if not os.path.exists(pth_path):
                continue
            ckpt = torch.load(pth_path, map_location="cpu")
            if "encoder" in ckpt and "decoder" in ckpt:
                # real segmentation-models-pytorch checkpoint (run_smp_model.py,
                # run_efficientnetb3_fpn.py, run_all_combinations.py all save this way)
                found.append({
                    "display_name": run_name, "ckpt_type": "smp_generic",
                    "path": pth_path, "encoder": ckpt["encoder"], "decoder": ckpt["decoder"],
                    "history_path": os.path.join(run_dir, "history.json"),
                })
            else:
                # no encoder/decoder keys -> the vendored fork's MiT-B3+pscse checkpoint
                found.append({
                    "display_name": run_name, "ckpt_type": "vendored_pscse",
                    "path": pth_path,
                    "history_path": os.path.join(run_dir, "history.json"),
                })
    return found


def _reset_smp_module_cache():
    """The vendored fork and the real pip package both register themselves as
    top-level module 'segmentation_models_pytorch'. Whichever gets imported
    FIRST in this process stays cached in sys.modules and silently poisons
    every later `import segmentation_models_pytorch` -- for a DIFFERENT
    checkpoint type -- with the wrong package. Must purge the cache (and any
    vendored-fork paths we added) before switching between checkpoint types."""
    for mod_name in list(sys.modules):
        if mod_name == "segmentation_models_pytorch" or mod_name.startswith("segmentation_models_pytorch."):
            del sys.modules[mod_name]
    sys.path[:] = [p for p in sys.path
                   if os.path.join("Codes", "segmentation_models_pytorch") not in p
                   and not p.rstrip("/\\").endswith("Codes")]


def build_model(entry, repo_root, device):
    """Returns (predict_fn, preprocess_fn)."""
    _reset_smp_module_cache()

    if entry["ckpt_type"] == "vendored_pscse":
        codes_dir = os.path.join(repo_root, "Codes")
        sys.path.insert(0, codes_dir)
        sys.path.insert(0, os.path.join(codes_dir, "segmentation_models_pytorch", "losses"))
        import segmentation_models_pytorch as smp_vendored
        model = smp_vendored.Unet(encoder_name="mit_b3", encoder_weights=None,
                                   classes=4, activation="sigmoid", decoder_attention_type="pscse")
        ckpt = torch.load(entry["path"], map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        preprocessing_fn = smp_vendored.encoders.get_preprocessing_fn("mit_b3", "imagenet")

        def to_tensor(x, **kwargs):
            return x.transpose(2, 0, 1).astype("float32")
        prep = albu.Compose([albu.Lambda(image=preprocessing_fn), albu.Lambda(image=to_tensor)])

        def preprocess(image_rgb):
            return torch.from_numpy(prep(image=image_rgb)["image"]).unsqueeze(0).to(device)

        def predict(tensor):
            with torch.no_grad():
                logits = model(tensor)
                return torch.argmax(logits, dim=1).squeeze().cpu().numpy()

        return predict, preprocess

    elif entry["ckpt_type"] == "smp_generic":
        import segmentation_models_pytorch as smp  # the REAL pip package, not the vendored fork
        extra_kwargs = {}
        if any(k in entry["encoder"] for k in ["swin", "vit"]):
            extra_kwargs["img_size"] = 256
        try:
            model = getattr(smp, entry["decoder"])(
                encoder_name=entry["encoder"], encoder_weights=None, classes=4,
                activation=None, **extra_kwargs)
        except TypeError:
            model = getattr(smp, entry["decoder"])(
                encoder_name=entry["encoder"], encoder_weights=None, classes=4, activation=None)
        ckpt = torch.load(entry["path"], map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        preprocessing_fn = smp.encoders.get_preprocessing_fn(entry["encoder"], "imagenet")

        def to_tensor(x, **kwargs):
            return x.transpose(2, 0, 1).astype("float32")
        prep = albu.Compose([albu.Lambda(image=preprocessing_fn), albu.Lambda(image=to_tensor)])

        def preprocess(image_rgb):
            return torch.from_numpy(prep(image=image_rgb)["image"]).unsqueeze(0).to(device)

        def predict(tensor):
            with torch.no_grad():
                logits = model(tensor)
                return torch.argmax(logits, dim=1).squeeze().cpu().numpy()

        return predict, preprocess

    elif entry["ckpt_type"] == "mask2former":
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
        model = Mask2FormerForUniversalSegmentation.from_pretrained(entry["path"]).to(device).eval()
        processor = Mask2FormerImageProcessor.from_pretrained(entry["path"])

        def preprocess(image_rgb):
            return processor(images=[image_rgb], return_tensors="pt")

        def predict(inputs):
            with torch.no_grad():
                outputs = model(pixel_values=inputs["pixel_values"].to(device),
                                 pixel_mask=inputs["pixel_mask"].to(device))
            pred = processor.post_process_semantic_segmentation(outputs, target_sizes=[(256, 256)])[0]
            return pred.cpu().numpy()

        return predict, preprocess

    raise ValueError(f"Unknown checkpoint type: {entry['ckpt_type']}")


def evaluate_and_render(entry, repo_root, report_dir, device):
    labeled_root = os.path.join(repo_root, "DFUTissue", "Labeled")
    data_root = os.path.join(labeled_root, "Padded")
    test_img_dir = os.path.join(data_root, "Images", "Test")
    test_mask_dir = os.path.join(data_root, "Annotations", "Test")
    list_IDs_test = read_names(os.path.join(labeled_root, "test_names.txt"))

    predict, preprocess = build_model(entry, repo_root, device)

    preds_dir = os.path.join(report_dir, "predictions", entry["display_name"])
    os.makedirs(preds_dir, exist_ok=True)

    ep = 1e-6
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(1, 4)}

    for name in list_IDs_test:
        image_bgr = cv2.imread(os.path.join(test_img_dir, name))
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(os.path.join(test_mask_dir, name), 0)

        inp = preprocess(image)
        pred = predict(inp)
        if pred.shape != gt_mask.shape:
            pred = cv2.resize(pred.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

        for c in range(1, 4):
            per_class[c]["tp"] += int(np.sum((pred == c) & (gt_mask == c)))
            per_class[c]["fp"] += int(np.sum((pred == c) & (gt_mask != c)))
            per_class[c]["fn"] += int(np.sum((pred != c) & (gt_mask == c)))

        gt_color, pred_color = colorize(gt_mask), colorize(pred)
        comparison = np.concatenate([image, gt_color, pred_color], axis=1)
        cv2.imwrite(os.path.join(preds_dir, name), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    stp = sum(d["tp"] for d in per_class.values())
    sfp = sum(d["fp"] for d in per_class.values())
    sfn = sum(d["fn"] for d in per_class.values())
    overall = {
        "iou": stp / (stp + sfp + sfn + ep) * 100,
        "dice": 2 * stp / (2 * stp + sfp + sfn + ep) * 100,
        "precision": stp / (stp + sfp + ep) * 100,
        "recall": stp / (stp + sfn + ep) * 100,
    }
    per_class_metrics = {}
    for c in range(1, 4):
        d = per_class[c]
        per_class_metrics[CLASS_NAMES[c]] = {
            "iou": d["tp"] / (d["tp"] + d["fp"] + d["fn"] + ep) * 100,
            "dice": 2 * d["tp"] / (2 * d["tp"] + d["fp"] + d["fn"] + ep) * 100,
            "precision": d["tp"] / (d["tp"] + d["fp"] + ep) * 100,
            "recall": d["tp"] / (d["tp"] + d["fn"] + ep) * 100,
        }

    history = None
    if os.path.exists(entry["history_path"]):
        with open(entry["history_path"]) as f:
            history = json.load(f).get("history")

    return {"overall": overall, "per_class": per_class_metrics, "history": history,
            "image_names": list_IDs_test, "preds_subdir": os.path.join("predictions", entry["display_name"])}


def build_html_report(report_dir, model_results):
    leaderboard_rows = "".join(
        f"<tr><td>{name}</td><td>{r['overall']['iou']:.2f}</td><td>{r['overall']['dice']:.2f}</td>"
        f"<td>{r['overall']['precision']:.2f}</td><td>{r['overall']['recall']:.2f}</td>"
        f"<td><a href='#model-{i}'>jump to section</a></td></tr>"
        for i, (name, r) in enumerate(model_results)
    )

    sections = []
    chart_scripts = []
    for i, (name, r) in enumerate(model_results):
        per_class_rows = "".join(
            f"<tr><td>{cname}</td><td>{m['iou']:.2f}</td><td>{m['dice']:.2f}</td>"
            f"<td>{m['precision']:.2f}</td><td>{m['recall']:.2f}</td></tr>"
            for cname, m in r["per_class"].items()
        )
        o = r["overall"]

        if r["history"]:
            chart_html = (f'<canvas id="loss{i}" height="80"></canvas>'
                          f'<canvas id="iou{i}" height="80"></canvas>')
            h = r["history"]
            chart_scripts.append(f"""
new Chart(document.getElementById('loss{i}'), {{ type: 'line',
  data: {{ labels: Array.from({{length: {len(h['train_loss'])}}}, (_, x) => x+1),
    datasets: [
      {{ label: 'Train Loss', data: {h['train_loss']}, borderColor: '#e74c3c', fill: false, pointRadius: 0 }},
      {{ label: 'Val Loss', data: {h['val_loss']}, borderColor: '#3498db', fill: false, pointRadius: 0 }}
    ]}},
  options: {{ plugins: {{ title: {{ display: true, text: '{name} — Loss' }} }} }} }});
new Chart(document.getElementById('iou{i}'), {{ type: 'line',
  data: {{ labels: Array.from({{length: {len(h['train_iou'])}}}, (_, x) => x+1),
    datasets: [
      {{ label: 'Train IoU', data: {h['train_iou']}, borderColor: '#e74c3c', fill: false, pointRadius: 0 }},
      {{ label: 'Val IoU', data: {h['val_iou']}, borderColor: '#3498db', fill: false, pointRadius: 0 }}
    ]}},
  options: {{ plugins: {{ title: {{ display: true, text: '{name} — IoU' }} }} }} }});
""")
        else:
            chart_html = ('<p style="color:#a00"><em>No training-curve data available for this run '
                          '(history.json was not found next to its checkpoint).</em></p>')

        preds_html = "".join(
            f'<div class="pred-row"><p>{img_name}</p><img src="{r["preds_subdir"]}/{img_name}" /></div>'
            for img_name in r["image_names"]
        )

        sections.append(f"""
<div class="model-section" id="model-{i}">
<h2>{i+1}. {name}</h2>
<h3>Training curves</h3>
{chart_html}
<h3>Test-set metrics (16 held-out images)</h3>
<table>
<tr><th>Class</th><th>IoU</th><th>Dice</th><th>Precision</th><th>Recall</th></tr>
<tr style="font-weight:bold"><td>Overall</td><td>{o['iou']:.2f}</td><td>{o['dice']:.2f}</td>
    <td>{o['precision']:.2f}</td><td>{o['recall']:.2f}</td></tr>
{per_class_rows}
</table>
<h3>Predictions (Original | Ground Truth | Prediction)</h3>
{preds_html}
</div>
""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>DFU Tissue Segmentation — All Models</title>
<style>
  body {{ font-family: sans-serif; margin: 30px; background: #fafafa; color: #222; }}
  .legend span {{ display: inline-block; width: 14px; height: 14px; margin-right: 6px; vertical-align: middle; }}
  table {{ border-collapse: collapse; margin: 15px 0; }}
  td, th {{ border: 1px solid #ccc; padding: 6px 12px; text-align: center; }}
  .pred-row {{ margin-bottom: 25px; }}
  .pred-row img {{ max-width: 100%; border: 1px solid #ccc; }}
  .pred-row p {{ font-weight: bold; margin-bottom: 4px; }}
  canvas {{ max-width: 800px; margin-bottom: 20px; }}
  .model-section {{ border-top: 3px solid #333; padding-top: 20px; margin-top: 40px; }}
</style></head>
<body>
<h1>DFU Tissue Segmentation — All Models Comparison</h1>
<p class="legend">
  <span style="background:black"></span>Background&nbsp;&nbsp;
  <span style="background:red"></span>Fibrin&nbsp;&nbsp;
  <span style="background:green"></span>Granulation&nbsp;&nbsp;
  <span style="background:blue"></span>Callus
</p>
<h2>Leaderboard (sorted by overall Dice)</h2>
<table>
<tr><th>Model</th><th>IoU</th><th>Dice</th><th>Precision</th><th>Recall</th><th></th></tr>
{leaderboard_rows}
</table>
{''.join(sections)}
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<script>
{''.join(chart_scripts)}
</script>
</body></html>
"""
    with open(os.path.join(report_dir, "report.html"), "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, required=True)
    parser.add_argument("--report_dir", type=str, default="./all_models_report")
    parser.add_argument("--search_dirs", type=str, nargs="*", default=SEARCH_DIRS,
                         help="Folders to scan for checkpoints/ subdirectories.")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    entries = discover_checkpoints(args.search_dirs)
    if not entries:
        print("No checkpoints found under:", args.search_dirs)
        print("Pass --search_dirs explicitly if you used custom --out_dir values.")
        return
    print(f"Found {len(entries)} model(s):")
    for e in entries:
        print(f"  - {e['display_name']}  ({e['ckpt_type']})")

    os.makedirs(args.report_dir, exist_ok=True)
    model_results = []
    for entry in entries:
        print(f"\nEvaluating {entry['display_name']} ({entry['ckpt_type']})...")
        try:
            result = evaluate_and_render(entry, args.repo_root, args.report_dir, DEVICE)
            model_results.append((entry["display_name"], result))
            print(f"  Overall IoU={result['overall']['iou']:.2f} Dice={result['overall']['dice']:.2f}"
                  + ("" if result["history"] else "  [no history.json found]"))
        except Exception as e:
            print(f"  SKIPPED due to error: {e}")

    model_results.sort(key=lambda item: item[1]["overall"]["dice"], reverse=True)
    build_html_report(args.report_dir, model_results)
    print(f"\nReport written to: {os.path.abspath(os.path.join(args.report_dir, 'report.html'))}")


if __name__ == "__main__":
    main()
