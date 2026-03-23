"""
auto_label.py — Automatically label tractor quotation images using Gemini 2.5 Flash
Usage:
    pip install google-generativeai pillow
    python auto_label.py --image_dir ./images --api_key YOUR_KEY --output labels.csv
"""

import os
import json
import argparse
import csv
import time
from pathlib import Path

import google.generativeai as genai
from PIL import Image

EXTRACTION_PROMPT = """
You are an expert document parser. This image is a tractor quotation / proforma invoice from an Indian dealer.

Extract the following fields and return ONLY a valid JSON object — no markdown, no explanation, no code fences:

{
  "dealer_name": "<company name from letterhead>",
  "model_name": "<tractor model name exactly as written>",
  "horse_power": 45,
  "asset_cost": 700000,
  "signature_present": true,
  "signature_bbox": [y_min, x_min, y_max, x_max],
  "stamp_present": false,
  "stamp_bbox": null,
  "notes": "<brief note about image quality, rotation, language, anything unusual>"
}

Rules:
- horse_power: integer only, e.g. 45. Use null if not found.
- asset_cost: integer, no commas or symbols, Grand Total row only. null if not found.
- signature_present: true if handwritten cursive mark near 'Authorised Signatory'. false otherwise.
- stamp_present: true ONLY for circular/oval rubber ink stamps. Printed logos = false.
- signature_bbox / stamp_bbox: [y_min, x_min, y_max, x_max] in 0-1000 scale. null if not present.

Return ONLY the JSON. No markdown. No explanation. No code fences.
"""


def safe_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return bool(val)


def denorm_bbox(bbox_norm, width, height):
    if not bbox_norm or not isinstance(bbox_norm, list) or len(bbox_norm) != 4:
        return None
    y0, x0, y1, x1 = bbox_norm
    return [
        int(x0 / 1000 * width),
        int(y0 / 1000 * height),
        int(x1 / 1000 * width),
        int(y1 / 1000 * height),
    ]


def label_image(image_path: Path, model) -> dict:
    img = Image.open(image_path)
    width, height = img.size

    response = model.generate_content(
        [EXTRACTION_PROMPT, img],
        generation_config={"temperature": 0}
    )

    raw = response.text.strip()
    print(f"    Raw Gemini response: {raw[:120]}...")

    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break

    data = json.loads(raw)

    return {
        "doc_id":        str(image_path.stem),
        "image_name":    image_path.name,
        "image_path":    str(image_path.resolve()),
        "dealer_name":   data.get("dealer_name", ""),
        "model_name":    data.get("model_name", ""),
        "horse_power":   data.get("horse_power"),
        "asset_cost":    data.get("asset_cost"),
        "sig_present":   safe_bool(data.get("signature_present", False)),
        "sig_bbox":      denorm_bbox(data.get("signature_bbox"), width, height),
        "stamp_present": safe_bool(data.get("stamp_present", False)),
        "stamp_bbox":    denorm_bbox(data.get("stamp_bbox"), width, height),
        "notes":         data.get("notes", ""),
        "difficulty":    "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--api_key",   required=True)
    parser.add_argument("--output",    default="labels.csv")
    parser.add_argument("--json_dir",  default="labels_json")
    parser.add_argument("--delay",     type=float, default=3.0)
    args = parser.parse_args()

    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"ERROR: image_dir '{image_dir}' does not exist.")
        return

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])

    if not images:
        print(f"ERROR: No images found in '{image_dir}'. Check the folder and file extensions.")
        return

    print(f"Found {len(images)} images in '{image_dir}'")
    print(f"Output CSV    : {args.output}")
    print(f"Per-image JSON: {args.json_dir}/\n")

    json_dir = Path(args.json_dir)
    json_dir.mkdir(exist_ok=True)

    csv_rows = []
    errors   = []

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}")
        try:
            label = label_image(img_path, model)

            with open(json_dir / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
                json.dump(label, f, indent=2, ensure_ascii=False)

            csv_rows.append(label)
            print(f"    OK — dealer={label['dealer_name']} | hp={label['horse_power']} | cost={label['asset_cost']} | sig={label['sig_present']} | stamp={label['stamp_present']}")

        except json.JSONDecodeError as e:
            print(f"    FAILED (JSON parse error): {e}")
            errors.append((img_path.name, f"JSON parse: {e}"))
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            errors.append((img_path.name, str(e)))

        if i < len(images):
            time.sleep(args.delay)

    # Always write CSV even if some images failed
    if csv_rows:
        fieldnames = [
            "doc_id", "image_name", "image_path",
            "dealer_name", "model_name", "horse_power", "asset_cost",
            "sig_present", "sig_bbox", "stamp_present", "stamp_bbox",
            "notes", "difficulty"
        ]
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nSaved {len(csv_rows)} rows  ->  '{args.output}'")
        print(f"Per-image JSONs          ->  '{args.json_dir}/'")
    else:
        print("\nNo rows to save — all images failed. Check errors above.")

    if errors:
        print(f"\n{len(errors)} image(s) failed:")
        for name, err in errors:
            print(f"  {name}: {err}")

    if csv_rows:
        print("\n── Quick Summary ──")
        print(f"  {'Image':<40} {'Sig':<6} {'Stamp':<7} {'HP':<6} {'Cost'}")
        print("  " + "-" * 72)
        for row in csv_rows:
            sig  = "YES" if row["sig_present"]  else "NO"
            stmp = "YES" if row["stamp_present"] else "NO"
            hp   = str(row["horse_power"]) if row["horse_power"] is not None else "?"
            cost = str(row["asset_cost"])  if row["asset_cost"]  is not None else "?"
            print(f"  {str(row['image_name']):<40} {sig:<6} {stmp:<7} {hp:<6} {cost}")


if __name__ == "__main__":
    main()