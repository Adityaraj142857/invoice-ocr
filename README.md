# Invoice OCR — Intelligent Document AI

End-to-end field extraction from Indian tractor loan quotation PDFs.
**Cross-platform: Mac (Apple Silicon + Intel) · Windows · Linux**

---

## Quick Start

### Mac / Linux
```bash
git clone <your-repo-url>
cd invoice-ocr
bash setup.sh
source venv/bin/activate
python executable.py --input data/train_docs/ --output sample_output/result.json
```

### Windows
```bat
git clone <your-repo-url>
cd invoice-ocr
setup.bat
venv\Scripts\activate
python executable.py --input data\train_docs\ --output sample_output\result.json
```

---

## Platform Behaviour (Automatic — no config changes needed)

| Feature | Mac Apple Silicon (M1/M2/M3) | Mac Intel | Windows (NVIDIA GPU) | Windows (CPU only) |
|---|---|---|---|---|
| VLM device | MPS (GPU) | CPU | CUDA | CPU |
| VLM dtype | float16 | float32 | float16 | float32 |
| PaddleOCR | Disabled (hangs on ARM) | Enabled | Enabled | Enabled |
| Speed (per doc) | ~10–15s | ~60–90s | ~5–10s | ~60–90s |

Everything is auto-detected at runtime from `device: "auto"` and `enabled: "auto"` in `configs/config.yaml`.

---

## Project Structure

```
invoice-OCR/
├── executable.py            ← Main entry point
├── setup.sh                 ← Mac/Linux one-command setup
├── setup.bat                ← Windows one-command setup
├── requirements.txt
├── README.md
├── configs/
│   └── config.yaml          ← All parameters (device: auto, ocr: auto)
├── data/
│   └── train_docs/          ← Put your input documents here
├── master_data/
│   ├── dealer_master.csv    ← Canonical dealer names (one per line)
│   └── model_master.csv     ← Canonical model names (one per line)
├── sample_output/
│   └── result.json          ← Output goes here
└── src/
    ├── ingestion/            ← PDF → image conversion + preprocessing
    ├── ocr/                  ← PaddleOCR wrapper (Windows/Linux)
    ├── vlm/                  ← Qwen2-VL local inference
    ├── detection/            ← Stamp + signature detection (OpenCV)
    ├── extraction/           ← Field parsing, consensus, confidence
    ├── matching/             ← RapidFuzz dealer/model name matching
    └── utils/
        ├── device_utils.py   ← Auto CUDA/MPS/CPU detection
        ├── image_utils.py
        └── json_utils.py
```

---

## Output Format

```json
{
  "doc_id": "invoice_001",
  "fields": {
    "dealer_name": "AMS Tractors",
    "model_name": "Powertrac Euro G28",
    "horse_power": 28,
    "asset_cost": 700000,
    "signature": {"present": true,  "bbox": [380, 820, 560, 900]},
    "stamp":     {"present": false, "bbox": null}
  },
  "confidence": 0.921,
  "processing_time_sec": 12.4,
  "cost_estimate_usd": 0.0
}
```

---

## First-time Model Download

The VLM model (~5GB) downloads automatically on first run.
To pre-download manually:
```bash
python -c "from huggingface_hub import login; login()"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-VL-2B-Instruct')"
```

---

## Configuration

Key settings in `configs/config.yaml`:

```yaml
vlm:
  model_id: "Qwen/Qwen2-VL-2B-Instruct"
  device: "auto"     # auto-detects CUDA → MPS → CPU
  dtype: "auto"      # auto-detects float16 (GPU) or float32 (CPU)
  max_new_tokens: 256
  run_verification: false

ocr:
  enabled: "auto"    # auto-disabled on Mac Apple Silicon, enabled elsewhere
```

To force a specific device, replace `"auto"` with `"cuda"`, `"mps"`, or `"cpu"`.

---

## Cost

Everything runs locally — **$0.00 per document**.

| Component | Tool | Cost |
|---|---|---|
| VLM extraction | Qwen2-VL-2B (local) | Free |
| OCR | PaddleOCR (local) | Free |
| Detection | OpenCV (local) | Free |
| Matching | RapidFuzz (local) | Free |
