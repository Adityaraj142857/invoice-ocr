mkdir -p data/ \
src/ingestion src/ocr src/vlm src/detection src/extraction src/matching src/utils \
master_data configs notebooks pseudo_labels sample_output

touch \
src/ingestion/__init__.py \
src/ingestion/pdf_converter.py \
src/ingestion/preprocessor.py \
src/ocr/__init__.py \
src/ocr/paddle_ocr.py \
src/ocr/ocr_utils.py \
src/vlm/__init__.py \
src/vlm/qwen_extractor.py \
src/vlm/prompts.py \
src/detection/__init__.py \
src/detection/stamp_detector.py \
src/detection/signature_detector.py \
src/extraction/__init__.py \
src/extraction/field_parser.py \
src/extraction/consensus.py \
src/extraction/confidence.py \
src/matching/__init__.py \
src/matching/fuzzy_matcher.py \
src/matching/master_loader.py \
src/utils/__init__.py \
src/utils/image_utils.py \
src/utils/json_utils.py \
master_data/dealer_master.csv \
master_data/model_master.csv \
configs/config.yaml \
notebooks/01_eda.ipynb \
notebooks/02_annotation.ipynb \
notebooks/03_evaluation.ipynb \
pseudo_labels/pseudo_gt.json \
sample_output/result.json \
executable.py \
requirements.txt \
README.md