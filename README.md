# Issam Demo Bundle (portable copy)

Self-contained copy of the **issamjebnouni** Arabic word SLR demo (100 classes, SignID 71–170).  
Original project folder is **not modified** — this bundle is a separate snapshot for sharing/testing.

## Contents

| Item | Purpose |
|------|---------|
| `Issam_One_Video_Sample.ipynb` | Quick test — one bundled video (Skeleton / SignID 71) |
| `Issam_Bundle_Samples_Batch_Test.ipynb` | **Verify all 100 classes** against bundled samples |
| `Issam_Word_Live_Test.ipynb` | Full guided live test + checklist (uses bundled refs) |
| `arsl_issam_demo.py` | CLI: `python arsl_issam_demo.py` or `--webcam` |
| `sample_videos/by_class/SignID0071.mp4` … `SignID0170.mp4` | **One KARSL clip per class** (100 files) |
| `sample_videos/manifest.csv` | Maps SignID → filename, English, Arabic |
| `sample_videos/skeleton_SignID0071.mp4` | Extra issam demo clip (same class as 0071) |
| `Holistic_keypoints_BiLSTM_model_3_signers_...h5` | Pre-trained BiLSTM (99.6% on 3-signer test) |
| `KARSL-502_Labels.xlsx` | KARSL labels (rows 71–170 used) |
| `scripts/copy_class_samples.py` | Re-copy samples from full KARSL dataset if needed |

## Quick start

**One-time:** install Python packages (not bundled — use your existing env or):

```bash
cd "Issam_Demo_Bundle"
pip install -r requirements.txt
```

1. Open **`Issam_Demo_Bundle`** in Jupyter / VS Code (works even if workspace root is `SLR Main`).
2. Run **`Issam_One_Video_Sample.ipynb`** cells 1 → 3 (single video smoke test).
3. Run **`Issam_Bundle_Samples_Batch_Test.ipynb`** to classify all 100 bundled clips.
4. For webcam testing, run **`Issam_Word_Live_Test.ipynb`** cells 1 → 5.

All notebooks import paths from **`bundle_config.py`** in this folder — no dependency on the original `Arabic-Word-level-Sign-Language-Recognition-main/` project.

### CLI (optional)

```bash
cd "Issam_Demo_Bundle"
python arsl_issam_demo.py
python arsl_issam_demo.py --webcam
```

### Refresh bundled samples from full KARSL

If you update the dataset on disk:

```bash
python scripts/copy_class_samples.py
```

Requires `E:/Downloads/Arabic Words Dataset` (or edit path in the script).

## Reference videos in live test

`Issam_Word_Live_Test.ipynb` uses **`sample_videos/by_class/`** first (works offline).  
If a class file is missing, it falls back to the full KARSL path in Cell 2.

## Notes

- Demo/reference only — graduation model (131 classes, 258-D) is in `ArSL_Word_Class_Guide.ipynb` (parent repo).
- Checklist from live test saves to `issam_live_checklist.json` in this folder.
