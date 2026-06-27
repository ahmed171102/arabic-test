"""Shared paths for Issam_Demo_Bundle — import from notebooks after _inject_bundle()."""
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent

MODEL_NAME = (
    "Holistic_keypoints_BiLSTM_model_3_signers___Date_Time_2023_05_30__12_48_47___"
    "Loss_0.026179302483797073___Accuracy_0.9962499737739563.h5"
)

MODEL_PATH = BUNDLE_ROOT / MODEL_NAME
LABELS_XLSX = BUNDLE_ROOT / "KARSL-502_Labels.xlsx"
BUNDLE_SAMPLES = BUNDLE_ROOT / "sample_videos" / "by_class"
SAMPLE_VIDEO = BUNDLE_ROOT / "sample_videos" / "skeleton_SignID0071.mp4"
MANIFEST_CSV = BUNDLE_ROOT / "sample_videos" / "manifest.csv"
CHECKLIST_PATH = BUNDLE_ROOT / "issam_live_checklist.json"
RESULTS_CSV = BUNDLE_ROOT / "sample_videos" / "batch_test_results.csv"

# Optional — only used if bundled sample missing for a class
DATASET_ROOT = Path(r"E:/Downloads/Arabic Words Dataset")

F_AVG = 48
N_FEATURES = 225


def inject_bundle_path():
    """Add bundle folder to sys.path so notebooks can import this module."""
    import sys

    root = str(BUNDLE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return BUNDLE_ROOT


def find_bundle_from_cwd():
    """Locate bundle when notebook cwd is workspace root or bundle folder."""
    import sys

    guesses = [
        Path.cwd(),
        Path.cwd() / "Words" / "ArSL Word (Arabic)" / "Issam_Demo_Bundle",
        BUNDLE_ROOT,
    ]
    guesses.extend(list(Path.cwd().parents)[:5])

    for p in guesses:
        p = p.resolve()
        if (p / "bundle_config.py").is_file() and (p / "KARSL-502_Labels.xlsx").is_file():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return p

    raise FileNotFoundError(
        "Cannot find Issam_Demo_Bundle. Open that folder in Jupyter / set kernel cwd there."
    )
