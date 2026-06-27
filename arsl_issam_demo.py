"""
Standalone demo for the issamjebnouni Arabic Word-level SLR model.

This reuses the EXACT preprocessing from the original notebook
(Holistic_keypoints_BiLSTM_model_3_signers.ipynb):

    - features: pose(33*3) + left_hand(21*3) + right_hand(21*3) = 225-D
    - normalization: pose relative to nose, each hand relative to its own wrist
    - f_avg = 48 frames per clip
    - labels: KArSL SignID 71..170 (100 body/medical words)

Two modes:
    python arsl_issam_demo.py --video skeleton.mp4      # proven, works out of the box
    python arsl_issam_demo.py --webcam                  # live (studio-trained, so expect lower accuracy at home)

Run this from inside the project folder (the one containing the .h5 files and KARSL-502_Labels.xlsx).
"""

import os
import sys
import argparse
import numpy as np
import cv2

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Make the Windows console print Arabic instead of crashing on cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import mediapipe as mp
import pandas as pd
import tensorflow as tf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
F_AVG = 48
N_FEATURES = 33 * 3 + 21 * 3 + 21 * 3  # 225
LABELS_XLSX = os.path.join(HERE, "KARSL-502_Labels.xlsx")
DEFAULT_MODEL = os.path.join(
    HERE,
    "Holistic_keypoints_BiLSTM_model_3_signers___Date_Time_2023_05_30__12_48_47___Loss_0.026179302483797073___Accuracy_0.9962499737739563.h5",
)
DEFAULT_VIDEO = os.path.join(HERE, "sample_videos", "skeleton_SignID0071.mp4")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Optional Arabic shaping for nicer console / overlay output
try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    def shape_ar(text: str) -> str:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
except Exception:
    def shape_ar(text: str) -> str:
        return text


# ---------------------------------------------------------------------------
# Preprocessing (identical to the notebook)
# ---------------------------------------------------------------------------
def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    return results


def adjust_landmarks(arr, center):
    arr_reshaped = arr.reshape(-1, 3)
    center_repeated = np.tile(center, (len(arr_reshaped), 1))
    arr_adjusted = (arr_reshaped - center_repeated).reshape(-1)
    return arr_adjusted


def extract_keypoints(results):
    pose = (
        np.array([[r.x, r.y, r.z] for r in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 3)
    )
    lh = (
        np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    nose = pose[:3]
    lh_wrist = lh[:3]
    rh_wrist = rh[:3]
    pose_adj = adjust_landmarks(pose, nose)
    lh_adj = adjust_landmarks(lh, lh_wrist)
    rh_adj = adjust_landmarks(rh, rh_wrist)
    return np.concatenate((pose_adj, lh_adj, rh_adj))


def pad_or_trim(seq, f_avg=F_AVG):
    seq = np.asarray(seq, dtype=np.float32)
    n = min(seq.shape[0], f_avg)
    seq = seq[:n, :]
    while seq.shape[0] < f_avg:
        seq = np.concatenate((seq, seq[-1:, :]), axis=0)
    return seq


# ---------------------------------------------------------------------------
# Labels + model loading
# ---------------------------------------------------------------------------
def load_words():
    df = pd.read_excel(LABELS_XLSX)
    karsl_100 = df[70:170].reset_index(drop=True)
    words_ar = np.array([v for v in karsl_100["Sign-Arabic"]])
    words_en = np.array([v for v in karsl_100["Sign-English"]])
    return words_ar, words_en


def build_model(n_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(F_AVG, N_FEATURES)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    return model


def load_model(model_path, n_classes):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("[model] loaded full model.")
        return model
    except Exception as e:
        print(f"[model] full load failed ({e}); rebuilding + load_weights ...")
        model = build_model(n_classes)
        model.load_weights(model_path)
        print("[model] rebuilt and weights loaded.")
        return model


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def predict_video(video_path, model, words, words_en, topk=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    seq = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = mediapipe_detection(frame, holistic)
            seq.append(extract_keypoints(results))
    cap.release()
    if not seq:
        print("No frames read.")
        return
    X = pad_or_trim(seq)[None, ...]
    probs = model.predict(X, verbose=0)[0]
    order = np.argsort(probs)[::-1][:topk]
    print(f"\n=== {os.path.basename(video_path)} ===")
    for rank, i in enumerate(order, 1):
        line = f"  {rank}. {words_en[i]:<22}  {words[i]}  ({probs[i]*100:.1f}%)"
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("utf-8", "replace").decode("ascii", "replace"))


def run_webcam(model, words, words_en, cam_index=0):
    from collections import deque

    buf = deque(maxlen=F_AVG)
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    pred_text = "..."
    print("Webcam: sign continuously. Press Q to quit.")
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            results = mediapipe_detection(frame, holistic)
            buf.append(extract_keypoints(results))

            has_hand = bool(results.left_hand_landmarks or results.right_hand_landmarks)

            if len(buf) == F_AVG and has_hand:
                X = np.array(buf, dtype=np.float32)[None, ...]
                probs = model.predict(X, verbose=0)[0]
                i = int(np.argmax(probs))
                pred_text = f"{words_en[i]}  {probs[i]*100:.0f}%"

            # draw landmarks for feedback
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(
                frame,
                shape_ar(pred_text),
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("ArSL issam demo (press Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="path to a video file to classify")
    ap.add_argument("--webcam", action="store_true", help="run live webcam mode")
    ap.add_argument("--cam", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="path to .h5 model")
    args = ap.parse_args()

    words, words_en = load_words()
    print(f"[labels] {len(words)} classes loaded (SignID 71..170).")
    model = load_model(args.model, len(words))

    if args.webcam:
        run_webcam(model, words, words_en, cam_index=args.cam)
    else:
        video = args.video or (DEFAULT_VIDEO if os.path.isfile(DEFAULT_VIDEO) else os.path.join(HERE, "skeleton.mp4"))
        predict_video(video, model, words, words_en)


if __name__ == "__main__":
    main()
