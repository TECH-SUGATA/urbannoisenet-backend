"""Retrain the UrbanNoiseNet classifier on UrbanSound8K with the OFFICIAL folds.

Usage:
    python train_model.py /path/to/UrbanSound8K

Why official folds: clips cut from the same recording sit in the same fold, so a
random 80/20 split leaks near-duplicate audio into the test set and inflates accuracy.

Outputs (next to this script): noise_classifier_model.pkl, label_encoder.pkl, metrics.txt
"""
import os
import sys

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

SR, N_MFCC = 22050, 40          # keep identical to main.py
root = sys.argv[1] if len(sys.argv) > 1 else "UrbanSound8K"
out = os.path.dirname(os.path.abspath(__file__))

meta = pd.read_csv(os.path.join(root, "metadata", "UrbanSound8K.csv"))
X, y, folds = [], [], []
for i, row in meta.iterrows():
    path = os.path.join(root, "audio", f"fold{row['fold']}", row["slice_file_name"])
    try:
        sig, _ = librosa.load(path, sr=SR, mono=True, duration=4)
        X.append(np.mean(librosa.feature.mfcc(y=sig, sr=SR, n_mfcc=N_MFCC).T, axis=0))
        y.append(row["class"])
        folds.append(row["fold"])
    except Exception as e:
        print("skip", path, e)
    if i % 500 == 0:
        print(f"features {i}/{len(meta)}")

X, folds = np.array(X), np.array(folds)
le = LabelEncoder()
y = le.fit_transform(y)

def forest():
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)

# 10-fold evaluation: each fold is held out once
scores = []
for f in sorted(set(folds)):
    clf = forest().fit(X[folds != f], y[folds != f])
    scores.append(accuracy_score(y[folds == f], clf.predict(X[folds == f])))
    print(f"fold {f}: {scores[-1]:.4f}")

# Detailed report on the last fold, then train the final model on everything
last = max(folds)
clf = forest().fit(X[folds != last], y[folds != last])
report = classification_report(y[folds == last], clf.predict(X[folds == last]), target_names=le.classes_)
final = forest().fit(X, y)

joblib.dump(final, os.path.join(out, "noise_classifier_model.pkl"))
joblib.dump(le, os.path.join(out, "label_encoder.pkl"))
summary = f"10-fold mean accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})\nClasses: {list(le.classes_)}\n\nFold {last} report:\n{report}"
open(os.path.join(out, "metrics.txt"), "w").write(summary)
print(summary)
