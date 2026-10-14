"""UrbanNoiseNet API (v1.1)

Endpoints
  GET  /          basic status
  GET  /health    status, the sound classes the model can recognise, and whether accounts are set up
  POST /auth/signup, POST /auth/login, GET /auth/me   accounts stored in Turso (see auth.py)
  POST /classify  multipart form: file (WAV recommended), optional lat, lng
"""
import io
import os

import joblib
import librosa
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import auth

BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, "noise_classifier_model.pkl"))
label_encoder = joblib.load(os.path.join(BASE, "label_encoder.pkl"))

SR = 22050                     # must match the sample rate used for training
MAX_BYTES = 10 * 1024 * 1024
# Phone mics are uncalibrated. 94 dB SPL is the usual reference for a full-scale
# reading of -0 dBFS on a MEMS mic at 1 Pa; change MIC_REF_DB after you calibrate.
MIC_REF_DB = float(os.getenv("MIC_REF_DB", "94"))
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(title="UrbanNoiseNet API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(auth.router)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    """Return one readable sentence instead of pydantic's list of errors."""
    e = exc.errors()[0]
    msg = str(e.get("msg", "Invalid request"))
    if e.get("type") == "missing":
        msg = f"{e['loc'][-1]} is required"
    return JSONResponse(status_code=422, content={"detail": msg.removeprefix("Value error, ")})


def classes() -> list[str]:
    return [str(c) for c in label_encoder.classes_]


def to_db(amplitude: float) -> float:
    return 20.0 * float(np.log10(max(amplitude, 1e-9)))


@app.get("/")
def root():
    return {"status": "UrbanNoiseNet API is running", "version": app.version}


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version, "classes": classes(), "auth": auth.configured()}


@app.post("/classify")
async def classify(
    file: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),  # accepted so older clients still work
    lat: float | None = Form(None),
    lng: float | None = Form(None),
):
    upload = file or audio
    if upload is None:
        raise HTTPException(422, "Send the audio as the form field 'file'.")
    data = await upload.read()
    if not data:
        raise HTTPException(400, "The audio file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The audio file is too large (10 MB max).")

    try:
        y, sr = librosa.load(io.BytesIO(data), sr=SR, mono=True, duration=4)
    except Exception:
        raise HTTPException(400, "Could not read the audio. Send a WAV file.")
    if y.size < SR // 4:
        raise HTTPException(400, "The clip is too short. Record at least 1 second.")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    x = np.mean(mfcc.T, axis=0).reshape(1, -1)
    proba = model.predict_proba(x)[0]
    order = np.argsort(proba)[::-1]
    names = label_encoder.inverse_transform(model.classes_[order])
    top = [
        {"label": str(n), "confidence": round(float(proba[i]) * 100, 2)}
        for n, i in list(zip(names, order))[:3]
    ]

    rms = float(np.sqrt(np.mean(y**2)))
    peak = float(np.max(np.abs(y)))
    level = round(to_db(rms) + MIC_REF_DB, 1)
    peak_est = round(to_db(peak) + MIC_REF_DB, 1)
    severity = "critical" if level >= 90 else "warning" if level >= 75 else "normal"

    return {
        "classification": top[0]["label"],
        "confidence": top[0]["confidence"],
        "top3": top,
        "level_db_est": level,       # estimated average level (uncalibrated)
        "peak_db_est": peak_est,     # estimated peak level (uncalibrated)
        "peak_db": peak_est,         # kept for older clients
        "peak_dbfs": round(to_db(peak), 1),
        "severity": severity,
        "gps": {"lat": lat, "lng": lng},
    }
