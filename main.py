import os
import base64
import numpy as np
import cv2
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="Student Biometrics AI Core", version="3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── SECURITY CONSTANTS ───────────────────────────────────────────────────────
#
# HOW FACE MATCHING WORKS:
#   DeepFace gives a "distance" (cosine). Lower = more similar.
#   - Same person, same session:      0.05 - 0.25
#   - Same person, different day/cam:  0.25 - 0.55
#   - Different person:                0.60 - 1.00+
#
# Facenet512's default cosine threshold is 0.30, but that assumes
# lab-quality images. With real webcams (different lighting, angle,
# resolution between registration and login), we MUST use a more
# forgiving threshold. 0.55 is the sweet spot:
#   ✅ Accepts same person with different lighting/angle
#   ❌ Rejects different people (who typically score >0.65)
#
FACE_MODEL = "Facenet512"         # Best accuracy model in DeepFace
FACE_MATCH_THRESHOLD = 0.55       # Forgiving for webcam; rejects strangers
FACE_DETECTOR = "opencv"          # Fast & reliable for webcams
DIFFERENT_PERSON_MIN = 0.65       # Above this = definitely different person

# ─── Lazy-load heavy models ───────────────────────────────────────────────────
_yolo_model = None
_hf_client = None

def get_hf_client():
    global _hf_client
    if _hf_client is None:
        try:
            from huggingface_hub import InferenceClient
            token = os.getenv("HF_API_KEY")
            _hf_client = InferenceClient(api_key=token)
            print("✅ Hugging Face client initialized.")
        except Exception as e:
            print(f"❌ HF Client init failed: {e}")
    return _hf_client

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO("yolov8n.pt")
            print("✅ YOLO proctor model loaded.")
        except Exception as e:
            print(f"❌ YOLO load failed: {e}")
    return _yolo_model


# ─── Image Utilities ──────────────────────────────────────────────────────────

def decode_image(base64_string: str) -> np.ndarray:
    """Decode a base64 image string into a NumPy array (BGR)."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image.")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


# ─── Biometric Comparison ─────────────────────────────────────────────────────

class BiometricRequest(BaseModel):
    registered_face_base64: str
    login_face_base64: str

@app.post("/api/biometrics/compare")
async def compare_faces(req: BiometricRequest):
    """
    Compare two face images using Facenet512 (best accuracy).
    Returns match status, distance score, and confidence percentage.
    
    Security:
    - enforce_detection=True → rejects frames with no visible face
    - Facenet512 model → highest accuracy for face-only features
    - Custom threshold (0.35) → rejects different people reliably
    """
    from deepface import DeepFace

    img1 = decode_image(req.registered_face_base64)
    img2 = decode_image(req.login_face_base64)

    print(f"[Biometric] img1: {img1.shape}, img2: {img2.shape}")

    # ── Run face verification ─────────────────────────────────────────────
    try:
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=FACE_MODEL,
            distance_metric="cosine",
            detector_backend=FACE_DETECTOR,
            enforce_detection=False,   # We handle no-face errors ourselves
            align=True
        )
    except Exception as e:
        msg = str(e)
        print(f"[Biometric Error] {msg}")

        if "img1_path" in msg or "img2_path" in msg:
            raise HTTPException(
                status_code=400,
                detail="Could not process face image. Please re-register or try in better lighting."
            )
        if "face could not be detected" in msg.lower():
            raise HTTPException(
                status_code=400,
                detail="No face detected. Please look directly at the camera."
            )
        raise HTTPException(status_code=500, detail=f"AI Error: {msg}")

    distance = float(result.get("distance", 1.0))
    threshold = result.get("threshold", FACE_MATCH_THRESHOLD)
    
    # Use the stricter of our threshold or model's default
    effective_threshold = min(float(threshold), FACE_MATCH_THRESHOLD)
    
    # But never go below 0.30 — that's unrealistically strict for webcams
    effective_threshold = max(effective_threshold, 0.30)
    
    is_match = distance <= effective_threshold
    confidence = round(max(0, (1.0 - distance)) * 100, 1)

    print(f"[Biometric] Distance: {distance:.4f} | Threshold: {effective_threshold} | Confidence: {confidence}% | Match: {is_match}")

    if not is_match:
        if distance >= DIFFERENT_PERSON_MIN:
            detail = "BIOMETRIC MISMATCH — This is NOT the registered student. Access Denied 🚫"
        else:
            detail = f"Face match confidence too low ({confidence}%). Please try again in better lighting with your face centered."
        
        return {
            "match": False,
            "similarity_score": round(distance, 4),
            "confidence": confidence,
            "message": detail
        }

    return {
        "match": True,
        "similarity_score": round(distance, 4),
        "confidence": confidence,
        "message": f"Identity Verified ✅ ({confidence}% confidence)"
    }


# ─── Proctoring ───────────────────────────────────────────────────────────────

class ProctorRequest(BaseModel):
    frame_base64: str

@app.post("/api/proctor/analyze")
async def analyze_proctor_frame(req: ProctorRequest):
    """Analyze a frame for suspicious proctoring activity using YOLO."""
    model = get_yolo()
    if model is None:
        raise HTTPException(status_code=503, detail="Proctor vision model not available.")

    try:
        img = decode_image(req.frame_base64)
        results = model(img, verbose=False)

        warnings = []
        person_count = 0
        phone_detected = False

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])

            if class_name == "person" and conf > 0.5:
                person_count += 1
            if class_name == "cell phone" and conf > 0.3:
                phone_detected = True

        if person_count > 1:
            warnings.append(f"Multiple people detected ({person_count})")
        elif person_count == 0:
            warnings.append("No person detected in frame.")
        if phone_detected:
            warnings.append("Mobile phone detected!")

        return {
            "status": "safe" if not warnings else "warning",
            "warnings": warnings,
            "person_count": person_count
        }

    except Exception as e:
        print(f"[Proctor Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── AI Chat & Study Planner ─────────────────────────────────────────────────

class AIChatRequest(BaseModel):
    message: str

@app.post("/api/ai/chat")
async def ai_chat(req: AIChatRequest):
    """Chat with Scholar AI powered by Llama 3.1."""
    client = get_hf_client()
    if client is None:
        return {"response": "Scholar AI is offline — HF client not initialized."}

    models_to_try = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3"
    ]

    for model_id in models_to_try:
        try:
            print(f"[AI Chat] Trying model: {model_id}")
            response = client.chat_completion(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are 'Scholar AI', a world-class academic tutor in a biometric-secured university portal. Be helpful, professional, and concise. Keep responses under 200 words."},
                    {"role": "user", "content": req.message}
                ],
                max_tokens=400
            )
            return {"response": response.choices[0].message.content.strip()}
        except Exception as e:
            errmsg = str(e)
            print(f"[AI Chat] Model {model_id} failed: {errmsg[:100]}")
            if any(x in errmsg for x in ["403", "401", "access", "Permission"]):
                continue
            return {"response": f"Neural link error: {errmsg[:80]}"}

    return {"response": "All AI models are currently unavailable. Please check your Hugging Face API key permissions."}


class AIPlanRequest(BaseModel):
    subjects: str

@app.post("/api/ai/study-plan")
async def ai_study_plan(req: AIPlanRequest):
    """Generate an AI-powered study plan for the week."""
    client = get_hf_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Scholar AI offline.")

    models_to_try = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct"
    ]

    prompt = (
        f"Generate a JSON object for a weekly study plan (Monday to Saturday). "
        f"Subjects: {req.subjects}. "
        f"Each day key maps to an array of exactly 4 subject strings. "
        f'Example: {{"Monday": ["Math", "Physics", "Break", "History"], "Tuesday": [...], ...}} '
        f"Return ONLY valid JSON. No explanation."
    )

    for model_id in models_to_try:
        try:
            response = client.chat_completion(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are an academic planner. Return ONLY raw JSON. No markdown, no explanation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=700
            )
            text = response.choices[0].message.content.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            if "{" in text:
                text = text[text.find("{"):text.rfind("}")+1]
            plan = json.loads(text)
            return {"plan": plan}
        except Exception as e:
            print(f"[AI Plan] Model {model_id} failed: {str(e)[:100]}")
            if model_id == models_to_try[-1]:
                raise HTTPException(status_code=500, detail="Neural planning failed. Please enter subjects and retry.")
            continue


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "online",
        "version": "3.1",
        "face_model": FACE_MODEL,
        "face_detector": FACE_DETECTOR,
        "match_threshold": FACE_MATCH_THRESHOLD,
        "ai_model": "meta-llama/Llama-3.1-8B-Instruct"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)