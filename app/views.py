from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required

def index(request):
    return render(request,'index.html')


def registration(request):
    if request.method == "POST":
        first_name = request.POST.get("first_name")
        last_name = request.POST.get("last_name")
        username = request.POST.get("username")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")

        # Check empty fields
        if not first_name or not last_name or not username or not password or not confirm_password:
            messages.error(request, "All fields are required!")
            return redirect("registration")

        # Check username already exist
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists!")
            return redirect("registration")

        # Check password match
        if password != confirm_password:
            messages.error(request, "Passwords do not match!")
            return redirect("registration")

        # Save user
        user = User.objects.create_user(
            username=username,
            password=password,
            first_name=first_name,
            last_name=last_name
        )
        user.save()

        messages.success(request, "Registration successful!")
        return redirect("loginform")

    return render(request, "registration.html")


def loginform(request):

    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        # Check empty fields
        if not username or not password:
            messages.info(request, "Both fields are required!")
            return redirect("loginform")

        # Authenticate
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, "Login successful!")
            return redirect("dashboard")   # change if needed
        else:
            messages.error(request, "Invalid username or password!")
            return redirect("loginform")

    return render(request, "loginform.html")

def logout_view(request):
    auth_logout(request)
    messages.success(request, "You have been logged out successfully!")
    return redirect('/')

# views.py (FULL CODE)

# views.py (FULL CODE with Camera Fixes)

from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.http import HttpResponse, StreamingHttpResponse, JsonResponse
from django.template.loader import get_template
from xhtml2pdf import pisa
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
from ultralytics import YOLO
import time
import threading

# --- MODEL & CONFIG ---
MODEL_FILENAME = os.path.join(settings.BASE_DIR, 'models/transfer_resnet50_trash_classifier.h5')
TARGET_SIZE = (384, 384)
LOADED_MODEL = None
YOLO_MODEL = None
YOLO_CLASS_NAMES = {}

# Global detection state
detection_lock = threading.Lock()
latest_detection = {
    "saved": False,
    "frame": None,
    "label": None,
    "confidence": 0.0,
    "image_path": None,
    "image_filename": None,
    "image_url": None,
    "best_yolo_conf": 0.0,
    "last_update": 0.0
}

# Load models
try:
    LOADED_MODEL = load_model(MODEL_FILENAME, compile=False)
    print("ResNet50 loaded (used only for uploaded images)")
except Exception as e:
    print("ResNet50 load error:", e)

try:
    YOLO_MODEL_PATH = os.path.join(settings.BASE_DIR, 'models/best.pt')
    YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
    YOLO_CLASS_NAMES = YOLO_MODEL.names
    print(f"YOLOv8 loaded with classes: {YOLO_CLASS_NAMES}")
except Exception as e:
    print("YOLOv8 LOAD ERROR:", e)

# --- RESNET PREDICTION (Only for uploaded images) ---
def predict_single_image(image_file_path):
    if not LOADED_MODEL:
        return "unknown", 0.0
    try:
        img = Image.open(image_file_path).convert('RGB').resize(TARGET_SIZE)
        arr = np.array(img, dtype=np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        pred = LOADED_MODEL.predict(arr, verbose=0)[0]
        classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        idx = np.argmax(pred)
        return classes[idx], float(pred[idx])
    except Exception as e:
        return "error", 0.0

# --- MAIN FIX: TRUST YOLO 100% FOR LIVE DETECTION ---
def save_detected_frame_and_classify(frame, yolo_label, yolo_conf, request):
    global latest_detection
    now = time.time()

    MIN_CONF = 0.60
    MIN_IMPROVEMENT = 0.10
    MIN_INTERVAL = 2.0

    with detection_lock:
        first_time = latest_detection["best_yolo_conf"] == 0.0
        should_update = (
            first_time or
            yolo_conf > latest_detection["best_yolo_conf"] + MIN_IMPROVEMENT or
            (yolo_conf > latest_detection["best_yolo_conf"] and now - latest_detection["last_update"] > MIN_INTERVAL)
        )

        if yolo_conf < MIN_CONF or not should_update:
            return

        # Save frame
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_uploads')
        os.makedirs(temp_dir, exist_ok=True)
        filename = f"live_{int(now)}.jpg"
        filepath = os.path.join(temp_dir, filename)
        cv2.imwrite(filepath, frame)

        # Use YOLO label directly (this is the fix!)
        final_label = yolo_label.upper().replace(" ", "_")
        final_conf = yolo_conf

        # Build absolute URL
        rel_url = settings.MEDIA_URL + f'temp_uploads/{filename}'
        abs_url = request.build_absolute_uri(rel_url)

        # Update latest detection
        latest_detection.update({
            "saved": True,
            "frame": frame.copy(),
            "label": final_label,
            "confidence": final_conf,
            "image_path": filepath,
            "image_filename": filename,
            "image_url": abs_url,
            "best_yolo_conf": yolo_conf,
            "last_update": now
        })

        print(f"SAVED → {final_label} ({final_conf:.1%}) from YOLO")

# --- CAMERA STREAM ---
def generate_frame(request):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera not available\r\n'
        return

    last_save_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()

        if YOLO_MODEL:
            results = YOLO_MODEL(frame, verbose=False, stream=True)
            try:
                result = next(results)
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = YOLO_CLASS_NAMES.get(cls_id, "unknown")

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, f"{label.upper()} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Save best detection
                    if conf > 0.60 and (now - last_save_time > 1.5):
                        save_detected_frame_and_classify(frame, label, conf, request)
                        last_save_time = now

            except StopIteration:
                pass

        # UI Text
        cv2.putText(frame, "Show waste item to camera", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if latest_detection["saved"]:
            display_label = latest_detection["label"].replace("_", " ")
            cv2.putText(frame, f"BEST: {display_label}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(frame, f"{latest_detection['confidence']*100:.1f}%", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, "Tap 'Go to Results'", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)

    cap.release()

# --- VIEWS ---
def camera_feed(request):
    if not YOLO_MODEL:
        return JsonResponse({"error": "YOLO model not loaded"}, status=503)
    return StreamingHttpResponse(generate_frame(request),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def start_camera_view(request):
    global latest_detection
    with detection_lock:
        latest_detection.update({
            "saved": False, "frame": None, "label": None, "confidence": 0.0,
            "image_path": None, "image_filename": None, "image_url": None,
            "best_yolo_conf": 0.0, "last_update": 0.0
        })
    return render(request, 'camera_view.html')

def goto_results_from_camera(request):
    global latest_detection
    with detection_lock:
        if not latest_detection["saved"]:
            messages.warning(request, "No item detected yet!")
            return redirect('start_camera_view')

        request.session['prediction_data'] = {
            'label': latest_detection["label"],
            'confidence': f"{latest_detection['confidence']*100:.1f}",
            'image_url': latest_detection["image_url"],
            'image_filename': latest_detection["image_filename"],
            'image_path': latest_detection["image_path"]
        }
        messages.success(request, f"{latest_detection['label'].replace('_', ' ')} detected!")
        return redirect('results')

def dashboard(request):
    return render(request, 'dashboard.html')

def upload_image(request):
    if request.method == 'POST' and LOADED_MODEL:
        file = request.FILES.get('property_image')
        if not file:
            messages.error(request, "No image uploaded")
            return redirect('dashboard')

        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_uploads')
        os.makedirs(temp_dir, exist_ok=True)
        filepath = os.path.join(temp_dir, file.name)
        with open(filepath, 'wb+') as f:
            for chunk in file.chunks():
                f.write(chunk)

        label, conf = predict_single_image(filepath)
        if conf > 0.3:
            url = request.build_absolute_uri(settings.MEDIA_URL + f'temp_uploads/{file.name}')
            request.session['prediction_data'] = {
                'label': label.upper(),
                'confidence': f"{conf*100:.1f}",
                'image_url': url,
                'image_filename': file.name,
                'image_path': filepath
            }
            messages.success(request, "Image analyzed!")
            return redirect('results')
        else:
            messages.error(request, "Low confidence")
            os.remove(filepath)
    return redirect('dashboard')

def results(request):
    data = request.session.get('prediction_data')
    if not data:
        messages.warning(request, "No results found")
        return redirect('dashboard')

    context = {
        'prediction': data,
        'recycling_info': get_recycling_info(data['label']),
        'nashik_centers': get_nashik_centers(data['label']),
        'now': __import__('datetime').datetime.now()
    }
    return render(request, 'results.html', context)

def download_pdf(request):
    data = request.session.get('prediction_data')
    if not data:
        return redirect('dashboard')

    context = {
        'prediction': data,
        'recycling_info': get_recycling_info(data['label']),
        'nashik_centers': get_nashik_centers(data['label']),
        'now': __import__('datetime').datetime.now(),
        'is_pdf': True
    }
    template = get_template('results_pdf.html')
    html = template.render(context)
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="Waste_{data["label"]}.pdf"'
    pisa.CreatePDF(html, dest=response)
    return response

# --- HELPERS (Updated for all YOLO classes) ---
def get_recycling_info(category):
    mapping = {
        'bag': {'recyclable': 'Sometimes', 'advice': 'Check local plastic bag recycling'},
        'plastic': {'recyclable': 'Conditional', 'advice': 'Check RIC 1-7'},
        'bottle': {'recyclable': 'Yes', 'advice': 'Rinse and recycle'},
        'can': {'recyclable': 'Yes', 'advice': 'Crush to save space'},
        'metal': {'recyclable': 'Yes', 'advice': 'Rinse cans'},
        'cardboard': {'recyclable': 'Yes', 'advice': 'Flatten boxes'},
        'paper': {'recyclable': 'Yes', 'advice': 'Keep dry'},
        'glass': {'recyclable': 'Yes', 'advice': 'Rinse, remove lids'},
        'trash': {'recyclable': 'No', 'advice': 'General waste'},
        'furniture': {'recyclable': 'Check locally', 'advice': 'Large item collection'},
        'yard': {'recyclable': 'Yes (compost)', 'advice': 'Green waste bin'}
    }
    key = category.lower().replace('_', ' ')
    return mapping.get(key, {'recyclable': 'Check locally', 'advice': 'Follow local rules'})

def get_nashik_centers(category):
    centers = [
        {"name": "Nashik Municipal Corporation (NMC) Recycling Unit", "address": "Pathardi Phata, Nashik Road", "contact": "+91 253 222 3300", "timing": "8 AM - 5 PM", "verified": True},
        {"name": "Saibaba Recycling Center", "address": "Satpur MIDC", "contact": "+91 98230 56789", "timing": "9 AM - 6 PM", "verified": True},
        {"name": "Green Earth Waste Management", "address": "Ambad MIDC", "contact": "greenearth@nashik.com", "timing": "24 Hours", "verified": True},
    ]
    if category.lower() == 'trash':
        return [{"name": "NMC Dumping Ground", "address": "Vilholigaon", "contact": "NMC Helpline", "timing": "6 AM - 6 PM", "verified": True}]
    return centers