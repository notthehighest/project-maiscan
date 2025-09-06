from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import firebase_admin
from firebase_admin import credentials, firestore, auth
import pyrebase
import datetime
import os
from werkzeug.utils import secure_filename
import secrets
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from dotenv import load_dotenv  # ✅ load env variables

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- FLASK SETUP ----------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(16))

UPLOAD_FOLDER = "static/user_image"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# ---------------- FIREBASE SETUP ----------------
cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS"))
firebase_admin.initialize_app(cred)
db = firestore.client()

# Pyrebase config from env
firebaseConfig = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID")
}

pb = pyrebase.initialize_app(firebaseConfig)
pb_auth = pb.auth()

# ---------------- FLASK-LOGIN SETUP ----------------
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "info"

class User(UserMixin):
    def __init__(self, uid, email):
        self.id = uid
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    try:
        user_record = auth.get_user(user_id)
        return User(uid=user_record.uid, email=user_record.email)
    except Exception as e:
        print("Error loading user:", e)
        return None

# ---------------- LOAD ML MODEL ----------------
try:
    model = load_model("maiscan_disease_model_final.keras")
    print("✓ Model Loaded Successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("base.html")

# -------- REGISTER --------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email and password are required.", "danger")
            return render_template("register.html")

        try:
            # ✅ Create user in Firebase Authentication
            user_record = auth.create_user(email=email, password=password)

            # ✅ Save extra data in Firestore
            db.collection("Users").document(user_record.uid).set({
                "email": email,
                "created_at": datetime.datetime.utcnow()
            })

            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))

        except Exception as e:
            print("Registration error:", e)
            flash("Registration failed: " + str(e), "danger")

    return render_template("register.html")

# -------- LOGIN --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email and password are required.", "danger")
            return render_template("login.html")

        try:
            # ✅ Authenticate with Firebase using Pyrebase
            user = pb_auth.sign_in_with_email_and_password(email, password)

            # ✅ Get Firebase user record
            user_record = auth.get_user(user["localId"])

            # Flask-Login user
            user_obj = User(uid=user_record.uid, email=user_record.email)
            login_user(user_obj)

            flash("Login successful!", "success")
            return redirect(url_for("maiscan"))

        except Exception as e:
            print("Login error:", e)
            flash("Invalid email or password.", "danger")

    return render_template("login.html")

# -------- LOGOUT --------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))

# -------- MAISCAN DASHBOARD --------
@app.route("/maiscan")
@login_required
def maiscan():
    try:
        # ✅ Fetch user’s uploads from Firestore
        uploads_ref = db.collection("UploadedImages").where("user_id", "==", current_user.id)
        uploads = [doc.to_dict() for doc in uploads_ref.stream()]

        # Disease stats
        disease_counts = {}
        for up in uploads:
            disease = up.get("disease_type", "Unknown")
            disease_counts[disease] = disease_counts.get(disease, 0) + 1

        total_images = sum(disease_counts.values())
        disease_count = sum(c for d, c in disease_counts.items() if "healthy" not in d.lower())
        most_common_disease = max(
            (d for d in disease_counts if "healthy" not in d.lower()),
            key=lambda d: disease_counts[d],
            default="None"
        )
        disease_types = list(disease_counts.keys())

    except Exception as e:
        print("Error loading dashboard:", e)
        uploads, disease_counts, total_images, disease_count, most_common_disease, disease_types = [], {}, 0, 0, "None", []

    return render_template(
        "mais.html",
        uploads=uploads,
        disease_counts=disease_counts,
        total_images=total_images,
        disease_count=disease_count,
        most_common_disease=most_common_disease,
        disease_types=disease_types
    )

# -------- PREDICTION --------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "image" not in request.files:
        flash("No image uploaded.", "danger")
        return redirect(url_for("maiscan"))

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        flash("Invalid file type.", "danger")
        return redirect(url_for("maiscan"))

    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Predict
        pred, output_page, confidence = pred_corn_disease(file_path)

        # ✅ Save metadata to Firestore (not the image itself)
        if pred != "Unknown Class":
            db.collection("UploadedImages").add({
                "filename": filename,
                "user_id": current_user.id,
                "disease_type": pred,
                "confidence": confidence,
                "upload_date": datetime.datetime.utcnow()
            })

        return render_template(output_page, pred_output=pred, user_image=file_path, confidence=confidence)

    except Exception as e:
        print("Prediction error:", e)
        flash("Error processing image.", "danger")
        return redirect(url_for("maiscan"))

# -------- PREDICTION FUNCTION --------
def pred_corn_disease(img_path):
    try:
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        pred_class = np.argmax(prediction)
        confidence = float(np.max(prediction))

        CONFIDENCE_THRESHOLD = 0.5
        if confidence < CONFIDENCE_THRESHOLD:
            return "Unknown Class", "invalid_image.html", confidence

        diseases = {
            0: ("Aphids", "aphids.html"),
            1: ("Armyworm", "armyworm.html"),
            2: ("Common Cutworm", "common_cutworm.html"),
            3: ("Common Rust", "common_rust.html"),
            4: ("Common Smut", "common_smut.html"),
            5: ("Corn Borer", "corn_borer.html"),
            6: ("Earwig", "earwig.html"),
            7: ("Fusarium Ear Rot", "fusarium_ear_rot.html"),
            8: ("Gray Leaf Spot", "gray_leaf_spot.html"),
            9: ("Healthy Corn", "healthycorn.html"),
            10: ("Healthy Leaf", "healthyleaf.html"),
            11: ("Leaf Blight", "leaf_blight.html"),
            12: ("Leafhopper", "leafhopper.html"),
        }

        return diseases.get(pred_class, ("Unknown Class", "invalid_image.html")) + (confidence,)

    except Exception as e:
        print("Error in prediction:", e)
        return "Error", "invalid_image.html", 0.0

# -------- ERROR HANDLERS --------
@app.errorhandler(404)
def not_found_error(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template("500.html"), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080, threaded=True)
