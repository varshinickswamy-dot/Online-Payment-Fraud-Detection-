from flask import Flask, render_template, request, redirect, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import joblib, os
import numpy as np

# ---------------- APP CONFIG ----------------
app = Flask(__name__)
app.secret_key = "securekey"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "/"

# ---------------- DATABASE MODELS ----------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(50))
    role = db.Column(db.String(10))

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    amount = db.Column(db.Float)
    result = db.Column(db.String(20))
    probability = db.Column(db.Float)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ---------------- INITIALIZE DB ----------------
with app.app_context():
    db.create_all()
    if not User.query.first():
        db.session.add(User(username="admin", password="admin", role="admin"))
        db.session.add(User(username="user", password="user", role="user"))
        db.session.commit()

# ---------------- LOAD MODELS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "fraud_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

print("✅ Models Loaded")

# ---------------- GLOBAL LAST PRED ----------------
last_prediction = {
    "result": "No prediction yet",
    "risk": "N/A",
    "probability": "N/A"
}

# ---------------- LOGIN ----------------
@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")

        user = User.query.filter_by(username=u, password=p).first()
        if user:
            login_user(user)
            return redirect("/admin" if user.role == "admin" else "/dashboard")

    return render_template("login.html")

# ---------------- USER DASHBOARD ----------------
@app.route("/dashboard", methods=["GET","POST"])
@login_required
def dashboard():

    result = None
    risk = None
    prob = None

    if request.method == "POST":

        try:
            amount = float(request.form.get("amount"))
            hour = float(request.form.get("hour"))
            merchant = float(request.form.get("merchant"))
            count = float(request.form.get("count"))
            device = float(request.form.get("device"))
            loc = float(request.form.get("loc"))
            intl = float(request.form.get("intl"))
        except:
            return render_template("user_dashboard.html",
                result="INVALID INPUT",
                risk="ERROR",
                prob=0
            )

        print("Received:", amount, hour, merchant, count, device, loc, intl)

        X = [amount, hour, merchant, count, device, loc, intl]

        scaled = scaler.transform([X])
        ml_p = model.predict_proba(scaled)[0][1]

        risk_score = (
            (amount/100000)*0.25 +
            merchant*0.2 +
            (count/20)*0.15 +
            (1-device)*0.15 +
            loc*0.15 +
            intl*0.1
        )

        p = (ml_p + risk_score) / 2
        p = max(0.01, min(p, 0.99))
        prob = round(p*100,2)

        if p < 0.25:
            risk = "LOW"
            result = "LEGIT"
        elif p < 0.6:
            risk = "MEDIUM"
            result = "LEGIT"
        else:
            risk = "HIGH"
            result = "FRAUD"

        global last_prediction
        last_prediction = {
            "result": result,
            "risk": risk,
            "probability": prob
        }

        db.session.add(Transaction(
            amount=amount,
            result=result,
            probability=prob
        ))
        db.session.commit()

    return render_template(
        "user_dashboard.html",
        result=result,
        risk=risk,
        prob=prob
    )

# ---------------- ADMIN ----------------
@app.route("/admin")
@login_required
def admin():
    tx = Transaction.query.order_by(Transaction.id.desc()).limit(10).all()
    return render_template("admin_dashboard.html", transactions=tx)

# ---------------- LIVE API ----------------
@app.route("/latest_prediction")
@login_required
def latest_prediction():
    return jsonify(last_prediction)

# ---------------- LOGOUT ----------------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)