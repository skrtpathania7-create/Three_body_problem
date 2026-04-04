"""
app.py  —  Three-Body Stability Prediction API
───────────────────────────────────────────────
Trains the model on startup, then serves predictions via HTTP.
Your friend calls this from the frontend.

Usage:
    python app.py
    → runs on http://localhost:5000

Endpoints:
    POST /predict        — predict stability from initial conditions
    GET  /health         — check the server is alive
    GET  /random         — get a random set of initial conditions to demo
"""

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── CORS headers (so the frontend can call from a different port) ──────────
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.route('/predict', methods=['OPTIONS'])
@app.route('/random',  methods=['OPTIONS'])
def options():
    return '', 204

# ── Physics sim (needed for the demo trajectory endpoint) ─────────────────
G, DT, MAX_STEPS = 1.0, 0.05, 800

def get_accel(p1, p2, m2):
    r = p2 - p1
    d = np.linalg.norm(r) + 0.1
    return G * m2 * r / d**3

def simulate_trajectory(pos, vel, steps=300):
    """Return (trajectory list, escaped bool)"""
    pos = np.array(pos, dtype=float).reshape(3, 2)
    vel = np.array(vel, dtype=float).reshape(3, 2)
    m   = np.array([10.0, 10.0, 10.0])
    traj = [pos.tolist()]
    for _ in range(steps):
        a = np.zeros((3, 2))
        a[0] = get_accel(pos[0], pos[1], m[1]) + get_accel(pos[0], pos[2], m[2])
        a[1] = get_accel(pos[1], pos[0], m[0]) + get_accel(pos[1], pos[2], m[2])
        a[2] = get_accel(pos[2], pos[0], m[0]) + get_accel(pos[2], pos[1], m[1])
        vel += a * DT
        pos += vel * DT
        traj.append(pos.tolist())
        if np.any(np.abs(pos) > 50):
            return traj, True
    return traj, False

# ── Train model on startup ─────────────────────────────────────────────────
print("Loading data and training model...")
try:
    df = pd.read_csv('orbital_data.csv')
    print(f"  Loaded {len(df)} simulations.")
except FileNotFoundError:
    raise FileNotFoundError(
        "orbital_data.csv not found. Run hakathon.py first to generate training data."
    )

X = df.drop('stable', axis=1)
y = df['stable']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)
sw   = compute_sample_weight('balanced', y_train)

model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu', max_iter=2000,
    random_state=42, early_stopping=True,
    validation_fraction=0.1
)
model.fit(X_tr, y_train, sample_weight=sw)

acc = accuracy_score(y_test, model.predict(X_te))
print(f"  Model ready. Test accuracy: {acc*100:.1f}%")
print("  Server starting at http://localhost:5000\n")

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({
        'status':   'ok',
        'accuracy': round(acc * 100, 1),
        'samples':  len(df)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body (JSON):
    {
        "x1": 3.2,  "y1": -1.5,
        "x2": -5.0, "y2": 2.1,
        "x3": 1.8,  "y3": 4.0,
        "vx1": 0.1, "vy1": -0.2,
        "vx2": 0.3, "vy2": 0.1,
        "vx3": -0.2,"vy3": 0.0,
        "simulate": true   // optional: also return trajectory for animation
    }
    """
    try:
        data = request.get_json()
        keys = ['x1','y1','x2','y2','x3','y3','vx1','vy1','vx2','vy2','vx3','vy3']
        features = np.array([[data[k] for k in keys]])
        features_scaled = scaler.transform(features)

        pred  = int(model.predict(features_scaled)[0])
        proba = model.predict_proba(features_scaled)[0]
        conf  = float(max(proba))

        result = {
            'prediction':  'STABLE' if pred == 1 else 'UNSTABLE',
            'stable':       pred == 1,
            'confidence':   round(conf * 100, 1),
            'stable_prob':  round(float(proba[1]) * 100, 1),
            'unstable_prob':round(float(proba[0]) * 100, 1),
        }

        # Optionally simulate trajectory for frontend animation
        if data.get('simulate', False):
            pos = [[data['x1'], data['y1']],
                   [data['x2'], data['y2']],
                   [data['x3'], data['y3']]]
            vel = [[data['vx1'], data['vy1']],
                   [data['vx2'], data['vy2']],
                   [data['vx3'], data['vy3']]]
            traj, escaped = simulate_trajectory(pos, vel, steps=400)
            result['trajectory'] = traj
            result['escaped']    = escaped

        return jsonify(result)

    except KeyError as e:
        return jsonify({'error': f'Missing field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/random')
def random_conditions():
    """
    GET /random
    Returns a random set of initial conditions + the model's prediction.
    Perfect for the demo — frontend can hit this to get pre-filled inputs.
    """
    np.random.seed()
    pos = np.random.uniform(-8, 8, (3, 2))
    vel = np.random.uniform(-0.4, 0.4, (3, 2))
    feat = np.hstack([pos.flatten(), vel.flatten()]).reshape(1, -1)
    feat_sc = scaler.transform(feat)
    pred  = int(model.predict(feat_sc)[0])
    proba = model.predict_proba(feat_sc)[0]

    return jsonify({
        'x1': round(float(pos[0,0]),3), 'y1': round(float(pos[0,1]),3),
        'x2': round(float(pos[1,0]),3), 'y2': round(float(pos[1,1]),3),
        'x3': round(float(pos[2,0]),3), 'y3': round(float(pos[2,1]),3),
        'vx1':round(float(vel[0,0]),3), 'vy1':round(float(vel[0,1]),3),
        'vx2':round(float(vel[1,0]),3), 'vy2':round(float(vel[1,1]),3),
        'vx3':round(float(vel[2,0]),3), 'vy3':round(float(vel[2,1]),3),
        'prediction':   'STABLE' if pred == 1 else 'UNSTABLE',
        'stable':        pred == 1,
        'confidence':    round(float(max(proba)) * 100, 1),
    })


if __name__ == '__main__':
    app.run(debug=False, port=5000)










