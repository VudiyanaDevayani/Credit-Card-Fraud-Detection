import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'unmyeong'

DATABASE = 'database.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            phone_number TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        sender_account TEXT,
        receiver_account TEXT,
        amount REAL,
        spending_deviation_score REAL,
        velocity_score REAL,
        geo_anomaly_score REAL,
        transaction_type_encoded INTEGER,
        merchant_category_encoded INTEGER,
        location_encoded INTEGER,
        device_used_encoded INTEGER,
        payment_channel_encoded INTEGER,
        fraud_probability REAL,
        true_label INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone_number = request.form['phone_number']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')

        conn = get_db_connection()
        user_check = conn.execute(
            'SELECT * FROM users WHERE username = ? OR email = ?',
            (username, email)
        ).fetchone()

        if user_check:
            flash('Username or email already exists. Please choose a different one.', 'error')
            conn.close()
            return render_template('register.html')
        hashed_password = generate_password_hash(password)

        try:
            conn.execute(
                'INSERT INTO users (username, email, phone_number, password) VALUES (?, ?, ?, ?)',
                (username, email, phone_number, hashed_password)
            )
            conn.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('An error occurred during registration. Please try again.', 'error')
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/home')
def home():
    if 'username' not in session:
        flash('Please log in to access the home page.', 'error')
        return redirect(url_for('login'))

    conn = get_db_connection()
    user = conn.execute(
        'SELECT username, email, phone_number FROM users WHERE username = ?',
        (session['username'],)
    ).fetchone()
    conn.close()

    return render_template('home.html', user=user)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("login"))

    if request.method == 'POST':
        sender = request.form['sender_account']
        receiver = request.form['receiver_account']
        amount_in = float(request.form['amount'])
        spend_in = float(request.form['spending_deviation_score'])
        velocity_in = float(request.form['velocity_score'])
        geo_in = float(request.form['geo_anomaly_score'])
        transaction_type_in = int(request.form['transaction_type_encoded'])
        merchant_in = int(request.form['merchant_category_encoded'])
        location_in = int(request.form['location_encoded'])
        device_in = int(request.form['device_used_encoded'])
        channel_in = int(request.form['payment_channel_encoded'])

        import joblib
        import torch
        import numpy as np
        import pandas as pd
        import lightgbm as lgb
        import torch.nn as nn
        from torch_geometric.nn import SAGEConv

        scaler = joblib.load("model/scaler.pkl")
        metadata = joblib.load("model/model_metadata.pkl")
        lgbm = lgb.Booster(model_file="model/lightgbm_model.txt")

        class AutoEncoder(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 32), nn.ReLU(),
                    nn.Linear(32, 16), nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(16, 32), nn.ReLU(),
                    nn.Linear(32, input_dim)
                )
            def forward(self, x):
                encoded = self.encoder(x)
                reconstructed = self.decoder(encoded)
                return reconstructed

        ae = AutoEncoder(metadata["ae_input_dim"])
        ae.load_state_dict(torch.load("model/autoencoder.pth", map_location="cpu"))
        ae.eval()

        class GNNModel(nn.Module):
            def __init__(self, in_channels=1, hidden=64, out_channels=32):
                super().__init__()
                self.conv1 = SAGEConv(in_channels, hidden)
                self.conv2 = SAGEConv(hidden, out_channels)
            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = torch.relu(x)
                x = self.conv2(x, edge_index)
                return x

        gnn = GNNModel(
            metadata["gnn_in_channels"],
            metadata["gnn_hidden"],
            metadata["gnn_output_dim"]
        )
        gnn.load_state_dict(torch.load("model/gnn_model.pth", map_location="cpu"))
        gnn.eval()

        df = pd.read_csv("model/credit_card.csv", low_memory=False)

        df["sender_account_id"] = df["sender_account"].astype("category").cat.codes
        df["receiver_account_id"] = df["receiver_account"].astype("category").cat.codes

        num_nodes = max(df["sender_account_id"].max(), df["receiver_account_id"].max()) + 1

        edge_index = torch.tensor([
            df["sender_account_id"].values,
            df["receiver_account_id"].values
        ], dtype=torch.long)

        x = torch.zeros((num_nodes, 1))

        with torch.no_grad():
            gnn_embeddings = gnn(x, edge_index).numpy()

        match = df[(df["sender_account"] == sender) & (df["receiver_account"] == receiver)]

        if len(match) > 0:
            row = match.iloc[0]
            amount = row["amount"]
            spend = row["spending_deviation_score"]
            velocity = row["velocity_score"]
            geo = row["geo_anomaly_score"]
            transaction_type = row["transaction_type_encoded"]
            merchant = row["merchant_category_encoded"]
            location = row["location_encoded"]
            device = row["device_used_encoded"]
            channel = row["payment_channel_encoded"]
            true_label = int(row["is_fraud_encoded"])
        else:
            amount = amount_in
            spend = spend_in
            velocity = velocity_in
            geo = geo_in
            transaction_type = transaction_type_in
            merchant = merchant_in
            location = location_in
            device = device_in
            channel = channel_in
            true_label = None

        sender_id = df[df["sender_account"] == sender]["sender_account_id"].iloc[0]
        receiver_id = df[df["receiver_account"] == receiver]["receiver_account_id"].iloc[0]

        sender_vec = gnn_embeddings[sender_id]
        receiver_vec = gnn_embeddings[receiver_id]

        scaled = scaler.transform([[amount, spend, velocity, geo,
                                    transaction_type, merchant, location,
                                    device, channel]])[0]

        tensor_in = torch.tensor(scaled, dtype=torch.float).unsqueeze(0)

        with torch.no_grad():
            recon = ae(tensor_in)
            ae_loss = torch.mean((recon - tensor_in) ** 2).item()

        features = np.concatenate([
            scaled,
            [ae_loss],
            sender_vec,
            receiver_vec
        ]).reshape(1, -1)

        prob = float(lgbm.predict(features)[0])
        label = 1 if prob > 0.5 else 0

        conn = get_db_connection()
        conn.execute('''
            INSERT INTO predictions
            (username, sender_account, receiver_account, amount,
             spending_deviation_score, velocity_score, geo_anomaly_score,
             transaction_type_encoded, merchant_category_encoded,
             location_encoded, device_used_encoded, payment_channel_encoded,
             fraud_probability, true_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session['username'], sender, receiver, amount_in,
            spend_in, velocity_in, geo_in,
            transaction_type_in, merchant_in, location_in,
            device_in, channel_in,
            prob, true_label
        ))
        conn.commit()
        conn.close()

        return render_template("predict.html", result=prob, label=label, true_label=true_label)

    return render_template("predict.html")


@app.route('/history')
def history():
    if 'username' not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("login"))

    conn = get_db_connection()
    rows = conn.execute('''
        SELECT *
        FROM predictions
        WHERE username = ?
        ORDER BY timestamp DESC
    ''', (session['username'],)).fetchall()
    conn.close()

    return render_template('history.html', rows=rows)

@app.route('/analytics')
def analytics():
    if 'username' not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("login"))

    conn = get_db_connection()

    result = conn.execute('''
        SELECT true_label, COUNT(*) AS count
        FROM predictions
        WHERE username = ?
        GROUP BY true_label
    ''', (session['username'],)).fetchall()
    conn.close()

    count_true = 0
    count_false = 0
    count_unknown = 0  

    for r in result:
        if r["true_label"] == 1:
            count_true = r["count"]
        elif r["true_label"] == 0:
            count_false = r["count"]
        else:
            count_unknown = r["count"]

    return render_template(
        "analytics.html",
        count_true=count_true,
        count_false=count_false,
        count_unknown=count_unknown
    )


@app.route('/datascience')
def datascience():
    return render_template('datascience.html')

@app.route('/exsisting')
def exsisting():
    return render_template('exsisting.html')

@app.route('/proposed')
def proposed():
    return render_template('proposed.html')

if __name__ == '__main__':
    app.run(debug=True)
