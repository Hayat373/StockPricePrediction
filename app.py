import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from flask import Flask, render_template, request
from data import fetch_stock_data, preprocess_data
from model import build_model, train_model, predict_stock_price
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        ticker = request.form.get('ticker').upper()
        try:
            data = fetch_stock_data(ticker)
            if len(data) < 60:
                error = f"Not enough data for {ticker}. At least 60 days of data required."
            else:
                X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
                model = build_model(sequence_length=60)
                model = train_model(model, X_train, y_train, epochs=5)  # Reduced epochs for speed
                prediction = predict_stock_price(model, data, scaler)
                prediction = [round(p, 2) for p in prediction]
        except Exception as e:
            error = f"Failed to fetch data for {ticker}: {str(e)}. Please check the ticker or try again later."
    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True, port=5001)