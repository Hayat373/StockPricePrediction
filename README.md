# Stock Price Prediction Web App
A web-based application for predicting future stock prices using historical data and a Long Short-Term Memory (LSTM) neural network. Built with Python, Flask, Pandas, and TensorFlow, this app allows users to input a stock ticker (e.g., AAPL) and receive predicted closing prices for the next 5 days. The app fetches real-time stock data from Yahoo Finance using the yfinance library, preprocesses it with Pandas, trains an LSTM model, and displays predictions via a Flask web interface.

## Features

Real-Time Data: Fetches historical stock data (e.g., 1 year of closing prices) from Yahoo Finance using yfinance.
LSTM Model: Uses TensorFlow to train a deep learning model for time-series stock price prediction.
Web Interface: Simple Flask-based UI for entering stock tickers and viewing predictions.
Error Handling: Robust handling for invalid tickers (e.g., ZZZZ) and insufficient data (less than 60 days).
Modular Design: Organized code structure with separate modules for data fetching (data.py), model training (model.py), and web serving (app.py).

## Tech Stack

Python: 3.x
Flask: Web framework for the user interface
Pandas: Data preprocessing and manipulation
TensorFlow: LSTM neural network for predictions
yfinance: API for fetching stock data
scikit-learn: Data scaling
HTML/CSS: Frontend interface





## Installation

Clone the Repository:
```bash 
git clone https://github.com/Hayat373/StockPricePrediction.git
cd StockPricePrediction
```


Set Up a Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```


Install Dependencies:
```bash
pip install -r requirements.txt
```


Run the App:
```bash
python app.py
```


Access the app at http://127.0.0.1:5001.



## Usage

Open http://127.0.0.1:5001 in your browser.

Enter a stock ticker (e.g., AAPL for Apple).

Click "Predict" to view the predicted closing prices for the next 5 days.

Example output for AAPL (based on data up to August 8, 2025, closing at $229.35):
Predicted Prices for Next 5 Days:
- Day 1: $230.XX
- Day 2: $231.XX
- Day 3: $229.XX
- Day 4: $232.XX
- Day 5: $231.XX






## File Structure

``` StockPricePrediction/
├── app.py              # Flask app and main logic
├── data.py             # Data fetching and preprocessing
├── model.py            # LSTM model training and prediction
├── requirements.txt    # Dependencies
├── static/
│   └── styles.css      # CSS for styling the web interface
├── templates/
│   └── index.html      # HTML template for the UI
└── .gitignore          # Git ignore file for excluding venv/, __pycache__/, etc.
```

## Dependencies
See requirements.txt for the full list. Key dependencies include:

```
Flask==2.3.2
pandas==2.0.3
tensorflow==2.15.0
yfinance==0.2.40
numpy==1.25.2
scikit-learn==1.3.0
```

## Limitations

Model Simplicity: Predictions are based solely on historical closing prices using a basic LSTM model, which may not capture all market factors (e.g., news, volume).
Performance: Training the LSTM model on each request can be slow. Consider saving the trained model (model.save('lstm_model.h5')) for production use.
Data Dependency: Requires an internet connection for yfinance to fetch data from Yahoo Finance.

## Future Improvements

Add support for additional features (e.g., trading volume, technical indicators like moving averages).
Implement model persistence to avoid retraining on each request.
Enhance the UI with visualizations (e.g., Chart.js for historical vs. predicted prices).
Support alternative data sources (e.g., Alpha Vantage) for redundancy.

## Testing

Test Data: Successfully tested with AAPL, fetching 250 days of data (August 9, 2024, to August 8, 2025, closing at $229.35).
Run Tests:
```bash
python data.py  # Test data fetching and preprocessing
python model.py  # Test model training and prediction
```


Web App: Test at http://127.0.0.1:5001 with valid tickers (e.g., AAPL, MSFT) and invalid ones (e.g., ZZZZ).

## Contributing
Contributions are welcome! Please:

 ## Fork the repository.
 ```
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.
```


