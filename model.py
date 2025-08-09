import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

def build_model(sequence_length):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def predict_stock_price(model, data, scaler, sequence_length=60, future_days=5):
    last_sequence = data[-sequence_length:]
    last_sequence = scaler.transform(last_sequence.reshape(-1, 1))
    last_sequence = np.reshape(last_sequence, (1, sequence_length, 1))
    
    predicted_prices = []
    for _ in range(future_days):
        pred = model.predict(last_sequence, verbose=0)
        predicted_prices.append(pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1, 0] = pred[0, 0]
    
    predicted_prices = np.array(predicted_prices).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices.flatten()

if __name__ == "__main__":
    from data import fetch_stock_data, preprocess_data
    try:
        data = fetch_stock_data("AAPL")
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
        model = build_model(sequence_length=60)
        model = train_model(model, X_train, y_train, epochs=1)  # Small epochs for testing
        predictions = predict_stock_price(model, data, scaler)
        print(f"Predicted prices: {predictions}")
    except Exception as e:
        print(f"Error in test: {e}")