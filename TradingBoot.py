from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

app = Flask(__name__)

# Load the Alpaca stock price data
# ... Load and preprocess your data ...

# Prepare the data for LSTM training
# ... Prepare your data using create_dataset function ...

# Build and train the LSTM model
# ... Build and train your LSTM model ...

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Normalize the input data
    input_data = np.array(data).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_input_data = scaler.fit_transform(input_data)

    # Reshape the input data for LSTM [samples, time steps, features]
    input_data = np.reshape(scaled_input_data, (scaled_input_data.shape[0], 1, 1))

    # Make the prediction
    predicted_price = model.predict(input_data)

    # Inverse scale the prediction
    predicted_price = scaler.inverse_transform(predicted_price)

    # Return the predicted price as JSON response
    return jsonify({'predicted_price': predicted_price[0][0]})

if __name__ == '__main__':
    app.run()
