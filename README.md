# Quantum ML In Practice with Qubits Workshop

## Name : VISHNU KM
## Reg no.: 212223240185

 

!pip install pennylane --quiet
     

!pip install pennylane
     
Requirement already satisfied: pennylane in /usr/local/lib/python3.11/dist-packages (0.41.1)
Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from pennylane) (2.2.5)
Requirement already satisfied: scipy in /usr/local/lib/python3.11/dist-packages (from pennylane) (1.15.3)
Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from pennylane) (3.4.2)
Requirement already satisfied: rustworkx>=0.14.0 in /usr/local/lib/python3.11/dist-packages (from pennylane) (0.16.0)
Requirement already satisfied: autograd in /usr/local/lib/python3.11/dist-packages (from pennylane) (1.8.0)
Requirement already satisfied: tomlkit in /usr/local/lib/python3.11/dist-packages (from pennylane) (0.13.2)
Requirement already satisfied: appdirs in /usr/local/lib/python3.11/dist-packages (from pennylane) (1.4.4)
Requirement already satisfied: autoray>=0.6.11 in /usr/local/lib/python3.11/dist-packages (from pennylane) (0.7.1)
Requirement already satisfied: cachetools in /usr/local/lib/python3.11/dist-packages (from pennylane) (5.5.2)
Requirement already satisfied: pennylane-lightning>=0.41 in /usr/local/lib/python3.11/dist-packages (from pennylane) (0.41.1)
Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from pennylane) (2.32.3)
Requirement already satisfied: typing-extensions in /usr/local/lib/python3.11/dist-packages (from pennylane) (4.13.2)
Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from pennylane) (24.2)
Requirement already satisfied: diastatic-malt in /usr/local/lib/python3.11/dist-packages (from pennylane) (2.15.2)
Requirement already satisfied: scipy-openblas32>=0.3.26 in /usr/local/lib/python3.11/dist-packages (from pennylane-lightning>=0.41->pennylane) (0.3.29.0.0)
Requirement already satisfied: astunparse in /usr/local/lib/python3.11/dist-packages (from diastatic-malt->pennylane) (1.6.3)
Requirement already satisfied: gast in /usr/local/lib/python3.11/dist-packages (from diastatic-malt->pennylane) (0.6.0)
Requirement already satisfied: termcolor in /usr/local/lib/python3.11/dist-packages (from diastatic-malt->pennylane) (3.1.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->pennylane) (3.4.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->pennylane) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->pennylane) (2.4.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->pennylane) (2025.4.26)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.11/dist-packages (from astunparse->diastatic-malt->pennylane) (0.45.1)
Requirement already satisfied: six<2.0,>=1.6.1 in /usr/local/lib/python3.11/dist-packages (from astunparse->diastatic-malt->pennylane) (1.17.0)

from IPython import get_ipython
from IPython.display import display
import matplotlib.patches as mpatches
import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt






     
Dataset points (X):
[[0.91698428 1.42690919]
 [1.97028309 3.07221562]
 [2.90262007 2.28991662]
 [2.25553479 1.40641531]
 [0.85141651 2.10075606]
 [1.32271876 2.2761343 ]
 [0.4166084  0.7132748 ]
 [2.68364791 0.71325635]
 [2.18852718 1.05950525]
 [1.92903045 2.89372922]
 [0.91124544 1.31169918]
 [2.88747624 1.02962703]
 [2.86650576 1.57060078]
 [0.91640578 1.3312293 ]
 [0.55760119 2.70773671]
 [2.23220169 1.75384493]
 [2.15835106 1.99777273]
 [0.85981769 1.40410182]
 [1.38029732 2.26326426]
 [2.30376512 0.26179465]
 [1.52156792 3.14159265]
 [1.27953505 0.08057915]
 [1.12405755 2.12665217]
 [0.04117036 1.36884346]
 [0.10503099 2.13994791]
 [0.49053405 0.68176591]
 [2.08788208 2.13241501]
 [2.17825923 1.07062916]
 [1.07873462 3.05142131]
 [2.79491777 0.52229491]
 [1.69747629 0.83115751]
 [1.49698214 0.90942503]
 [2.22099117 3.03655348]
 [0.67872048 0.39950378]
 [0.96289978 2.22363543]
 [1.92669459 2.24113115]
 [0.12245006 1.03023241]
 [1.95594335 0.0091382 ]
 [2.73632913 2.40787739]
 [2.25818251 1.79934794]
 [1.11321422 1.00216088]
 [1.78976139 0.14260179]
 [0.98983426 0.14556   ]
 [1.00480128 2.85253376]
 [0.95334465 1.1564307 ]
 [0.96602426 0.10414364]
 [2.38047099 2.80981082]
 [2.49986406 0.42436122]
 [1.42415584 2.28891528]
 [0.90718919 1.97955247]
 [0.05423024 1.34899969]
 [0.49568509 2.54645615]
 [2.30033352 1.32187394]
 [1.9092926  0.98214014]
 [0.         1.66136701]
 [1.61371914 2.41623445]
 [3.14159265 1.37631057]
 [1.38856312 3.05062797]
 [0.01200086 1.7738563 ]
 [2.93473026 2.1735505 ]
 [0.80690722 2.88402848]
 [1.73689286 0.03573035]
 [2.52874389 2.93103362]
 [1.07810188 0.85750493]
 [2.64870524 2.62071117]
 [1.05150891 2.14896304]
 [2.0020449  0.93133067]
 [1.19697581 0.82352396]
 [2.24697047 1.97349303]
 [2.18310453 1.60074137]
 [0.87901846 1.80207671]
 [2.84195659 0.85269497]
 [0.73756081 1.80570553]
 [1.71661214 2.42418261]
 [1.33805246 2.98902419]
 [1.15338002 0.93276468]
 [2.2034121  0.23856107]
 [2.21681913 2.2790428 ]
 [3.05066179 1.27904634]
 [1.98050894 0.91547812]
 [2.06344492 2.11931741]
 [0.78125866 1.5502163 ]
 [0.60190146 0.58272886]
 [0.20469752 2.54203132]
 [1.73214889 2.24545137]
 [1.57093237 2.35809578]
 [1.68544905 0.80290185]
 [1.49436612 0.        ]
 [0.25916857 0.85172544]
 [2.08598542 1.3009803 ]
 [3.04837233 1.62439803]
 [2.26563266 1.72748144]
 [0.3061137  2.26587594]
 [2.95836947 2.00779978]
 [0.17086742 1.87696641]
 [1.46585883 0.78530607]
 [2.17499765 1.44047052]
 [0.83445442 1.58687515]
 [1.46633533 0.99663749]
 [1.14259131 0.87972904]]

Dataset labels (Y):
[1 0 0 1 1 1 0 0 1 0 1 0 0 1 0 1 1 1 1 0 0 0 1 0 0 0 1 1 0 0 1 1 0 0 1 1 0
 0 0 1 1 0 0 0 1 0 0 0 1 1 0 0 1 1 0 1 0 0 0 0 0 0 0 1 0 1 1 1 1 1 1 0 1 1
 0 1 0 1 0 1 1 1 0 0 1 1 1 0 0 1 0 1 0 0 0 1 1 1 1 1]
Classical ML Accuracy (Logistic Regression): 0.49
Epoch 10: QML Accuracy = 0.53
Epoch 20: QML Accuracy = 0.53
Epoch 30: QML Accuracy = 0.53
Epoch 40: QML Accuracy = 0.54
Epoch 50: QML Accuracy = 0.55
Final QML Accuracy: 0.55

Generate concentric circles dataset

# Generate concentric circles dataset
X, Y = make_circles(n_samples=100, noise=0.05, factor=0.5)
scaler = MinMaxScaler()
X = scaler.fit_transform(X) * np.pi  # scale features to [0, pi]
     
Print dataset points and labels

# Print dataset points and labels
print("Dataset points (X):")
print(X)
print("\nDataset labels (Y):")
print(Y)
     

# Classical ML baseline: Logistic Regression
clf = LogisticRegression()
clf.fit(X, Y)
Y_pred_classical = clf.predict(X)
acc_classical = accuracy_score(Y, Y_pred_classical)
print("Classical ML Accuracy (Logistic Regression):", acc_classical)
     


# QML Setup
dev = qml.device("default.qubit", wires=2)
     

def feature_map(x):
    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(x[0], wires=1)
    qml.RY(x[1], wires=0)

     

def variational_layer(weights):
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(weights[2], wires=1)
     

@qml.qnode(dev)
def circuit(x, weights):
    feature_map(x)
    variational_layer(weights)
    return qml.expval(qml.PauliZ(0))
     

def variational_classifier(weights, x):
    return circuit(x, weights)
     

def cost(weights, X, Y):
    predictions = [variational_classifier(weights, x) for x in X]
    predictions = np.array(predictions, dtype=np.float64)
    labels = np.array(2 * Y - 1, dtype=np.float64)  # convert {0,1} to {-1,1}
    return np.mean((predictions - labels)**2)
     

# Training QML model
np.random.seed(42)
weights = np.random.randn(3, requires_grad=True)
opt = qml.GradientDescentOptimizer(stepsize=0.3)

epochs = 50
for i in range(epochs):
    weights = opt.step(lambda w: cost(w, X, Y), weights)
    if (i + 1) % 10 == 0:
        preds = [variational_classifier(weights, x) for x in X]
        preds = np.array(preds, dtype=np.float64)
        pred_labels = [1 if p > 0 else 0 for p in preds]
        acc_qml = accuracy_score(Y, pred_labels)
        print(f"Epoch {i+1}: QML Accuracy = {acc_qml:.2f}")
     

# Final QML accuracy and predictions
preds = [variational_classifier(weights, x) for x in X]
preds = np.array(preds, dtype=np.float64)
pred_labels_qml = [1 if p > 0 else 0 for p in preds]
acc_qml = accuracy_score(Y, pred_labels_qml)
print("Final QML Accuracy:", acc_qml)

# Plot predictions for classical ML and QML side-by-side


# Define legend patches
blue_patch = mpatches.Patch(color='blue', label='Class 0')
red_patch = mpatches.Patch(color='red', label='Class 1')

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

sc1 = axs[0].scatter(X[:, 0], X[:, 1], c=Y_pred_classical, cmap='coolwarm', s=40)
axs[0].set_title(f"Classical Logistic Regression\nAccuracy: {acc_classical:.2f}")
axs[0].set_xlabel("X1")
axs[0].set_ylabel("X2")
axs[0].grid(True)
axs[0].legend(handles=[blue_patch, red_patch])

sc2 = axs[1].scatter(X[:, 0], X[:, 1], c=pred_labels_qml, cmap='coolwarm', s=40)
axs[1].set_title(f"Quantum ML Classifier\nAccuracy: {acc_qml:.2f}")
axs[1].set_xlabel("X1")
axs[1].set_ylabel("X2")
axs[1].grid(True)
axs[1].legend(handles=[blue_patch, red_patch])

plt.tight_layout()
plt.show()
     

!pip install wbdata
     
Requirement already satisfied: wbdata in /usr/local/lib/python3.11/dist-packages (1.0.0)
Requirement already satisfied: appdirs<2.0,>=1.4 in /usr/local/lib/python3.11/dist-packages (from wbdata) (1.4.4)
Requirement already satisfied: backoff<3.0.0,>=2.2.1 in /usr/local/lib/python3.11/dist-packages (from wbdata) (2.2.1)
Requirement already satisfied: cachetools<6.0.0,>=5.3.2 in /usr/local/lib/python3.11/dist-packages (from wbdata) (5.5.2)
Requirement already satisfied: dateparser<2.0.0,>=1.2.0 in /usr/local/lib/python3.11/dist-packages (from wbdata) (1.2.1)
Requirement already satisfied: decorator<6.0.0,>=5.1.1 in /usr/local/lib/python3.11/dist-packages (from wbdata) (5.2.1)
Requirement already satisfied: requests<3.0,>=2.0 in /usr/local/lib/python3.11/dist-packages (from wbdata) (2.32.3)
Requirement already satisfied: shelved-cache<0.4.0,>=0.3.1 in /usr/local/lib/python3.11/dist-packages (from wbdata) (0.3.1)
Requirement already satisfied: tabulate<0.9.0,>=0.8.5 in /usr/local/lib/python3.11/dist-packages (from wbdata) (0.8.10)
Requirement already satisfied: python-dateutil>=2.7.0 in /usr/local/lib/python3.11/dist-packages (from dateparser<2.0.0,>=1.2.0->wbdata) (2.9.0.post0)
Requirement already satisfied: pytz>=2024.2 in /usr/local/lib/python3.11/dist-packages (from dateparser<2.0.0,>=1.2.0->wbdata) (2025.2)
Requirement already satisfied: regex!=2019.02.19,!=2021.8.27,>=2015.06.24 in /usr/local/lib/python3.11/dist-packages (from dateparser<2.0.0,>=1.2.0->wbdata) (2024.11.6)
Requirement already satisfied: tzlocal>=0.2 in /usr/local/lib/python3.11/dist-packages (from dateparser<2.0.0,>=1.2.0->wbdata) (5.3.1)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests<3.0,>=2.0->wbdata) (3.4.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests<3.0,>=2.0->wbdata) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests<3.0,>=2.0->wbdata) (2.4.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests<3.0,>=2.0->wbdata) (2025.4.26)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.7.0->dateparser<2.0.0,>=1.2.0->wbdata) (1.17.0)

import wbdata
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to fetch macroeconomic data from World Bank
def fetch_wb_data(indicator, country='USA', start_year=1960, end_year=2023):
    try:
        data = wbdata.get_data(indicator=indicator, country=country)
        df = pd.DataFrame([
            {'date': pd.to_datetime(d['date']), 'value': d['value']}
            for d in data if d['value'] is not None
        ])
        df = df.sort_values('date').set_index('date')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna()
        return df[(df.index.year >= start_year) & (df.index.year <= end_year)]
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Function to create additional features
def create_features(df):
    df = df.copy()
    df['lag_1'] = df['value'].shift(1)  # 1-year lag
    df['lag_2'] = df['value'].shift(2)  # 2-year lag
    df['rolling_mean'] = df['value'].rolling(window=3).mean()  # 3-year rolling mean
    df['rolling_std'] = df['value'].rolling(window=3).std()  # 3-year rolling std
    return df.dropna()

# Function to preprocess data
def preprocess_data(data, look_back=5, feature_columns=['value', 'lag_1', 'lag_2', 'rolling_mean', 'rolling_std']):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature_columns])

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i, 0])  # Predict 'value' column
    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test, scaler

# Function to build and train LSTM model
def build_lstm_model(X_train, y_train, look_back=5, feature_count=5):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(look_back, feature_count)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1, callbacks=[early_stopping])
    return model

# Function to make predictions
def make_predictions(model, X_test, scaler, feature_columns=['value', 'lag_1', 'lag_2', 'rolling_mean', 'rolling_std']):
    predictions = model.predict(X_test)
    # Inverse transform predictions (only for 'value' column)
    dummy = np.zeros((len(predictions), len(feature_columns)))
    dummy[:, 0] = predictions[:, 0]
    predictions = scaler.inverse_transform(dummy)[:, 0]
    return predictions

# Function to inverse transform actual values
def inverse_transform_actual(y_test, scaler, feature_columns=['value', 'lag_1', 'lag_2', 'rolling_mean', 'rolling_std']):
    dummy = np.zeros((len(y_test), len(feature_columns)))
    dummy[:, 0] = y_test
    return scaler.inverse_transform(dummy)[:, 0]

# Main execution
def main():
    # Fetch GDP growth data (annual %)
    indicator = 'NY.GDP.MKTP.KD.ZG'  # GDP growth (annual %)
    print("Fetching GDP growth data from World Bank...")
    gdp_data = fetch_wb_data(indicator)
    if gdp_data is None:
        return

    # Create features
    data_with_features = create_features(gdp_data)

    # Preprocess data
    look_back = 5
    feature_columns = ['value', 'lag_1', 'lag_2', 'rolling_mean', 'rolling_std']
    X_train, y_train, X_test, y_test, scaler = preprocess_data(data_with_features, look_back, feature_columns)

    # Train model
    print("Training LSTM model...")
    model = build_lstm_model(X_train, y_train, look_back, len(feature_columns))

    # Make predictions
    predictions = make_predictions(model, X_test, scaler, feature_columns)
    y_test_scaled = inverse_transform_actual(y_test, scaler, feature_columns)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
    print(f"Test RMSE: {rmse:.4f}")

    # Plot actual vs predicted
    test_dates = data_with_features.index[-len(y_test):]
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_scaled, label="Actual GDP Growth (%)")
    plt.plot(test_dates, predictions, label="Predicted GDP Growth (%)")
    plt.title("Actual vs Predicted GDP Growth")
    plt.xlabel("Year")
    plt.ylabel("GDP Growth (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Forecast future values (4 years ahead)
    future_steps = 4
    last_sequence = X_test[-1]
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(future_steps):
        current_sequence = np.reshape(current_sequence, (1, look_back, len(feature_columns)))
        next_pred = model.predict(current_sequence, verbose=0)
        future_predictions.append(next_pred[0, 0])
        # Shift sequence and append new prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred[0, 0]
        # Approximate other features (e.g., lags and rolling stats)
        for i in range(1, len(feature_columns)):
            current_sequence[0, -1, i] = current_sequence[0, -2, i]  # Carry forward last known value

    future_predictions = inverse_transform_actual(np.array(future_predictions), scaler, feature_columns)

    # Generate future dates
    last_date = data_with_features.index[-1]
    future_dates = [last_date + timedelta(days=365 * (i + 1)) for i in range(future_steps)]

    # Plot future forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, future_predictions, label="Forecasted GDP Growth (%)", marker='o')
    plt.title("Future GDP Growth Forecast")
    plt.xlabel("Year")
    plt.ylabel("GDP Growth (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

if _name_ == "_main_":
    main()
     
Fetching GDP growth data from World Bank...
Training LSTM model...
/usr/local/lib/python3.11/dist-packages/keras/src/layers/rnn/rnn.py:200: UserWarning: Do not pass an input_shape/input_dim argument to a layer. When using Sequential models, prefer using an Input(shape) object as the first layer in the model instead.
  super()._init_(**kwargs)
Epoch 1/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 6s 25ms/step - loss: 0.2493
Epoch 2/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: 0.0567
Epoch 3/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: 0.0846
Epoch 4/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 0.0923
Epoch 5/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: 0.0609
Epoch 6/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step - loss: 0.0682
Epoch 7/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0750
Epoch 8/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0541
Epoch 9/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: 0.0499
Epoch 10/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0606
Epoch 11/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: 0.0474
Epoch 12/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: 0.0459
Epoch 13/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: 0.0479
Epoch 14/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0454
Epoch 15/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0521
Epoch 16/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 0.0545
Epoch 17/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: 0.0454
Epoch 18/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: 0.0419 
Epoch 19/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: 0.0524
Epoch 20/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 0.0389
Epoch 21/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: 0.0357
Epoch 22/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0437
Epoch 23/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0447
Epoch 24/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 0.0467
Epoch 25/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0462
Epoch 26/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0472
Epoch 27/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0469
Epoch 28/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 0.0345 
Epoch 29/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 0.0548
Epoch 30/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: 0.0370
Epoch 31/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - loss: 0.0394 
Epoch 32/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 0.0418
Epoch 33/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: 0.0482
Epoch 34/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: 0.0483
Epoch 35/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0437
Epoch 36/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: 0.0459
Epoch 37/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step - loss: 0.0445
Epoch 38/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 0.0338
Epoch 39/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: 0.0413
Epoch 40/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 0.0388
Epoch 41/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: 0.0387
Epoch 42/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0461
Epoch 43/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0485
Epoch 44/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0541
Epoch 45/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0458
Epoch 46/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0398
Epoch 47/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: 0.0457
Epoch 48/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0428
Epoch 49/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 0.0458
Epoch 50/100
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: 0.0394
WARNING:tensorflow:5 out of the last 11 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x7e0985f009a0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 364ms/step
Test RMSE: 1.7360

WARNING:tensorflow:5 out of the last 11 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x7e0985f009a0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
