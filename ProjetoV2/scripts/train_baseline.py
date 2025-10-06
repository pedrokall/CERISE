import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def main():
    parser = argparse.ArgumentParser(description="Treinar modelo baseline (MLP) para prever RSS a partir de (rx_x, rx_y, rx_z)")
    parser.add_argument("--data", required=True, help="CSV com colunas rx_x, rx_y, rx_z, rss_db")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = df[["rx_x", "rx_y", "rx_z"]].values.astype(np.float32)
    y = df[["rss_db"]].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler_X.transform(X_test).astype(np.float32)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train).astype(np.float32)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(3,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train_scaled, y_train_scaled, epochs=150, batch_size=8, validation_split=0.2, verbose=0)

    preds_scaled = model.predict(X_test_scaled, verbose=0)
    preds_db = scaler_y.inverse_transform(preds_scaled)

    mae = mean_absolute_error(y_test, preds_db)
    print(f"MAE baseline: {mae:.2f} dB")


if __name__ == "__main__":
    main()
