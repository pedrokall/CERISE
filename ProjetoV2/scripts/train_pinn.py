import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense

C_MPS = 3e8


class PhysicsInformedNN(Model):
    def __init__(self, neural_network, scaler_y, tx_position, freq_hz=2.4e9, gamma=0.1):
        super().__init__()
        self.nn = neural_network
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.tx_position_tf = tf.constant(tx_position, dtype=tf.float32)
        self.y_mean_tf = tf.constant(scaler_y.mean_, dtype=tf.float32)
        self.y_scale_tf = tf.constant(scaler_y.scale_, dtype=tf.float32)
        self.freq_hz = tf.constant(freq_hz, dtype=tf.float32)

    def call(self, inputs):
        return self.nn(inputs["model_input"])  # dict inputs

    def _inverse_transform_y(self, y_scaled):
        return (y_scaled * self.y_scale_tf) + self.y_mean_tf

    def _compute_physics_loss(self, rx_positions_original, y_pred_scaled):
        y_pred_db = self._inverse_transform_y(y_pred_scaled)
        distance = tf.norm(self.tx_position_tf - rx_positions_original, axis=1)
        distance = tf.maximum(distance, 1e-6)
        fspl_db = 20. * tf.math.log(distance) / tf.math.log(10.) + \
                  20. * tf.math.log(self.freq_hz) / tf.math.log(10.) + \
                  20. * tf.math.log(4. * np.pi / C_MPS) / tf.math.log(10.)
        rss_fspl = -fspl_db
        residual = tf.nn.relu(tf.reshape(y_pred_db, [-1]) - rss_fspl)
        return tf.reduce_mean(residual)

    def train_step(self, data):
        inputs, y_true_scaled = data
        with tf.GradientTape() as tape:
            y_pred_scaled = self(inputs, training=True)
            data_loss = tf.reduce_mean(tf.square(y_true_scaled - y_pred_scaled))
            physics_loss = self._compute_physics_loss(inputs["physics_input"], y_pred_scaled)
            total_loss = data_loss + self.gamma * physics_loss
        grads = tape.gradient(total_loss, self.nn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.nn.trainable_variables))
        return {"loss": total_loss}


def main():
    parser = argparse.ArgumentParser(description="Treinar PINN para prever RSS com restrição FSPL")
    parser.add_argument("--data", required=True)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--freq", type=float, default=2.4e9)
    parser.add_argument("--tx", nargs=3, type=float, default=[21.18, -132.4, 18.76])
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

    base = Sequential([
        Dense(128, activation='relu', input_shape=(3,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1),
    ])

    pinn = PhysicsInformedNN(base, scaler_y, tx_position=np.array(args.tx, dtype=np.float32), freq_hz=args.freq, gamma=args.gamma)
    pinn.compile(optimizer='adam', loss='mean_squared_error')

    train_ds = tf.data.Dataset.from_tensor_slices((
        {"model_input": X_train_scaled, "physics_input": X_train},
        y_train_scaled
    )).batch(8)

    pinn.fit(train_ds, epochs=200, verbose=1)

    preds_scaled = pinn.predict({"model_input": X_test_scaled, "physics_input": X_test}, verbose=0)
    preds_db = scaler_y.inverse_transform(preds_scaled)

    mae = mean_absolute_error(y_test, preds_db)
    print(f"MAE PINN: {mae:.2f} dB")


if __name__ == "__main__":
    main()
