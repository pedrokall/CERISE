from typing import List, Tuple

import numpy as np
import tensorflow as tf
from sionna.rt import PlanarArray, Transmitter, Receiver, PathSolver


def setup_tx_rx(scene, tx_position: Tuple[float, float, float], rx_positions: List[Tuple[float, float, float]], display_radius: float = 2.0):
    scene.remove("tx")
    scene.add(Transmitter("tx", position=list(tx_position), display_radius=display_radius))

    for i, pos in enumerate(rx_positions):
        name = f"rx-{i}"
        scene.remove(name)
        scene.add(Receiver(name, position=[pos[0], pos[1], pos[2]], display_radius=display_radius))

    scene.tx_array = PlanarArray(num_cols=1, num_rows=1, pattern="iso", polarization="V")
    scene.rx_array = scene.tx_array


def compute_paths(scene, max_depth: int = 5):
    solver = PathSolver()
    return solver(scene, max_depth=max_depth)


def rss_from_paths(paths) -> tf.Tensor:
    # paths.a shape: [num_rx, 1, 1, 1, num_paths]
    atenuacoes_tensor = paths.a[0]
    atenuacoes_squeezed = tf.squeeze(atenuacoes_tensor)
    if len(atenuacoes_squeezed.shape) != 2:
        atenuacoes_squeezed = tf.squeeze(atenuacoes_tensor, axis=[1, 2, 3])
    power_per_path = tf.square(tf.abs(atenuacoes_squeezed))
    total_power_per_rx = tf.reduce_sum(power_per_path, axis=1)
    epsilon = 1e-10
    rss_db_per_rx = 10.0 * tf.math.log(total_power_per_rx + epsilon) / tf.math.log(10.0)
    return rss_db_per_rx
