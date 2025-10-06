import random
from typing import Iterable, List, Sequence, Tuple

import mitsuba as mi
import numpy as np
import pandas as pd
from pathlib import Path


def select_non_consecutive_vertices(vertices: Sequence[Tuple[float, float, float]],
                                    num_cars: int,
                                    min_gap: int = 2,
                                    seed: int = 42) -> List[Tuple[float, float, float]]:
    rng = random.Random(seed)
    total_vertices = len(vertices)
    if num_cars <= 0 or total_vertices == 0:
        return []

    selected_indices: List[int] = []
    current_index = rng.randint(0, min_gap)
    selected_indices.append(current_index)

    for _ in range(1, num_cars):
        next_index = current_index + min_gap + rng.randint(1, 3)
        if next_index >= total_vertices:
            next_index = (next_index - total_vertices) + rng.randint(0, min_gap)
        selected_indices.append(next_index)
        current_index = next_index

    return [vertices[i] for i in selected_indices]


def calculate_car_orientations_improved(car_positions: Iterable[Tuple[float, float, float]],
                                        all_vertices: Sequence[Tuple[float, float, float]],
                                        search_range: int = 5) -> List[Tuple[float, float, float]]:
    orientations: List[Tuple[float, float, float]] = []
    vertices_xy = np.array([v[:2] for v in all_vertices], dtype=np.float32)

    for pos in car_positions:
        car_xy = np.array(pos[:2], dtype=np.float32)
        dists = np.linalg.norm(vertices_xy - car_xy, axis=1)
        nearest_idx = int(np.argmin(dists))

        start = max(0, nearest_idx - search_range)
        end = min(len(all_vertices) - 1, nearest_idx + search_range)
        prev_vertex = np.array(all_vertices[start][:2], dtype=np.float32)
        next_vertex = np.array(all_vertices[end][:2], dtype=np.float32)
        direction = next_vertex - prev_vertex
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = direction / norm
        look_at_2d = car_xy + 3.0 * direction
        orientations.append((float(look_at_2d[0]), float(look_at_2d[1]), float(pos[2])))

    return orientations


def apply_positions_and_orientations(cars, positions, look_at_points) -> None:
    for car, pos, look_at in zip(cars, positions, look_at_points):
        car.position = mi.Point3f(float(pos[0]), float(pos[1]), float(pos[2]))
        car.look_at(mi.Point3f(float(look_at[0]), float(look_at[1]), float(look_at[2])))


def load_receiver_mesh_positions(csv_path: str, max_cars: int = None) -> List[Tuple[float, float, float]]:
    """
    Carrega posições de carros do arquivo receivers_mesh.csv
    
    Args:
        csv_path: Caminho para o arquivo receivers_mesh.csv
        max_cars: Número máximo de carros a carregar (None = todos)
        
    Returns:
        Lista de posições (x, y, z)
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
    
    # Carrega o CSV
    df = pd.read_csv(csv_path)
    
    # Verifica se tem as colunas necessárias
    required_cols = ['rx_id', 'x', 'y', 'z']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV deve conter as colunas: {required_cols}")
    
    # Limita o número de carros se especificado
    if max_cars is not None and max_cars > 0:
        df = df.head(max_cars)
    
    # Converte para lista de tuplas
    positions = [(row['x'], row['y'], row['z']) for _, row in df.iterrows()]
    
    return positions


def get_receiver_ids_from_mesh(csv_path: str, max_cars: int = None) -> List[str]:
    """
    Obtém os IDs dos receptores do arquivo receivers_mesh.csv
    
    Args:
        csv_path: Caminho para o arquivo receivers_mesh.csv
        max_cars: Número máximo de carros a carregar (None = todos)
        
    Returns:
        Lista de IDs dos receptores
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if max_cars is not None and max_cars > 0:
        df = df.head(max_cars)
    
    return df['rx_id'].tolist()
