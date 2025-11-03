"""
Funções auxiliares para o modelo de propagação loss
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import os
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh não está instalado. Instale com: pip install trimesh")


def calculate_distance_3d(tx_x, tx_y, tx_z, rx_x, rx_y, rx_z):
    """
    Calcula distância euclidiana 3D entre Tx e Rx
    
    Args:
        tx_x, tx_y, tx_z: Coordenadas do transmissor (Tx)
        rx_x, rx_y, rx_z: Coordenadas do receptor (Rx)
    
    Returns:
        Distância em metros (array)
    """
    distance = np.sqrt(
        (rx_x - tx_x)**2 + (rx_y - tx_y)**2 + (rx_z - tx_z)**2
    )
    return distance


def calculate_free_space_loss(distance_m, frequency_mhz):
    """
    Calcula Free Space Path Loss (FSPL) em dB
    
    FSPL = 20*log10(d) + 20*log10(f) + 32.44
    onde d está em km e f está em MHz
    
    Args:
        distance_m: Distância em metros
        frequency_mhz: Frequência em MHz
    
    Returns:
        FSPL em dB (array)
    """
    distance_km = distance_m / 1000.0
    fspl = 20 * np.log10(distance_km) + 20 * np.log10(frequency_mhz) + 32.44
    return fspl


def load_training_data(frequency, tx_list, train_proportion=1.0, base_path=None):
    """
    Carrega dados de treinamento da pasta Training
    
    Args:
        frequency: Frequência escolhida ('800MHz', '7GHz', '28GHz')
        tx_list: Lista de números de Tx para carregar (ex: [1, 2, 3])
        train_proportion: Proporção de dados para treino (0.1 a 1.0)
        base_path: Caminho base do dataset (None para usar padrão)
    
    Returns:
        train_df: DataFrame com dados de treino
        val_df: DataFrame com dados de validação
    """
    if base_path is None:
        base_path = Path("Dataset_ITU-AIML_2025-KRI-program/(b)_[Training]Propagation-loss")
    
    frequency_path = base_path / frequency
    
    train_dfs = []
    val_dfs = []
    
    for tx_num in tx_list:
        # Construir nome do arquivo
        file_name = f"{frequency}_Tx_{tx_num}.csv"
        file_path = frequency_path / file_name
        
        if not file_path.exists():
            print(f"Arquivo não encontrado: {file_path}")
            continue
        
        # Carregar dados
        df = pd.read_csv(file_path)
        
        # Amostrar dados conforme proporção
        n_samples = len(df)
        n_train = int(n_samples * train_proportion)
        
        # Shuffle e separar
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_df_tx = df_shuffled.iloc[:n_train].copy()
        val_df_tx = df_shuffled.iloc[n_train:].copy()
        
        train_dfs.append(train_df_tx)
        val_dfs.append(val_df_tx)
    
    if not train_dfs:
        raise ValueError(f"Nenhum arquivo foi carregado para Tx: {tx_list}")
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    
    return train_df, val_df


def load_evaluation_data(frequency, base_path=None):
    """
    Carrega dados de avaliação da pasta Evaluation
    
    Args:
        frequency: Frequência escolhida ('800MHz', '7GHz', '28GHz')
        base_path: Caminho base do dataset (None para usar padrão)
    
    Returns:
        eval_df: DataFrame com todos os dados de avaliação
        eval_by_tx: Dicionário com dados separados por Tx
    """
    if base_path is None:
        base_path = Path("Dataset_ITU-AIML_2025-KRI-program/(b)_[Evaluation]Propagation-loss")
    
    frequency_path = base_path / frequency
    
    # Tx disponíveis na pasta de avaliação
    eval_tx = [5, 9, 12, 14, 20]
    
    eval_dfs = []
    eval_by_tx = {}
    
    for tx_num in eval_tx:
        file_name = f"{frequency}_Tx_{tx_num}.csv"
        file_path = frequency_path / file_name
        
        if not file_path.exists():
            print(f"Arquivo não encontrado: {file_path}")
            continue
        
        df = pd.read_csv(file_path)
        df['Tx_num'] = tx_num  # Adicionar identificador do Tx
        
        eval_dfs.append(df)
        eval_by_tx[tx_num] = df
    
    if not eval_dfs:
        raise ValueError(f"Nenhum arquivo de avaliação encontrado para {frequency}")
    
    eval_df = pd.concat(eval_dfs, ignore_index=True)
    
    return eval_df, eval_by_tx


def prepare_features(df, scaler=None, fit_scaler=False):
    """
    Prepara features para o modelo
    
    Args:
        df: DataFrame com dados
        scaler: Scaler pré-treinado (None para criar novo)
        fit_scaler: Se True, ajusta o scaler aos dados
    
    Returns:
        features: Array numpy com features normalizadas
        distances: Distâncias calculadas
        fspl: FSPL calculado
        ld: Ld calculado (PL - FSPL)
        scaler: Scaler ajustado
    """
    # Calcular distância 3D
    distances = calculate_distance_3d(
        df['Tx_UTM_X'].values,
        df['Tx_UTM_Y'].values,
        df['Tx_UTM_Z'].values,
        df['Rx_UTM_X'].values,
        df['Rx_UTM_Y'].values,
        df['Rx_UTM_Z'].values
    )
    
    # Determinar frequência a partir do caminho ou dados
    # Extrair frequência do nome da coluna ou passar como parâmetro
    # Por enquanto, vamos inferir do dataframe ou passar como parâmetro
    # Assumindo que a frequência será passada ou calculada antes
    
    # Criar features
    feature_dict = {
        'distance': distances,
        'tx_x': df['Tx_UTM_X'].values,
        'tx_y': df['Tx_UTM_Y'].values,
        'tx_z': df['Tx_UTM_Z'].values,
        'rx_x': df['Rx_UTM_X'].values,
        'rx_y': df['Rx_UTM_Y'].values,
        'rx_z': df['Rx_UTM_Z'].values,
        'height_diff': df['Tx_UTM_Z'].values - df['Rx_UTM_Z'].values,
        'tx_altitude': df['Tx_UTM_Z'].values,
        'rx_altitude': df['Rx_UTM_Z'].values,
    }
    
    # Adicionar coordenadas relativas
    feature_dict['relative_x'] = df['Rx_UTM_X'].values - df['Tx_UTM_X'].values
    feature_dict['relative_y'] = df['Rx_UTM_Y'].values - df['Tx_UTM_Y'].values
    
    features_df = pd.DataFrame(feature_dict)
    
    # Normalizar features
    if scaler is None:
        scaler = StandardScaler()
    
    if fit_scaler:
        features_scaled = scaler.fit_transform(features_df)
    else:
        features_scaled = scaler.transform(features_df)
    
    return features_scaled, distances, scaler


def prepare_features_with_frequency(df, frequency_mhz, scaler=None, fit_scaler=False, mesh=None, progress_callback=None):
    """
    Prepara features incluindo cálculo de FSPL e Ld, e features geométricas do OBJ
    
    Args:
        df: DataFrame com dados
        frequency_mhz: Frequência em MHz
        scaler: Scaler pré-treinado (None para criar novo)
        fit_scaler: Se True, ajusta o scaler aos dados
        mesh: Mesh trimesh dos edifícios (opcional)
        progress_callback: Função callback(opcional) para reportar progresso (progress, message)
    
    Returns:
        features: Array numpy com features normalizadas
        distances: Distâncias calculadas (em metros)
        fspl: FSPL calculado (em dB)
        ld: Ld calculado (PL - FSPL) (em dB)
        pl: PL real (em dB)
        scaler: Scaler ajustado
    """
    if progress_callback:
        progress_callback(0.1, "Calculando distâncias 3D...")
    
    # Calcular distância 3D
    distances = calculate_distance_3d(
        df['Tx_UTM_X'].values,
        df['Tx_UTM_Y'].values,
        df['Tx_UTM_Z'].values,
        df['Rx_UTM_X'].values,
        df['Rx_UTM_Y'].values,
        df['Rx_UTM_Z'].values
    )
    
    if progress_callback:
        progress_callback(0.3, f"Distâncias calculadas. Calculando FSPL para {len(distances):,} amostras...")
    
    # Calcular FSPL
    fspl = calculate_free_space_loss(distances, frequency_mhz)
    
    if progress_callback:
        progress_callback(0.4, "FSPL calculado. Calculando Ld...")
    
    # Calcular Ld
    pl = df['PL'].values
    ld = pl - fspl
    
    if progress_callback:
        progress_callback(0.5, "Ld calculado. Preparando features básicas...")
    
    # Criar features
    feature_dict = {
        'distance': distances,
        'tx_x': df['Tx_UTM_X'].values,
        'tx_y': df['Tx_UTM_Y'].values,
        'tx_z': df['Tx_UTM_Z'].values,
        'rx_x': df['Rx_UTM_X'].values,
        'rx_y': df['Rx_UTM_Y'].values,
        'rx_z': df['Rx_UTM_Z'].values,
        'height_diff': df['Tx_UTM_Z'].values - df['Rx_UTM_Z'].values,
        'tx_altitude': df['Tx_UTM_Z'].values,
        'rx_altitude': df['Rx_UTM_Z'].values,
    }
    
    # Adicionar coordenadas relativas
    feature_dict['relative_x'] = df['Rx_UTM_X'].values - df['Tx_UTM_X'].values
    feature_dict['relative_y'] = df['Rx_UTM_Y'].values - df['Tx_UTM_Y'].values
    
    # Adicionar frequência normalizada
    feature_dict['frequency_normalized'] = np.full(len(df), frequency_mhz / 28000.0)  # Normalizar por 28GHz
    
    # Adicionar features geométricas do OBJ se disponível
    if mesh is not None:
        if progress_callback:
            progress_callback(0.6, f"Calculando features geométricas do OBJ para {len(df):,} pares Tx-Rx...")
        
        tx_positions = np.column_stack([
            df['Tx_UTM_X'].values,
            df['Tx_UTM_Y'].values,
            df['Tx_UTM_Z'].values
        ])
        rx_positions = np.column_stack([
            df['Rx_UTM_X'].values,
            df['Rx_UTM_Y'].values,
            df['Rx_UTM_Z'].values
        ])
        
        # Wrapper para converter progresso relativo (0-1) para progresso geral (0.6-0.85)
        def geometric_progress_wrapper(rel_progress, message):
            # Mapear progresso relativo (0-1) para progresso geral (0.6-0.85)
            general_progress = 0.6 + rel_progress * 0.25
            if progress_callback:
                progress_callback(general_progress, message)
        
        geometric_features = calculate_geometric_features_batch(
            tx_positions, rx_positions, mesh, n_samples=10, progress_callback=geometric_progress_wrapper
        )
        
        if progress_callback:
            progress_callback(0.9, "Features geométricas calculadas. Adicionando ao conjunto de features...")
        
        # Adicionar features geométricas
        feature_dict['is_los'] = geometric_features['is_los'].astype(float)  # Converter bool para float
        feature_dict['intersection_count'] = geometric_features['intersection_count'].astype(float)
        feature_dict['max_obstacle_height'] = geometric_features['max_obstacle_height']
        
        # Preencher NaN em obstacle_distance com valor padrão (distância total se LOS)
        obstacle_distance = geometric_features['obstacle_distance'].copy()
        obstacle_distance[np.isnan(obstacle_distance)] = distances[np.isnan(obstacle_distance)]
        feature_dict['obstacle_distance'] = obstacle_distance
    else:
        # Se não há mesh, usar valores padrão
        feature_dict['is_los'] = np.ones(len(df))  # Assumir LOS
        feature_dict['intersection_count'] = np.zeros(len(df))
        feature_dict['max_obstacle_height'] = np.zeros(len(df))
        feature_dict['obstacle_distance'] = distances  # Distância total como padrão
    
    if progress_callback:
        progress_callback(0.95, "Normalizando features...")
    
    features_df = pd.DataFrame(feature_dict)
    
    # Normalizar features
    if scaler is None:
        scaler = StandardScaler()
    
    if fit_scaler:
        features_scaled = scaler.fit_transform(features_df)
    else:
        features_scaled = scaler.transform(features_df)
    
    if progress_callback:
        progress_callback(1.0, "✅ Features preparadas com sucesso!")
    
    return features_scaled, distances, fspl, ld, pl, scaler


def get_available_tx(frequency, base_path=None):
    """
    Retorna lista de Tx disponíveis para uma frequência
    
    Args:
        frequency: Frequência escolhida ('800MHz', '7GHz', '28GHz')
        base_path: Caminho base do dataset
    
    Returns:
        Lista de números de Tx disponíveis
    """
    if base_path is None:
        base_path = Path("Dataset_ITU-AIML_2025-KRI-program/(b)_[Training]Propagation-loss")
    
    frequency_path = base_path / frequency
    
    if not frequency_path.exists():
        return []
    
    tx_list = []
    for file in frequency_path.glob(f"{frequency}_Tx_*.csv"):
        # Extrair número do Tx do nome do arquivo
        tx_num = int(file.stem.split('_Tx_')[1])
        tx_list.append(tx_num)
    
    return sorted(tx_list)


def frequency_to_mhz(frequency_str):
    """
    Converte string de frequência para MHz
    
    Args:
        frequency_str: '800MHz', '7GHz', ou '28GHz'
    
    Returns:
        Frequência em MHz (float)
    """
    if 'MHz' in frequency_str:
        return float(frequency_str.replace('MHz', ''))
    elif 'GHz' in frequency_str:
        return float(frequency_str.replace('GHz', '')) * 1000.0
    else:
        raise ValueError(f"Formato de frequência não reconhecido: {frequency_str}")


# Cache para o mesh carregado
_mesh_cache = None
_mesh_path_cache = None


def load_buildings_mesh(obj_path=None):
    """
    Carrega o arquivo OBJ dos edifícios usando trimesh
    
    Args:
        obj_path: Caminho para o arquivo buildings.obj (None para usar padrão)
    
    Returns:
        trimesh.Trimesh: Mesh dos edifícios ou None se não disponível
    """
    global _mesh_cache, _mesh_path_cache
    
    if not TRIMESH_AVAILABLE:
        return None
    
    if obj_path is None:
        obj_path = Path("Dataset_ITU-AIML_2025-KRI-program/(a)_3DMap_Data/buildings.obj")
    
    obj_path = Path(obj_path)
    
    # Usar cache se já foi carregado
    if _mesh_cache is not None and _mesh_path_cache == str(obj_path):
        return _mesh_cache
    
    if not obj_path.exists():
        print(f"Arquivo OBJ não encontrado: {obj_path}")
        return None
    
    try:
        mesh = trimesh.load(str(obj_path))
        if isinstance(mesh, trimesh.Scene):
            # Se for uma Scene, combinar todos os meshes
            mesh = trimesh.util.concatenate([m for m in mesh.geometry.values() if isinstance(m, trimesh.Trimesh)])
        
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"Tipo de mesh não suportado: {type(mesh)}")
            return None
        
        # Cachear o mesh
        _mesh_cache = mesh
        _mesh_path_cache = str(obj_path)
        
        return mesh
    except Exception as e:
        print(f"Erro ao carregar OBJ: {e}")
        return None


def check_line_of_sight(tx_pos, rx_pos, mesh, n_samples=50):
    """
    Verifica se há linha de visada (LOS) entre Tx e Rx
    
    Args:
        tx_pos: Posição do Tx como array [x, y, z]
        rx_pos: Posição do Rx como array [x, y, z]
        mesh: Mesh trimesh dos edifícios
        n_samples: Número de pontos ao longo da linha para verificar (reduzido para performance)
    
    Returns:
        dict com informações sobre LOS/NLOS:
        - is_los: bool (True se há linha de visada)
        - intersection_count: número de interseções com edifícios
        - max_obstacle_height: altura máxima do obstáculo acima da linha
        - obstacle_distance: distância até o primeiro obstáculo (None se LOS)
    """
    if mesh is None:
        # Se não há mesh, assumir LOS (não podemos verificar)
        return {
            'is_los': True,
            'intersection_count': 0,
            'max_obstacle_height': 0.0,
            'obstacle_distance': None
        }
    
    tx_pos = np.array(tx_pos)
    rx_pos = np.array(rx_pos)
    
    # Criar linha entre Tx e Rx
    line = np.array([tx_pos, rx_pos])
    
    # Verificar interseções com o mesh
    try:
        locations, ray_indices, face_indices = mesh.ray.intersects_location(
            ray_origins=line[:1],
            ray_directions=[line[1] - line[0]]
        )
        
        intersection_count = len(locations)
        is_los = intersection_count == 0
        
        # Calcular altura máxima do obstáculo acima da linha
        max_obstacle_height = 0.0
        obstacle_distance = None
        
        if intersection_count > 0:
            # Encontrar primeira interseção
            first_intersection = locations[0]
            obstacle_distance = np.linalg.norm(first_intersection - tx_pos)
            
            # OTIMIZAÇÃO: Usar altura máxima das interseções já encontradas
            # Ao invés de fazer mais ray intersections, usar informação das interseções existentes
            if len(locations) > 0:
                # Calcular altura da linha em cada ponto de interseção
                for loc in locations:
                    # Calcular parâmetro t (onde na linha está a interseção)
                    line_vec = rx_pos - tx_pos
                    line_len = np.linalg.norm(line_vec)
                    if line_len > 0:
                        intersection_vec = loc - tx_pos
                        t = np.dot(intersection_vec, line_vec) / (line_len ** 2)
                        t = np.clip(t, 0, 1)  # Garantir que está entre 0 e 1
                        
                        # Altura da linha neste ponto
                        line_height = tx_pos[2] + t * (rx_pos[2] - tx_pos[2])
                        
                        # Comparar com altura do obstáculo
                        if loc[2] > line_height:
                            obstacle_height = loc[2] - line_height
                            max_obstacle_height = max(max_obstacle_height, obstacle_height)
            
            # Verificação adicional apenas nos pontos extremos (Tx e Rx) se necessário
            # Isso reduz drasticamente o número de ray intersections
            try:
                # Verificar apenas no ponto médio se ainda não temos altura máxima
                if max_obstacle_height == 0.0:
                    midpoint = (tx_pos + rx_pos) / 2
                    ray_origin = np.array([midpoint[0], midpoint[1], 0])
                    ray_dir = np.array([0, 0, 1])
                    locs, _, _ = mesh.ray.intersects_location(
                        ray_origins=[ray_origin],
                        ray_directions=[ray_dir]
                    )
                    if len(locs) > 0:
                        max_height = max([loc[2] for loc in locs])
                        if max_height > midpoint[2]:
                            obstacle_height = max_height - midpoint[2]
                            max_obstacle_height = max(max_obstacle_height, obstacle_height)
            except:
                pass
        
        return {
            'is_los': is_los,
            'intersection_count': intersection_count,
            'max_obstacle_height': max_obstacle_height,
            'obstacle_distance': obstacle_distance
        }
    except Exception as e:
        print(f"Erro ao verificar LOS: {e}")
        return {
            'is_los': True,
            'intersection_count': 0,
            'max_obstacle_height': 0.0,
            'obstacle_distance': None
        }


def calculate_geometric_features_batch(tx_positions, rx_positions, mesh, n_samples=20, progress_callback=None):
    """
    Calcula features geométricas em batch para múltiplos pares Tx-Rx
    
    Args:
        tx_positions: Array de posições Tx [N, 3]
        rx_positions: Array de posições Rx [N, 3]
        mesh: Mesh trimesh dos edifícios
        n_samples: Número de amostras para cálculo de features
        progress_callback: Função callback(opcional) para reportar progresso (progress, message)
    
    Returns:
        dict com arrays de features:
        - is_los: array [N] de bools
        - intersection_count: array [N] de ints
        - max_obstacle_height: array [N] de floats
        - obstacle_distance: array [N] de floats (None se LOS)
    """
    n = len(tx_positions)
    
    is_los = np.zeros(n, dtype=bool)
    intersection_count = np.zeros(n, dtype=int)
    max_obstacle_height = np.zeros(n)
    obstacle_distance = np.full(n, np.nan)
    
    # Processar em batches para melhor performance
    batch_size = 100  # Processar 100 por vez
    
    total_batches = (n + batch_size - 1) // batch_size
    
    if progress_callback:
        progress_callback(0.0, f"Iniciando cálculo de features geométricas para {n:,} pares Tx-Rx...")
    
    for i in range(0, n, batch_size):
        end_idx = min(i + batch_size, n)
        batch_num = i // batch_size + 1
        
        if progress_callback:
            progress = batch_num / total_batches
            progress_callback(progress, f"Processando batch {batch_num}/{total_batches} ({end_idx:,}/{n:,} pares)")
        
        for j in range(i, end_idx):
            los_info = check_line_of_sight(
                tx_positions[j],
                rx_positions[j],
                mesh,
                n_samples=n_samples
            )
            
            is_los[j] = los_info['is_los']
            intersection_count[j] = los_info['intersection_count']
            max_obstacle_height[j] = los_info['max_obstacle_height']
            if los_info['obstacle_distance'] is not None:
                obstacle_distance[j] = los_info['obstacle_distance']
    
    if progress_callback:
        progress_callback(1.0, f"✅ Cálculo de features geométricas concluído!")
    
    return {
        'is_los': is_los,
        'intersection_count': intersection_count,
        'max_obstacle_height': max_obstacle_height,
        'obstacle_distance': obstacle_distance
    }

