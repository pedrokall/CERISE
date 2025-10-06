#!/usr/bin/env python3
"""
Script para gerar dataset RICO com todas as features possíveis do Sionna RT
- Features básicas: posições, RSS
- Features de propagação: delays, ângulos, caminhos
- Features derivadas: distâncias, dispersões, características ambientais
- Features estatísticas: variâncias, correlações, métricas de canal
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import circvar

# Adiciona o diretório pai ao path para importar o módulo tcc_project
sys.path.insert(0, str(Path(__file__).parent.parent))

from tcc_project.scene import load_scene_with_options, create_car_objects, add_objects
from tcc_project.geometry import load_receiver_mesh_positions, calculate_car_orientations_improved, apply_positions_and_orientations
from tcc_project.sim import setup_tx_rx, compute_paths, rss_from_paths
from tcc_project.utils import setup_logger, log_step, ensure_output_dir


def extract_comprehensive_features(paths, rx_positions, tx_position=(21.18, -132.4, 18.76)):
    """
    Extrai TODAS as features possíveis do Sionna RT e derivadas
    """
    logger = setup_logger("feature_extraction")
    c = 3e8  # velocidade da luz
    frequency = 2.4e9  # 2.4 GHz
    
    log_step(logger, f"🔬 Extraindo features abrangentes para {len(rx_positions)} receptores")
    
    features_list = []
    
    for rx_idx in range(len(rx_positions)):
        rx_pos = rx_positions[rx_idx]
        feature_dict = {}
        
        # ===== 1. FEATURES BÁSICAS =====
        feature_dict['rx_x'] = rx_pos[0]
        feature_dict['rx_y'] = rx_pos[1] 
        feature_dict['rx_z'] = rx_pos[2]
        
        # ===== 2. FEATURES GEOMÉTRICAS =====
        # Distâncias
        dist_3d = np.sqrt((rx_pos[0] - tx_position[0])**2 + 
                         (rx_pos[1] - tx_position[1])**2 + 
                         (rx_pos[2] - tx_position[2])**2)
        dist_2d = np.sqrt((rx_pos[0] - tx_position[0])**2 + 
                         (rx_pos[1] - tx_position[1])**2)
        
        feature_dict['distance_euclidean_3d'] = dist_3d
        feature_dict['distance_euclidean_2d'] = dist_2d
        feature_dict['height_difference'] = rx_pos[2] - tx_position[2]
        
        # Ângulos geométricos
        feature_dict['azimuth_geometric'] = np.arctan2(rx_pos[1] - tx_position[1], 
                                                      rx_pos[0] - tx_position[0])
        feature_dict['elevation_geometric'] = np.arctan2(feature_dict['height_difference'], dist_2d)
        
        # Free Space Path Loss teórico
        fspl_theoretical = 20 * np.log10(dist_3d) + 20 * np.log10(frequency) - 147.55
        feature_dict['fspl_theoretical'] = fspl_theoretical
        
        # ===== 3. FEATURES DOS CAMINHOS DE PROPAGAÇÃO =====
        try:
            # Delays (tau)
            tau_values = paths.tau.numpy()[rx_idx, 0, :]
            valid_mask = tau_values > 0
            valid_taus = tau_values[valid_mask]
            
            if len(valid_taus) > 0:
                # Estatísticas básicas dos delays
                feature_dict['tau_min'] = np.min(valid_taus)
                feature_dict['tau_max'] = np.max(valid_taus)
                feature_dict['tau_mean'] = np.mean(valid_taus)
                feature_dict['tau_std'] = np.std(valid_taus)
                feature_dict['tau_range'] = np.max(valid_taus) - np.min(valid_taus)
                feature_dict['num_paths'] = len(valid_taus)
                
                # Distâncias dos caminhos
                distances = valid_taus * c
                feature_dict['distance_min_path'] = np.min(distances)
                feature_dict['distance_max_path'] = np.max(distances)
                feature_dict['distance_mean_path'] = np.mean(distances)
                feature_dict['distance_std_path'] = np.std(distances)
                
                # Excess path length
                feature_dict['excess_path_length'] = feature_dict['distance_min_path'] - dist_3d
                feature_dict['excess_path_length_mean'] = feature_dict['distance_mean_path'] - dist_3d
                
                # RMS Delay Spread
                mean_delay = np.mean(valid_taus)
                rms_delay_spread = np.sqrt(np.mean((valid_taus - mean_delay)**2))
                feature_dict['rms_delay_spread'] = rms_delay_spread
                
            else:
                # Valores padrão para quando não há caminhos válidos
                for key in ['tau_min', 'tau_max', 'tau_mean', 'tau_std', 'tau_range', 'num_paths',
                           'distance_min_path', 'distance_max_path', 'distance_mean_path', 'distance_std_path',
                           'excess_path_length', 'excess_path_length_mean', 'rms_delay_spread']:
                    feature_dict[key] = 0
                    
        except Exception as e:
            logger.warning(f"Erro ao extrair features de delay para RX{rx_idx}: {e}")
            for key in ['tau_min', 'tau_max', 'tau_mean', 'tau_std', 'tau_range', 'num_paths',
                       'distance_min_path', 'distance_max_path', 'distance_mean_path', 'distance_std_path',
                       'excess_path_length', 'excess_path_length_mean', 'rms_delay_spread']:
                feature_dict[key] = 0
        
        # ===== 4. FEATURES DOS ÂNGULOS =====
        try:
            # Ângulos de transmissão e recepção
            theta_t = paths.theta_t.numpy()[rx_idx, 0, :][valid_mask] if 'valid_mask' in locals() else []
            phi_t = paths.phi_t.numpy()[rx_idx, 0, :][valid_mask] if 'valid_mask' in locals() else []
            theta_r = paths.theta_r.numpy()[rx_idx, 0, :][valid_mask] if 'valid_mask' in locals() else []
            phi_r = paths.phi_r.numpy()[rx_idx, 0, :][valid_mask] if 'valid_mask' in locals() else []
            
            if len(theta_t) > 0:
                # Ângulos médios
                feature_dict['theta_tx_mean'] = np.mean(theta_t)
                feature_dict['phi_tx_mean'] = np.mean(phi_t)
                feature_dict['theta_rx_mean'] = np.mean(theta_r)
                feature_dict['phi_rx_mean'] = np.mean(phi_r)
                
                # Dispersão angular
                feature_dict['theta_tx_std'] = np.std(theta_t)
                feature_dict['phi_tx_std'] = np.std(phi_t)
                feature_dict['theta_rx_std'] = np.std(theta_r)
                feature_dict['phi_rx_std'] = np.std(phi_r)
                
                # Angular spread (medida mais sofisticada)
                if len(theta_t) > 1:
                    feature_dict['angular_spread_tx'] = np.sqrt(np.var(theta_t) + np.var(phi_t))
                    feature_dict['angular_spread_rx'] = np.sqrt(np.var(theta_r) + np.var(phi_r))
                else:
                    feature_dict['angular_spread_tx'] = 0
                    feature_dict['angular_spread_rx'] = 0
                    
                # Ângulo do caminho dominante (primeiro caminho)
                feature_dict['dominant_theta_tx'] = theta_t[0] if len(theta_t) > 0 else 0
                feature_dict['dominant_phi_tx'] = phi_t[0] if len(phi_t) > 0 else 0
                feature_dict['dominant_theta_rx'] = theta_r[0] if len(theta_r) > 0 else 0
                feature_dict['dominant_phi_rx'] = phi_r[0] if len(phi_r) > 0 else 0
                
            else:
                for key in ['theta_tx_mean', 'phi_tx_mean', 'theta_rx_mean', 'phi_rx_mean',
                           'theta_tx_std', 'phi_tx_std', 'theta_rx_std', 'phi_rx_std',
                           'angular_spread_tx', 'angular_spread_rx',
                           'dominant_theta_tx', 'dominant_phi_tx', 'dominant_theta_rx', 'dominant_phi_rx']:
                    feature_dict[key] = 0
                    
        except Exception as e:
            logger.warning(f"Erro ao extrair features de ângulos para RX{rx_idx}: {e}")
            for key in ['theta_tx_mean', 'phi_tx_mean', 'theta_rx_mean', 'phi_rx_mean',
                       'theta_tx_std', 'phi_tx_std', 'theta_rx_std', 'phi_rx_std',
                       'angular_spread_tx', 'angular_spread_rx',
                       'dominant_theta_tx', 'dominant_phi_tx', 'dominant_theta_rx', 'dominant_phi_rx']:
                feature_dict[key] = 0
        
        # ===== 5. FEATURES DE INTERAÇÕES =====
        try:
            if hasattr(paths, 'interactions'):
                interactions = paths.interactions.numpy()[0, rx_idx, 0, :] if paths.interactions.shape[0] > 0 else []
                
                if len(interactions) > 0:
                    # Número de interações por tipo (valores específicos podem variar)
                    feature_dict['num_interactions'] = len(interactions)
                    feature_dict['num_reflections'] = np.sum(interactions == 1) if len(interactions) > 0 else 0
                    feature_dict['num_diffractions'] = np.sum(interactions == 2) if len(interactions) > 0 else 0
                    feature_dict['num_transmissions'] = np.sum(interactions == 0) if len(interactions) > 0 else 0
                    
                    # Caminho dominante
                    feature_dict['dominant_interaction_type'] = interactions[0] if len(interactions) > 0 else 0
                else:
                    for key in ['num_interactions', 'num_reflections', 'num_diffractions', 
                               'num_transmissions', 'dominant_interaction_type']:
                        feature_dict[key] = 0
            else:
                for key in ['num_interactions', 'num_reflections', 'num_diffractions', 
                           'num_transmissions', 'dominant_interaction_type']:
                    feature_dict[key] = 0
                    
        except Exception as e:
            logger.warning(f"Erro ao extrair features de interações para RX{rx_idx}: {e}")
            for key in ['num_interactions', 'num_reflections', 'num_diffractions', 
                       'num_transmissions', 'dominant_interaction_type']:
                feature_dict[key] = 0
        
        # ===== 6. FEATURES DE CONTEXTO AMBIENTAL =====
        # Densidade local de receptores
        positions_array = np.array(rx_positions)
        distances_to_others = cdist([rx_pos], positions_array)[0]
        
        # Receptores próximos (dentro de diferentes raios)
        feature_dict['neighbors_50m'] = np.sum(distances_to_others < 50) - 1  # -1 para excluir ele mesmo
        feature_dict['neighbors_100m'] = np.sum(distances_to_others < 100) - 1
        feature_dict['neighbors_200m'] = np.sum(distances_to_others < 200) - 1
        
        # Distância ao receptor mais próximo
        distances_to_others_nonzero = distances_to_others[distances_to_others > 0]
        feature_dict['distance_to_nearest_rx'] = np.min(distances_to_others_nonzero) if len(distances_to_others_nonzero) > 0 else 0
        
        # ===== 7. FEATURES DERIVADAS AVANÇADAS =====
        # Indicadores de linha de visada (baseado no caminho mais curto vs euclidiano)
        if feature_dict['excess_path_length'] < 10:  # Margem de 10 metros
            feature_dict['los_indicator'] = 1
        else:
            feature_dict['los_indicator'] = 0
            
        # Fator de multipercurso
        if feature_dict['num_paths'] > 0:
            feature_dict['multipath_factor'] = feature_dict['tau_range'] / feature_dict['tau_min'] if feature_dict['tau_min'] > 0 else 0
        else:
            feature_dict['multipath_factor'] = 0
            
        # REMOVIDO: approximate_snr causa data leakage
        # Era: tx_power_dbm + fspl_theoretical (correlação com target)
        
        # Coordenadas normalizadas (útil para ML)
        feature_dict['rx_x_normalized'] = (rx_pos[0] - tx_position[0]) / 1000  # Normalizado por km
        feature_dict['rx_y_normalized'] = (rx_pos[1] - tx_position[1]) / 1000
        feature_dict['rx_z_normalized'] = (rx_pos[2] - tx_position[2]) / 100   # Normalizado por 100m
        
        features_list.append(feature_dict)
    
    log_step(logger, f"✅ Features extraídas: {len(features_list[0])} features por receptor")
    return pd.DataFrame(features_list)


def main():
    parser = argparse.ArgumentParser(description="Gerar dataset rico com todas as features do Sionna RT")
    parser.add_argument("--scene", type=str, default="universitario.xml", help="Arquivo de cena")
    parser.add_argument("--mesh-csv", type=str, default="receivers_mesh.csv", help="Arquivo de posições")
    parser.add_argument("--num-cars", type=int, default=1000, help="Número de carros para simular")
    parser.add_argument("--output-name", type=str, default="dataset_rico", help="Nome base do dataset")
    parser.add_argument("--tx-power", type=float, default=20.0, help="Potência do TX em dBm")
    
    args = parser.parse_args()
    
    logger = setup_logger(__name__)
    
    try:
        log_step(logger, f"🚀 GERANDO DATASET RICO COM {args.num_cars} CARROS")
        
        # 1. Configurar cena
        log_step(logger, f"📁 Carregando cena: {args.scene}")
        scene = load_scene_with_options(args.scene)
        
        # 2. Carregar posições
        log_step(logger, f"📍 Carregando posições de {args.mesh_csv}")
        car_positions = load_receiver_mesh_positions(args.mesh_csv, args.num_cars)
        num_cars = len(car_positions)
        log_step(logger, f"✅ {num_cars} posições carregadas")
        
        # 3. Criar e posicionar carros
        log_step(logger, "🚗 Criando e posicionando carros")
        cars = create_car_objects(num_cars)
        add_objects(scene, cars)
        
        all_positions = load_receiver_mesh_positions(args.mesh_csv, 2000)
        orientations = calculate_car_orientations_improved(car_positions, all_positions)
        apply_positions_and_orientations(cars, car_positions, orientations)
        
        # 4. Configurar TX/RX
        log_step(logger, "📡 Configurando transmissor e receptores")
        tx_position = (21.18, -132.4, 18.76)
        rx_positions = [(x, y, z + 3.0) for (x, y, z) in car_positions]
        setup_tx_rx(scene, tx_position, rx_positions)
        
        # 5. Calcular caminhos de propagação
        log_step(logger, "🌐 Calculando caminhos de propagação (max_depth=5)")
        paths = compute_paths(scene, max_depth=5)
        log_step(logger, "✅ Caminhos calculados")
        
        # 6. Calcular RSS
        log_step(logger, "📊 Calculando RSS")
        rss_db = rss_from_paths(paths)
        
        # 7. Extrair todas as features
        log_step(logger, "🔬 Extraindo features abrangentes...")
        features_df = extract_comprehensive_features(paths, rx_positions, tx_position)
        
        # 8. Adicionar RSS ao dataset (TARGET)
        features_df['rss_db'] = rss_db.numpy()
        
        # 9. REMOVER features que causam data leakage
        # NÃO adicionar path_loss, approximate_snr, etc. que derivam do target
        # Manter apenas features independentes do RSS
        
        # 10. Estatísticas do dataset
        log_step(logger, "📊 ESTATÍSTICAS DO DATASET RICO:")
        logger.info(f"  📋 Total de features: {len(features_df.columns)}")
        logger.info(f"  🚗 Total de amostras: {len(features_df)}")
        logger.info(f"  📈 RSS range: {features_df['rss_db'].min():.2f} to {features_df['rss_db'].max():.2f} dBm")
        logger.info(f"  🌐 Caminhos médios por RX: {features_df['num_paths'].mean():.1f}")
        logger.info(f"  📡 LOS ratio: {features_df['los_indicator'].mean():.2%}")
        
        # 11. Salvar dataset
        output_dir = ensure_output_dir("data")
        dataset_filename = f"{args.output_name}_{num_cars}_features.csv"
        dataset_path = output_dir / dataset_filename
        
        features_df.to_csv(dataset_path, index=False)
        log_step(logger, f"💾 Dataset rico salvo em: {dataset_path}")
        
        # 12. Salvar metadados das features
        metadata = {
            'feature_name': features_df.columns.tolist(),
            'feature_type': ['target' if col == 'rss_db' else 'feature' for col in features_df.columns],
            'feature_category': []
        }
        
        # Categorizar features
        for col in features_df.columns:
            if col in ['rx_x', 'rx_y', 'rx_z', 'distance_euclidean_3d', 'distance_euclidean_2d']:
                metadata['feature_category'].append('geometric')
            elif 'tau' in col or 'delay' in col or 'distance' in col and 'path' in col:
                metadata['feature_category'].append('propagation_delay')
            elif 'theta' in col or 'phi' in col or 'angular' in col or 'azimuth' in col or 'elevation' in col:
                metadata['feature_category'].append('angular')
            elif 'interaction' in col or 'reflection' in col or 'diffraction' in col:
                metadata['feature_category'].append('interaction')
            elif 'neighbor' in col or 'density' in col or 'nearest' in col:
                metadata['feature_category'].append('environmental')
            elif col == 'rss_db':
                metadata['feature_category'].append('target')
            else:
                metadata['feature_category'].append('derived')
        
        metadata_df = pd.DataFrame(metadata)
        metadata_path = output_dir / f"{args.output_name}_{num_cars}_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        log_step(logger, f"📋 Metadados salvos em: {metadata_path}")
        
        # 13. Resumo final
        log_step(logger, "🎉 DATASET RICO GERADO COM SUCESSO!")
        logger.info("=" * 80)
        logger.info("📊 RESUMO DO DATASET RICO:")
        logger.info(f"  📋 Features por categoria:")
        for category in metadata_df['feature_category'].unique():
            count = metadata_df['feature_category'].value_counts()[category]
            logger.info(f"    {category}: {count} features")
        logger.info(f"  📁 Arquivos gerados:")
        logger.info(f"    Dataset: {dataset_path}")
        logger.info(f"    Metadados: {metadata_path}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Erro durante geração do dataset rico: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
