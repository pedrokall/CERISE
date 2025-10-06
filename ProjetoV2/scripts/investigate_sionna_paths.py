#!/usr/bin/env python3
"""
Script para investigar que informações estão disponíveis no objeto paths do Sionna RT
"""

import sys
from pathlib import Path
import numpy as np

# Adiciona o diretório pai ao path para importar o módulo tcc_project
sys.path.insert(0, str(Path(__file__).parent.parent))

from tcc_project.scene import load_scene_with_options, create_car_objects, add_objects
from tcc_project.geometry import load_receiver_mesh_positions, calculate_car_orientations_improved, apply_positions_and_orientations
from tcc_project.sim import setup_tx_rx, compute_paths, rss_from_paths
from tcc_project.utils import setup_logger, log_step

def main():
    logger = setup_logger(__name__)
    
    try:
        log_step(logger, "🔍 INVESTIGANDO OBJETO PATHS DO SIONNA RT")
        
        # Carregar cena simples
        log_step(logger, "📁 Carregando cena")
        scene = load_scene_with_options("universitario.xml")
        
        # Configurar poucos carros para análise
        log_step(logger, "🚗 Criando 5 carros para teste")
        car_positions = load_receiver_mesh_positions("receivers_mesh.csv", 5)
        cars = create_car_objects(5)
        add_objects(scene, cars)
        
        # Aplicar posições
        all_positions = load_receiver_mesh_positions("receivers_mesh.csv", 100)
        orientations = calculate_car_orientations_improved(car_positions, all_positions)
        apply_positions_and_orientations(cars, car_positions, orientations)
        
        # Configurar TX/RX
        tx_pos = (21.18, -132.4, 18.76)
        rx_positions = [(x, y, z + 3.0) for (x, y, z) in car_positions]
        setup_tx_rx(scene, tx_pos, rx_positions)
        
        # Calcular paths
        log_step(logger, "📡 Calculando paths...")
        paths = compute_paths(scene, max_depth=3)
        
        log_step(logger, "🔍 ANALISANDO OBJETO PATHS:")
        logger.info("=" * 60)
        
        # Investigar atributos do paths
        logger.info(f"Tipo do objeto paths: {type(paths)}")
        logger.info(f"Atributos disponíveis: {[attr for attr in dir(paths) if not attr.startswith('_')]}")
        
        # Analisar atributos principais
        try:
            if hasattr(paths, 'a'):
                logger.info(f"paths.a type: {type(paths.a)}")
                if hasattr(paths.a, 'shape'):
                    logger.info(f"paths.a shape: {paths.a.shape}")
                elif isinstance(paths.a, (list, tuple)):
                    logger.info(f"paths.a length: {len(paths.a)}")
                    if len(paths.a) > 0:
                        logger.info(f"paths.a[0] type: {type(paths.a[0])}")
                        if hasattr(paths.a[0], 'shape'):
                            logger.info(f"paths.a[0] shape: {paths.a[0].shape}")
            
            if hasattr(paths, 'tau'):
                logger.info(f"paths.tau type: {type(paths.tau)}")
                if hasattr(paths.tau, 'shape'):
                    logger.info(f"paths.tau shape: {paths.tau.shape}")
                    logger.info(f"paths.tau (delays) primeiros 5 valores: {paths.tau.numpy().flatten()[:5]}")
                    
                    # tau é o delay, pode ser convertido para distância
                    # distância = tau * velocidade_da_luz
                    c = 3e8  # velocidade da luz em m/s
                    distances_from_tau = paths.tau.numpy() * c
                    logger.info(f"Distâncias calculadas de tau (primeiros 5): {distances_from_tau.flatten()[:5]}")
                elif isinstance(paths.tau, (list, tuple)):
                    logger.info(f"paths.tau length: {len(paths.tau)}")
                    
            if hasattr(paths, 'theta_t'):
                logger.info(f"paths.theta_t type: {type(paths.theta_t)}")
                if hasattr(paths.theta_t, 'shape'):
                    logger.info(f"paths.theta_t shape: {paths.theta_t.shape}")
                    
            if hasattr(paths, 'phi_t'):
                logger.info(f"paths.phi_t type: {type(paths.phi_t)}")
                if hasattr(paths.phi_t, 'shape'):
                    logger.info(f"paths.phi_t shape: {paths.phi_t.shape}")
                    
            if hasattr(paths, 'theta_r'):
                logger.info(f"paths.theta_r type: {type(paths.theta_r)}")
                if hasattr(paths.theta_r, 'shape'):
                    logger.info(f"paths.theta_r shape: {paths.theta_r.shape}")
                    
            if hasattr(paths, 'phi_r'):
                logger.info(f"paths.phi_r type: {type(paths.phi_r)}")
                if hasattr(paths.phi_r, 'shape'):
                    logger.info(f"paths.phi_r shape: {paths.phi_r.shape}")
                    
            # Verificar outros atributos interessantes
            if hasattr(paths, 'vertices'):
                logger.info(f"paths.vertices type: {type(paths.vertices)}")
                if hasattr(paths.vertices, 'shape'):
                    logger.info(f"paths.vertices shape: {paths.vertices.shape}")
                    
            if hasattr(paths, 'interactions'):
                logger.info(f"paths.interactions type: {type(paths.interactions)}")
                if hasattr(paths.interactions, 'shape'):
                    logger.info(f"paths.interactions shape: {paths.interactions.shape}")
                    
        except Exception as e:
            logger.error(f"Erro ao analisar atributos: {e}")
        
        logger.info("=" * 60)
        
        # Calcular distâncias euclidianas manualmente para comparação
        log_step(logger, "📏 Calculando distâncias euclidianas manuais:")
        for i, rx_pos in enumerate(rx_positions):
            dist_euclidean = np.sqrt(
                (rx_pos[0] - tx_pos[0])**2 + 
                (rx_pos[1] - tx_pos[1])**2 + 
                (rx_pos[2] - tx_pos[2])**2
            )
            logger.info(f"RX{i}: Posição {rx_pos}, Distância euclidiana: {dist_euclidean:.2f}m")
        
        # Calcular RSS para comparação
        rss_db = rss_from_paths(paths)
        logger.info(f"RSS calculado: {rss_db.numpy()}")
        
        log_step(logger, "✅ Investigação completa!")
        
    except Exception as e:
        logger.error(f"❌ Erro durante investigação: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
