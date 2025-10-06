#!/usr/bin/env python3
"""
Script completo para gerar simulação com 1000 carros:
1. Visualização dos caminhos de propagação (como na imagem)
2. Dataset CSV para treinamento de ML
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Adiciona o diretório pai ao path para importar o módulo tcc_project
sys.path.insert(0, str(Path(__file__).parent.parent))

from tcc_project.scene import load_scene_with_options, create_car_objects, add_objects
from tcc_project.geometry import load_receiver_mesh_positions, calculate_car_orientations_improved, apply_positions_and_orientations
from tcc_project.sim import setup_tx_rx, compute_paths, rss_from_paths
from tcc_project.visualization import plot_coverage_paths, plot_rss_heatmap
from tcc_project.utils import setup_logger, log_step, log_progress, ensure_output_dir


def main():
    parser = argparse.ArgumentParser(description="Gerar simulação completa com 1000 carros")
    parser.add_argument("--scene", type=str, default="universitario.xml", help="Arquivo de cena (.xml)")
    parser.add_argument("--mesh-csv", type=str, default="receivers_mesh.csv", help="Arquivo CSV com posições dos carros")
    parser.add_argument("--max-cars", type=int, default=1000, help="Número de carros para simulação")
    parser.add_argument("--output-name", type=str, default="simulation_1000", help="Nome base para arquivos de saída")
    parser.add_argument("--skip-paths-render", action="store_true", help="Pular renderização dos caminhos")
    parser.add_argument("--skip-heatmap", action="store_true", help="Pular geração do heatmap")
    parser.add_argument("--skip-dataset", action="store_true", help="Pular geração do dataset CSV")
    
    args = parser.parse_args()
    
    logger = setup_logger(__name__)
    
    try:
        log_step(logger, f"🚀 INICIANDO SIMULAÇÃO COM {args.max_cars} CARROS")
        
        # 1. Carrega cena
        log_step(logger, f"📁 Carregando cena: {args.scene}")
        scene = load_scene_with_options(args.scene)
        
        # 2. Carrega posições do mesh
        log_step(logger, f"📍 Carregando {args.max_cars} posições do receivers_mesh.csv")
        car_positions = load_receiver_mesh_positions(args.mesh_csv, args.max_cars)
        num_cars = len(car_positions)
        log_step(logger, f"✅ {num_cars} posições carregadas")
        
        # 3. Cria e adiciona carros à cena
        log_step(logger, f"🚗 Criando {num_cars} carros e adicionando à cena")
        cars = create_car_objects(num_cars)
        add_objects(scene, cars)
        
        # 4. Aplica posições e orientações
        log_step(logger, "🧭 Calculando orientações e aplicando posições")
        all_positions = load_receiver_mesh_positions(args.mesh_csv, 2000)  # Para orientação
        orientations = calculate_car_orientations_improved(car_positions, all_positions)
        apply_positions_and_orientations(cars, car_positions, orientations)
        
        # 5. Configura transmissor e receptores
        log_step(logger, "📡 Configurando transmissor e receptores")
        tx_pos = (21.18, -132.4, 18.76)  # Posição otimizada
        rx_positions = [(x, y, z + 3.0) for (x, y, z) in car_positions]  # Eleva receptores 3m
        setup_tx_rx(scene, tx_pos, rx_positions)
        log_step(logger, f"📡 TX configurado em {tx_pos}")
        log_step(logger, f"📱 {len(rx_positions)} RX configurados (altura +3m)")
        
        # 6. Calcula caminhos de propagação
        log_step(logger, "🌐 Calculando caminhos de propagação (max_depth=5)...")
        paths = compute_paths(scene, max_depth=5)
        log_step(logger, "✅ Caminhos de propagação calculados")
        
        # 7. Renderiza caminhos de propagação (como na imagem)
        if not args.skip_paths_render:
            log_step(logger, "🎨 Renderizando cena com caminhos de propagação...")
            camera_position = (484.48, -212.68, 328.85)
            camera_look_at = (83.83, -94.6, -0.0667)
            
            paths_output = plot_coverage_paths(
                scene=scene,
                paths=paths,
                camera_position=camera_position,
                camera_look_at=camera_look_at,
                output_name=f"{args.output_name}_paths",
                save_to_outputs=True
            )
            
            if paths_output:
                log_step(logger, f"✅ Visualização dos caminhos salva em: {paths_output}")
        
        # 8. Calcula RSS para todas as posições
        log_step(logger, "📊 Calculando RSS (Received Signal Strength)...")
        rss_db = rss_from_paths(paths)
        
        # Converte para arrays numpy
        positions_array = np.array(car_positions, dtype=np.float32)
        rss_array = np.reshape(rss_db.numpy(), (-1, 1))
        
        # Estatísticas RSS
        rss_values = rss_array.flatten()
        log_step(logger, "📈 ESTATÍSTICAS RSS:")
        logger.info(f"  📊 RSS médio: {np.mean(rss_values):.2f} dB")
        logger.info(f"  📉 RSS mínimo: {np.min(rss_values):.2f} dB")
        logger.info(f"  📈 RSS máximo: {np.max(rss_values):.2f} dB")
        logger.info(f"  📏 Desvio padrão: {np.std(rss_values):.2f} dB")
        
        # 9. Gera heatmap interpolado
        if not args.skip_heatmap:
            log_step(logger, "🗺️  Gerando heatmap de cobertura...")
            heatmap_output = plot_rss_heatmap(
                positions_array, 
                rss_array,
                output_name=f"{args.output_name}_heatmap",
                save_to_outputs=True,
                interpolated=True
            )
            
            if heatmap_output:
                log_step(logger, f"✅ Heatmap salvo em: {heatmap_output}")
        
        # 10. Gera dataset CSV para ML
        if not args.skip_dataset:
            log_step(logger, "💾 Gerando dataset CSV para machine learning...")
            
            # Prepara dados para CSV
            dataset_data = []
            for i, (pos, rss_val) in enumerate(zip(rx_positions, rss_values)):
                dataset_data.append({
                    'rx_x': pos[0],
                    'rx_y': pos[1], 
                    'rx_z': pos[2],
                    'rss_db': rss_val
                })
            
            # Cria DataFrame
            df = pd.DataFrame(dataset_data)
            
            # Salva CSV
            output_dir = ensure_output_dir("data")
            csv_filename = f"dataset_{args.output_name}_{num_cars}_carros.csv"
            csv_path = output_dir / csv_filename
            
            df.to_csv(csv_path, index=False, float_format='%.6f')
            
            log_step(logger, f"✅ Dataset CSV salvo em: {csv_path}")
            log_step(logger, f"📋 Dataset contém {len(df)} amostras")
            log_step(logger, f"📊 Colunas: {list(df.columns)}")
        
        log_step(logger, "🎉 SIMULAÇÃO COMPLETA FINALIZADA COM SUCESSO!")
        
        # Resumo final
        logger.info("="*50)
        logger.info("📋 RESUMO DA SIMULAÇÃO:")
        logger.info(f"  🚗 Carros simulados: {num_cars}")
        logger.info(f"  📡 Transmissor: {tx_pos}")
        logger.info(f"  📱 Receptores: {len(rx_positions)} (altura +3m)")
        logger.info(f"  📊 RSS médio: {np.mean(rss_values):.2f} dB")
        logger.info(f"  📁 Arquivos salvos em: outputs/")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"❌ Erro durante simulação: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
