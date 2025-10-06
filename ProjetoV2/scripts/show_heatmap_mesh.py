#!/usr/bin/env python3
"""
Script para gerar apenas o heatmap de RSS usando receivers_mesh.csv
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Adiciona o diretório pai ao path para importar o módulo tcc_project
sys.path.insert(0, str(Path(__file__).parent.parent))

from tcc_project.scene import load_scene_with_options, create_car_objects, add_objects
from tcc_project.geometry import load_receiver_mesh_positions, calculate_car_orientations_improved, apply_positions_and_orientations
from tcc_project.sim import setup_tx_rx, compute_paths, rss_from_paths
from tcc_project.visualization import plot_rss_heatmap
from tcc_project.utils import setup_logger, log_step


def main():
    parser = argparse.ArgumentParser(description="Gerar heatmap de RSS usando receivers_mesh.csv")
    parser.add_argument("--scene", type=str, required=True, help="Arquivo de cena (.xml)")
    parser.add_argument("--mesh-csv", type=str, default="receivers_mesh.csv", help="Arquivo receivers_mesh.csv")
    parser.add_argument("--max-cars", type=int, default=1000, help="Número máximo de carros")
    parser.add_argument("--output-name", type=str, default="heatmap_mesh", help="Nome base dos arquivos")
    
    args = parser.parse_args()
    logger = setup_logger(__name__)
    
    try:
        # 1. Carrega cena
        log_step(logger, f"Carregando cena: {args.scene}")
        scene = load_scene_with_options(args.scene)
        
        # 2. Carrega posições do mesh
        log_step(logger, f"Carregando {args.max_cars} posições do mesh")
        car_positions = load_receiver_mesh_positions(args.mesh_csv, args.max_cars)
        num_cars = len(car_positions)
        log_step(logger, f"✅ {num_cars} posições carregadas")
        
        # 3. Cria e adiciona carros
        log_step(logger, "Criando e adicionando carros à cena")
        cars = create_car_objects(num_cars)
        add_objects(scene, cars)
        
        # 4. Aplica posições
        log_step(logger, "Aplicando posições aos carros")
        all_positions = load_receiver_mesh_positions(args.mesh_csv, 2000)  # Para orientação
        orientations = calculate_car_orientations_improved(car_positions, all_positions)
        apply_positions_and_orientations(cars, car_positions, orientations)
        
        # 5. Configura TX/RX
        log_step(logger, "Configurando transmissor e receptores")
        tx_pos = (21.18, -132.4, 18.76)  # Mesma posição do script que funciona
        rx_positions = [(x, y, z + 3.0) for (x, y, z) in car_positions]  # Eleva receptores 3m
        setup_tx_rx(scene, tx_pos, rx_positions)
        
        # 6. Calcula caminhos
        log_step(logger, "Calculando caminhos de propagação (max_depth=5)...")
        paths = compute_paths(scene, max_depth=5)
        log_step(logger, "✅ Caminhos calculados")
        
        # 7. Calcula RSS e gera heatmap
        log_step(logger, "Calculando RSS e gerando heatmap...")
        rss_db = rss_from_paths(paths)
        
        positions_array = np.array(car_positions, dtype=np.float32)
        rss_array = np.reshape(rss_db.numpy(), (-1, 1))
        
        heatmap_output = plot_rss_heatmap(
            positions_array, 
            rss_array,
            output_name=args.output_name,
            interpolated=True
        )
        
        if heatmap_output:
            log_step(logger, f"✅ Heatmap salvo em: {heatmap_output}")
        
        # Estatísticas RSS
        rss_values = rss_array.flatten()
        logger.info("ESTATÍSTICAS RSS:")
        logger.info(f"  RSS médio: {np.mean(rss_values):.2f} dB")
        logger.info(f"  RSS mínimo: {np.min(rss_values):.2f} dB")
        logger.info(f"  RSS máximo: {np.max(rss_values):.2f} dB")
        logger.info(f"  Desvio padrão: {np.std(rss_values):.2f} dB")
        
        log_step(logger, "🎉 Heatmap gerado com sucesso!")
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
