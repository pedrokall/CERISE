#!/usr/bin/env python3
"""
Script para visualizar posições dos carros usando receivers_mesh.csv
"""

import argparse
import sys
from pathlib import Path

# Adiciona o diretório pai ao path para importar o módulo tcc_project
sys.path.insert(0, str(Path(__file__).parent.parent))

from tcc_project.geometry import load_receiver_mesh_positions
from tcc_project.visualization import plot_car_positions_from_mesh
from tcc_project.utils import setup_logger, log_step


def main():
    parser = argparse.ArgumentParser(description="Visualizar posições dos carros usando receivers_mesh.csv")
    parser.add_argument("--mesh-csv", type=str, default="receivers_mesh.csv",
                       help="Caminho para o arquivo receivers_mesh.csv (padrão: receivers_mesh.csv)")
    parser.add_argument("--max-cars", type=int, default=None,
                       help="Número máximo de carros a mostrar (padrão: todos)")
    parser.add_argument("--output-name", type=str, default="car_positions_mesh",
                       help="Nome base para o arquivo de saída (padrão: car_positions_mesh)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(__name__)
    
    try:
        log_step(logger, f"Carregando posições de carros de {args.mesh_csv}")
        
        # Carrega posições do receivers_mesh.csv
        car_positions = load_receiver_mesh_positions(args.mesh_csv, args.max_cars)
        
        num_cars = len(car_positions)
        log_step(logger, f"Carregadas {num_cars} posições de carros")
        
        # Gera o plot
        log_step(logger, "Gerando visualização das posições dos carros...")
        
        output_path = plot_car_positions_from_mesh(
            car_positions=car_positions,
            output_name=args.output_name,
            save_to_outputs=True
        )
        
        if output_path:
            log_step(logger, f"✅ Visualização salva em: {output_path}")
        else:
            log_step(logger, "⚠️  Erro ao salvar a visualização")
            
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
