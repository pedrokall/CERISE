import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Adicionar o diretório pai ao path para encontrar tcc_project
sys.path.insert(0, str(Path(__file__).parent.parent))

from tcc_project.scene import load_scene_with_options, create_car_objects, add_objects
from tcc_project.geometry import select_non_consecutive_vertices, calculate_car_orientations_improved, apply_positions_and_orientations
from tcc_project.sim import setup_tx_rx, compute_paths, rss_from_paths
from tcc_project.utils import setup_logger, log_step, log_progress, ensure_output_dir


def main():
    parser = argparse.ArgumentParser(description="Gerar dataset RX (posições) e RSS a partir de uma cena Sionna RT.")
    parser.add_argument("--scene", required=True, help="Caminho para o arquivo .xml da cena (ex.: universitario.xml)")
    parser.add_argument("--vertices-json", required=True, help="Arquivo JSON com lista de vértices [[x,y,z], ...]")
    parser.add_argument("--num-cars", type=int, default=100, help="Número de carros/receptores")
    parser.add_argument("--output", default="dataset_rx_e_potencia.csv", help="Arquivo CSV de saída")
    parser.add_argument("--seed", type=int, default=42, help="Semente para reprodutibilidade")
    parser.add_argument("--preview", action="store_true", help="Usar preview ao invés de render")
    parser.add_argument("--save-to-outputs", action="store_true", help="Salvar CSV na pasta outputs/data")
    args = parser.parse_args()

    logger = setup_logger("generate_dataset")
    
    try:
        log_step(logger, "INICIALIZANDO GERAÇÃO DE DATASET")
        
        # Carregar cena
        log_step(logger, f"Carregando cena: {args.scene}")
        scene = load_scene_with_options(args.scene, merge_shapes=True, preview=args.preview)
        log_step(logger, f"Cena carregada com {len(scene.objects)} objetos")

        # Carregar vértices
        log_step(logger, f"Carregando vértices de: {args.vertices_json}")
        vertices = json.loads(Path(args.vertices_json).read_text(encoding="utf-8"))
        vertices = [tuple(map(float, v)) for v in vertices]
        log_step(logger, f"Carregados {len(vertices)} vértices")

        # Criar carros e posicioná-los
        log_step(logger, f"Criando {args.num_cars} carros")
        cars = create_car_objects(args.num_cars)
        add_objects(scene, cars)
        log_step(logger, f"Carros adicionados. Total de objetos: {len(scene.objects)}")

        log_step(logger, "Selecionando posições dos carros")
        car_positions = select_non_consecutive_vertices(vertices, args.num_cars, min_gap=2, seed=args.seed)
        
        log_step(logger, "Calculando orientações dos carros")
        orientations = calculate_car_orientations_improved(car_positions, vertices)
        
        log_step(logger, "Aplicando posições e orientações")
        apply_positions_and_orientations(cars, car_positions, orientations)

        # Definir TX e RX (+3m no z para receptores)
        tx_position = (21.18, -132.4, 18.76)
        rx_positions = [(x, y, z + 3.0) for (x, y, z) in car_positions]
        
        log_step(logger, f"Configurando TX em {tx_position} e {len(rx_positions)} RXs")
        setup_tx_rx(scene, tx_position, rx_positions)

        # Calcular caminhos e RSS
        log_step(logger, "Calculando caminhos de propagação (max_depth=5)")
        paths = compute_paths(scene, max_depth=5)
        
        log_step(logger, "Calculando RSS a partir dos caminhos")
        rss_db_per_rx = rss_from_paths(paths)

        X_receptores = np.array(rx_positions, dtype=np.float32)
        y_rss = np.reshape(rss_db_per_rx.numpy(), (-1, 1))

        # Salvar CSV
        if args.save_to_outputs:
            output_dir = ensure_output_dir("data")
            output_path = output_dir / args.output
        else:
            output_path = args.output
            
        log_step(logger, f"Salvando dataset em: {output_path}")
        df = pd.DataFrame(X_receptores, columns=["rx_x", "rx_y", "rx_z"])
        df["rss_db"] = y_rss
        df.to_csv(output_path, index=False)
        
        # Estatísticas
        rss_values = y_rss.flatten()
        logger.info("ESTATÍSTICAS DO DATASET:")
        logger.info(f"  Shape X: {X_receptores.shape}")
        logger.info(f"  Shape y: {y_rss.shape}")
        logger.info(f"  RSS médio: {np.mean(rss_values):.2f} dB")
        logger.info(f"  RSS mínimo: {np.min(rss_values):.2f} dB")
        logger.info(f"  RSS máximo: {np.max(rss_values):.2f} dB")
        logger.info(f"  Valores -100 dB: {np.sum(rss_values <= -99)}/{len(rss_values)}")
        
        log_step(logger, f"CONCLUÍDO - Dataset salvo em: {output_path}")
        
    except Exception as e:
        logger.error(f"Erro durante geração do dataset: {e}")
        raise


if __name__ == "__main__":
    main()
