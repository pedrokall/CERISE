import argparse
import sys
from pathlib import Path

# Adicionar o diretório pai ao path para encontrar tcc_project
sys.path.insert(0, str(Path(__file__).parent.parent))

from tcc_project.scene import load_scene_with_options, create_car_objects, add_objects
from tcc_project.visualization import render_scene
from tcc_project.utils import setup_logger, log_step


def main():
    parser = argparse.ArgumentParser(description="Renderizar cena Sionna RT")
    parser.add_argument("--scene", required=True, help="Arquivo XML da cena")
    parser.add_argument("--output-name", default="scene_render", help="Nome base do arquivo de saída")
    parser.add_argument("--camera-x", type=float, default=484.48, help="Posição X da câmera")
    parser.add_argument("--camera-y", type=float, default=-212.68, help="Posição Y da câmera")
    parser.add_argument("--camera-z", type=float, default=328.85, help="Posição Z da câmera")
    parser.add_argument("--look-at-x", type=float, default=83.83, help="Look-at X da câmera")
    parser.add_argument("--look-at-y", type=float, default=-94.6, help="Look-at Y da câmera")
    parser.add_argument("--look-at-z", type=float, default=-0.0667, help="Look-at Z da câmera")
    parser.add_argument("--merge-shapes", action="store_true", default=True, help="Mesclar objetos da cena")
    parser.add_argument("--add-cars", type=int, default=0, help="Adicionar N carros à cena")
    args = parser.parse_args()

    logger = setup_logger("render_scene")
    
    try:
        log_step(logger, "INICIALIZANDO RENDERIZAÇÃO")
        log_step(logger, f"Carregando cena: {args.scene}")
        
        if not Path(args.scene).exists():
            logger.error(f"Arquivo de cena não encontrado: {args.scene}")
            sys.exit(1)
        
        # Carregar cena
        scene = load_scene_with_options(
            args.scene, 
            merge_shapes=args.merge_shapes,
            preview=False
        )
        
        log_step(logger, f"Cena carregada com {len(scene.objects)} objetos")
        
        # Adicionar carros se solicitado
        if args.add_cars > 0:
            log_step(logger, f"Adicionando {args.add_cars} carros à cena")
            cars = create_car_objects(args.add_cars)
            add_objects(scene, cars)
            log_step(logger, f"Carros adicionados. Total de objetos: {len(scene.objects)}")
        
        # Renderizar
        camera_pos = (args.camera_x, args.camera_y, args.camera_z)
        look_at = (args.look_at_x, args.look_at_y, args.look_at_z)
        
        log_step(logger, f"Renderizando com câmera em {camera_pos}")
        
        output_path = render_scene(
            scene, 
            camera_position=camera_pos,
            camera_look_at=look_at,
            output_name=args.output_name
        )
        
        if output_path:
            log_step(logger, f"CONCLUÍDO - Render salvo em: {output_path}")
        else:
            logger.error("Falha ao salvar o render")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Erro durante renderização: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
