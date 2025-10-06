from typing import List, Tuple, Optional
from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sionna.rt import Scene, Camera
from .utils import ensure_output_dir, get_timestamp


logger = logging.getLogger(__name__)


def render_scene(scene: Scene, 
                 camera_position: Tuple[float, float, float] = (484.48, -212.68, 328.85),
                 camera_look_at: Tuple[float, float, float] = (83.83, -94.6, -0.0667),
                 output_name: str = "scene_render",
                 save_to_outputs: bool = True) -> Optional[str]:
    """Renderiza a cena e salva a imagem."""
    logger.info(f"Renderizando cena com câmera em {camera_position}")
    
    cam = Camera(position=list(camera_position), look_at=list(camera_look_at))
    fig_or_img = scene.render(camera=cam)
    
    if save_to_outputs:
        output_dir = ensure_output_dir("renders")
        timestamp = get_timestamp()
        filename = f"{output_name}_{timestamp}.png"
        filepath = output_dir / filename
        
        try:
            if hasattr(fig_or_img, 'savefig'):
                # É uma Figure do matplotlib
                fig_or_img.savefig(filepath, dpi=150, bbox_inches="tight")
                plt.close(fig_or_img)
            else:
                # É um array NumPy
                plt.imsave(filepath, np.clip(np.asarray(fig_or_img), 0.0, 1.0))
            
            logger.info(f"Render salvo em: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Erro ao salvar render: {e}")
            return None
    
    return None


def plot_car_positions(car_positions: List[Tuple[float, float, float]],
                       all_vertices: List[Tuple[float, float, float]],
                       output_name: str = "car_positions",
                       save_to_outputs: bool = True) -> Optional[str]:
    """Cria gráfico de distribuição dos carros nos vértices."""
    logger.info(f"Criando gráfico de posições para {len(car_positions)} carros")
    
    # Extrair coordenadas
    all_x = [v[0] for v in all_vertices]
    all_y = [v[1] for v in all_vertices]
    car_x = [pos[0] for pos in car_positions]
    car_y = [pos[1] for pos in car_positions]
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot dos vértices e carros
    ax.scatter(all_x, all_y, c='lightblue', s=10, alpha=0.6, label='Todos os vértices da rua')
    ax.scatter(car_x, car_y, c='red', s=50, marker='^', label=f'Posições dos carros ({len(car_positions)})')
    
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.set_title('Distribuição dos Carros nos Vértices da Rua')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    if save_to_outputs:
        output_dir = ensure_output_dir("plots")
        timestamp = get_timestamp()
        filename = f"{output_name}_{timestamp}.png"
        filepath = output_dir / filename
        
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Gráfico de posições salvo em: {filepath}")
        return str(filepath)
    
    plt.show()
    return None


def plot_coverage_paths(scene: Scene, paths,
                        camera_position: Tuple[float, float, float] = (484.48, -212.68, 328.85),
                        camera_look_at: Tuple[float, float, float] = (83.83, -94.6, -0.0667),
                        output_name: str = "coverage_paths",
                        save_to_outputs: bool = True) -> Optional[str]:
    """Renderiza a cena com caminhos de propagação visíveis."""
    logger.info("Renderizando cena com caminhos de propagação")
    
    cam = Camera(position=list(camera_position), look_at=list(camera_look_at))
    fig_or_img = scene.render(camera=cam, paths=paths)
    
    if save_to_outputs:
        output_dir = ensure_output_dir("renders")
        timestamp = get_timestamp()
        filename = f"{output_name}_{timestamp}.png"
        filepath = output_dir / filename
        
        try:
            if hasattr(fig_or_img, 'savefig'):
                fig_or_img.savefig(filepath, dpi=150, bbox_inches="tight")
                plt.close(fig_or_img)
            else:
                plt.imsave(filepath, np.clip(np.asarray(fig_or_img), 0.0, 1.0))
            
            logger.info(f"Render com caminhos salvo em: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Erro ao salvar render com caminhos: {e}")
            return None
    
    return None


def plot_car_positions_from_mesh(car_positions: List[Tuple[float, float, float]], output_name: str = "car_positions_mesh", save_to_outputs: bool = True) -> Optional[str]:
    """
    Plota as posições dos carros carregadas do receivers_mesh.csv
    
    Args:
        car_positions: Lista de posições (x, y, z) dos carros
        output_name: Nome base para o arquivo de saída
        save_to_outputs: Se deve salvar na pasta outputs/plots/
        
    Returns:
        Caminho do arquivo salvo ou None se erro
    """
    logger.info(f"Plotando {len(car_positions)} posições de carros do mesh")
    
    try:
        # Extrai coordenadas X e Y
        x_coords = [pos[0] for pos in car_positions]
        y_coords = [pos[1] for pos in car_positions]
        
        # Cria o plot
        plt.figure(figsize=(12, 10))
        plt.scatter(x_coords, y_coords, c='red', marker='^', s=30, alpha=0.7, 
                   label=f'Posições dos carros ({len(car_positions)})')
        
        plt.xlabel('Coordenada X', fontsize=12)
        plt.ylabel('Coordenada Y', fontsize=12)
        plt.title('Distribuição dos Carros - Receivers Mesh', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Define limites dos eixos com margem
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)
        
        plt.tight_layout()
        
        # Salva o arquivo
        if save_to_outputs:
            output_dir = ensure_output_dir("plots")
            timestamp = get_timestamp()
            filename = f"{output_name}_{timestamp}.png"
            output_path = output_dir / filename
        else:
            filename = f"{output_name}.png"
            output_path = Path(filename)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot salvo em: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Erro ao plotar posições do mesh: {e}")
        plt.close()
        return None


def plot_rss_heatmap(positions: np.ndarray, rss_values: np.ndarray,
                     output_name: str = "rss_heatmap",
                     save_to_outputs: bool = True,
                     interpolated: bool = True) -> Optional[str]:
    """Cria mapa de calor do RSS."""
    logger.info("Criando mapa de calor do RSS")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if interpolated:
        # Criar mapa interpolado suave como na segunda imagem
        from scipy.interpolate import griddata
        
        # Definir grid para interpolação
        x_min, x_max = positions[:, 0].min() - 50, positions[:, 0].max() + 50
        y_min, y_max = positions[:, 1].min() - 50, positions[:, 1].max() + 50
        
        xi = np.linspace(x_min, x_max, 200)
        yi = np.linspace(y_min, y_max, 200)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolar valores RSS
        rss_flat = rss_values.flatten()
        zi = griddata((positions[:, 0], positions[:, 1]), rss_flat, 
                     (xi_grid, yi_grid), method='cubic', fill_value=rss_flat.min())
        
        # Criar contour plot suavizado
        contour = ax.contourf(xi_grid, yi_grid, zi, levels=50, cmap='plasma', alpha=0.8)
        
        # Adicionar pontos dos receptores
        ax.scatter(positions[:, 0], positions[:, 1], 
                  c='black', s=15, alpha=0.7, label='Receptores')
        
        cbar = plt.colorbar(contour, ax=ax)
        ax.legend()
    else:
        # Mapa de pontos discretos (original)
        scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                            c=rss_values.flatten(), 
                            cmap='viridis', s=60, alpha=0.8)
        cbar = plt.colorbar(scatter, ax=ax)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Mapa de Cobertura - RSRP')
    ax.grid(True, alpha=0.3)
    
    cbar.set_label('RSRP (dBm)')
    
    if save_to_outputs:
        output_dir = ensure_output_dir("plots")
        timestamp = get_timestamp()
        suffix = "_interpolated" if interpolated else "_scatter"
        filename = f"{output_name}{suffix}_{timestamp}.png"
        filepath = output_dir / filename
        
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Mapa de calor salvo em: {filepath}")
        return str(filepath)
    
    plt.show()
    return None
