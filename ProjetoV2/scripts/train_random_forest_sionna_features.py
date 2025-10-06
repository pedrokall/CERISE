#!/usr/bin/env python3
"""
Script para treinar Random Forest usando features REAIS extraídas do Sionna RT
- Features do Sionna: delays (tau), ângulos, número de caminhos, distâncias dos caminhos
- Features geométricas: distância euclidiana, altura relativa, coordenadas polares
- Comparação com modelo baseline
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Adiciona o diretório pai ao path para importar o módulo tcc_project
sys.path.insert(0, str(Path(__file__).parent.parent))

from tcc_project.scene import load_scene_with_options, create_car_objects, add_objects
from tcc_project.geometry import load_receiver_mesh_positions, calculate_car_orientations_improved, apply_positions_and_orientations
from tcc_project.sim import setup_tx_rx, compute_paths, rss_from_paths
from tcc_project.utils import setup_logger, log_step, ensure_output_dir


def extract_sionna_features(paths, rx_positions, tx_position=(21.18, -132.4, 18.76)):
    """
    Extrai features REAIS do objeto paths do Sionna RT
    
    Args:
        paths: Objeto paths retornado pelo Sionna RT
        rx_positions: Lista de posições dos receptores
        tx_position: Posição do transmissor
    
    Returns:
        DataFrame com features extraídas
    """
    c = 3e8  # velocidade da luz em m/s
    
    features = []
    
    for rx_idx in range(len(rx_positions)):
        rx_pos = rx_positions[rx_idx]
        
        # Features geométricas básicas
        feature_dict = {
            'rx_x': rx_pos[0],
            'rx_y': rx_pos[1], 
            'rx_z': rx_pos[2]
        }
        
        # 1. Distância euclidiana 3D (geometria pura)
        feature_dict['distance_euclidean'] = np.sqrt(
            (rx_pos[0] - tx_position[0])**2 + 
            (rx_pos[1] - tx_position[1])**2 + 
            (rx_pos[2] - tx_position[2])**2
        )
        
        # 2. Distância horizontal (2D)
        feature_dict['distance_horizontal'] = np.sqrt(
            (rx_pos[0] - tx_position[0])**2 + 
            (rx_pos[1] - tx_position[1])**2
        )
        
        # 3. Diferença de altura
        feature_dict['height_diff'] = rx_pos[2] - tx_position[2]
        
        # 4. Coordenadas polares
        feature_dict['polar_angle'] = np.arctan2(
            rx_pos[1] - tx_position[1], 
            rx_pos[0] - tx_position[0]
        )
        feature_dict['polar_radius'] = feature_dict['distance_horizontal']
        
        # 5. Ângulo de elevação
        feature_dict['elevation_angle'] = np.arctan2(
            feature_dict['height_diff'], 
            feature_dict['distance_horizontal']
        )
        
        # 6. FEATURES DO SIONNA RT - Delays (tau)
        try:
            tau_values = paths.tau.numpy()[rx_idx, 0, :]  # [rx_idx, tx=0, paths]
            valid_taus = tau_values[tau_values > 0]  # Remove valores inválidos (-1.0)
            
            if len(valid_taus) > 0:
                # Delay do caminho direto (menor delay)
                feature_dict['tau_direct'] = np.min(valid_taus)
                # Delay médio de todos os caminhos
                feature_dict['tau_mean'] = np.mean(valid_taus)
                # Delay máximo (caminho mais longo)
                feature_dict['tau_max'] = np.max(valid_taus)
                # Spread dos delays (dispersão temporal)
                feature_dict['tau_spread'] = np.max(valid_taus) - np.min(valid_taus)
                # Número de caminhos válidos
                feature_dict['num_paths'] = len(valid_taus)
                
                # Distâncias calculadas dos delays
                feature_dict['distance_direct'] = feature_dict['tau_direct'] * c
                feature_dict['distance_mean'] = feature_dict['tau_mean'] * c
                feature_dict['distance_max'] = feature_dict['tau_max'] * c
                
                # Excess path length (diferença entre caminho direto e euclidiano)
                feature_dict['excess_path_length'] = feature_dict['distance_direct'] - feature_dict['distance_euclidean']
                
            else:
                # Se não há caminhos válidos, usar valores padrão
                feature_dict['tau_direct'] = 0
                feature_dict['tau_mean'] = 0
                feature_dict['tau_max'] = 0
                feature_dict['tau_spread'] = 0
                feature_dict['num_paths'] = 0
                feature_dict['distance_direct'] = 0
                feature_dict['distance_mean'] = 0
                feature_dict['distance_max'] = 0
                feature_dict['excess_path_length'] = 0
        except Exception as e:
            print(f"Erro ao extrair features de tau para RX{rx_idx}: {e}")
            # Valores padrão em caso de erro
            for key in ['tau_direct', 'tau_mean', 'tau_max', 'tau_spread', 'num_paths', 
                       'distance_direct', 'distance_mean', 'distance_max', 'excess_path_length']:
                feature_dict[key] = 0
        
        # 7. FEATURES DO SIONNA RT - Ângulos
        try:
            # Ângulos de transmissão
            theta_t = paths.theta_t.numpy()[rx_idx, 0, :]
            phi_t = paths.phi_t.numpy()[rx_idx, 0, :]
            
            # Ângulos de recepção  
            theta_r = paths.theta_r.numpy()[rx_idx, 0, :]
            phi_r = paths.phi_r.numpy()[rx_idx, 0, :]
            
            # Usar apenas ângulos de caminhos válidos
            valid_mask = tau_values > 0
            if np.any(valid_mask):
                feature_dict['theta_t_mean'] = np.mean(theta_t[valid_mask])
                feature_dict['phi_t_mean'] = np.mean(phi_t[valid_mask])
                feature_dict['theta_r_mean'] = np.mean(theta_r[valid_mask])
                feature_dict['phi_r_mean'] = np.mean(phi_r[valid_mask])
                
                # Spread angular
                feature_dict['theta_t_spread'] = np.std(theta_t[valid_mask])
                feature_dict['phi_t_spread'] = np.std(phi_t[valid_mask])
            else:
                for key in ['theta_t_mean', 'phi_t_mean', 'theta_r_mean', 'phi_r_mean',
                           'theta_t_spread', 'phi_t_spread']:
                    feature_dict[key] = 0
                    
        except Exception as e:
            print(f"Erro ao extrair features de ângulos para RX{rx_idx}: {e}")
            for key in ['theta_t_mean', 'phi_t_mean', 'theta_r_mean', 'phi_r_mean',
                       'theta_t_spread', 'phi_t_spread']:
                feature_dict[key] = 0
        
        features.append(feature_dict)
    
    return pd.DataFrame(features)


def generate_enhanced_dataset(scene_file="universitario.xml", mesh_file="receivers_mesh.csv", 
                            num_cars=1000, tx_position=(21.18, -132.4, 18.76)):
    """
    Gera dataset com features do Sionna RT
    """
    logger = setup_logger("dataset_generation")
    
    log_step(logger, f"🚗 Gerando dataset com {num_cars} carros usando features do Sionna RT")
    
    # Carregar cena
    scene = load_scene_with_options(scene_file)
    
    # Carregar posições dos carros
    car_positions = load_receiver_mesh_positions(mesh_file, num_cars)
    cars = create_car_objects(len(car_positions))
    add_objects(scene, cars)
    
    # Aplicar posições e orientações
    all_positions = load_receiver_mesh_positions(mesh_file, 2000)
    orientations = calculate_car_orientations_improved(car_positions, all_positions)
    apply_positions_and_orientations(cars, car_positions, orientations)
    
    # Configurar TX/RX
    rx_positions = [(x, y, z + 3.0) for (x, y, z) in car_positions]
    setup_tx_rx(scene, tx_position, rx_positions)
    
    # Calcular paths
    log_step(logger, "📡 Calculando paths com Sionna RT...")
    paths = compute_paths(scene, max_depth=5)
    
    # Calcular RSS
    log_step(logger, "📊 Calculando RSS...")
    rss_db = rss_from_paths(paths)
    
    # Extrair features do Sionna
    log_step(logger, "🔬 Extraindo features do Sionna RT...")
    features_df = extract_sionna_features(paths, rx_positions, tx_position)
    
    # Adicionar RSS ao DataFrame
    features_df['rss_db'] = rss_db.numpy()
    
    log_step(logger, f"✅ Dataset gerado com {len(features_df)} amostras e {len(features_df.columns)} features")
    
    return features_df


def main():
    parser = argparse.ArgumentParser(description="Treinar Random Forest com features do Sionna RT")
    parser.add_argument("--generate-data", action="store_true",
                       help="Gerar novo dataset com features do Sionna RT")
    parser.add_argument("--data", type=str, default="outputs/data/dataset_sionna_features.csv",
                       help="Caminho para o dataset CSV com features do Sionna")
    parser.add_argument("--num-cars", type=int, default=500,
                       help="Número de carros para gerar dataset (se --generate-data)")
    parser.add_argument("--n-estimators", type=int, default=75,
                       help="Número de árvores no Random Forest")
    parser.add_argument("--max-depth", type=int, default=10,
                       help="Profundidade máxima das árvores")
    parser.add_argument("--min-samples-split", type=int, default=5,
                       help="Número mínimo de amostras para dividir um nó")
    parser.add_argument("--min-samples-leaf", type=int, default=2,
                       help="Número mínimo de amostras em uma folha")
    parser.add_argument("--output-name", type=str, default="rf_sionna_features",
                       help="Nome base para os arquivos de saída")
    
    args = parser.parse_args()
    
    logger = setup_logger(__name__)
    
    try:
        log_step(logger, "🚀 INICIANDO TREINAMENTO COM FEATURES DO SIONNA RT")
        
        # 1. Gerar ou carregar dataset
        if args.generate_data:
            log_step(logger, "🔄 Gerando novo dataset...")
            df = generate_enhanced_dataset(num_cars=args.num_cars)
            
            # Salvar dataset
            data_dir = ensure_output_dir("data")
            dataset_path = data_dir / "dataset_sionna_features.csv"
            df.to_csv(dataset_path, index=False)
            log_step(logger, f"💾 Dataset salvo em: {dataset_path}")
        else:
            log_step(logger, f"📁 Carregando dataset existente: {args.data}")
            df = pd.read_csv(args.data)
        
        log_step(logger, f"📊 Dataset shape: {df.shape}")
        log_step(logger, f"📋 Features disponíveis: {list(df.columns)}")
        
        # 2. Preparar features e target
        target_col = 'rss_db'
        feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        log_step(logger, f"🎯 Usando {len(feature_cols)} features para predição")
        log_step(logger, f"📈 RSS range: {y.min():.2f} to {y.max():.2f} dBm")
        
        # 3. Dividir dados
        log_step(logger, "🔄 Dividindo dados (50% treino, 25% val, 25% teste)")
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.333, random_state=42)
        
        log_step(logger, f"📊 Treino: {len(X_train)}, Validação: {len(X_val)}, Teste: {len(X_test)}")
        
        # 4. Normalizar features
        log_step(logger, "⚙️ Normalizando features")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 5. Treinar modelo
        log_step(logger, f"🌳 Treinando Random Forest com features do Sionna RT")
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        log_step(logger, "✅ Modelo treinado com sucesso")
        
        # 6. Fazer predições
        log_step(logger, "🎯 Fazendo predições")
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # 7. Calcular métricas
        log_step(logger, "📊 Calculando métricas")
        
        def calculate_metrics(y_true, y_pred):
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
        
        train_metrics = calculate_metrics(y_train, y_train_pred)
        val_metrics = calculate_metrics(y_val, y_val_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)
        
        # 8. Exibir resultados
        logger.info("=" * 80)
        logger.info("📊 RESULTADOS COM FEATURES DO SIONNA RT:")
        logger.info("=" * 80)
        logger.info(f"🔹 TREINO   - MSE: {train_metrics['mse']:.3f}, MAE: {train_metrics['mae']:.3f}, R²: {train_metrics['r2']:.3f}")
        logger.info(f"🔸 VALIDAÇÃO - MSE: {val_metrics['mse']:.3f}, MAE: {val_metrics['mae']:.3f}, R²: {val_metrics['r2']:.3f}")
        logger.info(f"🔹 TESTE    - MSE: {test_metrics['mse']:.3f}, MAE: {test_metrics['mae']:.3f}, R²: {test_metrics['r2']:.3f}")
        
        # Gap de overfitting
        overfitting_gap = train_metrics['r2'] - test_metrics['r2']
        logger.info(f"📉 Gap de overfitting (R² treino - R² teste): {overfitting_gap:.3f}")
        logger.info("=" * 80)
        
        # 9. Análise de importância das features
        log_step(logger, "🔬 Analisando importância das features")
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("🏆 TOP 10 FEATURES MAIS IMPORTANTES:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"   {i+1:2d}. {row['feature']:20s}: {row['importance']:.3f}")
        
        # 10. Criar pastas organizadas para plots
        base_output_dir = ensure_output_dir("ml_analysis")
        learning_curves_dir = ensure_output_dir("ml_analysis/learning_curves")
        predictions_dir = ensure_output_dir("ml_analysis/predictions")
        heatmaps_dir = ensure_output_dir("ml_analysis/heatmaps")
        feature_importance_dir = ensure_output_dir("ml_analysis/feature_importance")
        residuals_dir = ensure_output_dir("ml_analysis/residuals")
        loss_comparison_dir = ensure_output_dir("ml_analysis/loss_comparison")
        metrics_dir = ensure_output_dir("ml_analysis/metrics")
        
        log_step(logger, f"📁 Salvando plots organizados em subdiretórios de: {base_output_dir}")
        
        # 11. Gerar Learning Curve
        log_step(logger, "📈 Gerando learning curve")
        from sklearn.model_selection import learning_curve as sk_learning_curve
        
        train_sizes, train_scores, val_scores = sk_learning_curve(
            model, X_train_scaled, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10), random_state=42,
            scoring='r2'
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                         val_scores_mean + val_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training R²")
        plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation R²")
        plt.xlabel("Training Set Size")
        plt.ylabel("R² Score")
        plt.title("Learning Curve - Random Forest (Dataset Rico)")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.savefig(learning_curves_dir / f"{args.output_name}_learning_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 12. Gerar gráficos de predições vs valores reais
        log_step(logger, "🎯 Gerando gráficos de predições vs valores reais")
        
        def plot_predictions_vs_actual(y_true, y_pred, title, filename):
            plt.figure(figsize=(8, 8))
            plt.scatter(y_true, y_pred, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel("Valores Reais de RSS (dBm)")
            plt.ylabel("Valores Preditos de RSS (dBm)")
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            # Adicionar R² no gráfico
            from sklearn.metrics import r2_score
            r2 = r2_score(y_true, y_pred)
            plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.savefig(predictions_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()

        plot_predictions_vs_actual(y_train, y_train_pred, "Predições vs Reais (Treino)", f"{args.output_name}_predictions_train.png")
        plot_predictions_vs_actual(y_val, y_val_pred, "Predições vs Reais (Validação)", f"{args.output_name}_predictions_validation.png")
        plot_predictions_vs_actual(y_test, y_test_pred, "Predições vs Reais (Teste)", f"{args.output_name}_predictions_test.png")

        # 13. Gerar heatmap de predições interpolado
        log_step(logger, "🗺️ Gerando heatmap de predições interpolado")
        
        # Usar apenas as coordenadas x, y para o heatmap
        positions_for_heatmap = X_test[:, :2]  # rx_x e rx_y
        x_coords = positions_for_heatmap[:, 0]
        y_coords = positions_for_heatmap[:, 1]
        
        # Criar grade regular para interpolação
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Adicionar margem
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        
        # Grade de interpolação
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min - x_margin, x_max + x_margin, 100),
            np.linspace(y_min - y_margin, y_max + y_margin, 100)
        )
        
        # Interpolação usando griddata
        from scipy.interpolate import griddata
        grid_rss = griddata(
            (x_coords, y_coords), y_test_pred,
            (grid_x, grid_y), method='cubic', fill_value=np.nan
        )
        
        # Criar heatmap interpolado
        plt.figure(figsize=(12, 10))
        
        # Heatmap de fundo interpolado
        contour = plt.contourf(grid_x, grid_y, grid_rss, levels=50, cmap='viridis', alpha=0.8)
        plt.colorbar(contour, label='RSS Predito (dBm)')
        
        # Adicionar pontos dos receptores por cima
        scatter = plt.scatter(x_coords, y_coords, c=y_test_pred, cmap='viridis', 
                            s=20, alpha=0.9, edgecolors='white', linewidth=0.5, 
                            label='Receptores')
        
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Heatmap of Predicted RSS (Test Set)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(heatmaps_dir / f"{args.output_name}_heatmap_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 14. Gerar importância das features (top 15)
        log_step(logger, "🌳 Gerando importância das features")
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)
        sns.barplot(data=top_features, x='importance', y='feature', hue='feature', palette='viridis', legend=False)
        plt.title('Top 15 Features - Importância (Random Forest Dataset Rico)')
        plt.xlabel('Importância')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(feature_importance_dir / f"{args.output_name}_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 15. Gerar análise de resíduos
        log_step(logger, "📊 Gerando análise de resíduos")
        
        residuals = y_test - y_test_pred
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(residuals.mean(), color='red', linestyle='--', label=f'Média: {residuals.mean():.3f}')
        plt.xlabel("Resíduos (Real - Predito)")
        plt.ylabel("Frequência")
        plt.title("Distribuição dos Resíduos (Conjunto de Teste)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(residuals_dir / f"{args.output_name}_residuals_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 16. Gerar comparação de loss (MSE)
        log_step(logger, "📉 Gerando gráfico de loss")
        
        datasets = ['Treino', 'Validação', 'Teste']
        mse_values = [train_metrics['mse'], val_metrics['mse'], test_metrics['mse']]
        colors = ['blue', 'orange', 'green']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(datasets, mse_values, color=colors, alpha=0.7, edgecolor='black')
        plt.ylabel('Mean Squared Error')
        plt.title('Loss (MSE) por Conjunto de Dados')
        plt.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, mse_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(mse_values)*0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.savefig(loss_comparison_dir / f"{args.output_name}_loss_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 17. Salvar métricas
        log_step(logger, "💾 Salvando métricas")
        
        metrics_df = pd.DataFrame([{
            'model': 'sionna_features_rico',
            'train_mse': train_metrics['mse'],
            'train_mae': train_metrics['mae'],
            'train_r2': train_metrics['r2'],
            'val_mse': val_metrics['mse'],
            'val_mae': val_metrics['mae'],
            'val_r2': val_metrics['r2'],
            'test_mse': test_metrics['mse'],
            'test_mae': test_metrics['mae'],
            'test_r2': test_metrics['r2'],
            'overfitting_gap': overfitting_gap,
            'n_features': len(feature_cols),
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_samples_split': args.min_samples_split,
            'min_samples_leaf': args.min_samples_leaf
        }])
        
        metrics_df.to_csv(metrics_dir / f"{args.output_name}_metrics.csv", index=False)
        feature_importance.to_csv(metrics_dir / f"{args.output_name}_feature_importance.csv", index=False)
        
        log_step(logger, "🎉 ANÁLISE COM FEATURES DO SIONNA RT COMPLETA!")
        logger.info("=" * 80)
        logger.info("📋 RESUMO FINAL:")
        logger.info(f"🎯 R² no teste: {test_metrics['r2']:.3f}")
        logger.info(f"📉 MSE no teste: {test_metrics['mse']:.3f}")
        logger.info(f"🔬 Feature mais importante: {feature_importance.iloc[0]['feature']}")
        logger.info(f"📁 Resultados salvos em: {base_output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Erro durante análise com features do Sionna: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
