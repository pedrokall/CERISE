#!/usr/bin/env python3
"""
Script avançado para treinar Random Forest com Feature Engineering
- Features originais: rx_x, rx_y, rx_z
- Features engineered: distância euclidiana, altura relativa, coordenadas polares, densidade local
- Análise de importância das features
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
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist

# Adiciona o diretório pai ao path para importar o módulo tcc_project
sys.path.insert(0, str(Path(__file__).parent.parent))

from tcc_project.utils import setup_logger, log_step, ensure_output_dir


def create_advanced_features(df, tx_position=(21.18, -132.4, 18.76)):
    """
    Cria features avançadas para melhorar a performance do modelo
    
    Args:
        df: DataFrame com colunas rx_x, rx_y, rx_z, rss_db
        tx_position: Posição do transmissor (x, y, z)
    
    Returns:
        DataFrame com features adicionais
    """
    df_enhanced = df.copy()
    
    # 1. Distância euclidiana 3D do transmissor
    df_enhanced['distance_3d'] = np.sqrt(
        (df['rx_x'] - tx_position[0])**2 + 
        (df['rx_y'] - tx_position[1])**2 + 
        (df['rx_z'] - tx_position[2])**2
    )
    
    # 2. Distância horizontal (2D) do transmissor
    df_enhanced['distance_2d'] = np.sqrt(
        (df['rx_x'] - tx_position[0])**2 + 
        (df['rx_y'] - tx_position[1])**2
    )
    
    # 3. Diferença de altura relativa ao transmissor
    df_enhanced['height_diff'] = df['rx_z'] - tx_position[2]
    
    # 4. Coordenadas polares (ângulo e raio)
    df_enhanced['polar_angle'] = np.arctan2(
        df['rx_y'] - tx_position[1], 
        df['rx_x'] - tx_position[0]
    )
    df_enhanced['polar_radius'] = df_enhanced['distance_2d']
    
    # 5. Coordenadas esféricas (elevação)
    df_enhanced['elevation_angle'] = np.arctan2(
        df_enhanced['height_diff'], 
        df_enhanced['distance_2d']
    )
    
    # 6. Features quadráticas (para capturar não-linearidades)
    df_enhanced['rx_x_squared'] = df['rx_x']**2
    df_enhanced['rx_y_squared'] = df['rx_y']**2
    df_enhanced['rx_z_squared'] = df['rx_z']**2
    
    # 7. Interações entre coordenadas
    df_enhanced['xy_interaction'] = df['rx_x'] * df['rx_y']
    df_enhanced['xz_interaction'] = df['rx_x'] * df['rx_z']
    df_enhanced['yz_interaction'] = df['rx_y'] * df['rx_z']
    
    # 8. Densidade local (número de pontos próximos)
    positions = df[['rx_x', 'rx_y', 'rx_z']].values
    distances = cdist(positions, positions)
    # Contar pontos dentro de um raio de 50 metros
    df_enhanced['local_density'] = np.sum(distances < 50, axis=1) - 1  # -1 para excluir o próprio ponto
    
    # 9. Free Space Path Loss teórico (como referência física)
    frequency_ghz = 2.4  # Assumindo 2.4 GHz
    df_enhanced['fspl_theoretical'] = 20 * np.log10(df_enhanced['distance_3d']) + \
                                    20 * np.log10(frequency_ghz) + 32.45
    
    return df_enhanced


def plot_feature_importance_comparison(baseline_importance, advanced_importance, 
                                     baseline_features, advanced_features, output_path):
    """
    Plota comparação das importâncias das features entre modelo baseline e avançado
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Modelo baseline
    ax1.barh(baseline_features, baseline_importance)
    ax1.set_title("Feature Importance - Modelo Baseline")
    ax1.set_xlabel("Importância")
    
    # Modelo avançado - mostra apenas as top 15 features
    top_indices = np.argsort(advanced_importance)[-15:]
    top_features = [advanced_features[i] for i in top_indices]
    top_importance = advanced_importance[top_indices]
    
    ax2.barh(top_features, top_importance)
    ax2.set_title("Feature Importance - Modelo Avançado (Top 15)")
    ax2.set_xlabel("Importância")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_predictions_vs_actual_comparison(y_test, baseline_pred, advanced_pred, output_path):
    """
    Plota comparação das predições entre modelo baseline e avançado
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Modelo baseline
    ax1.scatter(y_test, baseline_pred, alpha=0.6, color='blue')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel("Valores Reais (dBm)")
    ax1.set_ylabel("Predições (dBm)")
    ax1.set_title("Modelo Baseline")
    ax1.grid(True, alpha=0.3)
    
    # Modelo avançado
    ax2.scatter(y_test, advanced_pred, alpha=0.6, color='green')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel("Valores Reais (dBm)")
    ax2.set_ylabel("Predições (dBm)")
    ax2.set_title("Modelo Avançado")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Treinar Random Forest com Feature Engineering Avançado")
    parser.add_argument("--data", type=str, default="outputs/data/dataset_simulacao_1000_1000_carros.csv",
                       help="Caminho para o dataset CSV")
    parser.add_argument("--n-estimators", type=int, default=75,
                       help="Número de árvores no Random Forest")
    parser.add_argument("--max-depth", type=int, default=10,
                       help="Profundidade máxima das árvores")
    parser.add_argument("--min-samples-split", type=int, default=5,
                       help="Número mínimo de amostras para dividir um nó")
    parser.add_argument("--min-samples-leaf", type=int, default=2,
                       help="Número mínimo de amostras em uma folha")
    parser.add_argument("--output-name", type=str, default="rf_advanced_features",
                       help="Nome base para os arquivos de saída")
    
    args = parser.parse_args()
    
    logger = setup_logger(__name__)
    
    try:
        log_step(logger, "🚀 INICIANDO TREINAMENTO RANDOM FOREST COM FEATURE ENGINEERING")
        
        # 1. Carregar e preparar dados
        log_step(logger, f"📁 Carregando dataset: {args.data}")
        df = pd.read_csv(args.data)
        
        log_step(logger, f"📊 Dataset original shape: {df.shape}")
        log_step(logger, f"📈 RSS range: {df['rss_db'].min():.2f} to {df['rss_db'].max():.2f} dBm")
        
        # 2. Criar features avançadas
        log_step(logger, "🔬 Criando features avançadas...")
        df_enhanced = create_advanced_features(df)
        
        log_step(logger, f"📊 Dataset com features avançadas shape: {df_enhanced.shape}")
        log_step(logger, f"🆕 Novas features criadas: {df_enhanced.shape[1] - df.shape[1]}")
        
        # Features originais vs avançadas
        original_features = ['rx_x', 'rx_y', 'rx_z']
        all_features = [col for col in df_enhanced.columns if col != 'rss_db']
        advanced_features = [col for col in all_features if col not in original_features]
        
        logger.info(f"📋 Features originais: {original_features}")
        logger.info(f"🆕 Features avançadas: {advanced_features}")
        
        # 3. Preparar dados para treinamento
        X_original = df_enhanced[original_features].values
        X_advanced = df_enhanced[all_features].values
        y = df_enhanced['rss_db'].values
        
        # Dividir dados
        log_step(logger, "🔄 Dividindo dados (50% treino, 25% val, 25% teste)")
        X_orig_train_val, X_orig_test, X_adv_train_val, X_adv_test, y_train_val, y_test = train_test_split(
            X_original, X_advanced, y, test_size=0.25, random_state=42
        )
        X_orig_train, X_orig_val, X_adv_train, X_adv_val, y_train, y_val = train_test_split(
            X_orig_train_val, X_adv_train_val, y_train_val, test_size=0.333, random_state=42
        )
        
        log_step(logger, f"📊 Treino: {len(X_orig_train)}, Validação: {len(X_orig_val)}, Teste: {len(X_orig_test)}")
        
        # 4. Normalizar features
        log_step(logger, "⚙️ Normalizando features")
        scaler_orig = StandardScaler()
        scaler_adv = StandardScaler()
        
        X_orig_train_scaled = scaler_orig.fit_transform(X_orig_train)
        X_orig_val_scaled = scaler_orig.transform(X_orig_val)
        X_orig_test_scaled = scaler_orig.transform(X_orig_test)
        
        X_adv_train_scaled = scaler_adv.fit_transform(X_adv_train)
        X_adv_val_scaled = scaler_adv.transform(X_adv_val)
        X_adv_test_scaled = scaler_adv.transform(X_adv_test)
        
        # 5. Treinar modelo baseline (features originais)
        log_step(logger, "🌳 Treinando modelo BASELINE (features originais)")
        baseline_model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        baseline_model.fit(X_orig_train_scaled, y_train)
        
        # 6. Treinar modelo avançado (todas as features)
        log_step(logger, "🚀 Treinando modelo AVANÇADO (com feature engineering)")
        advanced_model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        advanced_model.fit(X_adv_train_scaled, y_train)
        
        # 7. Fazer predições
        log_step(logger, "🎯 Fazendo predições com ambos os modelos")
        
        # Baseline
        baseline_train_pred = baseline_model.predict(X_orig_train_scaled)
        baseline_val_pred = baseline_model.predict(X_orig_val_scaled)
        baseline_test_pred = baseline_model.predict(X_orig_test_scaled)
        
        # Avançado
        advanced_train_pred = advanced_model.predict(X_adv_train_scaled)
        advanced_val_pred = advanced_model.predict(X_adv_val_scaled)
        advanced_test_pred = advanced_model.predict(X_adv_test_scaled)
        
        # 8. Calcular métricas
        log_step(logger, "📊 Calculando métricas de ambos os modelos")
        
        def calculate_metrics(y_true, y_pred):
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
        
        baseline_metrics = {
            'train': calculate_metrics(y_train, baseline_train_pred),
            'val': calculate_metrics(y_val, baseline_val_pred),
            'test': calculate_metrics(y_test, baseline_test_pred)
        }
        
        advanced_metrics = {
            'train': calculate_metrics(y_train, advanced_train_pred),
            'val': calculate_metrics(y_val, advanced_val_pred),
            'test': calculate_metrics(y_test, advanced_test_pred)
        }
        
        # 9. Exibir comparação de resultados
        logger.info("=" * 80)
        logger.info("📊 COMPARAÇÃO DE RESULTADOS:")
        logger.info("=" * 80)
        
        for dataset in ['train', 'val', 'test']:
            baseline = baseline_metrics[dataset]
            advanced = advanced_metrics[dataset]
            
            logger.info(f"🔹 {dataset.upper()}:")
            logger.info(f"   Baseline  - MSE: {baseline['mse']:.3f}, MAE: {baseline['mae']:.3f}, R²: {baseline['r2']:.3f}")
            logger.info(f"   Avançado  - MSE: {advanced['mse']:.3f}, MAE: {advanced['mae']:.3f}, R²: {advanced['r2']:.3f}")
            
            # Melhorias
            mse_improvement = ((baseline['mse'] - advanced['mse']) / baseline['mse']) * 100
            r2_improvement = ((advanced['r2'] - baseline['r2']) / baseline['r2']) * 100
            logger.info(f"   Melhoria  - MSE: {mse_improvement:+.1f}%, R²: {r2_improvement:+.1f}%")
            logger.info("")
        
        # 10. Criar pastas organizadas e salvar plots
        base_output_dir = ensure_output_dir("ml_analysis")
        feature_importance_dir = ensure_output_dir("ml_analysis/feature_importance")
        predictions_dir = ensure_output_dir("ml_analysis/predictions")
        metrics_dir = ensure_output_dir("ml_analysis/metrics")
        
        log_step(logger, f"📁 Salvando análises em: {base_output_dir}")
        
        # 11. Plot de importância das features
        log_step(logger, "📊 Gerando comparação de importância das features")
        plot_feature_importance_comparison(
            baseline_model.feature_importances_,
            advanced_model.feature_importances_,
            original_features,
            all_features,
            feature_importance_dir / f"{args.output_name}_feature_comparison.png"
        )
        
        # 12. Plot de predições vs reais
        log_step(logger, "🎯 Gerando comparação de predições")
        plot_predictions_vs_actual_comparison(
            y_test, baseline_test_pred, advanced_test_pred,
            predictions_dir / f"{args.output_name}_predictions_comparison.png"
        )
        
        # 13. Salvar métricas detalhadas
        log_step(logger, "💾 Salvando métricas detalhadas")
        
        metrics_comparison = []
        for dataset in ['train', 'val', 'test']:
            for model_type in ['baseline', 'advanced']:
                metrics = baseline_metrics[dataset] if model_type == 'baseline' else advanced_metrics[dataset]
                metrics_comparison.append({
                    'model': model_type,
                    'dataset': dataset,
                    'mse': metrics['mse'],
                    'mae': metrics['mae'],
                    'r2': metrics['r2'],
                    'n_features': len(original_features) if model_type == 'baseline' else len(all_features)
                })
        
        metrics_df = pd.DataFrame(metrics_comparison)
        metrics_df.to_csv(metrics_dir / f"{args.output_name}_comparison_metrics.csv", index=False)
        
        # 14. Análise de importância das features avançadas
        feature_importance_df = pd.DataFrame({
            'feature': all_features,
            'importance': advanced_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance_df.to_csv(metrics_dir / f"{args.output_name}_feature_importance.csv", index=False)
        
        log_step(logger, "🎉 ANÁLISE AVANÇADA COMPLETA!")
        
        # Resumo final
        logger.info("=" * 80)
        logger.info("📋 RESUMO FINAL:")
        test_improvement = ((baseline_metrics['test']['mse'] - advanced_metrics['test']['mse']) / baseline_metrics['test']['mse']) * 100
        logger.info(f"🎯 Melhoria no MSE (teste): {test_improvement:+.1f}%")
        logger.info(f"📊 R² baseline (teste): {baseline_metrics['test']['r2']:.3f}")
        logger.info(f"🚀 R² avançado (teste): {advanced_metrics['test']['r2']:.3f}")
        logger.info(f"🔬 Features mais importantes:")
        
        for i, row in feature_importance_df.head(5).iterrows():
            logger.info(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")
        
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Erro durante análise avançada: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
