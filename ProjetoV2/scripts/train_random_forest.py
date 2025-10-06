#!/usr/bin/env python3
"""
Script para treinar modelo Random Forest com análise completa
- 50% treinamento, 25% validação, 25% teste
- Learning curve, Heatmap de predições, Comparação vs valores reais
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata

# Adiciona o diretório pai ao path para importar o módulo tcc_project
sys.path.insert(0, str(Path(__file__).parent.parent))

from tcc_project.utils import setup_logger, log_step, ensure_output_dir


def plot_learning_curve(estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10)):
    """Plota learning curve do modelo"""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, 
        train_sizes=train_sizes, scoring='neg_mean_squared_error'
    )
    
    train_scores_mean = -train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)
    val_scores_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training MSE")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation MSE")
    
    plt.xlabel("Training Set Size")
    plt.ylabel("Mean Squared Error")
    plt.title("Learning Curve - Random Forest")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()


def plot_predictions_vs_actual(y_true, y_pred, title="Predições vs Valores Reais"):
    """Plota predições vs valores reais"""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    
    # Linha diagonal perfeita
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predição Perfeita')
    
    # Métricas
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    plt.xlabel("Valores Reais (dBm)")
    plt.ylabel("Predições (dBm)")
    plt.title(f"{title}\nMSE: {mse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adicionar linha de tendência
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), "b--", alpha=0.8, label=f'Tendência (y={z[0]:.3f}x+{z[1]:.3f})')
    plt.legend()
    
    return plt.gcf()


def plot_rss_heatmap_predictions(positions, predictions, title="Heatmap of Predicted RSS"):
    """Cria heatmap das predições RSS"""
    plt.figure(figsize=(12, 8))
    
    # Preparar dados para interpolação
    x = positions[:, 0]
    y = positions[:, 1]
    z = predictions.flatten()
    
    # Criar grid para interpolação
    x_min, x_max = x.min() - 50, x.max() + 50
    y_min, y_max = y.min() - 50, y.max() + 50
    
    xi = np.linspace(x_min, x_max, 200)
    yi = np.linspace(y_min, y_max, 200)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolar valores
    zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic', fill_value=z.min())
    
    # Criar contour plot
    contour = plt.contourf(xi_grid, yi_grid, zi, levels=50, cmap='plasma', alpha=0.8)
    
    # Adicionar pontos dos receptores
    scatter = plt.scatter(x, y, c=z, cmap='plasma', s=15, alpha=0.7, 
                         edgecolors='black', linewidth=0.5, label='Receptores')
    
    plt.colorbar(contour, label='RSS Predito (dBm)')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()


def plot_feature_importance(model, feature_names):
    """Plota importância das features"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xlabel("Features")
    plt.ylabel("Importância")
    plt.title("Importância das Features - Random Forest")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()


def plot_residuals(y_true, y_pred):
    """Plota análise de resíduos"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Resíduos vs Predições
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predições (dBm)')
    ax1.set_ylabel('Resíduos (dBm)')
    ax1.set_title('Resíduos vs Predições')
    ax1.grid(True, alpha=0.3)
    
    # Histograma dos resíduos
    ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Resíduos (dBm)')
    ax2.set_ylabel('Frequência')
    ax2.set_title('Distribuição dos Resíduos')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Treinar Random Forest para predição de RSS")
    parser.add_argument("--data", type=str, default="outputs/data/dataset_simulacao_1000_1000_carros.csv",
                       help="Caminho para o dataset CSV")
    parser.add_argument("--n-estimators", type=int, default=100,
                       help="Número de árvores no Random Forest")
    parser.add_argument("--max-depth", type=int, default=None,
                       help="Profundidade máxima das árvores")
    parser.add_argument("--min-samples-split", type=int, default=2,
                       help="Número mínimo de amostras para dividir um nó")
    parser.add_argument("--min-samples-leaf", type=int, default=1,
                       help="Número mínimo de amostras em uma folha")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Semente para reprodutibilidade")
    parser.add_argument("--output-name", type=str, default="random_forest_analysis",
                       help="Nome base para os arquivos de saída")
    
    args = parser.parse_args()
    
    logger = setup_logger(__name__)
    
    try:
        log_step(logger, "🚀 INICIANDO TREINAMENTO RANDOM FOREST")
        
        # 1. Carregar dados
        log_step(logger, f"📁 Carregando dataset: {args.data}")
        df = pd.read_csv(args.data)
        
        # Preparar features e target
        X = df[['rx_x', 'rx_y', 'rx_z']].values
        y = df['rss_db'].values
        
        log_step(logger, f"📊 Dataset shape: {df.shape}")
        log_step(logger, f"📈 RSS range: {y.min():.2f} to {y.max():.2f} dBm")
        
        # 2. Dividir dados: 50% treino, 25% validação, 25% teste
        log_step(logger, "🔄 Dividindo dados (50% treino, 25% val, 25% teste)")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.25, random_state=args.random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.33, random_state=args.random_state  # 0.33 de 0.75 = 0.25 do total
        )
        
        log_step(logger, f"📊 Treino: {len(X_train)}, Validação: {len(X_val)}, Teste: {len(X_test)}")
        
        # 3. Normalizar features
        log_step(logger, "🔧 Normalizando features")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 4. Treinar modelo
        log_step(logger, f"🌳 Treinando Random Forest (n_estimators={args.n_estimators}, max_depth={args.max_depth}, min_samples_split={args.min_samples_split}, min_samples_leaf={args.min_samples_leaf})")
        rf_model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        log_step(logger, "✅ Modelo treinado com sucesso")
        
        # 5. Fazer predições
        log_step(logger, "🎯 Fazendo predições")
        y_train_pred = rf_model.predict(X_train_scaled)
        y_val_pred = rf_model.predict(X_val_scaled)
        y_test_pred = rf_model.predict(X_test_scaled)
        
        # 6. Calcular métricas
        log_step(logger, "📊 Calculando métricas")
        
        # Métricas de treino
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Métricas de validação
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # Métricas de teste
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Log das métricas
        logger.info("="*50)
        logger.info("📊 MÉTRICAS DO MODELO:")
        logger.info(f"🔹 TREINO   - MSE: {train_mse:.3f}, MAE: {train_mae:.3f}, R²: {train_r2:.3f}")
        logger.info(f"🔸 VALIDAÇÃO - MSE: {val_mse:.3f}, MAE: {val_mae:.3f}, R²: {val_r2:.3f}")
        logger.info(f"🔹 TESTE    - MSE: {test_mse:.3f}, MAE: {test_mae:.3f}, R²: {test_r2:.3f}")
        logger.info("="*50)
        
        # 7. Criar pastas organizadas para plots
        base_output_dir = ensure_output_dir("ml_analysis")
        
        # Criar subdiretórios organizados
        learning_curves_dir = ensure_output_dir("ml_analysis/learning_curves")
        predictions_dir = ensure_output_dir("ml_analysis/predictions")
        heatmaps_dir = ensure_output_dir("ml_analysis/heatmaps")
        feature_importance_dir = ensure_output_dir("ml_analysis/feature_importance")
        residuals_dir = ensure_output_dir("ml_analysis/residuals")
        loss_comparison_dir = ensure_output_dir("ml_analysis/loss_comparison")
        metrics_dir = ensure_output_dir("ml_analysis/metrics")
        
        log_step(logger, f"📁 Salvando plots organizados em subdiretórios de: {base_output_dir}")
        
        # 8. Gerar plots organizados
        log_step(logger, "📈 Gerando learning curve")
        fig_learning = plot_learning_curve(rf_model, X_train_scaled, y_train)
        fig_learning.savefig(learning_curves_dir / f"{args.output_name}_learning_curve.png", dpi=300, bbox_inches='tight')
        plt.close(fig_learning)
        
        log_step(logger, "🎯 Gerando gráficos de predições vs valores reais")
        fig_pred_train = plot_predictions_vs_actual(y_train, y_train_pred, "Treino: Predições vs Valores Reais")
        fig_pred_train.savefig(predictions_dir / f"{args.output_name}_predictions_train.png", dpi=300, bbox_inches='tight')
        plt.close(fig_pred_train)
        
        fig_pred_val = plot_predictions_vs_actual(y_val, y_val_pred, "Validação: Predições vs Valores Reais")
        fig_pred_val.savefig(predictions_dir / f"{args.output_name}_predictions_validation.png", dpi=300, bbox_inches='tight')
        plt.close(fig_pred_val)
        
        fig_pred_test = plot_predictions_vs_actual(y_test, y_test_pred, "Teste: Predições vs Valores Reais")
        fig_pred_test.savefig(predictions_dir / f"{args.output_name}_predictions_test.png", dpi=300, bbox_inches='tight')
        plt.close(fig_pred_test)
        
        log_step(logger, "🗺️ Gerando heatmap de predições")
        fig_heatmap = plot_rss_heatmap_predictions(X_test, y_test_pred, "Heatmap of Predicted RSS (Test Set)")
        fig_heatmap.savefig(heatmaps_dir / f"{args.output_name}_heatmap_predictions.png", dpi=300, bbox_inches='tight')
        plt.close(fig_heatmap)
        
        log_step(logger, "🌳 Gerando importância das features")
        feature_names = ['rx_x', 'rx_y', 'rx_z']
        fig_importance = plot_feature_importance(rf_model, feature_names)
        fig_importance.savefig(feature_importance_dir / f"{args.output_name}_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close(fig_importance)
        
        log_step(logger, "📊 Gerando análise de resíduos")
        fig_residuals = plot_residuals(y_test, y_test_pred)
        fig_residuals.savefig(residuals_dir / f"{args.output_name}_residuals_analysis.png", dpi=300, bbox_inches='tight')
        plt.close(fig_residuals)
        
        # 9. Gerar gráfico de loss (MSE) por conjunto
        log_step(logger, "📉 Gerando gráfico de loss")
        fig_loss, ax = plt.subplots(figsize=(10, 6))
        
        datasets = ['Treino', 'Validação', 'Teste']
        mse_values = [train_mse, val_mse, test_mse]
        colors = ['blue', 'orange', 'green']
        
        bars = ax.bar(datasets, mse_values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Mean Squared Error')
        ax.set_title('Loss (MSE) por Conjunto de Dados')
        ax.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, mse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        fig_loss.savefig(loss_comparison_dir / f"{args.output_name}_loss_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig_loss)
        
        # 10. Salvar métricas em arquivo
        log_step(logger, "💾 Salvando métricas")
        metrics_dict = {
            'train_mse': train_mse, 'train_mae': train_mae, 'train_r2': train_r2,
            'val_mse': val_mse, 'val_mae': val_mae, 'val_r2': val_r2,
            'test_mse': test_mse, 'test_mae': test_mae, 'test_r2': test_r2,
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_samples_split': args.min_samples_split,
            'min_samples_leaf': args.min_samples_leaf,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }
        
        metrics_df = pd.DataFrame([metrics_dict])
        metrics_df.to_csv(metrics_dir / f"{args.output_name}_metrics.csv", index=False)
        
        log_step(logger, "🎉 ANÁLISE COMPLETA FINALIZADA!")
        
        # Resumo final
        logger.info("="*60)
        logger.info("📋 RESUMO FINAL:")
        logger.info(f"📁 Plots salvos em: {base_output_dir}")
        logger.info(f"🎯 Melhor performance: {'Treino' if train_r2 > max(val_r2, test_r2) else 'Validação' if val_r2 > test_r2 else 'Teste'}")
        logger.info(f"📊 R² no teste: {test_r2:.3f}")
        logger.info(f"📉 MSE no teste: {test_mse:.3f}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Erro durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
