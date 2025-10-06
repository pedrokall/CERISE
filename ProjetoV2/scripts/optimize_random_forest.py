#!/usr/bin/env python3
"""
Script para otimização sistemática do Random Forest
Testa diferentes combinações de parâmetros para encontrar o melhor modelo
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Adiciona o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tcc_project.utils import setup_logger, log_step, ensure_output_dir


def test_rf_parameters(X_train, X_val, X_test, y_train, y_val, y_test):
    """Testa diferentes combinações de parâmetros"""
    
    # Combinações de parâmetros para testar
    param_combinations = [
        # (n_estimators, max_depth, min_samples_split, min_samples_leaf)
        (50, 6, 2, 1),      # Regularização forte
        (50, 8, 2, 1),      # Regularização moderada
        (50, 10, 2, 1),     # Regularização leve
        (75, 6, 5, 2),      # Mais regularização em samples
        (75, 8, 5, 2),      # Equilibrado
        (75, 10, 5, 2),     # Menos regularização
        (100, 6, 10, 4),    # Muito regularizado
        (100, 8, 10, 4),    # Regularização moderada-alta
        (100, 10, 10, 4),   # Regularização moderada
        (100, 12, 2, 1),    # Árvores mais profundas, menos regularização
    ]
    
    results = []
    
    for i, (n_est, max_d, min_split, min_leaf) in enumerate(param_combinations):
        print(f"Testando combinação {i+1}/10: n_est={n_est}, max_depth={max_d}, min_split={min_split}, min_leaf={min_leaf}")
        
        # Treinar modelo
        rf = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        
        # Fazer predições
        y_train_pred = rf.predict(X_train)
        y_val_pred = rf.predict(X_val)
        y_test_pred = rf.predict(X_test)
        
        # Calcular métricas
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        # Calcular gap de overfitting
        overfitting_gap = train_r2 - test_r2
        
        results.append({
            'n_estimators': n_est,
            'max_depth': max_d,
            'min_samples_split': min_split,
            'min_samples_leaf': min_leaf,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'overfitting_gap': overfitting_gap,
            'avg_r2': (train_r2 + val_r2 + test_r2) / 3,
            'balance_score': test_r2 - (overfitting_gap * 0.5)  # Penaliza overfitting
        })
    
    return results


def plot_optimization_results(results, output_dir):
    """Plota resultados da otimização"""
    
    df = pd.DataFrame(results)
    
    # 1. R² por conjunto
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # R² comparison
    ax1 = axes[0, 0]
    x = range(len(results))
    ax1.plot(x, df['train_r2'], 'o-', label='Treino', color='blue')
    ax1.plot(x, df['val_r2'], 's-', label='Validação', color='orange')
    ax1.plot(x, df['test_r2'], '^-', label='Teste', color='green')
    ax1.set_xlabel('Configuração')
    ax1.set_ylabel('R²')
    ax1.set_title('R² por Configuração')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Overfitting gap
    ax2 = axes[0, 1]
    ax2.bar(x, df['overfitting_gap'], color='red', alpha=0.7)
    ax2.set_xlabel('Configuração')
    ax2.set_ylabel('Gap Overfitting (R² treino - R² teste)')
    ax2.set_title('Gap de Overfitting')
    ax2.grid(True, alpha=0.3)
    
    # Balance score
    ax3 = axes[1, 0]
    ax3.bar(x, df['balance_score'], color='purple', alpha=0.7)
    ax3.set_xlabel('Configuração')
    ax3.set_ylabel('Score Balanceado')
    ax3.set_title('Score Balanceado (R² teste - 0.5*gap)')
    ax3.grid(True, alpha=0.3)
    
    # Parâmetros
    ax4 = axes[1, 1]
    ax4.scatter(df['max_depth'], df['test_r2'], s=df['n_estimators'], 
               c=df['overfitting_gap'], cmap='RdYlBu_r', alpha=0.7)
    ax4.set_xlabel('Max Depth')
    ax4.set_ylabel('R² Teste')
    ax4.set_title('R² Teste vs Max Depth\n(tamanho=n_estimators, cor=gap)')
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Gap Overfitting')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'rf_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Otimização sistemática Random Forest")
    parser.add_argument("--data", type=str, default="outputs/data/dataset_simulacao_1000_1000_carros.csv",
                       help="Caminho para o dataset CSV")
    parser.add_argument("--output-name", type=str, default="rf_optimization",
                       help="Nome base para os arquivos de saída")
    
    args = parser.parse_args()
    
    logger = setup_logger(__name__)
    
    try:
        log_step(logger, "🔍 INICIANDO OTIMIZAÇÃO SISTEMÁTICA RANDOM FOREST")
        
        # 1. Carregar dados
        log_step(logger, f"📁 Carregando dataset: {args.data}")
        df = pd.read_csv(args.data)
        
        X = df[['rx_x', 'rx_y', 'rx_z']].values
        y = df['rss_db'].values
        
        # 2. Dividir dados (mesma divisão sempre)
        log_step(logger, "🔄 Dividindo dados (50% treino, 25% val, 25% teste)")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.33, random_state=42
        )
        
        # 3. Normalizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 4. Testar parâmetros
        log_step(logger, "🧪 Testando 10 combinações de parâmetros...")
        results = test_rf_parameters(X_train_scaled, X_val_scaled, X_test_scaled, 
                                   y_train, y_val, y_test)
        
        # 5. Encontrar melhor modelo
        df_results = pd.DataFrame(results)
        
        # Ordenar por score balanceado (melhor = maior)
        df_results_sorted = df_results.sort_values('balance_score', ascending=False)
        best_model = df_results_sorted.iloc[0]
        
        # 6. Salvar resultados
        output_dir = ensure_output_dir("ml_analysis")
        
        # Salvar tabela completa
        df_results_sorted.to_csv(output_dir / f"{args.output_name}_results.csv", index=False, float_format='%.4f')
        
        # Gerar plots
        plot_optimization_results(results, output_dir)
        
        # 7. Log dos resultados
        logger.info("="*60)
        logger.info("🏆 RESULTADOS DA OTIMIZAÇÃO:")
        logger.info("="*60)
        
        logger.info("📊 TOP 3 MODELOS (por score balanceado):")
        for i, (_, row) in enumerate(df_results_sorted.head(3).iterrows()):
            logger.info(f"#{i+1}: n_est={int(row['n_estimators'])}, max_depth={int(row['max_depth'])}, "
                       f"min_split={int(row['min_samples_split'])}, min_leaf={int(row['min_samples_leaf'])}")
            logger.info(f"    R² teste: {row['test_r2']:.3f}, Gap: {row['overfitting_gap']:.3f}, "
                       f"Score: {row['balance_score']:.3f}")
        
        logger.info("\n🎯 MELHOR MODELO:")
        logger.info(f"Parâmetros: n_estimators={int(best_model['n_estimators'])}, "
                   f"max_depth={int(best_model['max_depth'])}, "
                   f"min_samples_split={int(best_model['min_samples_split'])}, "
                   f"min_samples_leaf={int(best_model['min_samples_leaf'])}")
        logger.info(f"R² treino: {best_model['train_r2']:.3f}")
        logger.info(f"R² validação: {best_model['val_r2']:.3f}")
        logger.info(f"R² teste: {best_model['test_r2']:.3f}")
        logger.info(f"Gap overfitting: {best_model['overfitting_gap']:.3f}")
        logger.info(f"Score balanceado: {best_model['balance_score']:.3f}")
        
        # 8. Comando para treinar o melhor modelo
        logger.info("\n🚀 COMANDO PARA TREINAR O MELHOR MODELO:")
        cmd = (f"python scripts\\train_random_forest.py "
               f"--n-estimators {int(best_model['n_estimators'])} "
               f"--max-depth {int(best_model['max_depth'])} "
               f"--output-name rf_best_optimized")
        logger.info(cmd)
        
        logger.info("="*60)
        log_step(logger, "✅ OTIMIZAÇÃO CONCLUÍDA!")
        
    except Exception as e:
        logger.error(f"❌ Erro durante otimização: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
