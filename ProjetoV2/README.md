# Projeto TCC - Simulação Avançada de Propagação com Sionna RT

Simulação avançada de propagação de ondas de rádio em ambientes urbanos usando **Sionna RT** (Ray Tracing) e **Machine Learning** com **features físicas reais** extraídas da simulação para predição de RSS (Received Signal Strength).

## 🎯 Funcionalidades Principais

- **🌐 Simulação Ray-Tracing**: Sionna RT com cenas 3D realistas e multipercurso
- **🔬 Feature Engineering**: Extração de 50+ features físicas (delays, ângulos, interações)
- **🤖 Machine Learning**: RandomForest otimizado com **R² = 0.957**
- **📊 Visualização Avançada**: Heatmaps interpolados, análise de performance completa
- **🚗 Simulação em Larga Escala**: Suporte para 1000+ receptores simultâneos

## 🏆 Resultados de Performance

### **📊 Modelo Final (RandomForest Otimizado)**

| Métrica      | Treino | Validação | Teste           |
| ------------- | ------ | ----------- | --------------- |
| **R²** | 0.990  | 0.928       | **0.957** |
| **MSE** | 0.187  | 1.096       | **0.783** |
| **MAE** | 0.152  | 0.417       | **0.323** |

### **🔍 Features Mais Importantes**

1. **`theta_rx_std` (64.1%)** - 📡 Dispersão angular no receptor
2. **`excess_path_length` (11.1%)** - 📏 Excess path length (NLOS)
3. **`num_reflections` (5.5%)** - 🏢 Número de reflexões

## 📋 Pré-requisitos

- **Python 3.10** (compatibilidade com Mitsuba/Dr.Jit)
- **16GB+ RAM** (para simulações com 1000+ carros)
- **CUDA** (opcional, para aceleração GPU)

## 🚀 Instalação Rápida

```bash
# 1. Criar ambiente virtual Python 3.10
python -3.10 -m venv .venv

# 2. Ativar ambiente
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Instalar projeto
pip install -e .
```

## 📁 Estrutura Organizada

```
ProjetoV2/
├── 🔧 tcc_project/                 # Módulo principal
│   ├── scene.py                   # Carregamento de cenas 3D
│   ├── geometry.py                # Posicionamento e orientação
│   ├── sim.py                     # Simulação Sionna RT
│   ├── visualization.py           # Heatmaps e plots avançados
│   └── utils.py                   # Logging e utilitários
├── 🚀 scripts/                    # Scripts executáveis
│   ├── generate_rich_dataset.py           # 🔬 Dataset com 50+ features
│   ├── train_random_forest_sionna_features.py  # 🤖 Treinamento ML completo
│   ├── generate_simulation_1000.py        # 🌐 Simulação 1000 carros
│   ├── optimize_random_forest.py          # ⚙️ Otimização hiperparâmetros
│   ├── render_scene.py                    # 🎨 Renderização 3D
│   ├── show_heatmap_mesh.py               # 🗺️ Heatmap de cobertura
│   └── show_positions_mesh.py             # 📍 Distribuição de carros
├── 📊 outputs/                    # Resultados organizados
│   ├── data/                      # 📈 Datasets CSV
│   ├── ml_analysis/               # 🤖 Análises ML por categoria
│   │   ├── learning_curves/       # 📈 Curvas de aprendizagem
│   │   ├── heatmaps/              # 🗺️ Heatmaps interpolados
│   │   ├── feature_importance/    # 🔍 Importância das features
│   │   ├── predictions/           # 🎯 Predições vs reais
│   │   ├── residuals/             # 📊 Análise de resíduos
│   │   └── metrics/               # 📋 Métricas detalhadas
│   ├── plots/                     # 🎨 Visualizações gerais
│   └── renders/                   # 🖼️ Renders 3D
├── 🏗️ meshes/                     # Geometria 3D (edifícios, estradas)
├── 🗺️ universitario.xml           # Cena principal Sionna RT
├── 📍 receivers_mesh.csv          # 2000 posições otimizadas
└── 📦 requirements.txt            # Dependências
```

## 🎮 Guia de Uso Completo

### 🔬 **1. Geração de Dataset Rico (⭐ RECOMENDADO)**

```bash
python scripts\generate_rich_dataset.py --num-cars 1000 --output-name dataset_final_1000
```

**🎯 Features Extraídas (50+ features):**

- **Geométricas**: distâncias 2D/3D, elevação, coordenadas polares
- **Propagação**: tau_min, tau_max, excess_path_length, RMS delay spread
- **Angulares**: theta/phi TX/RX, dispersão angular, ângulos dominantes
- **Interações**: num_reflections, num_diffractions, num_transmissions
- **Ambientais**: densidade local, vizinhança, indicador LOS
- **Derivadas**: multipath_factor, coordenadas normalizadas

### 🤖 **2. Treinamento de Modelo Otimizado**

```bash
python scripts\train_random_forest_sionna_features.py \
    --data outputs/data/dataset_final_1000_1000_features.csv \
    --n-estimators 75 --max-depth 10 --min-samples-split 5 --min-samples-leaf 2 \
    --output-name modelo_otimizado
```

**📊 Gera Automaticamente:**

- 📈 Learning curves com validação cruzada
- 🗺️ Heatmaps interpolados de predições
- 🔍 Feature importance (top 15)
- 🎯 Predições vs valores reais (treino/val/teste)
- 📊 Análise de resíduos e distribuição de erros
- 📉 Comparação de loss (MSE) por conjunto

### 🌐 **3. Simulação Completa**

```bash
python scripts\generate_simulation_1000.py \
    --scene universitario.xml --max-cars 1000 \
    --output-name simulacao_completa
```

**🚀 Inclui:**

- 🎨 Visualização 3D dos caminhos de propagação
- 🗺️ Heatmap interpolado de cobertura RSS
- 📈 Dataset CSV pronto para ML
- 📊 Estatísticas detalhadas da simulação

### ⚙️ **4. Otimização de Hiperparâmetros**

```bash
python scripts\optimize_random_forest.py \
    --data outputs/data/dataset_final_1000_1000_features.csv
```

**🎯 Encontra automaticamente:**

- Melhores valores de `n_estimators`, `max_depth`
- Parâmetros `min_samples_split`, `min_samples_leaf`
- Balance entre R² e overfitting gap

## 🎨 Scripts de Visualização

### **🖼️ Renderização 3D da Cena**

```bash
# Cena básica
python scripts\render_scene.py --scene universitario.xml --output-name cena_basica

# Com carros posicionados
python scripts\render_scene.py --scene universitario.xml --add-cars 100 --output-name cena_com_carros

# Câmera customizada
python scripts\render_scene.py --scene universitario.xml \
    --camera-x 500 --camera-y -200 --camera-z 400 \
    --look-at-x 0 --look-at-y 0 --look-at-z 0 \
    --output-name vista_personalizada
```

### **🗺️ Heatmap de Cobertura RSS**

```bash
# Heatmap básico
python scripts\show_heatmap_mesh.py --scene universitario.xml --max-cars 500

# Heatmap detalhado
python scripts\show_heatmap_mesh.py \
    --scene universitario.xml --max-cars 1000 \
    --mesh-csv receivers_mesh.csv \
    --output-name cobertura_completa
```

### **📍 Distribuição de Carros**

```bash
# Todas as posições
python scripts\show_positions_mesh.py --output-name distribuicao_completa

# Subset específico
python scripts\show_positions_mesh.py --max-cars 500 --output-name distribuicao_500
```

## 📈 Estrutura de Saída

### **📊 Datasets (`outputs/data/`)**

```
data/
├── dataset_limpo_1000_1000_features.csv    # 🔬 Dataset final (51 features)
├── dataset_limpo_1000_1000_metadata.csv    # 📋 Metadados e categorias
└── [outros datasets de desenvolvimento]
```

### **🤖 Análises ML (`outputs/ml_analysis/`)**

```
ml_analysis/
├── 📈 learning_curves/      # Curvas de aprendizagem
├── 🎯 predictions/          # Predições vs valores reais
├── 🗺️ heatmaps/            # Heatmaps interpolados de cobertura
├── 🔍 feature_importance/   # Importância das features
├── 📊 residuals/           # Análise de resíduos
├── 📉 loss_comparison/     # Comparação de loss/MSE
├── 📋 metrics/            # Métricas detalhadas (CSV)
└── 📚 documentation/      # Explicação dos plots
```

## ⚙️ Configuração da Simulação

### **📡 Parâmetros do Sistema**

- **TX Position**: `(21.18, -132.4, 18.76)` metros
- **TX Power**: 20 dBm
- **Frequency**: 2.4 GHz
- **RX Height**: +3m acima dos carros
- **Max Depth**: 5 reflexões/difrações

### **🎛️ Parâmetros do Modelo**

- **Algorithm**: RandomForest
- **N Estimators**: 75
- **Max Depth**: 10
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2

## 🚀 Workflows Recomendados

### **Workflow 1: Análise Completa (⭐ RECOMENDADO)**

```bash
# 1. Gerar dataset rico
python scripts\generate_rich_dataset.py --num-cars 1000 --output-name projeto_final

# 2. Treinar modelo otimizado
python scripts\train_random_forest_sionna_features.py \
    --data outputs/data/dataset_projeto_final_1000_features.csv \
    --n-estimators 75 --max-depth 10 --output-name modelo_final

# 3. Visualizar resultados
# Plots automáticos salvos em outputs/ml_analysis/
```

### **Workflow 2: Simulação + Visualização**

```bash
# 1. Simulação completa
python scripts\generate_simulation_1000.py --max-cars 1000

# 2. Heatmap individual
python scripts\show_heatmap_mesh.py --scene universitario.xml --max-cars 1000

# 3. Render da cena
python scripts\render_scene.py --scene universitario.xml --add-cars 100
```

### **Workflow 3: Otimização de Modelo**

```bash
# 1. Gerar dataset
python scripts\generate_rich_dataset.py --num-cars 1000

# 2. Encontrar melhores parâmetros
python scripts\optimize_random_forest.py --data outputs/data/dataset_*.csv

# 3. Treinar com parâmetros otimizados
python scripts\train_random_forest_sionna_features.py \
    --data outputs/data/dataset_*.csv \
    --n-estimators 75 --max-depth 10 --min-samples-split 5 --min-samples-leaf 2
```

## 🐛 Solução de Problemas

### **❌ "ModuleNotFoundError: No module named 'tcc_project'"**

```bash
# Solução definitiva
pip install -e .

# Alternativa (se VS Code)
# Ctrl+Shift+P → "Python: Select Interpreter" → .venv\Scripts\python.exe
```

### **⚠️ "LLVM API initialization failed"**

- **Não é erro crítico** - apenas desabilita otimizações Dr.Jit/LLVM
- Simulação funciona normalmente, apenas mais lenta
- **Para resolver**: instalar LLVM e configurar `DRJIT_LIBLLVM_PATH`

### **🐌 Simulação muito lenta**

```bash
# Reduzir número de carros para teste
python scripts\generate_rich_dataset.py --num-cars 100

# Reduzir max-depth (editar scripts: max_depth=3)
```

### **💾 Falta de memória RAM**

- **Recomendado**: 16GB+ para 1000 carros
- **Alternativa**: Processar em batches menores (200-500 carros)

## ⭐ Destaques do Projeto

### **🔬 Dataset Rico (51 features físicas)**

- **Sem data leakage** - features independentes do target
- **Fisicamente interpretáveis** - baseadas em física real de propagação
- **Alta performance** - R² = 0.957 sem overfitting

### **🤖 Modelo RandomForest Otimizado**

- **Hiperparâmetros otimizados** via grid search
- **Análise completa** - 6 tipos de plots automáticos
- **Heatmaps interpolados** - visualização suave e profissional

### **🌐 Simulação em Larga Escala**

- **1000+ carros** simultâneos
- **Multipercurso completo** - reflexões, difrações, transmissões
- **Saída organizada** - plots categorizados automaticamente

## 📚 Referências Técnicas

- **Sionna RT**: [nvidia.github.io/sionna](https://nvidia.github.io/sionna/)
- **Mitsuba 3**: [mitsuba-renderer.org](https://mitsuba-renderer.org/)
- **Dr.Jit**: [drjit.readthedocs.io](https://drjit.readthedocs.io/)
- **RandomForest**: [scikit-learn.org](https://scikit-learn.org/)

## 🤝 Contribuição

1. **Fork** do repositório
2. **Branch**: `git checkout -b feature/nova-feature`
3. **Commit**: `git commit -am 'Adiciona feature X'`
4. **Push**: `git push origin feature/nova-feature`
5. **Pull Request** com descrição detalhada

## 📄 Licença

MIT License - veja `LICENSE` para detalhes completos.
