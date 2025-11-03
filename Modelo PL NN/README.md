# Modelo de Propagação Loss com PINN

Modelo Physics-Informed Neural Network (PINN) para prever perda de propagação de sinal radioelétrico, implementado em PyTorch com interface interativa Streamlit.

## Descrição

Este projeto implementa um modelo de machine learning que prevê a perda de propagação (Propagation Loss - PL) seguindo a fórmula simplificada do modelo Vale:

**PL = FSPL + Ld**

Onde:
- **FSPL**: Free Space Path Loss (calculado analiticamente)
- **Ld**: Perda por difração (prevista pelo modelo PINN)

O modelo PINN incorpora conhecimento físico através de uma função de loss que combina:
- Loss de dados: erro entre Ld real e predito
- Loss física: garante que PL = FSPL + Ld seja respeitado

## Estrutura do Projeto

```
.
├── requirements.txt              # Dependências do projeto
├── utils.py                     # Funções auxiliares (cálculo de distância, FSPL, carregamento de dados)
├── propagation_loss_model.py    # Classe PINNPropagationLoss em PyTorch
├── ui_app.py                    # Interface Streamlit
└── README.md                     # Este arquivo
```

## Requisitos

- Python 3.8 ou superior
- Windows (testado, mas deve funcionar em Linux/Mac também)
- PyTorch (compatível com Windows)

## Instalação

1. Clone ou baixe este repositório

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. **Configure o Dataset** (veja seção abaixo)

## Dataset

⚠️ **IMPORTANTE**: O dataset `Dataset_ITU-AIML_2025-KRI-program` não está incluído neste repositório devido ao seu tamanho.

### Como Obter o Dataset

1. Baixe o dataset do **ITU-AIML 2025 KRI Program** (Site-Specific Radio Propagation Challenge)
2. Extraia o dataset na raiz do projeto para que a estrutura fique assim:
   ```
   .
   ├── Dataset_ITU-AIML_2025-KRI-program/
   │   ├── (a)_3DMap_Data/
   │   ├── (b)_[Training]Propagation-loss/
   │   │   ├── 800MHz/
   │   │   ├── 7GHz/
   │   │   └── 28GHz/
   │   ├── (b)_[Evaluation]Propagation-loss/
   │   │   ├── 800MHz/
   │   │   ├── 7GHz/
   │   │   └── 28GHz/
   │   ├── LICENSE.txt
   │   └── ITU_AI_for_Good_Desafio_Resumo.txt
   ├── requirements.txt
   ├── utils.py
   ├── propagation_loss_model.py
   ├── ui_app.py
   └── README.md
   ```

### Estrutura Esperada do Dataset

O projeto espera encontrar os arquivos CSV nas seguintes pastas:
- **Treinamento**: `Dataset_ITU-AIML_2025-KRI-program/(b)_[Training]Propagation-loss/[FREQUENCY]/[FREQUENCY]_Tx_[NUMBER].csv`
- **Avaliação**: `Dataset_ITU-AIML_2025-KRI-program/(b)_[Evaluation]Propagation-loss/[FREQUENCY]/[FREQUENCY]_Tx_[NUMBER].csv`

Onde `[FREQUENCY]` pode ser: `800MHz`, `7GHz` ou `28GHz`

## Como Usar

1. Certifique-se de que o dataset está na pasta correta (veja seção Dataset acima)

2. Execute a aplicação Streamlit:
```bash
streamlit run ui_app.py
```

3. A interface será aberta no navegador automaticamente

## Funcionalidades da Interface

### Controles (Sidebar)

- **Seleção de Frequência**: Escolha entre 800MHz, 7GHz ou 28GHz
- **Seleção de Transmitters**: Multi-select para escolher quais Tx usar para treinamento
- **Proporção de Treinamento**: Slider para selecionar proporção de dados por Tx (0.1 a 1.0)
- **Parâmetros do Modelo**:
  - Número de camadas ocultas
  - Neurônios por camada
  - Taxa de aprendizado
  - Lambda (peso do termo físico)
  - Épocas de treinamento

### Visualizações

Após treinar o modelo, você terá acesso a:

1. **Heatmaps**: Visualização espacial de PL real, predito e erro
2. **Estatísticas**: Histogramas, boxplots, scatter plots e QQ-plots
3. **Feature Importance**: Análise de importância das features
4. **Métricas Detalhadas**: Tabelas e gráficos de evolução do treinamento
5. **Avaliação**: Métricas e visualizações usando dados da pasta Evaluation

## Métricas Calculadas

- **RMSE**: Root Mean Square Error (dB)
- **MAE**: Mean Absolute Error (dB)
- **R²**: Coeficiente de determinação
- **MAPE**: Mean Absolute Percentage Error (%)

## Modelo PINN

O modelo Physics-Informed Neural Network:

- **Arquitetura**: Rede neural feedforward com múltiplas camadas ocultas
- **Entrada**: Features normalizadas (distância, coordenadas, alturas, frequência)
- **Saída**: Ld (perda por difração)
- **Loss**: Combinação de loss de dados e loss física

### Fórmula de Loss

```
L_total = L_data + λ * L_physics

onde:
- L_data = MSE(Ld_real, Ld_pred)
- L_physics = MSE(PL_real, FSPL + Ld_pred)
- λ = lambda_physics (peso físico)
```

## Dados Suportados

O modelo suporta três frequências:
- 800 MHz
- 7 GHz
- 28 GHz

Cada frequência possui múltiplos transmitters (Tx) disponíveis para treinamento.

## Notas Técnicas

- O modelo usa PyTorch para implementação da rede neural
- Features são normalizadas usando StandardScaler do scikit-learn
- Visualizações são geradas com Plotly para interatividade
- Heatmaps usam interpolação cúbica para suavização

## Estrutura de Dados

Os arquivos CSV devem conter as seguintes colunas:
- `Tx_UTM_X`, `Tx_UTM_Y`, `Tx_UTM_Z`: Coordenadas do transmissor
- `Rx_UTM_X`, `Rx_UTM_Y`, `Rx_UTM_Z`: Coordenadas do receptor
- `PL`: Perda de propagação real (target)

## Desenvolvido para

- Dataset ITU-AIML 2025 KRI Program
- Desafio de Site-Specific Radio Propagation

## Notas sobre o Repositório

- O arquivo `.gitignore` está configurado para ignorar a pasta do dataset
- A pasta `__pycache__` e outros arquivos temporários também são ignorados
- Para contribuir, certifique-se de não commitar o dataset

## Licença

Verifique a licença do dataset em `Dataset_ITU-AIML_2025-KRI-program/LICENSE.txt` (após baixar o dataset)

