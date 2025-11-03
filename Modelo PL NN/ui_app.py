"""
Interface Streamlit para modelo de propagação loss PINN
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

from utils import (
    load_training_data,
    load_evaluation_data,
    prepare_features_with_frequency,
    get_available_tx,
    frequency_to_mhz,
    load_buildings_mesh
)
from propagation_loss_model import PINNPropagationLoss

# Configuração da página
st.set_page_config(
    page_title="Propagation Loss Model",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título
st.title("📡 Modelo de Propagação Loss com PINN")
st.markdown("---")

# Inicializar sessão state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'val_data' not in st.session_state:
    st.session_state.val_data = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'mesh' not in st.session_state:
    st.session_state.mesh = None

# Sidebar - Controles
st.sidebar.header("⚙️ Configurações")

# Seleção de frequência
frequency = st.sidebar.radio(
    "Frequência",
    options=['800MHz', '7GHz', '28GHz'],
    index=0
)

# Carregar Tx disponíveis
try:
    available_tx = get_available_tx(frequency)
    if not available_tx:
        st.sidebar.error(f"Nenhum Tx encontrado para {frequency}")
        st.stop()
except Exception as e:
    st.sidebar.error(f"Erro ao carregar Tx: {e}")
    st.stop()

# Multi-select de Tx
st.sidebar.subheader("Seleção de Transmitters")
selected_tx = st.sidebar.multiselect(
    "Selecione os Transmitters para treinamento:",
    options=available_tx,
    default=available_tx[:5] if len(available_tx) >= 5 else available_tx
)

if not selected_tx:
    st.sidebar.warning("Selecione pelo menos um Transmitter")
    st.stop()

# Proporção de treinamento
train_proportion = st.sidebar.slider(
    "Proporção de dados para treino (por Tx):",
    min_value=0.1,
    max_value=1.0,
    value=0.8,
    step=0.1
)

st.sidebar.markdown("---")
st.sidebar.subheader("Parâmetros do Modelo PINN")

# Parâmetros do modelo
hidden_layers = st.sidebar.slider(
    "Número de camadas ocultas:",
    min_value=2,
    max_value=6,
    value=3,
    step=1
)

neurons_per_layer = st.sidebar.slider(
    "Neurônios por camada:",
    min_value=32,
    max_value=256,
    value=64,
    step=32
)

learning_rate = st.sidebar.select_slider(
    "Taxa de aprendizado:",
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    value=0.001
)

lambda_physics = st.sidebar.slider(
    "Lambda (peso físico):",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)

epochs = st.sidebar.slider(
    "Épocas de treinamento:",
    min_value=50,
    max_value=500,
    value=100,
    step=50
)

# Botão de treinamento
st.sidebar.markdown("---")
train_button = st.sidebar.button("🚀 Treinar Modelo", type="primary", use_container_width=True)

# Função para plotar heatmaps
def plot_heatmaps(pl_real, pl_pred, rx_x, rx_y, tx_num=None):
    """Cria heatmaps de PL real, predito e erro"""
    error = np.abs(pl_real - pl_pred)
    
    # Criar grid para interpolação
    x_min, x_max = rx_x.min(), rx_x.max()
    y_min, y_max = rx_y.min(), rx_y.max()
    
    grid_resolution = 100
    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolar valores
    z_real = griddata((rx_x, rx_y), pl_real, (xi_grid, yi_grid), method='cubic', fill_value=np.nan)
    z_pred = griddata((rx_x, rx_y), pl_pred, (xi_grid, yi_grid), method='cubic', fill_value=np.nan)
    z_error = griddata((rx_x, rx_y), error, (xi_grid, yi_grid), method='cubic', fill_value=np.nan)
    
    # Criar subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('PL Real', 'PL Predito', 'Erro Absoluto'),
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}]]
    )
    
    # PL Real
    fig.add_trace(
        go.Heatmap(
            x=xi, y=yi, z=z_real,
            colorscale='Viridis',
            colorbar=dict(title="PL (dB)", x=0.28, len=0.33),
            showscale=True
        ),
        row=1, col=1
    )
    
    # PL Predito
    fig.add_trace(
        go.Heatmap(
            x=xi, y=yi, z=z_pred,
            colorscale='Viridis',
            colorbar=dict(title="PL (dB)", x=0.63, len=0.33),
            showscale=True
        ),
        row=1, col=2
    )
    
    # Erro
    fig.add_trace(
        go.Heatmap(
            x=xi, y=yi, z=z_error,
            colorscale='Reds',
            colorbar=dict(title="Erro (dB)", x=1.0, len=0.33),
            showscale=True
        ),
        row=1, col=3
    )
    
    fig.update_xaxes(title_text="Rx UTM X (m)", row=1, col=1)
    fig.update_xaxes(title_text="Rx UTM X (m)", row=1, col=2)
    fig.update_xaxes(title_text="Rx UTM X (m)", row=1, col=3)
    
    fig.update_yaxes(title_text="Rx UTM Y (m)", row=1, col=1)
    fig.update_yaxes(title_text="Rx UTM Y (m)", row=1, col=2)
    fig.update_yaxes(title_text="Rx UTM Y (m)", row=1, col=3)
    
    title = f"Heatmaps de Propagação Loss"
    if tx_num:
        title += f" - Tx {tx_num}"
    fig.update_layout(title=title, height=400)
    
    return fig

# Função para plotar estatísticas
def plot_statistics(pl_real, pl_pred):
    """Cria gráficos estatísticos"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Histograma', 'Boxplot', 'Scatter Plot', 'QQ-Plot'),
        specs=[[{'type': 'histogram'}, {'type': 'box'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Histograma
    fig.add_trace(
        go.Histogram(x=pl_real, name='PL Real', opacity=0.7, nbinsx=50),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=pl_pred, name='PL Predito', opacity=0.7, nbinsx=50),
        row=1, col=1
    )
    
    # Boxplot
    fig.add_trace(
        go.Box(y=pl_real, name='PL Real', boxmean='sd'),
        row=1, col=2
    )
    fig.add_trace(
        go.Box(y=pl_pred, name='PL Predito', boxmean='sd'),
        row=1, col=2
    )
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=pl_real, y=pl_pred,
            mode='markers',
            marker=dict(size=3, opacity=0.5),
            name='Dados'
        ),
        row=2, col=1
    )
    
    # Linha ideal (y = x)
    min_val = min(pl_real.min(), pl_pred.min())
    max_val = max(pl_real.max(), pl_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Ideal'
        ),
        row=2, col=1
    )
    
    # QQ-plot
    from scipy import stats
    sorted_real = np.sort(pl_real)
    sorted_pred = np.sort(pl_pred)
    quantiles = np.linspace(0, 1, len(sorted_real))
    
    fig.add_trace(
        go.Scatter(
            x=sorted_real, y=sorted_pred,
            mode='markers',
            marker=dict(size=3, opacity=0.5),
            name='Quantis'
        ),
        row=2, col=2
    )
    
    # Linha ideal no QQ-plot
    min_q = min(sorted_real.min(), sorted_pred.min())
    max_q = max(sorted_real.max(), sorted_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[min_q, max_q],
            y=[min_q, max_q],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Ideal',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="PL (dB)", row=1, col=1)
    fig.update_xaxes(title_text="PL Real (dB)", row=2, col=1)
    fig.update_xaxes(title_text="PL Real (dB)", row=2, col=2)
    
    fig.update_yaxes(title_text="Frequência", row=1, col=1)
    fig.update_yaxes(title_text="PL (dB)", row=1, col=2)
    fig.update_yaxes(title_text="PL Predito (dB)", row=2, col=1)
    fig.update_yaxes(title_text="PL Predito (dB)", row=2, col=2)
    
    fig.update_layout(title="Análise Estatística", height=700)
    
    return fig

# Função para plotar feature importance
def plot_feature_importance(feature_importance):
    """Plota importância das features"""
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    # Ordenar por importância
    sorted_idx = np.argsort(importance)[::-1]
    features_sorted = [features[i] for i in sorted_idx]
    importance_sorted = [importance[i] for i in sorted_idx]
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance_sorted,
            y=features_sorted,
            orientation='h',
            marker=dict(color=importance_sorted, colorscale='Viridis')
        )
    ])
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importância Normalizada",
        yaxis_title="Features",
        height=400
    )
    
    return fig

# Função para plotar métricas de treinamento
def plot_training_history(history):
    """Plota histórico de treinamento"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss Total', 'Loss de Dados', 'Loss Física', 'Validação'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    epochs = range(len(history['train_loss']))
    
    # Loss Total
    fig.add_trace(
        go.Scatter(x=list(epochs), y=history['train_loss'], name='Train', mode='lines'),
        row=1, col=1
    )
    if history['val_loss']:
        fig.add_trace(
            go.Scatter(x=list(epochs), y=history['val_loss'], name='Val', mode='lines'),
            row=1, col=1
        )
    
    # Loss de Dados
    fig.add_trace(
        go.Scatter(x=list(epochs), y=history['train_loss_data'], name='Train Data', mode='lines'),
        row=1, col=2
    )
    if history['val_loss_data']:
        fig.add_trace(
            go.Scatter(x=list(epochs), y=history['val_loss_data'], name='Val Data', mode='lines'),
            row=1, col=2
        )
    
    # Loss Física
    fig.add_trace(
        go.Scatter(x=list(epochs), y=history['train_loss_physics'], name='Train Physics', mode='lines'),
        row=2, col=1
    )
    if history['val_loss_physics']:
        fig.add_trace(
            go.Scatter(x=list(epochs), y=history['val_loss_physics'], name='Val Physics', mode='lines'),
            row=2, col=1
        )
    
    # Comparação Train vs Val
    if history['val_loss']:
        fig.add_trace(
            go.Scatter(x=list(epochs), y=history['train_loss'], name='Train', mode='lines'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=list(epochs), y=history['val_loss'], name='Val', mode='lines'),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text="Época", row=1, col=1)
    fig.update_xaxes(title_text="Época", row=1, col=2)
    fig.update_xaxes(title_text="Época", row=2, col=1)
    fig.update_xaxes(title_text="Época", row=2, col=2)
    
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=2)
    
    fig.update_layout(title="Histórico de Treinamento", height=700)
    
    return fig

# Treinamento do modelo
if train_button:
    status_container = st.status("🔄 Iniciando treinamento do modelo...", expanded=True)
    
    try:
        with status_container:
            st.write("📂 **Etapa 1/6:** Carregando dados de treinamento...")
            st.write(f"- Frequência selecionada: {frequency}")
            st.write(f"- Transmitters selecionados: {selected_tx}")
            st.write(f"- Proporção de treinamento: {train_proportion*100:.1f}%")
            
            # Carregar dados
            train_df, val_df = load_training_data(frequency, selected_tx, train_proportion)
            
            st.write(f"✅ Dados carregados:")
            st.write(f"   - Treino: {len(train_df):,} amostras")
            if len(val_df) > 0:
                st.write(f"   - Validação: {len(val_df):,} amostras")
            else:
                st.write(f"   - Validação: Sem dados (proporção = 1.0)")
            
            st.write("")
            st.write("🧮 **Etapa 2/6:** Calculando features e FSPL...")
            
            # Calcular frequência em MHz
            freq_mhz = frequency_to_mhz(frequency)
            st.write(f"- Frequência convertida: {freq_mhz} MHz")
            
            st.write("")
            st.write("📊 **Etapa 2.5/6:** Carregando mapa 3D (OBJ)...")
            
            # Carregar mesh dos edifícios (usar cache se disponível)
            if 'mesh' in st.session_state and st.session_state.mesh is not None:
                mesh = st.session_state.mesh
                st.write(f"✅ Arquivo OBJ carregado (do cache)")
            else:
                mesh = load_buildings_mesh()
                st.session_state.mesh = mesh
                if mesh is not None:
                    st.write(f"✅ Arquivo OBJ carregado:")
                    st.write(f"   - Vértices: {len(mesh.vertices):,}")
                    st.write(f"   - Faces: {len(mesh.faces):,}")
                    st.write(f"   - Bounds: {mesh.bounds}")
                else:
                    st.write(f"⚠️ Arquivo OBJ não encontrado ou trimesh não disponível")
                    st.write(f"   - Usando apenas features básicas (sem geometria de edifícios)")
            
            st.write("")
            st.write("🧮 Continuando cálculo de features...")
            
            # Criar barra de progresso e área de status para features
            feature_progress_bar = st.progress(0)
            feature_status_text = st.empty()
            
            # Callback para atualizar progresso do cálculo de features
            def update_feature_progress(progress, message):
                feature_progress_bar.progress(progress)
                feature_status_text.text(f"📊 {message}")
            
            # Preparar features (incluindo features geométricas do OBJ)
            X_train, dist_train, fspl_train, ld_train, pl_train, scaler = prepare_features_with_frequency(
                train_df, freq_mhz, fit_scaler=True, mesh=mesh, progress_callback=update_feature_progress
            )
            
            feature_progress_bar.progress(1.0)
            feature_status_text.empty()
            
            st.write(f"✅ Features calculadas:")
            st.write(f"   - Dimensão das features: {X_train.shape[1]} (incluindo {4 if mesh else 0} features geométricas do OBJ)")
            st.write(f"   - Distância média: {np.mean(dist_train)/1000:.2f} km")
            st.write(f"   - FSPL médio: {np.mean(fspl_train):.2f} dB")
            st.write(f"   - Ld médio: {np.mean(ld_train):.2f} dB")
            st.write(f"   - PL médio: {np.mean(pl_train):.2f} dB")
            
            if mesh is not None:
                st.write(f"   - Features geométricas do OBJ: LOS/NLOS, interseções, altura de obstáculos")
            
            if len(val_df) > 0:
                st.write("")
                st.write("🧮 Preparando features de validação...")
                
                # Criar barra de progresso para validação
                val_progress_bar = st.progress(0)
                val_status_text = st.empty()
                
                def update_val_progress(progress, message):
                    val_progress_bar.progress(progress)
                    val_status_text.text(f"📊 {message}")
                
                X_val, dist_val, fspl_val, ld_val, pl_val, _ = prepare_features_with_frequency(
                    val_df, freq_mhz, scaler=scaler, fit_scaler=False, mesh=mesh, progress_callback=update_val_progress
                )
                
                val_progress_bar.progress(1.0)
                val_status_text.empty()
                st.write(f"   - Validação preparada: {len(X_val):,} amostras")
            else:
                X_val, dist_val, fspl_val, ld_val, pl_val = None, None, None, None, None
            
            st.write("")
            st.write("🏗️ **Etapa 3/6:** Criando arquitetura do modelo PINN...")
            
            # Nomes das features (incluindo features geométricas do OBJ)
            feature_names = [
                'distance', 'tx_x', 'tx_y', 'tx_z', 'rx_x', 'rx_y', 'rx_z',
                'height_diff', 'tx_altitude', 'rx_altitude', 'relative_x', 'relative_y',
                'frequency_normalized'
            ]
            
            # Adicionar features geométricas do OBJ
            if mesh is not None:
                feature_names.extend([
                    'is_los', 'intersection_count', 'max_obstacle_height', 'obstacle_distance'
                ])
            
            # Criar modelo
            model = PINNPropagationLoss(
                input_dim=X_train.shape[1],
                hidden_layers=hidden_layers,
                neurons_per_layer=neurons_per_layer,
                learning_rate=learning_rate,
                lambda_physics=lambda_physics
            )
            
            st.write(f"✅ Modelo criado:")
            st.write(f"   - Arquitetura: {hidden_layers} camadas ocultas")
            st.write(f"   - Neurônios por camada: {neurons_per_layer}")
            st.write(f"   - Taxa de aprendizado: {learning_rate}")
            st.write(f"   - Lambda físico: {lambda_physics}")
            st.write(f"   - Total de parâmetros: {sum(p.numel() for p in model.parameters()):,}")
            
            st.write("")
            st.write("🚀 **Etapa 5/6:** Treinando modelo PINN...")
            st.write(f"   - Épocas: {epochs}")
            st.write(f"   - Loss combina: L_data + {lambda_physics} * L_physics")
            st.write(f"   - O modelo prevê Ld (difração) e usa PL = FSPL + Ld")
            
            # Criar barra de progresso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Callback para atualizar progresso
            def update_progress(epoch, total_epochs, train_loss, val_loss=None):
                progress = (epoch + 1) / total_epochs
                progress_bar.progress(progress)
                
                if val_loss is not None:
                    status_text.text(f"Época {epoch+1}/{total_epochs} - Loss Treino: {train_loss:.4f} | Loss Val: {val_loss:.4f}")
                else:
                    status_text.text(f"Época {epoch+1}/{total_epochs} - Loss Treino: {train_loss:.4f}")
            
            # Modificar train_model para aceitar callback (vamos fazer uma versão simplificada)
            # Por enquanto, vamos apenas treinar e mostrar progresso básico
            
            # Treinar
            history = model.train_model(
                X_train, ld_train, fspl_train, pl_train,
                X_val, ld_val, fspl_val, pl_val,
                epochs=epochs,
                verbose=False
            )
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Treinamento concluído!")
            
            st.write("")
            st.write("💾 **Etapa 6/6:** Salvando resultados...")
            
            # Salvar no session state
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.train_data = {
                'X': X_train, 'pl': pl_train, 'fspl': fspl_train,
                'rx_x': train_df['Rx_UTM_X'].values,
                'rx_y': train_df['Rx_UTM_Y'].values
            }
            if X_val is not None:
                st.session_state.val_data = {
                    'X': X_val, 'pl': pl_val, 'fspl': fspl_val,
                    'rx_x': val_df['Rx_UTM_X'].values,
                    'rx_y': val_df['Rx_UTM_Y'].values
                }
            else:
                st.session_state.val_data = None
            st.session_state.history = history
            st.session_state.feature_names = feature_names
            
            st.write("✅ Resultados salvos!")
            
        status_container.update(label="✅ Modelo treinado com sucesso!", state="complete", expanded=False)
        st.success("🎉 Treinamento concluído! Veja as métricas e visualizações abaixo.")
        
    except Exception as e:
        status_container.update(label="❌ Erro ao treinar modelo", state="error", expanded=False)
        st.error(f"Erro ao treinar modelo: {e}")
        st.exception(e)

# Exibir resultados se modelo foi treinado
if st.session_state.model is not None:
    model = st.session_state.model
    scaler = st.session_state.scaler
    train_data = st.session_state.train_data
    val_data = st.session_state.val_data
    history = st.session_state.history
    feature_names = st.session_state.feature_names
    
    # Métricas principais
    st.header("📊 Métricas do Modelo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Métricas de treino
    train_metrics = model.evaluate(train_data['X'], train_data['pl'], train_data['fspl'])
    
    with col1:
        st.metric("RMSE (Treino)", f"{train_metrics['RMSE']:.2f} dB")
    with col2:
        st.metric("MAE (Treino)", f"{train_metrics['MAE']:.2f} dB")
    with col3:
        st.metric("R² (Treino)", f"{train_metrics['R²']:.4f}")
    with col4:
        st.metric("MAPE (Treino)", f"{train_metrics['MAPE']:.2f}%")
    
    if val_data is not None:
        val_metrics = model.evaluate(val_data['X'], val_data['pl'], val_data['fspl'])
        
        st.markdown("#### Validação")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE (Validação)", f"{val_metrics['RMSE']:.2f} dB")
        with col2:
            st.metric("MAE (Validação)", f"{val_metrics['MAE']:.2f} dB")
        with col3:
            st.metric("R² (Validação)", f"{val_metrics['R²']:.4f}")
        with col4:
            st.metric("MAPE (Validação)", f"{val_metrics['MAPE']:.2f}%")
    
    # Tabs para visualizações
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Heatmaps", "📈 Estatísticas", "🎯 Feature Importance",
        "📉 Métricas Detalhadas", "🔍 Avaliação"
    ])
    
    with tab1:
        st.subheader("Heatmaps de Propagação Loss")
        
        # Heatmaps de treino
        ld_train_pred, pl_train_pred = model.predict(train_data['X'], train_data['fspl'])
        
        st.markdown("#### Dados de Treino")
        fig_heatmap_train = plot_heatmaps(
            train_data['pl'], pl_train_pred,
            train_data['rx_x'], train_data['rx_y']
        )
        st.plotly_chart(fig_heatmap_train, use_container_width=True)
        
        if val_data is not None:
            ld_val_pred, pl_val_pred = model.predict(val_data['X'], val_data['fspl'])
            
            st.markdown("#### Dados de Validação")
            fig_heatmap_val = plot_heatmaps(
                val_data['pl'], pl_val_pred,
                val_data['rx_x'], val_data['rx_y']
            )
            st.plotly_chart(fig_heatmap_val, use_container_width=True)
    
    with tab2:
        st.subheader("Análise Estatística")
        
        ld_train_pred, pl_train_pred = model.predict(train_data['X'], train_data['fspl'])
        
        st.markdown("#### Dados de Treino")
        fig_stats_train = plot_statistics(train_data['pl'], pl_train_pred)
        st.plotly_chart(fig_stats_train, use_container_width=True)
        
        if val_data is not None:
            ld_val_pred, pl_val_pred = model.predict(val_data['X'], val_data['fspl'])
            
            st.markdown("#### Dados de Validação")
            fig_stats_val = plot_statistics(val_data['pl'], pl_val_pred)
            st.plotly_chart(fig_stats_val, use_container_width=True)
    
    with tab3:
        st.subheader("Feature Importance")
        
        # Análise de importância
        feature_importance = model.feature_importance_analysis(
            train_data['X'], train_data['fspl'], feature_names
        )
        
        fig_importance = plot_feature_importance(feature_importance)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with tab4:
        st.subheader("Métricas Detalhadas e Histórico de Treinamento")
        
        # Tabela de métricas
        metrics_df = pd.DataFrame({
            'Métrica': ['RMSE', 'MAE', 'R²', 'MAPE'],
            'Treino': [
                train_metrics['RMSE'],
                train_metrics['MAE'],
                train_metrics['R²'],
                train_metrics['MAPE']
            ]
        })
        
        if val_data is not None:
            metrics_df['Validação'] = [
                val_metrics['RMSE'],
                val_metrics['MAE'],
                val_metrics['R²'],
                val_metrics['MAPE']
            ]
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # Gráfico de histórico
        if history:
            fig_history = plot_training_history(history)
            st.plotly_chart(fig_history, use_container_width=True)
    
    with tab5:
        st.subheader("Avaliação com Dados de Evaluation")
        
        try:
            # Carregar dados de avaliação
            eval_df, eval_by_tx = load_evaluation_data(frequency)
            freq_mhz = frequency_to_mhz(frequency)
            
            # Carregar mesh se ainda não foi carregado (usar session state)
            if 'mesh' in st.session_state and st.session_state.mesh is not None:
                mesh = st.session_state.mesh
            else:
                mesh = load_buildings_mesh()
                st.session_state.mesh = mesh
            
            # Preparar features
            X_eval, dist_eval, fspl_eval, ld_eval, pl_eval, _ = prepare_features_with_frequency(
                eval_df, freq_mhz, scaler=scaler, fit_scaler=False, mesh=mesh
            )
            
            # Avaliar modelo
            eval_metrics = model.evaluate(X_eval, pl_eval, fspl_eval)
            
            st.markdown("#### Métricas Agregadas")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("RMSE", f"{eval_metrics['RMSE']:.2f} dB")
            with col2:
                st.metric("MAE", f"{eval_metrics['MAE']:.2f} dB")
            with col3:
                st.metric("R²", f"{eval_metrics['R²']:.4f}")
            with col4:
                st.metric("MAPE", f"{eval_metrics['MAPE']:.2f}%")
            
            # Métricas por Tx
            st.markdown("#### Métricas por Transmitter")
            tx_metrics = []
            
            for tx_num, tx_df in eval_by_tx.items():
                X_tx, _, fspl_tx, _, pl_tx, _ = prepare_features_with_frequency(
                    tx_df, freq_mhz, scaler=scaler, fit_scaler=False, mesh=mesh
                )
                metrics_tx = model.evaluate(X_tx, pl_tx, fspl_tx)
                metrics_tx['Tx'] = tx_num
                tx_metrics.append(metrics_tx)
            
            tx_metrics_df = pd.DataFrame(tx_metrics)
            tx_metrics_df = tx_metrics_df[['Tx', 'RMSE', 'MAE', 'R²', 'MAPE']]
            st.dataframe(tx_metrics_df, use_container_width=True)
            
            # Visualizações de avaliação
            ld_eval_pred, pl_eval_pred = model.predict(X_eval, fspl_eval)
            
            st.markdown("#### Heatmaps de Avaliação")
            fig_eval_heatmap = plot_heatmaps(
                pl_eval, pl_eval_pred,
                eval_df['Rx_UTM_X'].values, eval_df['Rx_UTM_Y'].values
            )
            st.plotly_chart(fig_eval_heatmap, use_container_width=True)
            
            st.markdown("#### Estatísticas de Avaliação")
            fig_eval_stats = plot_statistics(pl_eval, pl_eval_pred)
            st.plotly_chart(fig_eval_stats, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro ao avaliar dados: {e}")
            st.exception(e)

else:
    st.info("👈 Configure os parâmetros na sidebar e clique em 'Treinar Modelo' para começar")

