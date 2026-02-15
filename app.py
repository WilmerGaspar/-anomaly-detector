"""
Anomaly Detector - Versión Web con Streamlit
Aplicación web ligera para análisis de anomalías.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import sys
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from report_generator import generate_report

# Configurar página
st.set_page_config(
    page_title="Anomaly Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar apariencia
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .score-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        color: white;
        margin: 1rem 0;
    }
    .score-low { background-color: #28a745; }
    .score-medium { background-color: #ffc107; color: #333; }
    .score-high { background-color: #dc3545; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .plugin-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🔬 Anomaly Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistema de Análisis de Anomalías con Deep Learning</p>', unsafe_allow_html=True)

# Sidebar - Configuración
st.sidebar.title("⚙️ Configuración")

# Modo de análisis
mode = st.sidebar.selectbox(
    "Modo de Análisis",
    ["balanced", "explorer", "conservative"],
    help="• Explorer: Máxima sensibilidad\n• Balanced: Equilibrio recomendado\n• Conservative: Solo anomalías claras"
)

# Plugins activos
st.sidebar.markdown("### 🔌 Plugins Activos")
plugins = {
    'fractal_base': st.sidebar.checkbox('Fractal (D0/D1/D2)', value=True),
    'kolmogorov_1941': st.sidebar.checkbox('Turbulencia (Kolmogorov)', value=True),
    'lyapunov_stability': st.sidebar.checkbox('Caos (Lyapunov)', value=True),
    'persistent_homology': st.sidebar.checkbox('Topología', value=True),
    'renormalization_group': st.sidebar.checkbox('Criticalidad', value=True)
    ,
    'anisotropy': st.sidebar.checkbox('Anisotropía', value=True),
    'entropy': st.sidebar.checkbox('Entropía', value=True)
}

active_plugins = [k for k, v in plugins.items() if v]

# Opciones de salida
st.sidebar.markdown("### 📄 Opciones")
generate_pdf = st.sidebar.checkbox('Generar PDF', value=True)
show_charts = st.sidebar.checkbox('Mostrar gráficos', value=True)

# Información en sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Información")
st.sidebar.info("""
**Formatos soportados:**
• JPG, PNG, TIFF
• FITS, FIT (Astronomía)

**Versión:** 1.0.0 Web
""")

# Carga de archivo
st.markdown("### 📁 Cargar Imagen")
uploaded_file = st.file_uploader(
    "Arrastra o selecciona una imagen",
    type=['jpg', 'jpeg', 'png', 'tiff', 'tif', 'fits', 'fit'],
    help="Soporta imágenes estándar y archivos FITS astronómicos"
)

# Funciones de utilidad
@st.cache_data
def load_and_preprocess_image(uploaded_file):
    """Carga y preprocesa la imagen."""
    try:
        # Verificar si es FITS
        if uploaded_file.name.lower().endswith(('.fits', '.fit')):
            from astropy.io import fits
            
            # Leer FITS
            with fits.open(uploaded_file) as hdul:
                # Buscar extensión SCI o primera con datos
                data = None
                for hdu in hdul:
                    if hasattr(hdu, 'data') and hdu.data is not None:
                        if hasattr(hdu, 'name') and hdu.name == 'SCI':
                            data = hdu.data
                            break
                
                if data is None:
                    for hdu in hdul:
                        if hasattr(hdu, 'data') and hdu.data is not None:
                            data = hdu.data
                            break
                
                if data is None:
                    return None, None, "No se encontraron datos en el archivo FITS"
                
                # Normalizar
                data = np.array(data, dtype=np.float32)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                
                p2, p98 = np.percentile(data, [2, 98])
                if p98 > p2:
                    data = (data - p2) / (p98 - p2)
                data = np.clip(data, 0.0, 1.0)
                
                # Metadatos
                header = hdul[0].header
                metadata = {
                    'filename': uploaded_file.name,
                    'format': 'FITS',
                    'width': data.shape[1],
                    'height': data.shape[0],
                    'is_fits': True
                }
                
                if 'DATE-OBS' in header:
                    metadata['date_obs'] = str(header['DATE-OBS'])
                if 'EXPTIME' in header:
                    metadata['exposure_time'] = float(header['EXPTIME'])
                if 'FILTER' in header:
                    metadata['filter'] = str(header['FILTER'])
                
                return data, metadata, None
        
        else:
            # Imagen estándar
            image = Image.open(uploaded_file)
            
            metadata = {
                'filename': uploaded_file.name,
                'format': image.format,
                'width': image.width,
                'height': image.height,
                'is_fits': False
            }
            
            # Convertir a escala de grises
            if image.mode != 'L':
                image = image.convert('L')
            
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            return img_array, metadata, None
            
    except Exception as e:
        return None, None, str(e)

def create_score_gauge(score):
    """Crea gráfico de medidor circular."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Score de Anomalía", 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#90EE90'},
                {'range': [40, 70], 'color': '#FFD700'},
                {'range': [70, 100], 'color': '#FF6B6B'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def run_analysis(image, active_plugins, mode):
    """Ejecuta el análisis completo."""
    results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. Fractal
    if 'fractal_base' in active_plugins:
        status_text.text("🌀 Calculando dimensión fractal...")
        from plugins.fractal_base import FractalBase
        plugin = FractalBase()
        results['fractal_base'] = plugin.analyze(image)
        progress_bar.progress(20)
    
    # 2. Kolmogorov
    if 'kolmogorov_1941' in active_plugins:
        status_text.text("🌊 Analizando espectro de turbulencia...")
        from plugins.kolmogorov_1941 import Kolmogorov1941
        plugin = Kolmogorov1941()
        results['kolmogorov_1941'] = plugin.analyze(image)
        progress_bar.progress(40)
    
    # 3. Lyapunov
    if 'lyapunov_stability' in active_plugins:
        status_text.text("🔄 Calculando exponentes de Lyapunov...")
        from plugins.lyapunov_stability import LyapunovStability
        plugin = LyapunovStability()
        results['lyapunov_stability'] = plugin.analyze(image)
        progress_bar.progress(60)
    
    # 4. Homología
    if 'persistent_homology' in active_plugins:
        status_text.text("🔄 Analizando topología...")
        from plugins.persistent_homology import PersistentHomology
        plugin = PersistentHomology()
        results['persistent_homology'] = plugin.analyze(image)
        progress_bar.progress(80)
    
    # 5. Renormalization
    if 'renormalization_group' in active_plugins:
        status_text.text("⚛️ Analizando criticalidad...")
        from plugins.renormalization_group import RenormalizationGroup
        plugin = RenormalizationGroup()
        results['renormalization_group'] = plugin.analyze(image)
        progress_bar.progress(80)

    # 6. Anisotropía
    if 'anisotropy' in active_plugins:
        status_text.text("📐 Calculando anisotropía...")
        from plugins.anisotropy import calculate_anisotropy
        results['anisotropy'] = calculate_anisotropy(image)
        progress_bar.progress(90)

    # 7. Entropía
    if 'entropy' in active_plugins:
        status_text.text("🔎 Calculando entropía e información...")
        from plugins.entropy import calculate_entropy
        results['entropy'] = calculate_entropy(image)
        progress_bar.progress(100)
    
    status_text.empty()
    progress_bar.empty()
    
    return results

def compute_global_score(plugin_results, mode):
    """Calcula score global."""
    scores = []
    
    if 'fractal_base' in plugin_results:
        fractal = plugin_results['fractal_base']
        scores.append(fractal.get('complexity_score', 0.5))
    
    if 'kolmogorov_1941' in plugin_results:
        turb = plugin_results['kolmogorov_1941']
        scores.append(turb.get('intermittency_factor', 0) * 0.7 + turb.get('turbulence_intensity', 0.5) * 0.3)
    
    if 'lyapunov_stability' in plugin_results:
        chaos = plugin_results['lyapunov_stability']
        scores.append(chaos.get('chaos_strength', 0))
    
    if 'persistent_homology' in plugin_results:
        topo = plugin_results['persistent_homology']
        scores.append(topo.get('complexity_score', 0.5))
    
    if 'renormalization_group' in plugin_results:
        rg = plugin_results['renormalization_group']
        scores.append(rg.get('criticality_score', 0))
    
    if scores:
        score = np.mean(scores)
    else:
        score = 0.5
    
    # Ajustar por modo
    if mode == 'explorer':
        score = min(score * 1.2, 1.0)
    elif mode == 'conservative':
        score = score * 0.8
    
    return np.clip(score, 0.0, 1.0)

def monte_carlo_validation(score, n_simulations=10000):
    """Validación Monte Carlo.

    Devuelve p-value, marcadores de significancia, y estadísticas de la
    distribución nula (media, std, error estándar y CI 95%).
    """
    # No fijar la semilla globalmente aquí para evitar resultados deterministas
    null_dist = np.random.beta(2, 5, n_simulations)
    p_value = np.mean(null_dist >= score)

    null_mean = float(np.mean(null_dist))
    null_std = float(np.std(null_dist, ddof=1))
    null_se = float(null_std / np.sqrt(max(1, n_simulations)))
    ci_lower = float(null_mean - 1.96 * null_se)
    ci_upper = float(null_mean + 1.96 * null_se)

    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "ns"

    return {
        'p_value': p_value,
        'stars': stars,
        'is_significant': p_value < 0.05,
        'null_mean': null_mean,
        'null_std': null_std,
        'null_se': null_se,
        'null_ci95': [ci_lower, ci_upper],
        'n_simulations': n_simulations
    }

# Procesamiento principal
if uploaded_file is not None:
    # Cargar imagen
    with st.spinner('Cargando imagen...'):
        image, metadata, error = load_and_preprocess_image(uploaded_file)
    
    if error:
        st.error(f"❌ Error cargando imagen: {error}")
    else:
        # Mostrar información
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Archivo", metadata['filename'])
        with col2:
            st.metric("Dimensiones", f"{metadata['width']} x {metadata['height']}")
        with col3:
            fmt = "FITS" if metadata['is_fits'] else metadata['format']
            st.metric("Formato", fmt)
        
        # Mostrar imagen
        st.markdown("### 🖼️ Vista Previa")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)
        
        # Botón de análisis
        if st.button('🚀 INICIAR ANÁLISIS', type='primary', use_container_width=True):
            
            # Ejecutar análisis
            with st.spinner('Analizando... Esto puede tomar unos minutos'):
                plugin_results = run_analysis(image, active_plugins, mode)
            
            # Calcular score global
            global_score = compute_global_score(plugin_results, mode)
            mc_results = monte_carlo_validation(global_score)
            
            # Mostrar resultados principales
            st.markdown("---")
            st.markdown("## 📊 RESULTADOS DEL ANÁLISIS")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Score visual
                st.plotly_chart(create_score_gauge(global_score), use_container_width=True)
            
            with col2:
                # Detalles
                st.markdown("### 📈 Métricas Principales")
                
                score_pct = global_score * 100
                if score_pct >= 70:
                    level = "🔴 ALTA"
                    color_class = "score-high"
                elif score_pct >= 40:
                    level = "🟡 MEDIA"
                    color_class = "score-medium"
                else:
                    level = "🟢 BAJA"
                    color_class = "score-low"
                
                st.markdown(f'''
                <div class="score-box {color_class}">
                    {score_pct:.1f}%<br>
                    <span style="font-size: 1.2rem;">{level}</span>
                </div>
                ''', unsafe_allow_html=True)
                
                # Significancia
                st.markdown(f"**Significancia estadística:** p = {mc_results['p_value']:.4f} {mc_results['stars']}")
                
                if mc_results['is_significant']:
                    st.success("✅ Anomalía estadísticamente significativa")
                else:
                    st.info("ℹ️ Dentro de rangos normales")
                
                # Generar y mostrar informe automático (Resumen ejecutivo, hallazgos, advertencias, conclusión)
                try:
                    results_for_report = {
                        'timestamp': datetime.now().isoformat(),
                        'filename': metadata.get('filename'),
                        'mode': mode,
                        'global_score': global_score,
                        'monte_carlo': mc_results,
                        'plugin_results': plugin_results
                    }
                    report_text = generate_report(results_for_report)
                    with st.expander('📝 Informe automático (Resumen)'):
                        st.markdown(report_text)
                except Exception as e:
                    st.warning(f"No se pudo generar el informe automático: {e}")
            
            # Resultados por plugin
            st.markdown("---")
            st.markdown("## 🔬 Resultados por Plugin")
            
            tabs = st.tabs([name.replace('_', ' ').title() for name in active_plugins])
            
            for i, plugin_name in enumerate(active_plugins):
                with tabs[i]:
                    if plugin_name in plugin_results:
                        result = plugin_results[plugin_name]
                        
                        # Crear columnas para métricas
                        metric_cols = st.columns(3)
                        
                        # Mostrar métricas clave según el plugin
                        if plugin_name == 'fractal_base':
                            with metric_cols[0]:
                                st.metric("Dimensión D0", f"{result.get('d0', 0):.3f}")
                            with metric_cols[1]:
                                st.metric("Dimensión D1", f"{result.get('d1', 0):.3f}")
                            with metric_cols[2]:
                                st.metric("Dimensión D2", f"{result.get('d2', 0):.3f}")
                            
                            st.metric("Multifractalidad", f"{result.get('multifractality_index', 0):.3f}")
                            st.metric("Complejidad", f"{result.get('complexity_score', 0):.3f}")
                            
                            # Gráfico log-log si hay datos
                            if 'log_scales' in result and 'log_counts' in result:
                                fig = px.scatter(
                                    x=result['log_scales'], 
                                    y=result['log_counts'],
                                    title="Dimensión Fractal (Log-Log)",
                                    labels={'x': 'log(ε)', 'y': 'log(N)'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        elif plugin_name == 'kolmogorov_1941':
                            with metric_cols[0]:
                                st.metric("Exponente β", f"{result.get('beta', 0):.3f}")
                            with metric_cols[1]:
                                st.metric("Intermittencia", f"{result.get('intermittency_factor', 0):.3f}")
                            with metric_cols[2]:
                                st.metric("Intensidad", f"{result.get('turbulence_intensity', 0):.3f}")
                            
                            # Gráfico espectro
                            if 'k_values' in result and 'spectrum' in result:
                                fig = px.line(
                                    x=result['k_values'][:50], 
                                    y=result['spectrum'][:50],
                                    title="Espectro de Potencias",
                                    labels={'x': 'k (número de onda)', 'y': 'E(k)'}
                                )
                                fig.update_yaxes(type="log")
                                fig.update_xaxes(type="log")
                                st.plotly_chart(fig, use_container_width=True)
                        
                        elif plugin_name == 'lyapunov_stability':
                            with metric_cols[0]:
                                st.metric("Max Lyapunov", f"{result.get('max_lyapunov', 0):.3f}")
                            with metric_cols[1]:
                                st.metric("Caótico", "Sí" if result.get('is_chaotic', False) else "No")
                            with metric_cols[2]:
                                st.metric("Entropía KS", f"{result.get('ks_entropy', 0):.3f}")
                            
                            st.info(f"**Tipo:** {result.get('dynamics_type', 'Desconocido')}")
                        
                        elif plugin_name == 'persistent_homology':
                            with metric_cols[0]:
                                st.metric("Betti 0", result.get('betti_0', 0))
                            with metric_cols[1]:
                                st.metric("Betti 1", result.get('betti_1', 0))
                            with metric_cols[2]:
                                st.metric("Entropía Topológica", f"{result.get('topological_entropy', 0):.3f}")
                        
                        elif plugin_name == 'renormalization_group':
                            with metric_cols[0]:
                                st.metric("Longitud Corr.", f"{result.get('correlation_length', 0):.1f}")
                            with metric_cols[1]:
                                st.metric("Crítico", "Sí" if result.get('is_critical', False) else "No")
                            with metric_cols[2]:
                                st.metric("Score", f"{result.get('criticality_score', 0):.3f}")
                            
                            st.info(f"**Clase:** {result.get('universality_class', 'Desconocida')}")
            
            # Descargar resultados
            st.markdown("---")
            st.markdown("## 💾 Descargar Resultados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON
                import json
                results_json = json.dumps({
                    'timestamp': datetime.now().isoformat(),
                    'filename': metadata['filename'],
                    'mode': mode,
                    'global_score': global_score,
                    'monte_carlo': mc_results,
                    'plugin_results': plugin_results,
                    'report': locals().get('report_text', '')
                }, indent=2, default=str)
                
                st.download_button(
                    label="📥 Descargar JSON",
                    data=results_json,
                    file_name=f"results_{metadata['filename']}.json",
                    mime="application/json"
                )
            
            with col2:
                if generate_pdf:
                    st.info("📄 La generación de PDF se realiza en la versión de escritorio")
                    st.markdown("Para obtener PDFs profesionales, descarga la versión completa")

else:
    # Pantalla inicial
    st.markdown("---")
    st.markdown("### 👋 Bienvenido a Anomaly Detector Web")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🔬 Plugins Matemáticos
        - Dimensión fractal
        - Turbulencia
        - Caos y estabilidad
        - Topología
        - Criticalidad
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Análisis
        - Score de anomalía
        - Validación Monte Carlo
        - Gráficos interactivos
        - Resultados en tiempo real
        """)
    
    with col3:
        st.markdown("""
        #### 🚀 Formatos
        - JPG, PNG, TIFF
        - FITS (Hubble/JWST)
        - Metadatos astronómicos
        - Sin instalación
        """)
    
    st.info("💡 **Para comenzar:** Arrastra o selecciona una imagen arriba ↑")
    
    # Ejemplo
    st.markdown("---")
    st.markdown("### 📝 Ejemplo de Uso")
    st.code("""
1. Selecciona una imagen (JPG, PNG o FITS)
2. Elige el modo de análisis (Explorer/Balanced/Conservative)
3. Activa los plugins que necesites
4. Click en "Iniciar Análisis"
5. Revisa los resultados y descarga el JSON
    """)