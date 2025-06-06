import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.formula.api import ols
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Análisis de Ventas - Aires Acondicionados",

    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Ventas de Aires Acondicionados")

# Load data
df = pd.read_csv('forecasting_data.csv')
df['month'] = pd.to_datetime(df['month'], format='%m/%d/%Y')

# Sidebar for model selection and date range
st.sidebar.header("Configuración del Análisis")

# Date range slider
min_date = df['month'].min()
max_date = df['month'].max()
selected_date = st.sidebar.date_input(
    "Selecciona la fecha hasta la cual quieres analizar:",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

# Filter data based on selected date
df = df[df['month'] <= pd.to_datetime(selected_date)]

model_approach = st.sidebar.selectbox(
    "Selecciona el enfoque para tratar el outlier (Enero 2023):",
    ["Variables Dummy", "Reemplazo por Promedio"],
    help="Compara diferentes estrategias para manejar la falta de inventario"
)

# Key metrics in header
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Período", f"{df['month'].min().strftime('%Y-%m')} - {df['month'].max().strftime('%Y-%m')}")
with col2:
    st.metric("Total Observaciones", df.shape[0])
with col3:
    st.metric("Venta Promedio", f"{df['sales'].mean():.0f}")
with col4:
    st.metric("Venta Máxima", f"{df['sales'].max():.0f}")

# Business Question 1: Seasonality Analysis
st.markdown("---")
st.header("¿Cuáles son los meses de mayor y menor venta?")

df['month_name'] = df['month'].dt.month_name()
monthly_avg = df.groupby('month_name')['sales'].mean().sort_values(ascending=False)

col1, col2 = st.columns([2, 1])

with col1:
    # Monthly sales chart
    months_ordered = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
    monthly_ordered = df.groupby('month_name')['sales'].mean().reindex(months_ordered)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Month': monthly_ordered.index,
        'Sales': monthly_ordered.values
    })
    
    fig_monthly = px.bar(
        plot_df,
        x='Month',
        y='Sales',
        title="Ventas Promedio por Mes",
        labels={'x': 'Mes', 'y': 'Ventas Promedio'},
        color='Sales',
        color_continuous_scale='Blues'
    )
  
    st.plotly_chart(fig_monthly, use_container_width=True)

# with col2:
#     st.subheader("🏆 Ranking de Meses")
#     for i, (month, sales) in enumerate(monthly_avg.head(3).items()):
#         if i == 0:
#             st.success(f"🥇 **{month}**: {sales:.0f}")
#         elif i == 1:
#             st.info(f"🥈 **{month}**: {sales:.0f}")
#         else:
#             st.warning(f"🥉 **{month}**: {sales:.0f}")
        
#     st.subheader("📉 Menor Venta")
#     worst_month = monthly_avg.tail(1)
#     st.error(f"**{worst_month.index[0]}**: {worst_month.values[0]:.0f}")
        
#     # Key insight
#     best_month = monthly_avg.idxmax()
#     worst_month_name = monthly_avg.idxmin()
#     st.info(f"💡 **Insight**: {best_month} vende {(monthly_avg.max()/monthly_avg.min()):.1f}x más que {worst_month_name}")

# Business Question 2: Trend Analysis
st.markdown("---")
st.header("¿Cuál es la tendencia general de las ventas?")

# Decomposition
decomposed = seasonal_decompose(
    df.set_index('month')['sales'],
    model='additive',
    period=12
)

col1, col2 = st.columns([3, 1])

with col1:
    fig_trend = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Ventas Originales', 'Tendencia Extraída'],
        vertical_spacing=0.1
    )
    
    fig_trend.add_trace(
        go.Scatter(x=df['month'], y=df['sales'], mode='lines+markers', 
                  name='Ventas', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    fig_trend.add_trace(
        go.Scatter(x=decomposed.trend.index, y=decomposed.trend.values, 
                  mode='lines', name='Tendencia', line=dict(color='red', width=3)),
        row=2, col=1
    )
    
    fig_trend.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_trend, use_container_width=True)

# with col2:
#     st.subheader("📊 Análisis de Tendencia")
        
#     # Calculate trend slope
#     trend_values = decomposed.trend.dropna()
#     trend_slope = np.polyfit(range(len(trend_values)), trend_values, 1)[0]
        
#     if trend_slope > 0:
#         st.success(f"📈 **Tendencia Creciente**\n+{trend_slope:.1f} unidades/mes")
#     else:
#         st.error(f"📉 **Tendencia Decreciente**\n{trend_slope:.1f} unidades/mes")
        
#     # Growth percentage
#     first_year_avg = df[df['month'].dt.year == df['month'].dt.year.min()]['sales'].mean()
#     last_year_avg = df[df['month'].dt.year == df['month'].dt.year.max()]['sales'].mean()
#     growth_pct = ((last_year_avg - first_year_avg) / first_year_avg) * 100
        
#     st.metric("Crecimiento Total", f"{growth_pct:.1f}%")
        
#     st.info("💡 **Insight**: El negocio muestra una tendencia sostenida de crecimiento")

# Business Question 3: Impact Analysis
st.markdown("---")
st.header("¿Qué impacto tuvo la falta de inventario en enero 2023?")

# Calculate impact
eneros_historicos = df[(df['month'].dt.month == 1) & (df['month'].dt.year != 2023)]
prom_enero = eneros_historicos['sales'].mean()
venta_enero_2023 = df[df['month'] == '2023-01-01']['sales'].values[0]
diferencia_abs = prom_enero - venta_enero_2023
diferencia_pct = (diferencia_abs / prom_enero) * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Promedio Histórico Enero", f"{prom_enero:.0f}", help="Promedio de ventas en enero (excluye 2023)")
with col2:
    st.metric("Ventas Enero 2023", f"{venta_enero_2023:.0f}", f"-{diferencia_abs:.0f}", delta_color="inverse")
with col3:
    st.metric("Impacto Negativo", f"{diferencia_pct:.1f}%", help="Porcentaje de pérdida vs promedio histórico")

# Impact visualization
impact_data = pd.DataFrame({
    'Categoría': ['Promedio Histórico Enero', 'Enero 2023 (Real)', 'Pérdida por Falta Inventario'],
    'Ventas': [prom_enero, venta_enero_2023, diferencia_abs],
    'Color': ['Normal', 'Afectado', 'Pérdida']
})

fig_impact = px.bar(
    impact_data, 
    x='Categoría', 
    y='Ventas',
    color='Color',
    title="Impacto de la Falta de Inventario - Enero 2023",
    color_discrete_map={'Normal': '#2ecc71', 'Afectado': '#e74c3c', 'Pérdida': '#f39c12'}
)

st.plotly_chart(fig_impact, use_container_width=True)

# Model Selection and Training
st.markdown("---")
st.header("Construcción del Modelo Predictivo")

# Data preprocessing based on selected approach
if model_approach == "Variables Dummy":
    st.success("Variables Dummy")
    # st.write("✅ Mantiene el dato real y modela el impacto del evento")
    
    # Advanced approach with dummy variables
    falta_inventario = pd.to_datetime('2023-01-01')
    pos_falta_inventario = pd.to_datetime('2023-02-01')
    
    df_training = df[df['month'] <= '2024-12-01'].copy()
    df_training['falta_inventario'] = (df_training['month'] == falta_inventario).astype(int)
    df_training['post_falta_inventario'] = (df_training['month'] == pos_falta_inventario).astype(int)
    df_training['peak_season'] = ((df_training['month'].dt.month >= 9) & 
                                (df_training['month'].dt.month <= 12)).astype(int)
    
    formula = 'sales ~ trend + months + lag1 + falta_inventario + post_falta_inventario + peak_season'
    
else:
    st.info("Reemplazo por Promedio")
    # st.write("⚠️ Reemplaza el outlier, pero pierde información del evento")
    
    # Basic approach with outlier replacement
    falta_inventario = pd.to_datetime('2023-01-01')
    eneros = df[(df['month'].dt.month == 1)]
    promedio_enero = eneros['sales'].mean()
    
    df_outliers = df.copy()
    df_outliers.loc[df_outliers['month'] == falta_inventario, 'sales'] = promedio_enero
    
    df_training = df_outliers[df_outliers['month'] <= '2024-12-01'].copy()
    
    formula = 'sales ~ trend + months + lag1'

# Model training
df_model = df_training.copy()
df_model['trend'] = range(1, len(df_model) + 1)
df_model['months'] = df_model['month'].dt.month_name()
df_model['lag1'] = df_model['sales'].shift(1)
df_model = df_model.dropna()

model = ols(formula=formula, data=df_model).fit()
df_model['predicted'] = model.predict()

# Model performance
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("R² Score", f"{model.rsquared:.3f}")
with col2:
    st.metric("R² Ajustado", f"{model.rsquared_adj:.3f}")
with col3:
    st.metric("RMSE", f"{np.sqrt(np.mean((df_model['sales'] - df_model['predicted'])**2)):.1f}")
with col4:
    st.metric("MAE", f"{np.mean(np.abs(df_model['sales'] - df_model['predicted'])):.1f}")

# Model validation visualization
col1, col2 = st.columns(2)

with col1:
    fig_scatter = px.scatter(
        df_model, x='sales', y='predicted',
        title='Ventas Reales vs Predichas'
    )
    min_val = min(df_model['sales'].min(), df_model['predicted'].min())
    max_val = max(df_model['sales'].max(), df_model['predicted'].max())
    fig_scatter.add_shape(
        type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val,
        line=dict(color='#e74c3c', dash='dash')
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    fig_residuals = px.scatter(
        x=df_model['predicted'], y=model.resid,
        title='Análisis de Residuos'
    )
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="#e74c3c")
  
    st.plotly_chart(fig_residuals, use_container_width=True)

# Business Question 4: Forecasting
st.markdown("---")
st.header("¿Cuánto estimamos vender en los próximos 7 meses?")

# Future predictions
last_date = df_model['month'].max()
future_months = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=7, freq='MS')

future_df = pd.DataFrame({'month': future_months})
future_df['trend'] = range(df_model['trend'].max() + 1, df_model['trend'].max() + 8)
future_df['months'] = future_df['month'].dt.month_name()

if model_approach == "Variables Dummy":
    future_df['falta_inventario'] = 0
    future_df['post_falta_inventario'] = 0
    future_df['peak_season'] = ((future_df['month'].dt.month >= 9) & 
                               (future_df['month'].dt.month <= 12)).astype(int)

# Rolling predictions
last_lag = df_model.iloc[-1]['sales']
preds = []

for i, row in future_df.iterrows():
    if model_approach == "Variables Dummy":
        row_data = {
            'trend': row['trend'], 'months': row['months'], 'lag1': last_lag,
            'falta_inventario': row['falta_inventario'],
            'post_falta_inventario': row['post_falta_inventario'],
            'peak_season': row['peak_season']
        }
    else:
        row_data = {'trend': row['trend'], 'months': row['months'], 'lag1': last_lag}
    
    pred = model.predict(pd.DataFrame([row_data]))[0]
    preds.append(pred)
    last_lag = pred

future_df['predicted'] = preds

# Results summary
col1, col2 = st.columns([2, 1])

with col1:
    # Combined historical and future plot
    df_pred_hist = df_model[['month', 'sales', 'predicted']].copy()
    df_pred_hist['tipo'] = 'Histórico'
    future_df['sales'] = None
    future_df['tipo'] = 'Futuro'
    
    df_pred_all = pd.concat([
        df_pred_hist[['month', 'sales', 'predicted', 'tipo']], 
        future_df[['month', 'sales', 'predicted', 'tipo']]
    ], ignore_index=True)
    
    fig_forecast = go.Figure()
    
    # Historical sales
    historical_data = df_pred_all[df_pred_all['tipo'] == 'Histórico']
    fig_forecast.add_trace(go.Scatter(
        x=historical_data['month'], y=historical_data['sales'],
        mode='lines+markers', name='Ventas Históricas',
        line=dict(color='#3498db', width=2)
    ))
    
    # Historical predictions
    fig_forecast.add_trace(go.Scatter(
        x=historical_data['month'], y=historical_data['predicted'],
        mode='lines', name='Predicción Histórica',
        line=dict(color='#f39c12', width=1, dash='dot')
    ))
    
    # Future predictions
    future_data = df_pred_all[df_pred_all['tipo'] == 'Futuro']
    fig_forecast.add_trace(go.Scatter(
        x=future_data['month'], y=future_data['predicted'],
        mode='lines+markers', name='Predicción Futura',
        line=dict(color='#e74c3c', width=3), marker=dict(size=8)
    ))
    

    st.plotly_chart(fig_forecast, use_container_width=True)

with col2:
    st.subheader("Proyección Detallada")
    
    # Format predictions nicely
    forecast_display = future_df[['month', 'predicted']].copy()
    forecast_display['month'] = forecast_display['month'].dt.strftime('%Y-%m')
    forecast_display['predicted'] = forecast_display['predicted'].round(0).astype(int)
    forecast_display.columns = ['Mes', 'Ventas Proyectadas']
    
    st.dataframe(forecast_display, use_container_width=True)
    
    # Summary metrics
    total_forecast = future_df['predicted'].sum()
    avg_forecast = future_df['predicted'].mean()
    
    st.metric("Total 7 Meses", f"{total_forecast:.0f}")
    st.metric("Promedio Mensual", f"{avg_forecast:.0f}")
    
   

# Model diagnostics
with st.expander("🔍 Diagnósticos del Modelo (Validación Estadística)"):
    col1, col2, col3 = st.columns(3)
    
    residuals = model.resid
    
    # Statistical tests
    shapiro_stat, shapiro_p = shapiro(residuals)
    _, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
    ljung_result = acorr_ljungbox(residuals, lags=12, return_df=True)
    min_ljung_p = ljung_result['lb_pvalue'].min()
    
    with col1:
       
        st.metric(
            f"Normalidad",
            f"{shapiro_p:.3f}",
            help="Shapiro-Wilk test. p>0.05 = Normal"
        )
    
    with col2:
       
        st.metric(
            f"Homocedasticidad",
            f"{bp_p:.3f}",
            help="Breusch-Pagan test. p>0.05 = Homocedástico"
        )
    
    with col3:
     
        st.metric(
            f"Autocorrelación",
            f"{min_ljung_p:.3f}",
            help="Ljung-Box test. p>0.05 = No autocorrelación"
        )
