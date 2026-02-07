import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import io
import pickle
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Demand Forecasting System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None

# Create models directory if it doesn't exist
MODELS_DIR = "saved_models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Helper functions for pickle
def save_model(model, product_id, metadata):
    """Save trained model and metadata to pickle file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{MODELS_DIR}/model_{product_id}_{timestamp}.pkl"
    
    model_data = {
        'model': model,
        'product_id': product_id,
        'metadata': metadata,
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    return filename

def load_model(filepath):
    """Load model from pickle file"""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def get_saved_models():
    """Get list of saved model files"""
    if not os.path.exists(MODELS_DIR):
        return []
    
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    return sorted(model_files, reverse=True)

# Header
st.markdown('<p class="main-header">📊 Retail Demand Forecasting System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Inventory Planning with Model Persistence</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/business-report.png", width=80)
    st.title("⚙️ Configuration")
    
    # Model Management Section
    st.markdown("---")
    st.markdown("### 💾 Model Management")
    
    mode = st.radio(
        "Select Mode",
        options=["Train New Model", "Load Saved Model"],
        help="Train a new model or load a previously saved one"
    )
    
    if mode == "Load Saved Model":
        saved_models = get_saved_models()
        
        if saved_models:
            selected_model = st.selectbox(
                "Select Saved Model",
                options=saved_models,
                help="Choose a previously trained model"
            )
            
            if st.button("📂 Load Model", type="primary"):
                with st.spinner("Loading model..."):
                    model_data = load_model(os.path.join(MODELS_DIR, selected_model))
                    st.session_state.forecast_data = model_data.get('metadata', {})
                    st.session_state.model_trained = True
                    st.session_state.loaded_model = model_data['model']
                    st.success(f"✅ Model loaded: {model_data['product_id']}")
                    st.info(f"📅 Saved: {model_data['saved_at']}")
        else:
            st.info("No saved models found. Train a new model first.")
    
    st.markdown("---")
    st.markdown("### 📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload Historical Sales Data (CSV)",
        type=['csv'],
        help="Upload a CSV file with columns: Date, Product ID, Units Sold"
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Model Settings")
    
    forecast_days = st.slider(
        "Forecast Horizon (Days)",
        min_value=7,
        max_value=90,
        value=30,
        step=7
    )
    
    train_split = st.slider(
        "Train/Test Split (%)",
        min_value=60,
        max_value=90,
        value=80,
        step=5
    )
    
    seasonality_mode = st.selectbox(
        "Seasonality Mode",
        options=['multiplicative', 'additive'],
        index=0
    )
    
    # Save model option
    save_trained_model = st.checkbox("💾 Save model after training", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 Display Options")
    
    show_components = st.checkbox("Show Trend Components", value=True)
    show_insights = st.checkbox("Show Business Insights", value=True)

# Main content
if uploaded_file is None and mode == "Train New Model":
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.info("👋 Welcome! Please upload your historical sales data to begin.")
        
        st.markdown("### 📋 Expected Data Format")
        st.markdown("""
        Your CSV file should contain:
        - **Date**: Sales transaction date
        - **Product ID**: Unique identifier
        - **Units Sold**: Quantity sold
        """)
        
        sample_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5),
            'Product ID': ['P001'] * 5,
            'Units Sold': [120, 135, 142, 128, 156]
        })
        
        st.dataframe(sample_data, use_container_width=True)

elif mode == "Train New Model" and uploaded_file is not None:
    try:
        # Load and validate data
        df = pd.read_csv(uploaded_file)
        
        required_columns = ['Date', 'Product ID', 'Units Sold']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing columns: {', '.join(missing_columns)}")
            st.stop()
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        st.success(f"✅ Dataset loaded: {len(df):,} records")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Products", f"{df['Product ID'].nunique():,}")
        with col3:
            st.metric("Date Range", f"{(df['Date'].max() - df['Date'].min()).days} days")
        with col4:
            st.metric("Total Units", f"{df['Units Sold'].sum():,.0f}")
        
        st.markdown("---")
        
        # Product selection
        st.markdown("### 🎯 Select Product for Forecasting")
        
        product_stats = df.groupby('Product ID').agg({
            'Units Sold': ['sum', 'mean', 'count']
        }).round(2)
        product_stats.columns = ['Total Sales', 'Avg Daily Sales', 'Data Points']
        product_stats = product_stats.sort_values('Total Sales', ascending=False)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            product_options = df['Product ID'].unique().tolist()
            selected_product = st.selectbox("Choose Product", options=product_options)
            
            st.info(f"""
            **Statistics:**
            - Total: {product_stats.loc[selected_product, 'Total Sales']:,.0f} units
            - Avg Daily: {product_stats.loc[selected_product, 'Avg Daily Sales']:.1f} units
            - Data Points: {int(product_stats.loc[selected_product, 'Data Points'])}
            """)
        
        with col2:
            st.markdown("#### Top 5 Products by Sales")
            st.dataframe(product_stats.head(), use_container_width=True)
        
        st.markdown("---")
        
        # Train model button
        if st.button("🚀 Train Model & Generate Forecast", type="primary", use_container_width=True):
            with st.spinner("Training Prophet model..."):
                
                # Prepare data
                df_product = df[df["Product ID"] == selected_product].copy()
                df_product = df_product[df_product["Units Sold"] > 0]
                
                df_daily = df_product.groupby("Date")["Units Sold"].sum().reset_index()
                df_prophet = df_daily.rename(columns={"Date": "ds", "Units Sold": "y"})
                
                # Train-test split
                train_size = int(len(df_prophet) * (train_split / 100))
                train_df = df_prophet.iloc[:train_size]
                test_df = df_prophet.iloc[train_size:]
                
                # Train model
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=0.05
                )
                model.fit(train_df)
                
                # Generate forecast
                future = model.make_future_dataframe(periods=forecast_days)
                forecast = model.predict(future)
                
                # Evaluate
                forecast_result = forecast[["ds", "yhat"]].copy()
                test_merge = pd.merge(test_df, forecast_result, on="ds", how="inner")
                
                if len(test_merge) > 0:
                    mae = mean_absolute_error(test_merge["y"], test_merge["yhat"])
                    rmse = np.sqrt(mean_squared_error(test_merge["y"], test_merge["yhat"]))
                    mape = np.mean(np.abs((test_merge["y"] - test_merge["yhat"]) / test_merge["y"])) * 100
                else:
                    mae = rmse = mape = 0
                
                # Prepare metadata
                metadata = {
                    'model': model,
                    'forecast': forecast,
                    'train_df': train_df,
                    'test_df': test_df,
                    'test_merge': test_merge,
                    'df_prophet': df_prophet,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'product_id': selected_product,
                    'forecast_days': forecast_days,
                    'train_split': train_split,
                    'seasonality_mode': seasonality_mode
                }
                
                # Save model to pickle if requested
                if save_trained_model:
                    model_filename = save_model(model, selected_product, metadata)
                    st.success(f"💾 Model saved: {model_filename}")
                
                # Save to session state
                st.session_state.model_trained = True
                st.session_state.forecast_data = metadata
                
            st.success("✅ Model trained successfully!")
            st.rerun()
        
        # Display results if model is trained
        if st.session_state.model_trained and st.session_state.forecast_data:
            data = st.session_state.forecast_data
            
            st.markdown("---")
            st.markdown("## 📈 Forecast Results")
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MAE", f"{data['mae']:.2f}", help="Mean Absolute Error")
            
            with col2:
                st.metric("RMSE", f"{data['rmse']:.2f}", help="Root Mean Squared Error")
            
            with col3:
                st.metric("MAPE", f"{data['mape']:.2f}%", 
                         delta=f"{100-data['mape']:.1f}% accuracy",
                         help="Mean Absolute Percentage Error")
            
            with col4:
                if data['mape'] < 10:
                    performance = "🌟 EXCELLENT"
                elif data['mape'] < 20:
                    performance = "✅ GOOD"
                else:
                    performance = "⚠️ FAIR"
                st.metric("Performance", performance)
            
            # Visualizations
            st.markdown("### 📊 Demand Forecast Visualization")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot 1: Full forecast
            ax1.plot(data['train_df']["ds"], data['train_df']["y"], 
                    label="Training Data", color='#2E86AB', linewidth=2, alpha=0.8)
            ax1.plot(data['test_df']["ds"], data['test_df']["y"], 
                    label="Actual Test Data", color='#06A77D', linewidth=2, marker='o', markersize=4)
            ax1.plot(data['forecast']["ds"], data['forecast']["yhat"], 
                    label="Forecast", linestyle="--", color='#D62828', linewidth=2)
            ax1.fill_between(data['forecast']["ds"], 
                            data['forecast']["yhat_lower"], 
                            data['forecast']["yhat_upper"], 
                            alpha=0.2, color='#D62828', label='95% CI')
            ax1.axvline(x=data['train_df']["ds"].iloc[-1], color='black', 
                       linestyle=':', linewidth=2, label='Train/Test Split')
            ax1.set_xlabel("Date", fontsize=12)
            ax1.set_ylabel("Units Sold", fontsize=12)
            ax1.set_title(f"Demand Forecast for Product {data['product_id']}", 
                         fontsize=14, fontweight='bold')
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Test set zoom
            if len(data['test_merge']) > 0:
                ax2.plot(data['test_merge']["ds"], data['test_merge']["y"], 
                        label="Actual", color='#06A77D', linewidth=2, marker='o', markersize=6)
                ax2.plot(data['test_merge']["ds"], data['test_merge']["yhat"], 
                        label="Predicted", color='#D62828', linewidth=2, marker='s', markersize=6)
                ax2.fill_between(data['test_merge']["ds"], 
                                data['test_merge']["y"], 
                                data['test_merge']["yhat"], 
                                alpha=0.3, color='gray')
                ax2.set_xlabel("Date", fontsize=12)
                ax2.set_ylabel("Units Sold", fontsize=12)
                ax2.set_title(f"Test Set (MAE: {data['mae']:.2f}, MAPE: {data['mape']:.1f}%)", 
                             fontsize=12, fontweight='bold')
                ax2.legend(loc='best', fontsize=10)
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Components
            if show_components:
                st.markdown("### 📉 Trend & Seasonality Components")
                fig2 = data['model'].plot_components(data['forecast'])
                st.pyplot(fig2)
            
            # Business insights
            if show_insights:
                st.markdown("---")
                st.markdown("### 💡 Business Insights")
                
                forecast_future = data['forecast'].tail(data.get('forecast_days', 30))
                avg_demand = forecast_future["yhat"].mean()
                max_demand = forecast_future["yhat"].max()
                min_demand = forecast_future["yhat"].min()
                std_demand = forecast_future["yhat"].std()
                
                safety_stock = 1.65 * std_demand
                reorder_point = avg_demand * 7 + safety_stock
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Demand Statistics")
                    st.markdown(f"""
                    - **Average Daily:** {avg_demand:.0f} units
                    - **Peak Demand:** {max_demand:.0f} units
                    - **Minimum:** {min_demand:.0f} units
                    - **Variability:** {std_demand:.0f} units
                    """)
                
                with col2:
                    st.markdown("#### 📦 Inventory Recommendations")
                    st.markdown(f"""
                    - **Safety Stock:** {safety_stock:.0f} units
                    - **Reorder Point:** {reorder_point:.0f} units
                    - **Monthly Need:** {avg_demand * 30:.0f} units
                    """)
            
            # Download section
            st.markdown("---")
            st.markdown("### 📥 Downloads")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Forecast CSV
                forecast_output = data['forecast'][["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(data.get('forecast_days', 30))
                forecast_output.columns = ["Date", "Predicted_Demand", "Lower_Bound", "Upper_Bound"]
                
                csv_buffer = io.StringIO()
                forecast_output.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="📊 Forecast CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"forecast_{data['product_id']}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download trained model
                if os.path.exists(MODELS_DIR):
                    model_files = get_saved_models()
                    if model_files:
                        latest_model = model_files[0]
                        with open(os.path.join(MODELS_DIR, latest_model), 'rb') as f:
                            st.download_button(
                                label="💾 Download Model (.pkl)",
                                data=f,
                                file_name=latest_model,
                                mime="application/octet-stream"
                            )
            
            with col3:
                # Report
                report = f"""
DEMAND FORECASTING REPORT
{'='*60}

Product: {data['product_id']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PERFORMANCE
MAE:  {data['mae']:.2f} units
RMSE: {data['rmse']:.2f} units
MAPE: {data['mape']:.2f}%

FORECAST ({data.get('forecast_days', 30)} days)
Avg Demand: {avg_demand:.0f} units
Peak: {max_demand:.0f} units
Min: {min_demand:.0f} units

INVENTORY
Safety Stock: {safety_stock:.0f} units
Reorder Point: {reorder_point:.0f} units
{'='*60}
"""
                
                st.download_button(
                    label="📄 Report TXT",
                    data=report,
                    file_name=f"report_{data['product_id']}.txt",
                    mime="text/plain"
                )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>📊 <strong>Retail Demand Forecasting</strong> | Model Persistence with Pickle</p>
</div>
""", unsafe_allow_html=True)