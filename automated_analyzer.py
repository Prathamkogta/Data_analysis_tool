import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO
import re

# Configure page
st.set_page_config(
    page_title="Automated Data Analysis Tool",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .analysis-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .stSelectbox, .stMultiselect {
        width: 100%;
    }
    .header {
        color: #2c3e50;
        margin-bottom: 1.5rem;
    }
    .subheader {
        color: #3498db;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class AutomatedDataAnalyzer:
    def __init__(self):
        self.df = None
        self.initialize_ui()

    def initialize_ui(self):
        """Initialize the user interface"""
        st.title("📊 Automated Data Analysis Tool")
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            Upload your dataset and explore various analysis options
        </div>
        """, unsafe_allow_html=True)

    def clean_dataframe(self, df):
        """Handle empty rows, multi-index headers, and clean column names with duplicate handling"""
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Handle duplicate column names
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols == dup] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
        df.columns = cols
        
        # Check if the first row should be used as header
        if st.sidebar.checkbox("First row contains column names", value=True):
            if st.sidebar.checkbox("Use first row as header", value=True):
                # Set column names from first row
                new_header = df.iloc[0]
                df = df[1:]
                df.columns = new_header
                
                # Handle duplicates again after setting new header
                cols = pd.Series(df.columns)
                for dup in cols[cols.duplicated()].unique():
                    cols[cols == dup] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
                df.columns = cols
        
        # Clean column names (replace spaces, special chars, etc.)
        df.columns = [str(col).strip().replace(' ', '_').replace('.', '_').replace('-', '_') for col in df.columns]
        
        # Final check for duplicates after all cleaning
        cols = pd.Series(df.columns)
        if cols.duplicated().any():
            for dup in cols[cols.duplicated()].unique():
                cols[cols == dup] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
            df.columns = cols
        
        return df

    def load_data(self):
        """Handle data loading in the sidebar"""
        st.sidebar.header("Data Upload")
        data_source = st.sidebar.radio(
            "Select Data Source:",
            ["Use Sample Data", "Upload Your Own Data"]
        )

        if data_source == "Upload Your Own Data":
            uploaded_file = st.sidebar.file_uploader(
                "Upload your dataset (CSV, Excel, or JSON)",
                type=['csv', 'xlsx', 'xls', 'json']
            )
            if uploaded_file:
                try:
                    file_ext = uploaded_file.name.split('.')[-1].lower()
                    
                    # Skip rows option
                    skip_rows = 0
                    if st.sidebar.checkbox("Skip rows at top"):
                        skip_rows = st.sidebar.number_input("Number of rows to skip", min_value=0, max_value=100, value=0)
                    
                    if file_ext == 'csv':
                        # Try different encodings if needed
                        try:
                            self.df = pd.read_csv(uploaded_file, skiprows=skip_rows)
                        except UnicodeDecodeError:
                            self.df = pd.read_csv(uploaded_file, encoding='latin1', skiprows=skip_rows)
                        self.df = self.clean_dataframe(self.df)
                    elif file_ext in ['xls', 'xlsx']:
                        self.df = pd.read_excel(uploaded_file, skiprows=skip_rows)
                        self.df = self.clean_dataframe(self.df)
                    elif file_ext == 'json':
                        self.df = pd.read_json(uploaded_file)
                        if skip_rows > 0:
                            self.df = self.df.iloc[skip_rows:]
                        self.df = self.clean_dataframe(self.df)
                    
                    st.sidebar.success("Data loaded successfully!")
                except Exception as e:
                    st.sidebar.error(f"Error reading file: {str(e)}")
        else:
            # Load sample data with enhanced time series features
            try:
                dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
                self.df = pd.DataFrame({
                    'Timestamp': dates,
                    'Efficiency': np.sin(np.linspace(0, 10, 365)) * 10 + 80 + np.random.normal(0, 3, 365),
                    'Temperature': np.cos(np.linspace(0, 8, 365)) * 15 + 150 + np.random.normal(0, 5, 365),
                    'Pressure': np.sin(np.linspace(0, 6, 365)) * 8 + 50 + np.random.normal(0, 2, 365),
                    'Flow_Rate': np.cos(np.linspace(0, 12, 365)) * 20 + 100 + np.random.normal(0, 8, 365),
                    'Quality_Score': np.random.uniform(0.7, 1.0, 365)
                })
                st.sidebar.info("Sample data loaded. You can now explore the analysis options.")
            except Exception as e:
                st.sidebar.error(f"Error creating sample data: {str(e)}")

        if self.df is not None:
            st.sidebar.subheader("Dataset Preview")
            st.sidebar.dataframe(self.df.head(3))
            st.sidebar.write(f"Shape: {self.df.shape}")

    def show_main_menu(self):
        """Show the main analysis options menu"""
        st.header("Select Analysis Type", anchor=False)
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Basic Analysis")
            if st.button("📈 Basic Statistics", help="Show summary statistics for all numerical columns"):
                st.session_state.current_analysis = "basic_stats"
                st.rerun()
            
            if st.button("📊 Distribution Analysis", help="Show histograms and boxplots for numerical columns"):
                st.session_state.current_analysis = "distribution"
                st.rerun()
            
            if st.button("🔍 Missing Values Analysis", help="Analyze and handle missing values in the dataset"):
                st.session_state.current_analysis = "missing_values"
                st.rerun()

        with col2:
            st.markdown("### Advanced Analysis")
            if st.button("⏱️ Time Series Analysis", help="Perform various time series analyses"):
                st.session_state.current_analysis = "time_series"
                st.rerun()
            
            if st.button("🔄 Correlation Analysis", help="Analyze correlations between numerical features"):
                st.session_state.current_analysis = "correlation"
                st.rerun()
            
            if st.button("📑 Generate Full Report", help="Generate a comprehensive report with all analyses"):
                st.session_state.current_analysis = "full_report"
                st.rerun()

        if hasattr(st.session_state, 'current_analysis'):
            st.markdown("---")
            self.show_analysis_section()

    def show_analysis_section(self):
        """Show the appropriate analysis section based on user selection"""
        with st.container():
            st.markdown(f"<div class='analysis-section'>", unsafe_allow_html=True)
            
            if st.session_state.current_analysis == "basic_stats":
                self.show_basic_stats()
            elif st.session_state.current_analysis == "distribution":
                self.show_distribution_analysis()
            elif st.session_state.current_analysis == "missing_values":
                self.show_missing_values_analysis()
            elif st.session_state.current_analysis == "time_series":
                self.show_time_series_menu()
            elif st.session_state.current_analysis == "correlation":
                self.show_correlation_analysis()
            elif st.session_state.current_analysis == "full_report":
                self.generate_full_report()
            
            st.markdown("</div>", unsafe_allow_html=True)

    def detect_datetime_columns(self):
        """Enhanced datetime column detection with multiple format support"""
        datetime_cols = []
        
        # First check existing datetime columns
        existing_datetime_cols = self.df.select_dtypes(include=['datetime64']).columns
        if len(existing_datetime_cols) > 0:
            return list(existing_datetime_cols)
            
        # Common datetime column names to prioritize
        common_datetime_names = ['date', 'time', 'timestamp', 'datetime', 'created', 'modified']
        
        # Check columns with common names first
        potential_cols = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(dt_name in col_lower for dt_name in common_datetime_names):
                potential_cols.append(col)
        
        # Then check other string columns
        for col in self.df.select_dtypes(include=['object']).columns:
            if col not in potential_cols:
                sample = self.df[col].dropna().head(5).astype(str)
                if sample.empty:
                    continue
                    
                # Enhanced date pattern detection
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
                    r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
                    r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
                    r'\d{2}-\w{3}-\d{4}',  # DD-MMM-YYYY
                    r'\d{2}/\w{3}/\d{4}',  # DD/MMM/YYYY
                    r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
                ]
                
                has_date_pattern = any(any(re.search(pattern, str(val)) for pattern in date_patterns) for val in sample)
                
                if has_date_pattern:
                    potential_cols.append(col)
        
        # Try to convert each potential column with multiple date formats
        date_formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%d-%b-%Y', '%d/%b/%Y', '%Y-%m-%d %H:%M:%S',
            '%m-%d-%Y', '%m/%d/%y', '%b %d %Y', '%B %d %Y'
        ]
        
        for col in potential_cols:
            converted = False
            for date_format in date_formats:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], format=date_format, errors='raise')
                    datetime_cols.append(col)
                    converted = True
                    break
                except (ValueError, TypeError):
                    continue
            
            if not converted:
                # Try without format as last resort
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    if not self.df[col].isnull().all():
                        datetime_cols.append(col)
                except (ValueError, TypeError):
                    continue
        
        return datetime_cols

    def show_basic_stats(self):
        """Display basic statistical features"""
        st.subheader("Basic Statistical Features", anchor=False)
        
        if self.df is None:
            st.warning("No data loaded. Please upload a dataset first.")
            return
            
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found in the dataset.")
            return
            
        st.dataframe(self.df[numeric_cols].describe().T.style.background_gradient(cmap='Blues'))
        
        # Additional statistics
        st.subheader("Additional Statistics", anchor=False)
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Skewness:")
            st.dataframe(self.df[numeric_cols].skew().to_frame(name='Skewness').style.background_gradient(cmap='Reds'))
        
        with col2:
            st.write("Kurtosis:")
            st.dataframe(self.df[numeric_cols].kurtosis().to_frame(name='Kurtosis').style.background_gradient(cmap='Greens'))
        
        self.add_download_button(self.df[numeric_cols].describe().T, "basic_stats.csv")

    def show_distribution_analysis(self):
        """Show distribution analysis for numerical columns"""
        st.subheader("Distribution Analysis", anchor=False)
        
        if self.df is None:
            st.warning("No data loaded. Please upload a dataset first.")
            return
            
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found in the dataset.")
            return
            
        selected_col = st.selectbox("Select column for analysis:", numeric_cols)
        
        if selected_col:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig1 = px.histogram(
                    self.df, 
                    x=selected_col,
                    nbins=30,
                    title=f"Distribution of {selected_col}",
                    color_discrete_sequence=['#3498db']
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Boxplot
                fig2 = px.box(
                    self.df,
                    y=selected_col,
                    title=f"Boxplot of {selected_col}",
                    color_discrete_sequence=['#e74c3c']
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Create a combined figure for download
            combined_fig = go.Figure()
            combined_fig.add_trace(go.Histogram(
                x=self.df[selected_col],
                nbinsx=30,
                name="Histogram",
                marker_color='#3498db'
            ))
            combined_fig.add_trace(go.Box(
                y=self.df[selected_col],
                name="Boxplot",
                boxpoints="outliers",
                marker_color='#e74c3c'
            ))
            combined_fig.update_layout(
                title=f"Combined Distribution Analysis of {selected_col}",
                template="plotly_white"
            )
            
            self.add_plot_download_button(combined_fig, f"distribution_{selected_col}.html")

    def show_missing_values_analysis(self):
        """Analyze and handle missing values"""
        st.subheader("Missing Values Analysis", anchor=False)
        
        if self.df is None:
            st.warning("No data loaded. Please upload a dataset first.")
            return
            
        # Calculate missing values
        missing_vals = self.df.isnull().sum()
        missing_percentages = (missing_vals / len(self.df) * 100).round(2)
        
        # Create missing values summary
        missing_summary = pd.DataFrame({
            'Missing Values': missing_vals,
            'Percentage (%)': missing_percentages
        })
        
        st.dataframe(missing_summary.style.background_gradient(cmap='Oranges'))
        
        # Visualization
        fig = px.bar(
            missing_summary,
            x=missing_summary.index,
            y='Percentage (%)',
            title="Percentage of Missing Values by Column",
            color='Percentage (%)',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Handling options
        st.subheader("Handle Missing Values", anchor=False)
        handling_method = st.selectbox(
            "Select handling method:",
            ["None", "Drop rows with missing values", "Linear interpolation", "Forward fill", "Backward fill"]
        )
        
        if handling_method != "None":
            original_shape = self.df.shape
            
            if handling_method == "Drop rows with missing values":
                cleaned_df = self.df.dropna()
            elif handling_method == "Linear interpolation":
                cleaned_df = self.df.interpolate(method='linear')
            elif handling_method == "Forward fill":
                cleaned_df = self.df.fillna(method='ffill')
            else:  # Backward fill
                cleaned_df = self.df.fillna(method='bfill')
            
            st.write(f"Original shape: {original_shape}")
            st.write(f"New shape: {cleaned_df.shape}")
            
            if st.button("Apply changes to dataset"):
                self.df = cleaned_df
                st.success("Missing values handled successfully! The dataset has been updated.")
                st.rerun()
        
        self.add_download_button(missing_summary, "missing_values_summary.csv")

    def show_time_series_menu(self):
        """Show time series analysis menu with enhanced options"""
        st.subheader("Time Series Analysis", anchor=False)
        
        if self.df is None:
            st.warning("No data loaded. Please upload a dataset first.")
            return
            
        # Auto-detect datetime columns with enhanced detection
        datetime_cols = self.detect_datetime_columns()
        
        if len(datetime_cols) == 0:
            st.warning("No datetime columns detected. Please ensure your data includes a timestamp column.")
            return
            
        # Step 1: Select time column
        time_col = st.selectbox(
            "Select timestamp column:", 
            datetime_cols,
            key="time_col_select"
        )
        
        # Step 2: Select analysis type
        analysis_type = st.selectbox(
            "Select time series analysis type:",
            [
                "Basic Time Series Plot",
                "Trend Analysis",
                "Seasonal Decomposition",
                "Compare First vs Second Half",
                "Hourly/Daily/Monthly Analysis",
                "Weekday vs Weekend Analysis",
                "Rolling Statistics"
            ],
            key="ts_analysis_type"
        )
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for time series analysis.")
            return
            
        # Step 3: Show appropriate analysis based on selection
        if analysis_type == "Basic Time Series Plot":
            self.show_basic_time_series(time_col, numeric_cols)
        elif analysis_type == "Trend Analysis":
            self.show_trend_analysis(time_col, numeric_cols)
        elif analysis_type == "Seasonal Decomposition":
            self.show_seasonal_decomposition(time_col, numeric_cols)
        elif analysis_type == "Compare First vs Second Half":
            self.compare_halves(time_col, numeric_cols)
        elif analysis_type == "Hourly/Daily/Monthly Analysis":
            self.show_periodic_analysis(time_col, numeric_cols)
        elif analysis_type == "Weekday vs Weekend Analysis":
            self.weekday_weekend_analysis(time_col, numeric_cols)
        elif analysis_type == "Rolling Statistics":
            self.show_rolling_stats(time_col, numeric_cols)

    def show_basic_time_series(self, time_col, numeric_cols):
        """Show basic time series plot with enhanced options"""
        st.subheader("Basic Time Series Visualization", anchor=False)
        
        selected_cols = st.multiselect(
            "Select columns to plot:",
            numeric_cols,
            default=[numeric_cols[0]] if len(numeric_cols) > 0 else [],
            key="ts_cols_select"
        )
        
        if selected_cols:
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly
            
            for i, col in enumerate(selected_cols):
                fig.add_trace(go.Scatter(
                    x=self.df[time_col],
                    y=self.df[col],
                    name=col,
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            fig.update_layout(
                title="Time Series Analysis",
                xaxis_title="Time",
                yaxis_title="Value",
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Time range information
            min_date = self.df[time_col].min()
            max_date = self.df[time_col].max()
            st.write(f"Time Range: {min_date} to {max_date}")
            st.write(f"Time Span: {max_date - min_date}")
            
            self.add_plot_download_button(fig, "time_series_plot.html")

    def show_trend_analysis(self, time_col, numeric_cols):
        """Show trend analysis with moving averages"""
        st.subheader("Trend Analysis", anchor=False)
        
        selected_col = st.selectbox(
            "Select column for trend analysis:", 
            numeric_cols,
            key="trend_col_select"
        )
        
        window_size = st.slider(
            "Select window size for moving average:",
            min_value=3,
            max_value=30,
            value=7,
            key="trend_window"
        )
        
        fig = go.Figure()
        
        # Original data
        fig.add_trace(go.Scatter(
            x=self.df[time_col],
            y=self.df[selected_col],
            name="Original",
            line=dict(color='#3498db', width=1),
            opacity=0.3
        ))
        
        # Moving average
        rolling_mean = self.df[selected_col].rolling(window=window_size).mean()
        fig.add_trace(go.Scatter(
            x=self.df[time_col],
            y=rolling_mean,
            name=f"{window_size}-period Moving Average",
            line=dict(color='#e74c3c', width=2)
        ))
        
        fig.update_layout(
            title=f"Trend Analysis for {selected_col}",
            xaxis_title="Time",
            yaxis_title=selected_col,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        self.add_plot_download_button(fig, "trend_analysis.html")

    def show_seasonal_decomposition(self, time_col, numeric_cols):
        """Show seasonal decomposition of time series"""
        st.subheader("Seasonal Decomposition", anchor=False)
        
        selected_col = st.selectbox(
            "Select column for decomposition:", 
            numeric_cols,
            key="decomp_col_select"
        )
        
        period = st.selectbox(
            "Select seasonal period:",
            [7, 30, 90, 365],
            index=1,
            key="seasonal_period"
        )
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Ensure the data is sorted by time
            temp_df = self.df.set_index(time_col).sort_index()
            
            # Perform decomposition
            decomposition = seasonal_decompose(temp_df[selected_col], period=period, model='additive')
            
            # Create figure with subplots
            fig = go.Figure()
            
            # Original
            fig.add_trace(go.Scatter(
                x=temp_df.index,
                y=temp_df[selected_col],
                name="Observed",
                line=dict(color='#3498db')
            ))
            
            # Trend
            fig.add_trace(go.Scatter(
                x=temp_df.index,
                y=decomposition.trend,
                name="Trend",
                line=dict(color='#e74c3c')
            ))
            
            # Seasonal
            fig.add_trace(go.Scatter(
                x=temp_df.index,
                y=decomposition.seasonal,
                name="Seasonal",
                line=dict(color='#2ecc71')
            ))
            
            # Residual
            fig.add_trace(go.Scatter(
                x=temp_df.index,
                y=decomposition.resid,
                name="Residual",
                line=dict(color='#f39c12')
            ))
            
            fig.update_layout(
                title=f"Seasonal Decomposition of {selected_col}",
                xaxis_title="Time",
                template="plotly_white",
                height=800
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Individual components
            st.subheader("Individual Components", anchor=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=temp_df.index,
                    y=decomposition.trend,
                    name="Trend",
                    line=dict(color='#e74c3c')
                ))
                fig_trend.update_layout(title="Trend Component")
                st.plotly_chart(fig_trend, use_container_width=True)
                
            with col2:
                fig_seasonal = go.Figure()
                fig_seasonal.add_trace(go.Scatter(
                    x=temp_df.index,
                    y=decomposition.seasonal,
                    name="Seasonal",
                    line=dict(color='#2ecc71')
                ))
                fig_seasonal.update_layout(title="Seasonal Component")
                st.plotly_chart(fig_seasonal, use_container_width=True)
            
            self.add_plot_download_button(fig, "seasonal_decomposition.html")
            
        except ImportError:
            st.error("This feature requires statsmodels. Please install it with: pip install statsmodels")
        except Exception as e:
            st.error(f"Error in seasonal decomposition: {str(e)}")

    def compare_halves(self, time_col, numeric_cols):
        """Compare first and second half trends"""
        st.subheader("First Half vs Second Half Comparison", anchor=False)
        
        mid_point = self.df[time_col].min() + (self.df[time_col].max() - self.df[time_col].min()) / 2
        
        first_half = self.df[self.df[time_col] <= mid_point]
        second_half = self.df[self.df[time_col] > mid_point]
        
        st.write(f"First Half Period: {first_half[time_col].min()} to {first_half[time_col].max()}")
        st.write(f"Second Half Period: {second_half[time_col].min()} to {second_half[time_col].max()}")
        
        selected_col = st.selectbox(
            "Select column to compare:", 
            numeric_cols,
            key="half_compare_col"
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=first_half[time_col],
            y=first_half[selected_col],
            name=f"{selected_col} (First Half)",
            line=dict(color='#3498db')
        ))
        fig.add_trace(go.Scatter(
            x=second_half[time_col],
            y=second_half[selected_col],
            name=f"{selected_col} (Second Half)",
            line=dict(color='#e74c3c')
        ))
        fig.update_layout(
            title="First Half vs Second Half Comparison",
            xaxis_title="Date",
            yaxis_title=selected_col,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical comparison
        comparison_data = {
            "Metric": ["Mean", "Median", "Std Dev", "Min", "Max"],
            "First Half": [
                first_half[selected_col].mean(),
                first_half[selected_col].median(),
                first_half[selected_col].std(),
                first_half[selected_col].min(),
                first_half[selected_col].max()
            ],
            "Second Half": [
                second_half[selected_col].mean(),
                second_half[selected_col].median(),
                second_half[selected_col].std(),
                second_half[selected_col].min(),
                second_half[selected_col].max()
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.write("Statistical Comparison:")
        st.dataframe(comparison_df.style.background_gradient(cmap='Blues'))
        
        self.add_plot_download_button(fig, "half_comparison.html")
        self.add_download_button(comparison_df, "half_comparison_stats.csv")

    def show_periodic_analysis(self, time_col, numeric_cols):
        """Show hourly/daily/monthly analysis"""
        st.subheader("Periodic Analysis", anchor=False)
        
        analysis_period = st.selectbox(
            "Select analysis period:",
            ["Hourly", "Daily", "Monthly"],
            key="period_select"
        )
        
        selected_col = st.selectbox(
            "Select column for analysis:", 
            numeric_cols,
            key="periodic_col_select"
        )
        
        try:
            temp_df = self.df.copy()
            temp_df[time_col] = pd.to_datetime(temp_df[time_col])
            
            if analysis_period == "Hourly":
                temp_df['period'] = temp_df[time_col].dt.hour
                period_name = "Hour"
            elif analysis_period == "Daily":
                temp_df['period'] = temp_df[time_col].dt.day
                period_name = "Day"
            else:  # Monthly
                temp_df['period'] = temp_df[time_col].dt.month
                period_name = "Month"
            
            # Group by period
            period_stats = temp_df.groupby('period')[selected_col].agg(['mean', 'min', 'max', 'std']).reset_index()
            
            fig = go.Figure()
            
            # Mean line
            fig.add_trace(go.Scatter(
                x=period_stats['period'],
                y=period_stats['mean'],
                name="Mean",
                line=dict(color='#3498db', width=2)
            ))
            
            # Range area
            fig.add_trace(go.Scatter(
                x=period_stats['period'],
                y=period_stats['min'],
                name="Min",
                line=dict(color='rgba(0,0,0,0)')
            ))
            
            fig.add_trace(go.Scatter(
                x=period_stats['period'],
                y=period_stats['max'],
                name="Max",
                line=dict(color='rgba(0,0,0,0)'),
                fill='tonexty',
                fillcolor='rgba(52, 152, 219, 0.2)'
            ))
            
            fig.update_layout(
                title=f"{analysis_period} Analysis of {selected_col}",
                xaxis_title=period_name,
                yaxis_title=selected_col,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show statistics table
            st.write(f"{analysis_period} Statistics:")
            st.dataframe(period_stats.style.background_gradient(cmap='Blues'))
            
            self.add_plot_download_button(fig, f"{analysis_period.lower()}_analysis.html")
            self.add_download_button(period_stats, f"{analysis_period.lower()}_stats.csv")
            
        except Exception as e:
            st.error(f"Error in {analysis_period.lower()} analysis: {str(e)}")

    def weekday_weekend_analysis(self, time_col, numeric_cols):
        """Compare weekday vs weekend trends"""
        st.subheader("Weekday vs Weekend Analysis", anchor=False)
        
        try:
            # Extract day of week
            self.df['day_of_week'] = self.df[time_col].dt.dayofweek
            self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6])  # 5=Saturday, 6=Sunday
            
            weekday_data = self.df[~self.df['is_weekend']]
            weekend_data = self.df[self.df['is_weekend']]
            
            st.write(f"Weekday Records: {len(weekday_data)}")
            st.write(f"Weekend Records: {len(weekend_data)}")
            
            selected_col = st.selectbox(
                "Select column for comparison:", 
                numeric_cols,
                key="weekend_col_select"
            )
            
            # Create box plots for comparison
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=weekday_data[selected_col],
                name="Weekday",
                boxpoints='outliers',
                marker_color='#3498db'
            ))
            fig.add_trace(go.Box(
                y=weekend_data[selected_col],
                name="Weekend",
                boxpoints='outliers',
                marker_color='#e74c3c'
            ))
            fig.update_layout(
                title=f"Weekday vs Weekend: {selected_col}",
                yaxis_title=selected_col,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical comparison
            comparison_data = {
                "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "Count"],
                "Weekday": [
                    weekday_data[selected_col].mean(),
                    weekday_data[selected_col].median(),
                    weekday_data[selected_col].std(),
                    weekday_data[selected_col].min(),
                    weekday_data[selected_col].max(),
                    len(weekday_data)
                ],
                "Weekend": [
                    weekend_data[selected_col].mean(),
                    weekend_data[selected_col].median(),
                    weekend_data[selected_col].std(),
                    weekend_data[selected_col].min(),
                    weekend_data[selected_col].max(),
                    len(weekend_data)
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.write("Statistical Comparison:")
            st.dataframe(comparison_df.style.background_gradient(cmap='Blues'))
            
            # Calculate percent difference
            if not weekday_data.empty and not weekend_data.empty:
                weekday_mean = weekday_data[selected_col].mean()
                weekend_mean = weekend_data[selected_col].mean()
                if weekday_mean != 0:
                    percent_diff = ((weekend_mean - weekday_mean) / weekday_mean) * 100
                    st.write(f"Weekend vs Weekday Percent Difference: {percent_diff:.2f}%")
            
            self.add_plot_download_button(fig, "weekday_weekend_comparison.html")
            self.add_download_button(comparison_df, "weekday_weekend_stats.csv")
            
        except Exception as e:
            st.error(f"Error in weekday/weekend analysis: {str(e)}")

    def show_rolling_stats(self, time_col, numeric_cols):
        """Show rolling statistics analysis"""
        st.subheader("Rolling Statistics", anchor=False)
        
        selected_col = st.selectbox(
            "Select column for rolling analysis:", 
            numeric_cols,
            key="rolling_col_select"
        )
        
        window_size = st.slider(
            "Select window size:",
            min_value=3,
            max_value=90,
            value=7,
            key="rolling_window"
        )
        
        stat_type = st.selectbox(
            "Select statistic to calculate:",
            ["Mean", "Median", "Standard Deviation", "Minimum", "Maximum"],
            key="rolling_stat"
        )
        
        try:
            temp_df = self.df.set_index(time_col).sort_index()
            
            if stat_type == "Mean":
                rolling_stat = temp_df[selected_col].rolling(window=window_size).mean()
            elif stat_type == "Median":
                rolling_stat = temp_df[selected_col].rolling(window=window_size).median()
            elif stat_type == "Standard Deviation":
                rolling_stat = temp_df[selected_col].rolling(window=window_size).std()
            elif stat_type == "Minimum":
                rolling_stat = temp_df[selected_col].rolling(window=window_size).min()
            else:  # Maximum
                rolling_stat = temp_df[selected_col].rolling(window=window_size).max()
            
            fig = go.Figure()
            
            # Original data
            fig.add_trace(go.Scatter(
                x=temp_df.index,
                y=temp_df[selected_col],
                name="Original",
                line=dict(color='#3498db', width=1),
                opacity=0.3
            ))
            
            # Rolling statistic
            fig.add_trace(go.Scatter(
                x=temp_df.index,
                y=rolling_stat,
                name=f"{window_size}-period {stat_type}",
                line=dict(color='#e74c3c', width=2)
            ))
            
            fig.update_layout(
                title=f"{window_size}-period Rolling {stat_type} of {selected_col}",
                xaxis_title="Time",
                yaxis_title=selected_col,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a dataframe with the rolling stats
            rolling_df = pd.DataFrame({
                'Date': temp_df.index,
                'Original': temp_df[selected_col],
                f'Rolling_{stat_type}': rolling_stat
            })
            
            self.add_plot_download_button(fig, "rolling_stats.html")
            self.add_download_button(rolling_df, "rolling_stats.csv")
            
        except Exception as e:
            st.error(f"Error in rolling statistics: {str(e)}")

    def show_correlation_analysis(self):
        """Show correlation analysis"""
        st.subheader("Correlation Analysis", anchor=False)
        
        if self.df is None:
            st.warning("No data loaded. Please upload a dataset first.")
            return
            
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            st.warning("At least two numeric columns are required for correlation analysis.")
            return
            
        # Correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        # Heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            title="Correlation Matrix Heatmap",
            color_continuous_scale="RdBu",
            aspect="auto",
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlations
        st.subheader("Top Correlations", anchor=False)
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                correlations.append({
                    'Variables': f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}",
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        correlations_df = pd.DataFrame(correlations)
        st.dataframe(correlations_df.nlargest(5, 'Correlation').style.background_gradient(cmap='RdBu', subset=['Correlation']))
        
        # Pairplot for selected features
        st.subheader("Feature Relationships", anchor=False)
        selected_cols = st.multiselect(
            "Select features for pairplot:",
            numeric_cols,
            default=list(numeric_cols)[:4] if len(numeric_cols) > 4 else numeric_cols,
            key="pairplot_cols"
        )
        
        if len(selected_cols) >= 2:
            fig = px.scatter_matrix(
                self.df[selected_cols],
                dimensions=selected_cols,
                title="Feature Pairplot",
                color=selected_cols[0] if len(selected_cols) > 0 else None
            )
            st.plotly_chart(fig, use_container_width=True)
            self.add_plot_download_button(fig, "feature_pairplot.html")
        
        self.add_download_button(corr_matrix, "correlation_matrix.csv")
        self.add_download_button(correlations_df.nlargest(5, 'Correlation'), "top_correlations.csv")

    def generate_full_report(self):
        """Generate a comprehensive report with all analyses"""
        st.header("📑 Comprehensive Data Analysis Report", anchor=False)
        
        if self.df is None:
            st.warning("No data loaded. Please upload a dataset first.")
            return
            
        with st.spinner("Generating comprehensive report..."):
            # Create a buffer to store the report
            report_buffer = []
            
            # Add basic information
            report_buffer.append("# Comprehensive Data Analysis Report\n")
            report_buffer.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report_buffer.append(f"Dataset shape: {self.df.shape}\n")
            report_buffer.append(f"Columns: {', '.join(self.df.columns)}\n")
            
            # 1. Basic Statistics
            report_buffer.append("\n## 1. Basic Statistics\n")
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                basic_stats = self.df[numeric_cols].describe().T
                report_buffer.append(basic_stats.to_markdown())
            else:
                report_buffer.append("No numeric columns found for statistical analysis.")
            
            # 2. Missing Values
            report_buffer.append("\n## 2. Missing Values Analysis\n")
            missing_summary = pd.DataFrame({
                'Missing Values': self.df.isnull().sum(),
                'Percentage (%)': (self.df.isnull().sum() / len(self.df) * 100).round(2)
            })
            report_buffer.append(missing_summary.to_markdown())
            
            # 3. Distribution Analysis
            report_buffer.append("\n## 3. Distribution Analysis\n")
            if len(numeric_cols) > 0:
                for col in numeric_cols:
                    report_buffer.append(f"\n### {col}\n")
                    stats = {
                        'Mean': self.df[col].mean(),
                        'Median': self.df[col].median(),
                        'Std Dev': self.df[col].std(),
                        'Min': self.df[col].min(),
                        'Max': self.df[col].max(),
                        'Skewness': self.df[col].skew(),
                        'Kurtosis': self.df[col].kurtosis()
                    }
                    report_buffer.append(pd.Series(stats).to_markdown())
            else:
                report_buffer.append("No numeric columns found for distribution analysis.")
            
            # 4. Time Series Analysis (if applicable)
            datetime_cols = self.detect_datetime_columns()
            if len(datetime_cols) > 0 and len(numeric_cols) > 0:
                report_buffer.append("\n## 4. Time Series Analysis\n")
                time_col = datetime_cols[0]
                
                # Time range
                min_date = self.df[time_col].min()
                max_date = self.df[time_col].max()
                report_buffer.append(f"\nTime Range: {min_date} to {max_date}")
                report_buffer.append(f"\nTime Span: {max_date - min_date}\n")
                
                # First vs second half comparison
                mid_point = min_date + (max_date - min_date) / 2
                first_half = self.df[self.df[time_col] <= mid_point]
                second_half = self.df[self.df[time_col] > mid_point]
                
                report_buffer.append("\n### First Half vs Second Half Comparison\n")
                report_buffer.append(f"First Half Period: {first_half[time_col].min()} to {first_half[time_col].max()}")
                report_buffer.append(f"\nSecond Half Period: {second_half[time_col].min()} to {second_half[time_col].max()}\n")
                
                for col in numeric_cols[:3]:  # Limit to first 3 numeric columns for brevity
                    comparison_data = {
                        "Metric": ["Mean", "Median", "Std Dev", "Min", "Max"],
                        "First Half": [
                            first_half[col].mean(),
                            first_half[col].median(),
                            first_half[col].std(),
                            first_half[col].min(),
                            first_half[col].max()
                        ],
                        "Second Half": [
                            second_half[col].mean(),
                            second_half[col].median(),
                            second_half[col].std(),
                            second_half[col].min(),
                            second_half[col].max()
                        ]
                    }
                    report_buffer.append(f"\n#### {col}\n")
                    report_buffer.append(pd.DataFrame(comparison_data).to_markdown())
            
            # 5. Correlation Analysis
            if len(numeric_cols) >= 2:
                report_buffer.append("\n## 5. Correlation Analysis\n")
                corr_matrix = self.df[numeric_cols].corr()
                
                # Top correlations
                correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i):
                        correlations.append({
                            'Variables': f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}",
                            'Correlation': corr_matrix.iloc[i, j]
                        })
                
                correlations_df = pd.DataFrame(correlations)
                report_buffer.append("\n### Top 5 Correlations\n")
                report_buffer.append(correlations_df.nlargest(5, 'Correlation').to_markdown())
            
            # Combine all parts of the report
            full_report = "\n".join(report_buffer)
            
            # Display the report in the app
            st.markdown(full_report)
            
            # Download button for the report
            st.download_button(
                label="📥 Download Full Report",
                data=full_report,
                file_name="data_analysis_report.md",
                mime="text/markdown"
            )

    def add_download_button(self, data, filename):
        """Add a download button for dataframes"""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            csv = data.to_csv(index=True)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="display: inline-block; padding: 0.5rem 1rem; background-color: #3498db; color: white; border-radius: 5px; text-decoration: none;">📥 Download {filename}</a>'
            st.markdown(href, unsafe_allow_html=True)

    def add_plot_download_button(self, fig, filename):
        """Add a download button for plotly figures"""
        html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        b64 = base64.b64encode(html.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="{filename}" style="display: inline-block; padding: 0.5rem 1rem; background-color: #3498db; color: white; border-radius: 5px; text-decoration: none;">📥 Download Plot as HTML</a>'
        st.markdown(href, unsafe_allow_html=True)

    def run(self):
        """Main application flow"""
        self.load_data()
        
        if self.df is not None:
            if st.sidebar.button("← Back to Main Menu"):
                if 'current_analysis' in st.session_state:
                    del st.session_state['current_analysis']
                st.rerun()
            
            if 'current_analysis' not in st.session_state:
                self.show_main_menu()
            else:
                self.show_analysis_section()

if __name__ == "__main__":
    analyzer = AutomatedDataAnalyzer()
    analyzer.run()
