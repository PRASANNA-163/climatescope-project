import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

# Page configuration
st.set_page_config(
    page_title="ClimateScope Dashboard",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #E0F2FE 0%, #F3E8FF 50%, #FCE7F3 100%);
    }
    .stMetric {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
    }
    h1 {
        color: #1E293B;
        font-weight: 700;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style='background: linear-gradient(90deg, #2563EB 0%, #9333EA 50%, #EC4899 100%); 
            padding: 30px; border-radius: 15px; margin-bottom: 30px;'>
    <h1 style='color: white; margin: 0;'>🌦️ ClimateScope Dashboard</h1>
    <p style='color: #E0E7FF; margin: 5px 0 0 0; font-size: 16px;'>
        Interactive Global Climate & Air Quality Analytics
    </p>
</div>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('CleanedWeatherRepository.csv')
        dataMonth = pd.read_csv('CleanedWeatherRepositoryMonthly.csv')
        data['date'] = pd.to_datetime(data['date'])
        return data, dataMonth
    except Exception as e:
        st.error(f"⚠️ Error loading data files: {e}")
        st.info("Please make sure 'CleanedWeatherRepository.csv' and 'CleanedWeatherRepositoryMonthly.csv' are in the same directory as this script.")
        return None, None

# Load data directly
data, dataMonth = load_data()

if data is not None and dataMonth is not None:
    # Define numeric columns
    numeric_cols = ['temperature_celsius', 'feels_like_celsius', 'humidity', 'wind_kph', 
                    'gust_kph', 'pressure_mb', 'precip_mm', 'air_quality_PM2.5', 'air_quality_PM10']
    
    key_cols = ['temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm']
    
    pollutants = ['air_quality_Carbon_Monoxide', 'air_quality_Nitrogen_dioxide',
                  'air_quality_Sulphur_dioxide', 'air_quality_Ozone',
                  'air_quality_PM2.5', 'air_quality_PM10']
    
    # Calculate KPIs from your ACTUAL data
    avg_temp = data['temperature_celsius'].mean()
    avg_humidity = data['humidity'].mean()
    avg_wind = data['wind_kph'].mean()
    avg_precip = data['precip_mm'].mean()
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #FB923C 0%, #EF4444 100%); 
                    padding: 20px; border-radius: 15px; text-align: center;'>
            <div style='font-size: 36px; font-weight: bold; color: white;'>{avg_temp:.1f}°C</div>
            <div style='color: #FED7AA; font-weight: 500; margin-top: 5px;'>Avg Temperature</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #3B82F6 0%, #06B6D4 100%); 
                    padding: 20px; border-radius: 15px; text-align: center;'>
            <div style='font-size: 36px; font-weight: bold; color: white;'>{avg_humidity:.1f}%</div>
            <div style='color: #BAE6FD; font-weight: 500; margin-top: 5px;'>Avg Humidity</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #14B8A6 0%, #22C55E 100%); 
                    padding: 20px; border-radius: 15px; text-align: center;'>
            <div style='font-size: 36px; font-weight: bold; color: white;'>{avg_wind:.1f}</div>
            <div style='color: #A7F3D0; font-weight: 500; margin-top: 5px;'>Avg Wind (km/h)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%); 
                    padding: 20px; border-radius: 15px; text-align: center;'>
            <div style='font-size: 36px; font-weight: bold; color: white;'>{avg_precip:.1f}</div>
            <div style='color: #DDD6FE; font-weight: 500; margin-top: 5px;'>Avg Precip (mm)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title("🎛️ Navigation")
    
    analysis_type = st.sidebar.radio(
        "Select Analysis Type:",
        ["📈 Statistical Overview", 
         "📊 Distribution Analysis", 
         "🔗 Correlation Analysis",
         "📅 Monthly Trends",
         "💨 Air Quality Analysis",
         "🗺️ Geographic Analysis",
         "🌡️ Extreme Weather Events"]
    )
    
    st.sidebar.markdown("---")
    
    # Data filtering options
    st.sidebar.markdown("### 🔍 Data Filters")
    
    # Country filter
    countries = ['All Countries'] + sorted(data['country'].unique().tolist())
    selected_country = st.sidebar.selectbox("**Select Country**", countries)
    
    # Month filter
    months = ['All Months'] + sorted(data['month'].unique().tolist())
    selected_month = st.sidebar.selectbox("**Select Month**", months)
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Select different visualizations to explore climate data patterns**")

    # Filter data based on selections
    filtered_data = data
    if selected_country != 'All Countries':
        filtered_data = filtered_data[filtered_data['country'] == selected_country]
    if selected_month != 'All Months':
        filtered_data = filtered_data[filtered_data['month'] == selected_month]
    
    # STATISTICAL OVERVIEW
    if analysis_type == "📈 Statistical Overview":
        st.header("📈 Statistical Overview")
        
        viz_option = st.selectbox(
            "Select Visualization:",
            ["Mean vs Median Comparison",
             "Standard Deviation Analysis",
             "Mean and Standard Deviation Combined",
             "Min and Max Comparison"]
        )
        
        if viz_option == "Mean vs Median Comparison":
            st.subheader("📊 Mean vs Median Comparison for Weather Variables")
            
            stats_data = pd.DataFrame({
                'Mean': filtered_data[numeric_cols].mean().round(2),
                'Median': filtered_data[numeric_cols].median().round(2)
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=stats_data.index, y=stats_data['Mean'], 
                                name='Mean', marker_color='#FB923C'))
            fig.add_trace(go.Bar(x=stats_data.index, y=stats_data['Median'], 
                                name='Median', marker_color='#3B82F6'))
            
            fig.update_layout(
                title='Mean vs Median Comparison',
                xaxis_title='Variables', yaxis_title='Value',
                barmode='group', template='plotly_white',
                plot_bgcolor='rgba(250,250,250,1)',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Standard Deviation Analysis":
            st.subheader("📊 Standard Deviation of All Numeric Weather Variables")
            
            std_values = filtered_data[numeric_cols].std().sort_values(ascending=False)
            std_data = pd.DataFrame({'Variable': std_values.index, 'Standard Deviation': std_values.values})
            
            colors = ['#FB923C', '#3B82F6', '#14B8A6', '#6366F1', '#A855F7', '#EF4444', '#10B981', '#8B5CF6', '#F59E0B']
            fig = px.bar(std_data, x='Variable', y='Standard Deviation',
                        text='Standard Deviation', color='Variable',
                        color_discrete_sequence=colors,
                        title='Standard Deviation Analysis')
            
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(template='plotly_white', showlegend=False, height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Mean and Standard Deviation Combined":
            st.subheader("📊 Mean and Standard Deviation Combined")
            
            summary_data = pd.DataFrame({
                'Mean': filtered_data[numeric_cols].mean().round(2),
                'Std Dev': filtered_data[numeric_cols].std().round(2)
            }).reset_index().rename(columns={'index': 'Variable'})
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Mean', x=summary_data['Variable'], y=summary_data['Mean'],
                                marker_color='#FB923C', text=summary_data['Mean'], textposition='outside'))
            fig.add_trace(go.Bar(name='Standard Deviation', x=summary_data['Variable'], y=summary_data['Std Dev'],
                                marker_color='#3B82F6', text=summary_data['Std Dev'], textposition='outside'))
            
            fig.update_layout(title='Mean and Standard Deviation', barmode='group',
                            template='plotly_white', height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Min and Max Comparison":
            st.subheader("📊 Min and Max of Weather Variables")
            
            summary_data = pd.DataFrame({
                'Min': filtered_data[numeric_cols].min().round(2),
                'Max': filtered_data[numeric_cols].max().round(2)
            }).reset_index().rename(columns={'index': 'Variable'})
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Min', x=summary_data['Variable'], y=summary_data['Min'],
                                marker_color='#14B8A6', text=summary_data['Min'], textposition='outside'))
            fig.add_trace(go.Bar(name='Max', x=summary_data['Variable'], y=summary_data['Max'],
                                marker_color='#EF4444', text=summary_data['Max'], textposition='outside'))
            
            fig.update_layout(title='Min and Max Comparison', barmode='group',
                            template='plotly_white', height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    # DISTRIBUTION ANALYSIS
    elif analysis_type == "📊 Distribution Analysis":
        st.header("📊 Distribution Analysis")
        
        viz_option = st.selectbox(
            "Select Visualization:",
            ["Individual Variable Distribution",
             "Violin Plot (All Variables)",
             "Variable Stability Analysis"]
        )
        
        if viz_option == "Individual Variable Distribution":
            selected_var = st.selectbox("Select Variable:", key_cols)
            
            gradient_sets = {
                'temperature_celsius': ['#FB923C', '#FDBA74', '#FED7AA'],
                'humidity': ['#3B82F6', '#60A5FA', '#93C5FD'],
                'wind_kph': ['#14B8A6', '#34D399', '#6EE7B7'],
                'pressure_mb': ['#6366F1', '#8B5CF6', '#A78BFA'],
                'precip_mm': ['#A855F7', '#C084FC', '#D8B4FE']
            }
            
            fig = px.histogram(filtered_data, x=selected_var, nbins=40, marginal="box",
                              title=f"Distribution of {selected_var}",
                              color_discrete_sequence=gradient_sets.get(selected_var, ['#FB923C']))
            
            fig.update_layout(template='plotly_white', height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Violin Plot (All Variables)":
            st.subheader("🎻 Distribution & Spread of Weather Variables")
            
            fig = px.violin(filtered_data, y=key_cols, box=True, points="outliers",
                           color_discrete_sequence=['#FB923C', '#3B82F6', '#14B8A6', '#6366F1', '#A855F7'],
                           title="Violin Plot - Distribution Analysis")
            
            fig.update_layout(template='plotly_white', height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Variable Stability Analysis":
            st.subheader("📊 Variable Stability vs Volatility")
            
            std_values = filtered_data[key_cols].std().sort_values(ascending=False)
            std_data = pd.DataFrame({'Variable': std_values.index, 'Std Dev': std_values.values})
            
            fig = px.bar(std_data, x='Variable', y='Std Dev', color='Variable',
                        color_discrete_sequence=['#FB923C', '#3B82F6', '#14B8A6', '#6366F1', '#A855F7'],
                        title="Variable Stability Analysis")
            
            fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
            fig.update_layout(template='plotly_white', showlegend=False, height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    # CORRELATION ANALYSIS
    elif analysis_type == "🔗 Correlation Analysis":
        st.header("🔗 Correlation Analysis")
        
        viz_option = st.selectbox(
            "Select Visualization:",
            ["Full Correlation Heatmap",
             "Focused Correlation (Key Variables)"]
        )
        
        if viz_option == "Full Correlation Heatmap":
            st.subheader("🌍 Correlation Heatmap of Weather Variables")
            
            corr_matrix = filtered_data[numeric_cols].corr().round(2)
            
            fig = ff.create_annotated_heatmap(
                z=corr_matrix.values, x=list(corr_matrix.columns), y=list(corr_matrix.index),
                colorscale='Viridis', showscale=True,
                annotation_text=corr_matrix.values.round(2)
            )
            
            fig.update_layout(title='Correlation Heatmap', template='plotly_white',
                            width=950, height=950)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Focused Correlation (Key Variables)":
            st.subheader("🔍 Focused Correlation Heatmap")
            
            corr_matrix = filtered_data[key_cols].corr()
            
            fig = ff.create_annotated_heatmap(
                z=corr_matrix.values, x=key_cols, y=key_cols,
                colorscale='Plasma', showscale=True,
                annotation_text=corr_matrix.round(2).values
            )
            
            fig.update_layout(title='Key Variables Correlation', template='plotly_white',
                            width=800, height=800)
            st.plotly_chart(fig, use_container_width=True)
    
    # MONTHLY TRENDS
    elif analysis_type == "📅 Monthly Trends":
        st.header("📅 Monthly Trends Analysis")
        
        if 'month' not in filtered_data.columns:
            filtered_data['month'] = pd.to_datetime(filtered_data['date']).dt.month
        
        viz_option = st.selectbox(
            "Select Visualization:",
            ["Temperature Trends",
             "Humidity Trends",
             "Precipitation Trends",
             "Wind Speed Trends",
             "Pressure Trends"]
        )
        
        top_countries = filtered_data['country'].value_counts().nlargest(5).index
        
        if viz_option == "Temperature Trends":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌡️ Global Average Temperature")
                monthly_temp = filtered_data.groupby('month')['temperature_celsius'].mean().reset_index()
                fig = px.line(monthly_temp, x='month', y='temperature_celsius', markers=True,
                             color_discrete_sequence=['#FB923C'])
                fig.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🌍 Temperature by Country")
                subset = filtered_data[filtered_data['country'].isin(top_countries)]
                fig = px.line(subset, x='month', y='temperature_celsius', color='country',
                             markers=True, line_shape='spline',
                             color_discrete_sequence=['#FB923C', '#3B82F6', '#14B8A6', '#6366F1', '#A855F7'])
                fig.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Humidity Trends":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💧 Global Average Humidity")
                monthly_humidity = filtered_data.groupby('month')['humidity'].mean().reset_index()
                fig = px.line(monthly_humidity, x='month', y='humidity', markers=True,
                             color_discrete_sequence=['#3B82F6'])
                fig.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("💧 Humidity by Country")
                subset = filtered_data[filtered_data['country'].isin(top_countries)]
                fig = px.line(subset, x='month', y='humidity', color='country',
                             markers=True, line_shape='spline',
                             color_discrete_sequence=['#3B82F6', '#FB923C', '#14B8A6', '#6366F1', '#A855F7'])
                fig.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Precipitation Trends":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌧️ Global Average Precipitation")
                monthly_precip = filtered_data.groupby('month')['precip_mm'].mean().reset_index()
                fig = px.line(monthly_precip, x='month', y='precip_mm', markers=True,
                             color_discrete_sequence=['#A855F7'])
                fig.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🌧️ Precipitation by Country")
                subset = filtered_data[filtered_data['country'].isin(top_countries)]
                fig = px.line(subset, x='month', y='precip_mm', color='country',
                             markers=True, line_shape='spline',
                             color_discrete_sequence=['#A855F7', '#FB923C', '#3B82F6', '#14B8A6', '#6366F1'])
                fig.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Wind Speed Trends":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💨 Global Average Wind Speed")
                monthly_wind = filtered_data.groupby('month')['wind_kph'].mean().reset_index()
                fig = px.line(monthly_wind, x='month', y='wind_kph', markers=True,
                             color_discrete_sequence=['#14B8A6'])
                fig.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("💨 Wind Speed by Country")
                subset = filtered_data[filtered_data['country'].isin(top_countries)]
                fig = px.line(subset, x='month', y='wind_kph', color='country',
                             markers=True, line_shape='spline',
                             color_discrete_sequence=['#14B8A6', '#FB923C', '#3B82F6', '#6366F1', '#A855F7'])
                fig.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Pressure Trends":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚖️ Global Average Pressure")
                monthly_pressure = filtered_data.groupby('month')['pressure_mb'].mean().reset_index()
                fig = px.line(monthly_pressure, x='month', y='pressure_mb', markers=True,
                             color_discrete_sequence=['#6366F1'])
                fig.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("⚖️ Pressure by Country")
                subset = filtered_data[filtered_data['country'].isin(top_countries)]
                fig = px.line(subset, x='month', y='pressure_mb', color='country',
                             markers=True, line_shape='spline',
                             color_discrete_sequence=['#6366F1', '#FB923C', '#3B82F6', '#14B8A6', '#A855F7'])
                fig.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # AIR QUALITY ANALYSIS
    elif analysis_type == "💨 Air Quality Analysis":
        st.header("💨 Air Quality Analysis")
        
        viz_option = st.selectbox(
            "Select Visualization:",
            ["Interactive Global Air Quality Map",
             "Interactive Global Weather Map", 
             "Monthly Pollutant Heatmap",
             "Country Comparison",
             "Monthly Pollutant Trends"]
        )
        
        if viz_option == "Interactive Global Air Quality Map":
            st.subheader("🗺️ Interactive Global Air Quality Map")
            
            # Compute mean pollutant levels per country
            air_map_data = dataMonth.groupby('country')[pollutants].mean().reset_index()

            # Create initial choropleth (default pollutant)
            fig = go.Figure()

            fig.add_trace(go.Choropleth(
                locations=air_map_data['country'],
                locationmode='country names',
                z=air_map_data['air_quality_PM2.5'],
                colorscale='Viridis',
                colorbar_title="PM2.5 (µg/m³)",
            ))
            
            # Add dropdown menu for pollutant selection
            fig.update_layout(
                updatemenus=[
                    dict(
                        buttons=[
                            dict(
                                label=pollutant.replace("air_quality_", "").replace("_", " "),
                                method='update',
                                args=[
                                    {'z': [air_map_data[pollutant]]},
                                    {'colorbar.title.text': pollutant.replace("air_quality_", "").replace("_", " ")}
                                ]
                            )
                            for pollutant in pollutants
                        ],
                        direction='down',
                        showactive=True,
                        x=0.05,
                        y=1.15,
                        xanchor='left',
                        yanchor='top'
                    )
                ],
                title=dict(
                    text="Global Air Quality Levels (Select Pollutant)",
                    x=0.5,
                    font=dict(size=22, color='#0A043C')
                ),
                geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'),
                template='plotly_white',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Interactive Global Weather Map":
            st.subheader("🗺️ Interactive Global Weather Conditions Map")
            
            # Compute average weather parameters per country
            weather_params = ['temperature_celsius', 'humidity', 'pressure_mb', 'precip_mm']
            weather_data = dataMonth.groupby('country')[weather_params].mean().reset_index()

            # Create base figure (default: temperature)
            fig = go.Figure()

            fig.add_trace(go.Choropleth(
                locations=weather_data['country'],
                locationmode='country names',
                z=weather_data['temperature_celsius'],
                colorscale='Plasma',
                colorbar_title="Temperature (°C)",
            ))

            # Add dropdown menu
            fig.update_layout(
                updatemenus=[
                    dict(
                        buttons=[
                            dict(
                                label=param.replace("_", " ").title(),
                                method='update',
                                args=[
                                    {'z': [weather_data[param]]},
                                    {'colorbar.title.text': param.replace("_", " ").title()}
                                ]
                            )
                            for param in weather_params
                        ],
                        direction='down',
                        showactive=True,
                        x=0.05,
                        y=1.15,
                        xanchor='left',
                        yanchor='top'
                    )
                ],
                title=dict(
                    text="Global Weather Conditions (Select Parameter)",
                    x=0.5,
                    font=dict(size=22, color='#0A043C')
                ),
                geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'),
                template='plotly_white',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Monthly Pollutant Heatmap":
            st.subheader("🗓️ Monthly Air Pollutant Concentrations Heatmap")
            
            # Compute monthly averages
            monthly_air_matrix = dataMonth.groupby('month')[pollutants].mean().T

            fig = ff.create_annotated_heatmap(
                z=monthly_air_matrix.values,
                x=list(range(1, 13)),
                y=list(monthly_air_matrix.index),
                colorscale='Plasma',
                annotation_text=monthly_air_matrix.round(1).values,
                showscale=True
            )

            fig.update_layout(
                title="Monthly Air Pollutant Concentrations (Heatmap)",
                template='plotly_white',
                title_font=dict(size=22, color='#0A043C'),
                width=900,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Country Comparison":
            st.subheader("🌍 Comparison of Air Pollutants Across Top Countries")
            
            # Average per country
            country_air = dataMonth.groupby('country')[pollutants].mean().reset_index()
            top_countries = dataMonth['country'].value_counts().nlargest(5).index
            country_air = country_air[country_air['country'].isin(top_countries)]

            # Melt for Plotly
            country_air_melted = country_air.melt(id_vars='country', var_name='Pollutant', value_name='Concentration')

            fig = px.bar(
                country_air_melted,
                x='country',
                y='Concentration',
                color='Pollutant',
                barmode='group',
                title="Comparison of Air Pollutants Across Top 5 Countries",
                color_discrete_sequence=['#FB923C', '#3B82F6', '#14B8A6', '#6366F1', '#A855F7', '#EF4444']
            )

            fig.update_layout(
                xaxis_title="Country",
                yaxis_title="Average Concentration",
                template='plotly_white',
                title_font=dict(size=22, color='#0A043C'),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Monthly Pollutant Trends":
            st.subheader("📈 Monthly Trends of All Air Pollutants")
            
            monthly_air = dataMonth.groupby('month')[pollutants].mean().reset_index()
            air_melted = monthly_air.melt(id_vars='month', var_name='Pollutant', value_name='Concentration')
            
            fig = px.line(air_melted, x='month', y='Concentration', color='Pollutant',
                         markers=True, line_shape='spline',
                         color_discrete_sequence=['#FB923C', '#3B82F6', '#14B8A6', '#6366F1', '#A855F7', '#EF4444'])
            
            fig.update_layout(template='plotly_white', height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    # GEOGRAPHIC ANALYSIS
    elif analysis_type == "🗺️ Geographic Analysis":
        st.header("🗺️ Geographic Analysis")
        
        viz_option = st.selectbox(
            "Select Visualization:",
            ["Top 10 Countries by Rainfall",
             "Top Hottest Days",
             "Top Coldest Days"]
        )
        
        if viz_option == "Top 10 Countries by Rainfall":
            st.subheader("🌧️ Top 10 Countries by Average Rainfall")
            
            avg_rainfall = filtered_data.groupby('country')['precip_mm'].mean().reset_index()
            top10_rainfall = avg_rainfall.sort_values(by='precip_mm', ascending=False).head(10)
            
            fig = px.bar(top10_rainfall, x='country', y='precip_mm',
                        color='precip_mm', color_continuous_scale='Blues')
            
            fig.update_layout(template='plotly_white', height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Top Hottest Days":
            st.subheader("🔥 Top 5 Hottest Recorded Days by Country")
            
            top5_hottest = (filtered_data.loc[filtered_data.groupby('country')['temperature_celsius'].idxmax()]
                           .sort_values(by='temperature_celsius', ascending=False).head(5))
            
            fig = px.bar(top5_hottest, x='country', y='temperature_celsius',
                        color='temperature_celsius', text='location_name',
                        hover_data=['date', 'location_name'],
                        color_continuous_scale='Reds')
            
            fig.update_traces(textposition='outside')
            fig.update_layout(template='plotly_white', height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Top Coldest Days":
            st.subheader("❄️ Top 5 Coldest Recorded Days by Country")
            
            top5_coldest = (filtered_data.loc[filtered_data.groupby('country')['temperature_celsius'].idxmin()]
                           .sort_values(by='temperature_celsius', ascending=True).head(5))
            
            fig = px.bar(top5_coldest, x='country', y='temperature_celsius',
                        color='temperature_celsius', text='location_name',
                        hover_data=['date', 'location_name'],
                        color_continuous_scale='Blues_r')
            
            fig.update_traces(textposition='outside')
            fig.update_layout(template='plotly_white', height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    # EXTREME WEATHER EVENTS
    elif analysis_type == "🌡️ Extreme Weather Events":
        st.header("🌡️ Extreme Weather Events")
        
        st.subheader("⚠️ Extreme Weather Events Around the World")
        
        # Calculate thresholds
        temp_high = filtered_data['temperature_celsius'].mean() + 2 * filtered_data['temperature_celsius'].std()
        rain_high = filtered_data['precip_mm'].mean() + 2 * filtered_data['precip_mm'].std()
        wind_high = filtered_data['wind_kph'].mean() + 2 * filtered_data['wind_kph'].std()
        
        # Filter extremes
        extreme_temps = filtered_data[filtered_data['temperature_celsius'] > temp_high].assign(event='High Temperature')
        extreme_rain = filtered_data[filtered_data['precip_mm'] > rain_high].assign(event='Heavy Rainfall')
        extreme_wind = filtered_data[filtered_data['wind_kph'] > wind_high].assign(event='High Wind')
        
        extreme_events = pd.concat([extreme_temps, extreme_rain, extreme_wind])
        
        fig = px.scatter_geo(extreme_events, lat='latitude', lon='longitude',
                            color='event', hover_name='location_name',
                            color_discrete_map={
                                'High Temperature': '#FB923C',
                                'Heavy Rainfall': '#3B82F6',
                                'High Wind': '#14B8A6'
                            })
        
        fig.update_traces(marker=dict(size=6, opacity=0.8))
        fig.update_layout(template='plotly_white', height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔥 High Temperature Events", len(extreme_temps))
        with col2:
            st.metric("🌧️ Heavy Rainfall Events", len(extreme_rain))
        with col3:
            st.metric("💨 High Wind Events", len(extreme_wind))

else:
    st.error("❌ Unable to load data files. Please make sure 'CleanedWeatherRepository.csv' and 'CleanedWeatherRepositoryMonthly.csv' are in the same directory as this script.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%); 
            border-radius: 10px; color: #4B5563;'>
    📊 Dataset: Global Climate Data 2024 | Last Updated: October 2025 | 
    <span style='color: #7C3AED; font-weight: 600;'>ClimateScope Analytics Platform</span>
</div>
""", unsafe_allow_html=True)