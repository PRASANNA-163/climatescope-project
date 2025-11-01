import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats # Import for statistical test in Lunar section

# --- CORE CONFIGURATION AND STYLING ---

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
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1E293B;
        font-weight: 700;
    }
    /* Unique style for subheaders */
    h3 {
        color: #4C1D95; /* Deep Violet */
        border-bottom: 2px solid #C4B5FD;
        padding-bottom: 5px;
        margin-top: 15px;
    }
    
    /* MODIFIED: Smart Insights Panel Styling for Uniform Size and Gradient */
    .smart-insight-box {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%); 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #F59E0B;
        margin-bottom: 15px;
        /* Using a height suitable for its content in the Home Dashboard */
    }
    
    /* NEW: Custom Gradient Card for Specific Insights (Hottest, Windiest, Wettest) - UNIFORM SIZE */
    .insight-card-hot {
        background: linear-gradient(45deg, #FF7E5F 0%, #FEB47B 100%); /* Sunset Orange */
        padding: 15px; 
        border-radius: 12px;
        border-left: 5px solid #FF5733;
        margin-bottom: 15px;
        height: 140px; /* Uniform height */
        display: flex;
        flex-direction: column;
        justify-content: center;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        font-size: 16px;
    }
    .insight-card-wind {
        background: linear-gradient(45deg, #4ECDC4 0%, #556270 100%); /* Teal/Slate Grey */
        padding: 15px; 
        border-radius: 12px;
        border-left: 5px solid #00B4DB;
        margin-bottom: 15px;
        height: 140px; /* Uniform height */
        display: flex;
        flex-direction: column;
        justify-content: center;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        font-size: 16px;
    }
    .insight-card-rain {
        background: linear-gradient(45deg, #4776E6 0%, #8E54E9 100%); /* Blue/Violet */
        padding: 15px; 
        border-radius: 12px;
        border-left: 5px solid #47A9DA;
        margin-bottom: 15px;
        height: 140px; /* Uniform height */
        display: flex;
        flex-direction: column;
        justify-content: center;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        font-size: 16px;
    }
    .kpi-card {
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* NEW: Travel Comfort Card Styles */
    .comfort-card-good {
        background: linear-gradient(135deg, #ECFDF5 0%, #10B981 100%);
        padding: 15px;
        border-radius: 12px;
        color: #1F2937;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.2);
        border: 1px solid #059669;
        text-align: center;
    }
    .comfort-card-bad {
        background: linear-gradient(135deg, #FEF2F2 0%, #F87171 100%);
        padding: 15px;
        border-radius: 12px;
        color: #1F2937;
        box-shadow: 0 4px 10px rgba(248, 113, 113, 0.2);
        border: 1px solid #EF4444;
        text-align: center;
    }
    .activity-card {
        background: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    .moon-card {
        background: linear-gradient(135deg, #374151 0%, #111827 100%);
        padding: 15px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.5);
    }
    
    /* Fix for clumpy text in Plotly heatmap (only affects non-annotated heatmaps, but keep for completeness) */
    .js-plotly-plot .textpoint {
        white-space: pre-wrap !important;
    }
</style>
""", unsafe_allow_html=True)

# Header (Retained)
st.markdown("""
<div style='background: linear-gradient(90deg, #2563EB 0%, #9333EA 50%, #EC4899 100%); 
             padding: 30px; border-radius: 15px; margin-bottom: 30px;'>
    <h1 style='color: white; margin: 0;'>☀️ ClimateScope Dashboard 🌎</h1>
    <p style='color: #E0E7FF; margin: 5px 0 0 0; font-size: 16px;'>
        Interactive Global Climate & Air Quality Analytics: Exploring Trends and Anomalies
    </p>
</div>
""", unsafe_allow_html=True)

# --- ADVANCED VISUALIZATION FUNCTIONS (EXISTING) ---

@st.cache_data
def create_3d_scatter(data):
    # Sample data for performance
    sample_size = min(5000, len(data))
    sample_data = data.sample(sample_size, random_state=42)
    
    fig = px.scatter_3d(
        sample_data,
        x='temperature_celsius',
        y='humidity',
        z='pressure_mb',
        color='precip_mm',
        size='wind_kph',
        opacity=0.8,
        color_continuous_scale=px.colors.sequential.Rainbow,
        hover_data={'country': True, 'location_name': True, 'date': True, 'temperature_celsius': ':.1f', 'humidity': ':.1f', 'pressure_mb': ':.1f'},
        title='3D Correlation: Temp vs Humidity vs Pressure (Color=Precip, Size=Wind)'
    )
    
    fig.update_layout(
        template='plotly_white',
        height=700,
        scene=dict(
            xaxis_title='Temperature (°C)',
            yaxis_title='Humidity (%)',
            zaxis_title='Pressure (mb)'
        )
    )
    return fig

@st.cache_data
def create_animated_evolution(data):
    # Ensure we have month/year/temp for the animation frame
    data['year_month'] = data['date'].dt.to_period('M').astype(str)
    
    agg_data = data.groupby(['year_month', 'location_name', 'country']).agg(
        avg_temp=('temperature_celsius', 'mean'),
        avg_humidity=('humidity', 'mean'),
        avg_pressure=('pressure_mb', 'mean'),
        avg_wind=('wind_kph', 'mean'),
        avg_precip=('precip_mm', 'mean')
    ).reset_index()
    
    sample_data = agg_data.sample(min(3000, len(agg_data)), random_state=42)

    min_temp, max_temp = agg_data['avg_temp'].min() - 1, agg_data['avg_temp'].max() + 1
    min_humidity, max_humidity = agg_data['avg_humidity'].min() - 5, agg_data['avg_humidity'].max() + 5
    
    fig = px.scatter(
        sample_data,
        x='avg_temp',
        y='avg_humidity',
        animation_frame='year_month',
        color='avg_wind',
        size='avg_precip',
        hover_name='location_name',
        log_x=False,
        size_max=50,
        range_x=[min_temp, max_temp],
        range_y=[min_humidity, max_humidity],
        color_continuous_scale=px.colors.sequential.Plasma,
        title='Temperature vs Humidity Evolution (Color=Wind, Size=Precip)'
    )
    
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500
    fig.update_layout(template='plotly_white', height=650)
    
    return fig

# --- DATA LOADING AND PREPROCESSING (EXISTING) ---

@st.cache_data
def load_data():
    try:
        data = pd.read_csv('CleanedWeatherRepository.csv')
        dataMonth = pd.read_csv('CleanedWeatherRepositoryMonthly.csv')
        
        # --- Essential Preprocessing ---
        data['date'] = pd.to_datetime(data['date'])
        
        if 'month' not in data.columns:
            data['month'] = data['date'].dt.month
        data['month_name'] = data['month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))

        # Prepare dataMonth for correct plotting
        if 'month' not in dataMonth.columns:
            dataMonth['month'] = dataMonth['date'].apply(lambda x: pd.to_datetime(x).month) if 'date' in dataMonth.columns else dataMonth['month']
        dataMonth['month_name'] = dataMonth['month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
        
        # NEW: Parse Time Columns (Needed for Lunar/Travel Sections)
        if 'sunrise' in data.columns:
            # Safely handle potential mixed formats if necessary, but use existing conversion for consistency
            data['sunrise_dt'] = pd.to_datetime(data['sunrise'], format='%I:%M %p').dt.time
        if 'sunset' in data.columns:
            data['sunset_dt'] = pd.to_datetime(data['sunset'], format='%I:%M %p').dt.time
            data['sunset_datetime'] = data.apply(lambda row: datetime.combine(row['date'].date(), row['sunset_dt']), axis=1)
            data['sunrise_datetime'] = data.apply(lambda row: datetime.combine(row['date'].date(), row['sunrise_dt']), axis=1)
            # Ensure the order is correct to get a positive difference for daylight hours
            data['daylight_hours'] = (data['sunset_datetime'] - data['sunrise_datetime']).dt.total_seconds() / 3600
            
        return data, dataMonth
    except Exception as e:
        st.error(f"⚠️ Error loading data files: {e}")
        st.info("Please make sure 'CleanedWeatherRepository.csv' and 'CleanedWeatherRepositoryMonthly.csv' are in the same directory as this script.")
        return None, None

data, dataMonth = load_data()

if data is not None and dataMonth is not None:
    # Define columns (EXISTING)
    numeric_cols = ['temperature_celsius', 'feels_like_celsius', 'humidity', 'wind_kph', 
                    'gust_kph', 'pressure_mb', 'precip_mm', 'air_quality_PM2.5', 'air_quality_PM10']
    key_cols = ['temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm']
    pollutants = ['air_quality_Carbon_Monoxide', 'air_quality_Nitrogen_dioxide',
                  'air_quality_Sulphur_dioxide', 'air_quality_Ozone',
                  'air_quality_PM2.5', 'air_quality_PM10']
    

    # --- SIDEBAR NAVIGATION AND FILTERS (MODIFIED) ---
    st.sidebar.title("🎛️ Navigation")
    
    analysis_type = st.sidebar.radio(
        "Select Analysis Type:",
        ["🏠 **Home Dashboard**", 
         "📈 Statistical Overview", 
         "📊 Distribution Analysis", 
         "🔗 Correlation Analysis",
         "📅 Monthly Trends",
         "✨ **Advanced Visualizations**", 
         "🎯 **Climate Comparison**",
         "💨 Air Quality Analysis",
         "🗺️ Geographic Analysis",
         "🌡️ Extreme Weather Events",
         "🔬 **Pattern Recognition**",
         "🧳 **Travel & Lifestyle Recommendations**", # <--- NEW SECTION 1
         "🌙 **Lunar & Astronomical Insights**"     # <--- NEW SECTION 2
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Data Filters (Interactive)")
    
    # 1. Date Range Filter (MODIFIED to allow future date selection)
    min_date = data['date'].min().date()
    max_data_date = data['date'].max().date()
    
    # Allow selection up to 5 years into the future for planning purposes
    max_filter_date = max(max_data_date, datetime.now().date() + timedelta(days=5 * 365))
    
    date_range = st.sidebar.date_input(
        "**Select Date Range**", 
        (min_date, max_data_date), 
        min_value=min_date, 
        max_value=max_filter_date # Allow future dates
    )
    
    # 2. Country filter (EXISTING)
    countries = ['All Countries'] + sorted(data['country'].unique().tolist())
    selected_country = st.sidebar.selectbox("**Select Country**", countries) # <--- Variable defined here
    
    # 3. Month filter (EXISTING)
    months = ['All Months'] + ['January', 'February', 'March', 'April', 'May', 'June', 
                              'July', 'August', 'September', 'October', 'November', 'December']
    selected_month_name = st.sidebar.selectbox("**Select Month**", months)

    # 4. Temperature Range Filter (EXISTING)
    temp_min_global = data['temperature_celsius'].min()
    temp_max_global = data['temperature_celsius'].max()
    temp_range_filter = st.sidebar.slider(
        "**Filter by Temperature Range (°C)**",
        float(temp_min_global), float(temp_max_global), (float(temp_min_global), float(temp_max_global)), step=1.0
    )

    # 5. Climate Zone Filter (EXISTING)
    climate_zone_options = ['All Zones', 'Tropical (>25°C Avg)', 'Subtropical (18-25°C)', 'Temperate (10-18°C)', 'Cold (<10°C)']
    selected_zone = st.sidebar.selectbox("**Filter by Climate Zone**", climate_zone_options)

    # 6. Weather Condition Filter (EXISTING)
    weather_condition_options = ['All Conditions', '☀️ Sunny (Precip < 1mm)', '🌧️ Rainy (Precip > 5mm)', '❄️ Extreme Wind (> 50kph)']
    selected_condition = st.sidebar.selectbox("**Filter by Condition**", weather_condition_options)
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Use filters to refine global and regional patterns.**")

    # --- FILTER APPLICATION LOGIC (EXISTING, with accommodation for future date) ---
    filtered_data = data
    
    # Apply Date, Country, Month, Temperature Range Filters
    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        
        # Only filter for dates present in the *historical* data range
        filtered_data = filtered_data[
            (filtered_data['date'] >= start_date) & 
            (filtered_data['date'] <= end_date)
        ]

    # Note: Country filter applied below, BEFORE other filters to define kpi_data_source correctly
    
    # Apply Month, Temperature, Zone, and Condition Filters (Remaining logic)
    if selected_month_name != 'All Months':
        filtered_data = filtered_data[filtered_data['month_name'] == selected_month_name]

    filtered_data = filtered_data[
        (filtered_data['temperature_celsius'] >= temp_range_filter[0]) & 
        (filtered_data['temperature_celsius'] <= temp_range_filter[1])
    ]
    
    # Apply Climate Zone Filter
    if selected_zone != 'All Zones' and not filtered_data.empty:
        avg_temp_by_loc = filtered_data.groupby('location_name')['temperature_celsius'].mean().reset_index()
        locs = pd.Series([], dtype='object') 
        
        if selected_zone == 'Tropical (>25°C Avg)':
            locs = avg_temp_by_loc[avg_temp_by_loc['temperature_celsius'] > 25]['location_name']
        elif selected_zone == 'Subtropical (18-25°C)':
            locs = avg_temp_by_loc[(avg_temp_by_loc['temperature_celsius'] >= 18) & (avg_temp_by_loc['temperature_celsius'] <= 25)]['location_name']
        elif selected_zone == 'Temperate (10-18°C)':
            locs = avg_temp_by_loc[(avg_temp_by_loc['temperature_celsius'] >= 10) & (avg_temp_by_loc['temperature_celsius'] < 18)]['location_name']
        elif selected_zone == 'Cold (<10°C)':
            locs = avg_temp_by_loc[avg_temp_by_loc['temperature_celsius'] < 10]['location_name']
        
        filtered_data = filtered_data[filtered_data['location_name'].isin(locs)]

    # Apply Weather Condition Filter
    if selected_condition != 'All Conditions' and not filtered_data.empty:
        if selected_condition == '☀️ Sunny (Precip < 1mm)':
            filtered_data = filtered_data[filtered_data['precip_mm'] < 1]
        elif selected_condition == '🌧️ Rainy (Precip > 5mm)':
            filtered_data = filtered_data[filtered_data['precip_mm'] > 5]
        elif selected_condition == '❄️ Extreme Wind (> 50kph)':
            filtered_data = filtered_data[filtered_data['wind_kph'] > 50]
    
    # Final Country Filter for the main view (applied *after* all others for consistency)
    final_filtered_data = filtered_data
    if selected_country != 'All Countries':
        final_filtered_data = final_filtered_data[final_filtered_data['country'] == selected_country]


    # Final check for empty data
    if final_filtered_data.empty:
        st.error("❌ No data available based on the current combination of filters. Please adjust the sidebar filters.")


    # --- KPI CALCULATION AND DISPLAY (MOVED HERE TO FIX NAMEERROR) ---
    
    # Determine the dataset for KPI calculation: 
    # Use filtered_data for country/date specific KPIs, otherwise use full data 
    if selected_country != 'All Countries' or date_range[0] != min_date or date_range[1] != max_data_date:
        # Use the fully filtered data (final_filtered_data) if any filter is active
        kpi_data_source = final_filtered_data
    else:
        # Use the base data if no filters are applied (global average)
        kpi_data_source = data
        
    avg_temp = kpi_data_source['temperature_celsius'].mean() if not kpi_data_source.empty else 0
    avg_humidity = kpi_data_source['humidity'].mean() if not kpi_data_source.empty else 0
    avg_wind = kpi_data_source['wind_kph'].mean() if not kpi_data_source.empty else 0
    avg_precip = kpi_data_source['precip_mm'].mean() if not kpi_data_source.empty else 0
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='kpi-card' style='background: linear-gradient(135deg, #FB923C 0%, #EF4444 100%);'>
            <div style='font-size: 36px; font-weight: bold; color: white;'>{avg_temp:.1f}°C</div>
            <div style='color: #FED7AA; font-weight: 500; margin-top: 5px;'>Avg Temperature</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='kpi-card' style='background: linear-gradient(135deg, #3B82F6 0%, #06B6D4 100%);'>
            <div style='font-size: 36px; font-weight: bold; color: white;'>{avg_humidity:.1f}%</div>
            <div style='color: #BAE6FD; font-weight: 500; margin-top: 5px;'>Avg Humidity</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='kpi-card' style='background: linear-gradient(135deg, #14B8A6 0%, #22C55E 100%);'>
            <div style='font-size: 36px; font-weight: bold; color: white;'>{avg_wind:.1f}</div>
            <div style='color: #A7F3D0; font-weight: 500; margin-top: 5px;'>Avg Wind (km/h)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='kpi-card' style='background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%);'>
            <div style='font-size: 36px; font-weight: bold; color: white;'>{avg_precip:.1f}</div>
            <div style='color: #DDD6FE; font-weight: 500; margin-top: 5px;'>Avg Precip (mm)</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


    # --- ADVANCED HELPER FUNCTIONS (EXISTING) ---

    @st.cache_data
    def calculate_health_scores(df):
        if df.empty or df[numeric_cols].isnull().all().all():
            return 0, 0, 0, 0, "No data"

        # Calculate a simple metric for each score (0-100 scale)
        temp_std = df['temperature_celsius'].std()
        temp_stability_score = max(0, 100 - (temp_std * 5)) 

        pm_mean = df['air_quality_PM2.5'].mean()
        aqi_score = max(0, 100 - (pm_mean * 2)) 

        global_avg_precip = data['precip_mm'].mean()
        local_avg_precip = df['precip_mm'].mean()
        precip_diff_percent = abs(local_avg_precip - global_avg_precip) / global_avg_precip if global_avg_precip else 0
        precip_balance_score = max(0, 100 - (precip_diff_percent * 100)) 

        overall_score = np.mean([temp_stability_score, aqi_score, precip_balance_score])
        
        return int(temp_stability_score), int(aqi_score), int(precip_balance_score), int(overall_score), None

    @st.cache_data
    def generate_smart_insights(df):
        if df.empty:
            return ["No data available for insights."]

        insights = []

        # Insight 1: Hottest day recorded (Corrected Logic)
        hottest_day = df.loc[df['temperature_celsius'].idxmax()]
        insights.append(f"🔥 Hottest day: **{hottest_day['location_name']}, {hottest_day['country']}** at **{hottest_day['temperature_celsius']:.1f}°C** on **{hottest_day['date'].strftime('%Y-%m-%d')}**.")

        # Insight 2: Rainiest Region
        rain_sum_by_country = df.groupby('country')['precip_mm'].sum()
        if not rain_sum_by_country.empty:
            rainiest_country = rain_sum_by_country.idxmax()
            rain_total = rain_sum_by_country.max()
            insights.append(f"🌧️ Rainiest Region: **{rainiest_country}** recorded a total of **{rain_total:.1f}mm** of precipitation in the selected period.")
        else:
            insights.append("🌧️ Rainiest Region: No precipitation data found.")

        # Insight 3: Air Quality Alert
        pm_exceeded_count = df[df['air_quality_PM2.5'] > 25]['country'].nunique() # Threshold > 25 µg/m³
        if pm_exceeded_count > 0:
            insights.append(f"⚠️ Air Quality Alert: **{pm_exceeded_count} countries** recorded days with PM2.5 levels exceeding the warning threshold (> 25 µg/m³).")
        else:
            insights.append("✅ Air Quality: PM2.5 levels remain below the warning threshold across all selected regions.")

        # Insight 4: Windiest Location
        windiest_day = df.loc[df['wind_kph'].idxmax()]
        insights.append(f"💨 Windiest day: **{windiest_day['location_name']}, {windiest_day['country']}** with a wind speed of **{windiest_day['wind_kph']:.1f} kph**.")

        return insights[:5]
    
    # --------------------------------------------------------------------------
    # NEW HELPER FUNCTIONS FOR TRAVEL & LUNAR SECTIONS
    # --------------------------------------------------------------------------

    @st.cache_data
    def calculate_comfort_score(df, temp_ideal, precip_max, aqi_max, humidity_range, wind_max):
        # 1. Prepare data and group
        loc_cols = ['location_name', 'country', 'latitude', 'longitude']
        score_data = df.groupby(loc_cols).agg(
            avg_temp=('temperature_celsius', 'mean'),
            avg_precip=('precip_mm', 'mean'),
            avg_pm25=('air_quality_PM2.5', 'mean'),
            avg_humidity=('humidity', 'mean'),
            avg_wind=('wind_kph', 'mean')
        ).reset_index().dropna()
        
        if score_data.empty:
            return pd.DataFrame()

        # 2. Score Calculation (0-1)
        temp_center = np.mean(temp_ideal)
        temp_score = 1 - (abs(score_data['avg_temp'] - temp_center) / 10)
        temp_score = np.clip(temp_score, 0, 1)

        precip_score = 1 - (score_data['avg_precip'] / precip_max)
        precip_score = np.clip(precip_score, 0, 1)

        aqi_score = 1 - (score_data['avg_pm25'] / aqi_max)
        aqi_score = np.clip(aqi_score, 0, 1)
        
        wind_score = 1 - (score_data['avg_wind'] / wind_max)
        wind_score = np.clip(wind_score, 0, 1)

        humidity_center = np.mean(humidity_range)
        humidity_score = 1 - (abs(score_data['avg_humidity'] - humidity_center) / 30)
        humidity_score = np.clip(humidity_score, 0, 1)

        # 3. Combine scores (0-100)
        score_data['Comfort_Score'] = ((temp_score + precip_score + aqi_score + wind_score + humidity_score) / 5) * 100
        
        return score_data.sort_values('Comfort_Score', ascending=False)

    @st.cache_data
    def get_moon_phase_emoji(illumination):
        if illumination < 10:
            return '🌑 New Moon'
        elif illumination < 40:
            return '🌒 Waxing Crescent'
        elif illumination < 60:
            return '🌓 First Quarter'
        elif illumination < 90:
            return '🌔 Waxing Gibbous'
        elif illumination <= 100:
            return '🌕 Full Moon'
        else:
            return '✨ Unknown'

    # --------------------------------------------------------------------------
    # MAIN CONTENT BLOCKS (EXISTING)
    # --------------------------------------------------------------------------

    # HOME DASHBOARD SECTION
    if analysis_type == "🏠 **Home Dashboard**":
        # ... (EXISTING CODE FOR HOME DASHBOARD) ...
        st.header("🏠 Global Climate Health and Insights")
        
        # Use final_filtered_data here
        if final_filtered_data.empty:
            st.info("No data available for the selected filters.")
        else:
            # Smart Insights Panel
            st.subheader("💡 Smart Insights Panel (Top Extremes)")
            
            # --- MODIFIED: USE CUSTOM CSS FOR UNIFORM AND COLORFUL CARDS ---
            
            # Recalculate Extremes
            hottest_day = final_filtered_data.loc[final_filtered_data['temperature_celsius'].idxmax()]
            rain_sum_by_country = final_filtered_data.groupby('country')['precip_mm'].sum()
            rainiest_country = rain_sum_by_country.idxmax() if not rain_sum_by_country.empty else "N/A"
            rain_total = rain_sum_by_country.max() if not rain_sum_by_country.empty else 0
            windiest_day = final_filtered_data.loc[final_filtered_data['wind_kph'].idxmax()]
            
            # Create a list of insights with their card class
            insights_data = [
                {
                    'title': "🔥 Hottest Day Recorded",
                    'value': f"{hottest_day['temperature_celsius']:.1f}°C",
                    'location': f"{hottest_day['location_name']}, {hottest_day['country']} ({hottest_day['date'].strftime('%Y-%m-%d')})",
                    'class': 'insight-card-hot'
                },
                {
                    'title': "💨 Windiest Day Recorded",
                    'value': f"{windiest_day['wind_kph']:.1f} kph",
                    'location': f"{windiest_day['location_name']}, {windiest_day['country']} ({windiest_day['date'].strftime('%Y-%m-%d')})",
                    'class': 'insight-card-wind'
                },
                {
                    'title': "🌧️ Rainiest Region (Total Precip)",
                    'value': f"{rain_total:.1f} mm",
                    'location': f"Region: {rainiest_country}",
                    'class': 'insight-card-rain'
                },
            ]
            
            # Use columns with a fixed gap for presentation - ensures uniformity
            insight_cols = st.columns(len(insights_data), gap="medium")
            
            for i, insight in enumerate(insights_data):
                with insight_cols[i]:
                    st.markdown(f"""
                    <div class='{insight['class']}'>
                        <p style='font-size: 14px; margin: 0; font-weight: 300;'>{insight['title']}</p>
                        <p style='font-size: 32px; font-weight: 700; margin: 5px 0 0 0;'>{insight['value']}</p>
                        <p style='font-size: 12px; margin: 5px 0 0 0; font-weight: 400;'>{insight['location']}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # --- END MODIFIED INSIGHTS ---
            st.markdown("---")
            
            # Climate Health Score Cards (Gauges)
            st.subheader("❤️ Climate Health Score Cards (Simulated Index 0-100)")
            t_score, a_score, p_score, o_score, error = calculate_health_scores(final_filtered_data)

            score_cols = st.columns(4)
            scores = [
                ("Temperature Stability", t_score, '#F472B6', 'The lower the temperature variation, the higher the score.'),
                ("Air Quality Index", a_score, '#34D399', 'Based on PM2.5 levels: Higher score means better air quality.'),
                ("Precipitation Balance", p_score, '#3B82F6', 'Score reflects how close the rainfall is to the long-term global average.'),
                ("Overall Health", o_score, '#6366F1', 'Weighted average of key environmental indicators.'),
            ]
            
            for i, (label, score, color, help_text) in enumerate(scores):
                with score_cols[i]:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"<b>{label}</b>", 'font': {'size': 18, 'color': '#1F2937'}},
                        gauge={'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': 'darkgray'},
                               'bar': {'color': color},
                               'bgcolor': 'white',
                               'steps': [
                                   {'range': [0, 40], 'color': 'red'},
                                   {'range': [40, 70], 'color': 'yellow'},
                                   {'range': [70, 100], 'color': 'green'}
                               ],
                               'threshold': {'line': {'color': "darkred", 'width': 4}, 'thickness': 0.75, 'value': 85}}
                    ))
                    fig.update_layout(height=250, margin=dict(t=50, b=10, l=10, r=10), font={'color': '#1F2937', 'family': "Arial"})
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            
            # Top 5 Extremes Table (Corrected Logic)
            st.subheader("🏆 Top 5 Extreme Locations (All Data)")

            col_h, col_w, col_r = st.columns(3)
            
            with col_h:
                # Use the original (non-uniform) generate_smart_insights logic for the tables below the cards
                hottest = final_filtered_data.sort_values('temperature_celsius', ascending=False).head(5)[['location_name', 'country', 'temperature_celsius', 'date']].reset_index(drop=True)
                st.markdown("🌡️ **Hottest Locations**")
                st.dataframe(hottest.rename(columns={'temperature_celsius': 'Temp (°C)', 'location_name': 'Location'}), use_container_width=True, hide_index=True)
            
            with col_w:
                windiest = final_filtered_data.sort_values('wind_kph', ascending=False).head(5)[['location_name', 'country', 'wind_kph', 'date']].reset_index(drop=True)
                st.markdown("💨 **Windiest Locations**")
                st.dataframe(windiest.rename(columns={'wind_kph': 'Wind (kph)', 'location_name': 'Location'}), use_container_width=True, hide_index=True)

            with col_r:
                wettest = final_filtered_data.sort_values('precip_mm', ascending=False).head(5)[['location_name', 'country', 'precip_mm', 'date']].reset_index(drop=True)
                st.markdown("🌧️ **Wettest Locations**")
                st.dataframe(wettest.rename(columns={'precip_mm': 'Precip (mm)', 'location_name': 'Location'}), use_container_width=True, hide_index=True)


    # -------------------------------------------------------------------------
    # NEW SECTION 1: TRAVEL & LIFESTYLE RECOMMENDATIONS (PROMPT 1)
    # -------------------------------------------------------------------------

    elif analysis_type == "🧳 **Travel & Lifestyle Recommendations**":
        st.header("🧳 Travel & Lifestyle Recommendations")
        
        # Use final_filtered_data here
        if final_filtered_data.empty:
            st.error("❌ No data available based on the current filters. Please adjust the sidebar filters.")
            
        else:
            # --- Feature 1: Best Travel Destinations Finder ---
            st.subheader("🏆 Best Travel Destinations Finder (Comfort Score)")
            
            col_finder, col_settings = st.columns([3, 1])
            with col_settings:
                # Custom filters for scoring
                st.markdown("**Scoring Filters**")
                temp_range = st.slider("Ideal Temperature Range (°C)", 
                                       float(temp_min_global), float(temp_max_global), (18.0, 26.0), step=1.0)
                precip_max = st.slider("Max Avg Daily Precip (mm)", 0.0, 20.0, 5.0, step=0.5)
                aqi_max = st.slider("Max Avg PM2.5 (µg/m³)", 5.0, 50.0, 15.0, step=1.0)
                humidity_range = st.slider("Ideal Humidity (%)", 0, 100, (40, 70), step=5)
                wind_max = st.slider("Max Avg Wind Speed (kph)", 10.0, 60.0, 25.0, step=1.0)
                
            with col_finder:
                comfort_data = calculate_comfort_score(final_filtered_data, temp_range, precip_max, aqi_max, humidity_range, wind_max)
                
                if comfort_data.empty:
                    st.warning("No locations match the scoring filters. Try broadening the ranges.")
                else:
                    st.markdown(f"**Top 10 Locations for Comfort (Score 0-100):** (Based on {len(comfort_data)} unique locations)")
                    top_10 = comfort_data[['location_name', 'country', 'Comfort_Score', 'avg_temp', 'avg_precip', 'avg_pm25', 'avg_humidity', 'avg_wind']].head(10)
                    
                    st.dataframe(
                        top_10.rename(columns={
                            'Comfort_Score': 'Score',
                            'avg_temp': 'Temp (°C)',
                            'avg_precip': 'Precip (mm)',
                            'avg_pm25': 'PM2.5',
                            'avg_humidity': 'Humidity (%)',
                            'avg_wind': 'Wind (kph)',
                            'location_name': 'Location'
                        }).round(1),
                        use_container_width=True,
                        hide_index=True
                    )
            
            st.markdown("---")
            
            # --- Feature 2: Climate Comfort Calendar ---
            st.subheader("🗓️ Climate Comfort Calendar (Monthly Comparison)")
            
            available_countries = final_filtered_data['country'].unique().tolist()
            if len(available_countries) < 2:
                st.info("Select more countries in the sidebar to use the comparison calendar.")
            else:
                comp_countries = st.multiselect("Select Countries to Compare (Max 5)", available_countries, default=available_countries[:min(5, len(available_countries))])
                
                if comp_countries:
                    # Use original data filtered by selected countries for a full monthly view
                    calendar_data = data[data['country'].isin(comp_countries)].groupby(['country', 'month_name']).agg(
                        avg_temp=('temperature_celsius', 'mean'),
                        avg_precip=('precip_mm', 'mean'),
                        avg_pm25=('air_quality_PM2.5', 'mean')
                    ).reset_index().dropna()
                    
                    # Score calculation for heatmap (Simplified for month granularity)
                    calendar_data['Monthly_Score'] = (
                        (1 - (abs(calendar_data['avg_temp'] - 22) / 10)) + # Closer to 22C
                        (1 - (calendar_data['avg_precip'] / 10)) +          # Lower precip
                        (1 - (calendar_data['avg_pm25'] / 25))              # Lower PM2.5
                    ) / 3
                    calendar_data['Monthly_Score'] = np.clip(calendar_data['Monthly_Score'], 0, 1) * 100
                    
                    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                    
                    fig = px.imshow(
                        calendar_data.pivot_table(index='country', columns='month_name', values='Monthly_Score'),
                        x=month_order,
                        y=comp_countries,
                        color_continuous_scale=px.colors.sequential.Plotly3, # Green to Red
                        aspect="auto",
                        labels=dict(x="Month", y="Country", color="Comfort Score (0-100)"),
                        text_auto=".0f",
                        title="Monthly Comfort Score Heatmap for Selected Countries"
                    )
                    fig.update_layout(template='plotly_white', height=600)
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # --- Feature 3: Weather-Based Activity Recommendations ---
            st.subheader("🏞️ Activity Recommendations for Current Conditions")
            
            # Use the mean of the filtered data to define "current" conditions
            # Corrected to use column names present in the mean result
            cols_for_cond = ['temperature_celsius', 'precip_mm', 'humidity', 'air_quality_PM2.5']
            current_cond = final_filtered_data[cols_for_cond].mean()
            
            activity_list = []
            
            # Use correct column names for index lookup: 'precip_mm' instead of 'avg_precip' and 'humidity' instead of 'avg_humidity'
            if current_cond['temperature_celsius'] > 25 and current_cond['precip_mm'] < 1 and current_cond['humidity'] < 75:
                activity_list.append(("☀️ Beach / Pool Day", "🏖️", "Enjoy the sun and low rain!"))
            
            if current_cond['temperature_celsius'] > 15 and current_cond['temperature_celsius'] < 25 and current_cond['precip_mm'] < 2:
                activity_list.append(("🌳 Hiking & Cycling", "🚴", "Ideal temperate weather for exertion."))
            
            if current_cond['precip_mm'] > 5 and current_cond['temperature_celsius'] > 10:
                activity_list.append(("☕ Museums & Cafes", "🖼️", "Best for indoor activities due to rain."))
            
            if current_cond['air_quality_PM2.5'] > 25:
                 activity_list.append(("🏠 Indoor Activities ONLY", "😷", "**Air Quality Alert!** Minimize outdoor exposure."))
            else:
                activity_list.append(("📸 City Tours & Photography", "🏙️", "Good air quality and pleasant conditions for city walks."))
                
            if current_cond['temperature_celsius'] < 5:
                activity_list.append(("🏂 Winter Sports / Warm Up", "❄️", "Requires cold gear, seek snow if available!"))

            col_act = st.columns(len(activity_list))
            for i, (name, icon, desc) in enumerate(activity_list):
                with col_act[i % len(col_act)]:
                    st.markdown(f"""
                    <div class='activity-card'>
                        <div style='font-size: 24px;'>{icon}</div>
                        <p style='font-weight: bold; margin: 5px 0 0 0;'>{name}</p>
                        <p style='font-size: 12px; color: #6B7280;'>{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            
            # --- Feature 4: Seasonal Migration Advisor ---
            st.subheader("✈️ Seasonal Migration Advisor (Home vs Destination)")
            
            available_countries = dataMonth['country'].unique().tolist() # Use full data for comparison
            home_loc = st.selectbox("Select Your Home Country:", available_countries, index=0)
            dest_locs = [c for c in available_countries if c != home_loc]
            dest_loc = st.selectbox("Select Destination Country:", dest_locs, index=0)
            
            if home_loc and dest_loc:
                migration_data = dataMonth[dataMonth['country'].isin([home_loc, dest_loc])]
                
                # Dual-Axis Line Chart: Temperature & Precipitation
                fig = go.Figure()
                
                # Temperature Traces
                fig.add_trace(go.Scatter(
                    x=migration_data[migration_data['country'] == home_loc]['month'], 
                    y=migration_data[migration_data['country'] == home_loc]['temperature_celsius'], 
                    name=f'{home_loc} Temp (°C)', 
                    mode='lines+markers', 
                    yaxis='y1',
                    line=dict(color='#EF4444', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=migration_data[migration_data['country'] == dest_loc]['month'], 
                    y=migration_data[migration_data['country'] == dest_loc]['temperature_celsius'], 
                    name=f'{dest_loc} Temp (°C)', 
                    mode='lines+markers', 
                    yaxis='y1',
                    line=dict(color='#22C55E', width=3)
                ))
                
                # Precipitation Traces (Secondary Axis)
                fig.add_trace(go.Bar(
                    x=migration_data[migration_data['country'] == home_loc]['month'], 
                    y=migration_data[migration_data['country'] == home_loc]['precip_mm'], 
                    name=f'{home_loc} Precip (mm)', 
                    opacity=0.4, 
                    yaxis='y2',
                    marker_color='#FBBF24'
                ))
                fig.add_trace(go.Bar(
                    x=migration_data[migration_data['country'] == dest_loc]['month'], 
                    y=migration_data[migration_data['country'] == dest_loc]['precip_mm'], 
                    name=f'{dest_loc} Precip (mm)', 
                    opacity=0.4, 
                    yaxis='y2',
                    marker_color='#60A5FA'
                ))
                
                # Layout
                fig.update_layout(
                    title=f'Seasonal Climate Migration: {home_loc} vs {dest_loc}',
                    xaxis=dict(title='Month', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
                    yaxis=dict(title='Temperature (°C)', side='left', showgrid=False, range=[migration_data['temperature_celsius'].min() - 5, migration_data['temperature_celsius'].max() + 5]),
                    yaxis2=dict(title='Precipitation (mm)', overlaying='y', side='right', showgrid=True),
                    template='plotly_white',
                    height=600,
                    legend=dict(x=0, y=1.2, orientation="h")
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # -------------------------------------------------------------------------
    # NEW SECTION 2: LUNAR & ASTRONOMICAL INSIGHTS (PROMPT 2)
    # -------------------------------------------------------------------------

    elif analysis_type == "🌙 **Lunar & Astronomical Insights**":
        st.header("🌙 Lunar & Astronomical Insights")

        if 'moon_illumination' not in filtered_data.columns or final_filtered_data.empty or final_filtered_data['moon_illumination'].isnull().all():
            st.error("❌ Moon illumination data is missing or filtered data is empty. Please check the dataset columns and filters.")
            
        else:
            
            # Use final_filtered_data here
            moon_data = final_filtered_data.dropna(subset=['moon_illumination']).copy()
            moon_data['moon_phase_emoji'] = moon_data['moon_illumination'].apply(get_moon_phase_emoji)
            
            if moon_data.empty:
                 st.info("Filtered data has no moon illumination values. Please adjust the sidebar filters.")
                
            else:
                
                # FIX 1: Define the required emoji column before creating the heatmap pivot data
                # We use only the first two characters (the emoji) for clarity on the heatmap cells
                moon_data['moon_emoji_only'] = moon_data['moon_phase_emoji'].str[0:2].str.strip()
                
                # Now define the local copy and the day column
                heatmap_data = moon_data.copy()
                heatmap_data['day'] = heatmap_data['date'].dt.day
                
                
                # 1. Moon Phase Calendar Visualization
                st.subheader("📅 Moon Phase Calendar & Illumination Heatmap")
                
                col_moon_map, col_moon_tip = st.columns([3, 1])
                
                with col_moon_tip:
                    st.markdown(f"""
                    <div class='moon-card'>
                        <p style='font-size: 28px; font-weight: bold; margin: 0;'>🌙 Illumination Tip</p>
                        <hr style='border-top: 1px solid #6B7280;'>
                        <p style='font-size: 14px; margin: 0;'>
                            **🌑 New Moon:** Best for **stargazing** (lowest light pollution).<br>
                            **🌕 Full Moon:** Highest natural light, best for **night hikes**.<br>
                            **Illumination:** Average is **{moon_data['moon_illumination'].mean():.1f}%** for this period.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_moon_map:
                    # Pivot for illumination values
                    moon_calendar = heatmap_data.pivot_table(
                        index='month_name', 
                        columns='day', 
                        values='moon_illumination', 
                        aggfunc='mean'
                    ).reindex(index=months[1:], fill_value=np.nan)

                    # Annotations for moon phase (now using the clean 'moon_emoji_only' column)
                    annotation_data = heatmap_data.groupby(['month_name', 'day'])['moon_emoji_only'].first().reset_index()
                    annotation_data = annotation_data.pivot_table(
                        index='month_name', 
                        columns='day', 
                        values='moon_emoji_only',
                        aggfunc='first'
                    ).reindex(index=months[1:], fill_value='')
                    
                    # Prepare annotation text as a numpy array for create_annotated_heatmap
                    annotation_text_array = annotation_data.fillna('').astype(str).values


                    fig = ff.create_annotated_heatmap(
                        z=moon_calendar.values,
                        x=list(moon_calendar.columns),
                        y=list(moon_calendar.index),
                        annotation_text=annotation_text_array, # Use the emoji-only array
                        colorscale='twilight', 
                        showscale=True,
                        colorbar_title='Illumination (%)'
                    )
                    
                    # Fine-tune annotation text appearance for clarity
                    for i in range(len(fig.layout.annotations)):
                        fig.layout.annotations[i].font.size = 14 
                        fig.layout.annotations[i].yanchor = 'middle'
                        fig.layout.annotations[i].xanchor = 'center'
                        
                    fig.update_layout(title='Monthly Moon Illumination Calendar (Hover for Phase Name)', template='plotly_white', height=700)
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")

                # 2. Daylight Duration Analysis
                st.subheader("☀️ Daylight Duration Analysis")
                
                if 'daylight_hours' in moon_data.columns:
                    daylight_data = moon_data.groupby(['month', 'country'])['daylight_hours'].mean().reset_index().sort_values('month')
                    
                    # Line Chart: Daylight Trends
                    fig = px.line(
                        daylight_data,
                        x='month',
                        y='daylight_hours',
                        color='country',
                        markers=True,
                        title='Monthly Daylight Duration by Country'
                    )
                    fig.update_layout(
                        xaxis=dict(title='Month', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
                        yaxis_title="Average Daylight Hours",
                        template='plotly_white',
                        height=550
                    )
                    # Add Solstice/Equinox annotations
                    fig.add_vline(x=3.5, line_width=1, line_dash="dash", line_color="blue", annotation_text="Vernal Equinox (~Mar 20)")
                    fig.add_vline(x=6.5, line_width=1, line_dash="dash", line_color="red", annotation_text="Summer Solstice (~Jun 21)")
                    fig.add_vline(x=9.5, line_width=1, line_dash="dash", line_color="blue", annotation_text="Autumnal Equinox (~Sep 23)")
                    fig.add_vline(x=12.5, line_width=1, line_dash="dash", line_color="red", annotation_text="Winter Solstice (~Dec 21)")

                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.info("Daylight duration calculation requires 'sunrise' and 'sunset' columns in the dataset.")

                # 3. Moonlight & Weather Correlation
                st.subheader("🔗 Moonlight vs. Weather Correlation")
                
                corr_data = moon_data.sample(min(5000, len(moon_data)), random_state=42)
                
                fig = px.scatter(
                    corr_data, 
                    x='moon_illumination', 
                    y='temperature_celsius', 
                    color='precip_mm', 
                    size='air_quality_PM2.5', 
                    color_continuous_scale='Inferno',
                    hover_data={'location_name': True, 'precip_mm': ':.1f', 'moon_illumination': ':.0f'},
                    title='Moon Illumination vs. Temperature (Color=Precipitation, Size=Air Quality)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 6. Lunar-Influenced Activity Planner
                st.markdown("### 🔭 Lunar-Influenced Activity Planner")
                
                full_moon_days = moon_data[moon_data['moon_illumination'] >= 80]
                new_moon_days = moon_data[moon_data['moon_illumination'] <= 20]
                
                col_new, col_full = st.columns(2)
                
                with col_new:
                    st.markdown(f"**New Moon Activities (Total {len(new_moon_days)} Days)**")
                    st.markdown(f"""
                    <div class='activity-card' style='background: #374151; color: white;'>
                        <p style='font-size: 20px;'>🔭 Stargazing & Astrophotography</p>
                        <p style='font-size: 14px; color: #D1D5DB;'>Lowest ambient light, maximizing visibility of deep-sky objects.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_full:
                    st.markdown(f"**Full Moon Activities (Total {len(full_moon_days)} Days)**")
                    st.markdown(f"""
                    <div class='activity-card' style='background: #FBBF24; color: #1F2937;'>
                        <p style='font-size: 20px;'>🌕 Night Hiking & Outdoor Events</p>
                        <p style='font-size: 14px; color: #4B5563;'>Maximum visibility for non-technical outdoor adventures.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # 8. Moon-Weather Pattern Detection (Box Plots)
                st.markdown("### 📊 Weather Patterns by Moon Phase")
                
                moon_data['Phase_Category'] = pd.cut(
                    moon_data['moon_illumination'],
                    bins=[-1, 25, 50, 75, 101],
                    labels=['New (0-25%)', 'First Quarter (25-50%)', 'Full (50-75%)', 'Last Quarter (75-100%)']
                )
                
                var_box = st.selectbox("Select Variable for Phase Comparison:", key_cols)
                
                fig = px.box(
                    moon_data,
                    x='Phase_Category',
                    y=var_box,
                    color='Phase_Category',
                    notched=True,
                    points='outliers',
                    title=f'Distribution of {var_box.replace("_", " ").title()} Across Moon Phases'
                )
                fig.update_layout(template='plotly_white', height=600)
                st.plotly_chart(fig, use_container_width=True)


    # -------------------------------------------------------------------------
    # REST OF ORIGINAL SECTIONS FOLLOW (EXISTING)
    # -------------------------------------------------------------------------
    
    # ADVANCED VISUALIZATIONS SECTION
    elif analysis_type == "✨ **Advanced Visualizations**":
        # ... (EXISTING CODE) ...
        st.header("✨ Advanced Visualizations")
        
        if final_filtered_data.empty:
            st.info("No data available for the selected filters.")
        elif len(final_filtered_data) < 5:
              st.warning("Data filtered too heavily. Need more data points (e.g., >50) for meaningful advanced visualizations.")
        else:
            viz_options = [
                "3D Temperature-Humidity-Pressure Scatter",
                "Animated Climate Evolution",
            ]
            
            viz_option = st.selectbox("Select Advanced Visualization:", viz_options)
            
            if viz_option == "3D Temperature-Humidity-Pressure Scatter":
                st.plotly_chart(create_3d_scatter(final_filtered_data), use_container_width=True)
                
            elif viz_option == "Animated Climate Evolution":
                st.plotly_chart(create_animated_evolution(final_filtered_data), use_container_width=True)


    # STATISTICAL OVERVIEW (Retained)
    elif analysis_type == "📈 Statistical Overview":
        # ... (EXISTING CODE) ...
        st.header("📈 Statistical Overview")
        
        viz_option = st.selectbox(
            "Select Visualization:",
            ["Mean vs Median Comparison", "Standard Deviation Analysis", "Mean and Standard Deviation Combined", "Min and Max Comparison"]
        )
        
        if final_filtered_data.empty:
            st.info("No data available for the selected filters.")
        else:
            if viz_option == "Mean vs Median Comparison":
                st.subheader("📊 Mean vs Median Comparison for Weather Variables")
                stats_data = pd.DataFrame({'Mean': final_filtered_data[numeric_cols].mean().round(2), 'Median': final_filtered_data[numeric_cols].median().round(2)})
                fig = go.Figure()
                fig.add_trace(go.Bar(x=stats_data.index, y=stats_data['Mean'], name='Mean', marker_color='#FB923C'))
                fig.add_trace(go.Bar(x=stats_data.index, y=stats_data['Median'], name='Median', marker_color='#3B82F6'))
                fig.update_layout(title='Mean vs Median Comparison', xaxis_title='Variables', yaxis_title='Value', barmode='group', template='plotly_white', plot_bgcolor='rgba(250,250,250,1)', height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Standard Deviation Analysis":
                st.subheader("📊 Standard Deviation of All Numeric Weather Variables")
                std_values = final_filtered_data[numeric_cols].std().sort_values(ascending=False)
                std_data = pd.DataFrame({'Variable': std_values.index, 'Standard Deviation': std_values.values})
                colors = ['#FB923C', '#3B82F6', '#14B8A6', '#6366F1', '#A855F7', '#EF4444', '#10B981', '#8B5CF6', '#F59E0B']
                fig = px.bar(std_data, x='Variable', y='Standard Deviation', text='Standard Deviation', color='Variable', color_discrete_sequence=colors, title='Standard Deviation Analysis')
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.update_layout(template='plotly_white', showlegend=False, height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Mean and Standard Deviation Combined":
                st.subheader("📊 Mean and Standard Deviation Combined")
                summary_data = pd.DataFrame({'Mean': final_filtered_data[numeric_cols].mean().round(2), 'Std Dev': final_filtered_data[numeric_cols].std().round(2)}).reset_index().rename(columns={'index': 'Variable'})
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Mean', x=summary_data['Variable'], y=summary_data['Mean'], marker_color='#FB923C', text=summary_data['Mean'], textposition='outside'))
                fig.add_trace(go.Bar(name='Standard Deviation', x=summary_data['Variable'], y=summary_data['Std Dev'], marker_color='#3B82F6', text=summary_data['Std Dev'], textposition='outside'))
                fig.update_layout(title='Mean and Standard Deviation', barmode='group', template='plotly_white', height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Min and Max Comparison":
                st.subheader("📊 Min and Max of Weather Variables")
                summary_data = pd.DataFrame({'Min': final_filtered_data[numeric_cols].min().round(2), 'Max': final_filtered_data[numeric_cols].max().round(2)}).reset_index().rename(columns={'index': 'Variable'})
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Min', x=summary_data['Variable'], y=summary_data['Min'], marker_color='#14B8A6', text=summary_data['Min'], textposition='outside'))
                fig.add_trace(go.Bar(name='Max', x=summary_data['Variable'], y=summary_data['Max'], marker_color='#EF4444', text=summary_data['Max'], textposition='outside'))
                fig.update_layout(title='Min and Max Comparison', barmode='group', template='plotly_white', height=600)
                st.plotly_chart(fig, use_container_width=True)

    
    # DISTRIBUTION ANALYSIS (Retained logic)
    elif analysis_type == "📊 Distribution Analysis":
        # ... (EXISTING CODE) ...
        st.header("📊 Distribution Analysis")
        
        viz_option = st.selectbox(
            "Select Visualization:",
            ["Individual Variable Distribution", "Violin Plot (All Variables)", "Variable Stability Analysis", "Seasonal Anomaly Box Plots"] 
        )
        
        if final_filtered_data.empty:
            st.info("No data available for the selected filters.")
        else:
            if viz_option == "Individual Variable Distribution":
                selected_var = st.selectbox("Select Variable:", key_cols)
                gradient_sets = {'temperature_celsius': ['#FB923C', '#FDBA74', '#FED7AA'], 'humidity': ['#3B82F6', '#60A5FA', '#93C5FD'], 'wind_kph': ['#14B8A6', '#34D399', '#6EE7B7'], 'pressure_mb': ['#6366F1', '#8B5CF6', '#A78BFA'], 'precip_mm': ['#A855F7', '#C084FC', '#D8B4FE']}
                fig = px.histogram(final_filtered_data, x=selected_var, nbins=40, marginal="box", title=f"Distribution of {selected_var}", color_discrete_sequence=gradient_sets.get(selected_var, ['#FB923C']))
                fig.update_layout(template='plotly_white', height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Violin Plot (All Variables)":
                st.subheader("🎻 Distribution & Spread of Weather Variables")
                fig = px.violin(final_filtered_data, y=key_cols, box=True, points="outliers", color_discrete_sequence=['#FB923C', '#3B82F6', '#14B8A6', '#6366F1', '#A855F7'], title="Violin Plot - Distribution Analysis")
                fig.update_layout(template='plotly_white', height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Variable Stability Analysis":
                st.subheader("📊 Variable Stability vs Volatility")
                std_values = final_filtered_data[key_cols].std().sort_values(ascending=False)
                std_data = pd.DataFrame({'Variable': std_values.index, 'Std Dev': std_values.values})
                fig = px.bar(std_data, x='Variable', y='Std Dev', color='Variable', color_discrete_sequence=['#FB923C', '#3B82F6', '#14B8A6', '#6366F1', '#A855F7'], title="Variable Stability Analysis")
                fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                fig.update_layout(template='plotly_white', showlegend=False, height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Seasonal Anomaly Box Plots":
                st.subheader("❄️ Seasonal Anomaly Analysis using Box Plots")
                var_to_analyze = st.selectbox("Select Variable for Box Plot Analysis:", key_cols)
                month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                present_months = final_filtered_data['month_name'].unique().tolist()
                sorted_month_names = [m for m in month_order if m in present_months]
                fig = px.box(final_filtered_data, x='month_name', y=var_to_analyze, color='month_name', category_orders={"month_name": sorted_month_names}, notched=True, points='outliers', title=f'Monthly Distribution of {var_to_analyze.replace("_", " ").title()} (Outliers = Anomalies)', color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(template='plotly_white', showlegend=False, height=600)
                st.plotly_chart(fig, use_container_width=True)

    # CORRELATION ANALYSIS (Retained logic)
    elif analysis_type == "🔗 Correlation Analysis":
        # ... (EXISTING CODE) ...
        st.header("🔗 Correlation Analysis")
        
        viz_options = [
            "Full Correlation Heatmap", "Focused Correlation (Key Variables)", "Key Variable Scatter Plot (Original)"
        ]
        
        viz_option = st.selectbox("Select Visualization:", viz_options)
        
        if final_filtered_data.empty:
            st.info("No data available for the selected filters.")
        else:
            if viz_option == "Full Correlation Heatmap":
                st.subheader("🌍 Correlation Heatmap of Weather Variables")
                corr_matrix = final_filtered_data[numeric_cols].corr().round(2)
                fig = ff.create_annotated_heatmap(z=corr_matrix.values, x=list(corr_matrix.columns), y=list(corr_matrix.index), colorscale='Viridis', showscale=True, annotation_text=corr_matrix.values.round(2))
                fig.update_layout(title='Full Correlation Heatmap', template='plotly_white', width=950, height=950)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Focused Correlation (Key Variables)":
                st.subheader("🔍 Focused Correlation Heatmap")
                corr_matrix = final_filtered_data[key_cols].corr()
                fig = ff.create_annotated_heatmap(z=corr_matrix.values, x=key_cols, y=key_cols, colorscale='Plasma', showscale=True, annotation_text=corr_matrix.round(2).values)
                fig.update_layout(title='Key Variables Correlation', template='plotly_white', width=800, height=800)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Key Variable Scatter Plot (Original)": 
                st.subheader("Scatter Plot: Relationship between two Key Variables")
                scatter_x = st.selectbox("Select X-axis Variable:", key_cols, index=0, key='scatter_x')
                scatter_y = st.selectbox("Select Y-axis Variable:", key_cols, index=1, key='scatter_y')
                sample_size = min(5000, len(final_filtered_data))
                fig = px.scatter(final_filtered_data.sample(sample_size, random_state=42), x=scatter_x, y=scatter_y, color='month_name', hover_data=['country', 'location_name'], title=f'Scatter Plot: {scatter_x.replace("_", " ").title()} vs {scatter_y.replace("_", " ").title()}', color_discrete_sequence=px.colors.qualitative.Set1)
                fig.update_layout(template='plotly_white', height=600)
                st.plotly_chart(fig, use_container_width=True)
    
    # MONTHLY TRENDS (Retained logic)
    elif analysis_type == "📅 Monthly Trends":
        # ... (EXISTING CODE) ...
        st.header("📅 Monthly Trends Analysis")
        
        viz_option = st.selectbox(
            "Select Visualization:",
            ["Temperature Trends", "Humidity Trends", "Precipitation Trends", "Wind Speed Trends", "Pressure Trends", "Interactive Multi-Country Trend Comparison"] 
        )
        
        if final_filtered_data.empty:
            st.info("No data available for the selected filters.")
        else:
            data_for_top_countries = data if selected_country == 'All Countries' else final_filtered_data
            top_countries = data_for_top_countries['country'].value_counts().nlargest(5).index
            monthly_data_filtered = final_filtered_data.groupby('month')[key_cols].mean().reset_index()
            
            if viz_option == "Temperature Trends":
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🌡️ Global Average Temperature")
                    monthly_temp = monthly_data_filtered[['month', 'temperature_celsius']]
                    fig = px.line(monthly_temp, x='month', y='temperature_celsius', markers=True, color_discrete_sequence=['#FB923C'])
                    fig.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.subheader("🌍 Temperature by Top 5 Countries")
                    subset = final_filtered_data[final_filtered_data['country'].isin(top_countries)]
                    monthly_country_temp = subset.groupby(['month', 'country'])['temperature_celsius'].mean().reset_index()
                    fig = px.line(monthly_country_temp, x='month', y='temperature_celsius', color='country', markers=True, line_shape='spline', color_discrete_sequence=['#FB923C', '#3B82F6', '#14B8A6', '#6366F1', '#A855F7'])
                    fig.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Humidity Trends":
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("💧 Global Average Humidity")
                    monthly_humidity = monthly_data_filtered[['month', 'humidity']]
                    fig = px.line(monthly_humidity, x='month', y='humidity', markers=True, color_discrete_sequence=['#3B82F6'])
                    fig.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.subheader("💧 Humidity by Top 5 Countries")
                    subset = final_filtered_data[final_filtered_data['country'].isin(top_countries)]
                    monthly_country_humidity = subset.groupby(['month', 'country'])['humidity'].mean().reset_index()
                    fig = px.line(monthly_country_humidity, x='month', y='humidity', color='country', markers=True, line_shape='spline', color_discrete_sequence=['#3B82F6', '#FB923C', '#14B8A6', '#6366F1', '#A855F7'])
                    fig.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Precipitation Trends":
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🌧️ Global Average Precipitation")
                    monthly_precip = monthly_data_filtered[['month', 'precip_mm']]
                    fig = px.line(monthly_precip, x='month', y='precip_mm', markers=True, color_discrete_sequence=['#A855F7'])
                    fig.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.subheader("🌧️ Precipitation by Top 5 Countries")
                    subset = final_filtered_data[final_filtered_data['country'].isin(top_countries)]
                    monthly_country_precip = subset.groupby(['month', 'country'])['precip_mm'].mean().reset_index()
                    fig = px.line(monthly_country_precip, x='month', y='precip_mm', color='country', markers=True, line_shape='spline', color_discrete_sequence=['#A855F7', '#FB923C', '#3B82F6', '#14B8A6', '#6366F1'])
                    fig.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Wind Speed Trends":
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("💨 Global Average Wind Speed")
                    monthly_wind = monthly_data_filtered[['month', 'wind_kph']]
                    fig = px.line(monthly_wind, x='month', y='wind_kph', markers=True, color_discrete_sequence=['#14B8A6'])
                    fig.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.subheader("💨 Wind Speed by Top 5 Countries")
                    subset = final_filtered_data[final_filtered_data['country'].isin(top_countries)]
                    monthly_country_wind = subset.groupby(['month', 'country'])['wind_kph'].mean().reset_index()
                    fig = px.line(subset, x='month', y='wind_kph', color='country', markers=True, line_shape='spline', color_discrete_sequence=['#14B8A6', '#FB923C', '#3B82F6', '#6366F1', '#A855F7'])
                    fig.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Pressure Trends":
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("⚖️ Global Average Pressure")
                    monthly_pressure = monthly_data_filtered[['month', 'pressure_mb']]
                    fig = px.line(monthly_pressure, x='month', y='pressure_mb', markers=True, color_discrete_sequence=['#6366F1'])
                    fig.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.subheader("⚖️ Pressure by Top 5 Countries")
                    subset = final_filtered_data[final_filtered_data['country'].isin(top_countries)]
                    monthly_country_pressure = subset.groupby(['month', 'country'])['pressure_mb'].mean().reset_index()
                    fig = px.line(subset, x='month', y='pressure_mb', color='country', markers=True, line_shape='spline', color_discrete_sequence=['#6366F1', '#FB923C', '#3B82F6', '#14B8A6', '#A855F7'])
                    fig.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Interactive Multi-Country Trend Comparison":
                st.subheader("Compare Monthly Trends for Multiple Countries")
                available_countries = sorted(data['country'].unique().tolist())
                country_filter_data = dataMonth
                default_countries = available_countries[:min(3, len(available_countries))]
                selected_countries_comp = st.multiselect("Select Countries to Compare (Max 5 Recommended)", available_countries, default=[selected_country] if selected_country != 'All Countries' and selected_country in available_countries else default_countries)
                trend_variable = st.selectbox("Select Trend Variable:", key_cols)
                if selected_countries_comp:
                    comparison_data = country_filter_data[country_filter_data['country'].isin(selected_countries_comp)]
                    monthly_comparison = comparison_data.groupby(['month', 'country'])[trend_variable].mean().reset_index()
                    monthly_comparison = monthly_comparison.sort_values(by='month')
                    fig = px.line(monthly_comparison, x='month', y=trend_variable, color='country', title=f'Monthly Average {trend_variable.replace("_", " ").title()} Comparison', markers=True, line_shape='spline', color_discrete_sequence=px.colors.qualitative.Dark24) 
                    fig.update_layout(xaxis_title="Month", yaxis_title=trend_variable.replace("_", " ").title(), template='plotly_white', height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please select at least one country for comparison.")


    # CLIMATE COMPARISON (Retained logic)
    elif analysis_type == "🎯 **Climate Comparison**":
        # ... (EXISTING CODE) ...
        st.header("🎯 Multi-Country Climate Comparison Tool")
        
        available_countries = sorted(data['country'].unique().tolist())
        
        # User Selection for Comparison (Multiselect)
        selected_countries = st.multiselect(
            "**Select 2-5 Countries for Side-by-Side Comparison:**", 
            available_countries, 
            default=available_countries[:min(4, len(available_countries))]
        )
        
        if len(selected_countries) < 2 or final_filtered_data.empty:
            st.info("Please select at least two countries for comparison or adjust your filters.")
        else:
            # Aggregate data for comparison
            comp_data = dataMonth[dataMonth['country'].isin(selected_countries)].groupby('country')[key_cols].mean().reset_index()
            
            # --- FEATURE 1: COMPARISON CARDS ---
            st.subheader("📊 Side-by-Side Comparison Metrics")
            
            comp_tabs = st.tabs(selected_countries)
            
            for i, country in enumerate(selected_countries):
                country_data = comp_data[comp_data['country'] == country]
                if not country_data.empty:
                    with comp_tabs[i]:
                        country_data = country_data.iloc[0]
                        st.metric(label="🌡️ Avg Temperature", value=f"{country_data['temperature_celsius']:.1f}°C")
                        st.metric(label="💧 Avg Humidity", value=f"{country_data['humidity']:.1f}%")
                        st.metric(label="💨 Avg Wind Speed", value=f"{country_data['wind_kph']:.1f} kph")
                        st.metric(label="🌧️ Avg Precipitation", value=f"{country_data['precip_mm']:.1f} mm")
                else:
                    with comp_tabs[i]:
                        st.warning("No data for this country in the monthly aggregate.")

            st.markdown("---")

            # --- FEATURE 2: RADAR CHART (Spider Chart) ---
            # REMOVED ALL RADAR CHART CODE AND PLACEHOLDERS.


    # PATTERN RECOGNITION (K-Means Clustering)
    elif analysis_type == "🔬 **Pattern Recognition**":
        # ... (EXISTING CODE) ...
        st.header("🔬 Climate Pattern Recognition and Clustering")
        
        # Data preparation for clustering
        cluster_data_cols = ['temperature_celsius', 'humidity', 'precip_mm']
        cluster_data = final_filtered_data.dropna(subset=cluster_data_cols).copy()
        
        if cluster_data.empty or len(cluster_data) < 10:
            st.info("Insufficient data for meaningful clustering. Need at least 10 non-null data points.")
        else:
            # Sidebar controls for K-Means
            st.sidebar.markdown("### K-Means Controls")
            n_clusters = st.sidebar.slider("Select Number of Climate Clusters (K)", 2, 8, 4)
            
            # --- CLUSTERING LOGIC ---
            
            # 1. Scale data for K-Means
            X = cluster_data[cluster_data_cols]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 2. Run K-Means
            # Use max 1000 samples for faster clustering if dataset is massive
            X_sample = X_scaled[np.random.choice(X_scaled.shape[0], min(1000, X_scaled.shape[0]), replace=False), :]
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_data['Cluster_ID'] = kmeans.fit_predict(X_scaled)
            cluster_data['Cluster'] = 'Cluster ' + (cluster_data['Cluster_ID'] + 1).astype(str)
            
            # 3. Calculate Cluster Centers (Inverse transform for meaningful interpretation)
            centers_scaled = kmeans.cluster_centers_
            centers = scaler.inverse_transform(centers_scaled)
            centers_df = pd.DataFrame(centers, columns=cluster_data_cols)
            centers_df['Cluster'] = 'Center ' + (centers_df.index + 1).astype(str)

            # --- FEATURE 1: CLUSTER SCATTER PLOT ---
            st.subheader("🗺️ K-Means Climate Cluster Scatter (Temp vs Humidity)")
            st.markdown(f"**{n_clusters}** climate patterns identified across **{len(cluster_data['location_name'].unique())} locations**.")

            # Sample the scatter plot data for faster rendering
            sample_scatter = cluster_data.sample(min(5000, len(cluster_data)), random_state=42)
            
            fig = px.scatter(
                sample_scatter,
                x='temperature_celsius',
                y='humidity',
                color='Cluster',
                hover_data={'location_name': True, 'country': True, 'precip_mm': ':.1f'},
                title="Location Clustering by Temperature, Humidity, and Precipitation",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            # Add cluster centers as stars (Requires separate scatter plot)
            fig.add_trace(go.Scatter(
                x=centers_df['temperature_celsius'],
                y=centers_df['humidity'],
                mode='markers',
                marker=dict(symbol='star-open', size=15, color='black', line=dict(width=2)),
                name='Cluster Centers',
                hovertext=[f"Center {i+1}: Temp={c['temperature_celsius']:.1f}°C, Humidity={c['humidity']:.1f}%" for i, c in centers_df.iterrows()],
                showlegend=True
            ))

            fig.update_layout(template='plotly_white', height=650)
            st.plotly_chart(fig, use_container_width=True)

            # --- FEATURE 2: CLUSTER CHARACTERISTICS TABLE ---
            st.subheader("📋 Cluster Characteristics Table")
            st.markdown("Average characteristics of locations belonging to each pattern.")
            
            cluster_summary = cluster_data.groupby('Cluster')[cluster_data_cols].mean().reset_index()
            cluster_summary['Location Count'] = cluster_data.groupby('Cluster')['location_name'].nunique().values
            cluster_summary = cluster_summary.round(1)
            
            st.dataframe(cluster_summary.sort_values('temperature_celsius'), use_container_width=True, hide_index=True)
            
            st.markdown("""
            <div class='smart-insight-box'>
                <strong>💡 Key Insight:</strong> The clustering shows distinct climate patterns. For instance, a cluster with high Temp/high Precip suggests a **Tropical** climate, whereas low Temp/low Precip suggests a **Cold & Dry** climate.
            </div>
            """, unsafe_allow_html=True)

    # AIR QUALITY ANALYSIS (Retained logic)
    elif analysis_type == "💨 Air Quality Analysis":
        # ... (EXISTING CODE) ...
        st.header("💨 Air Quality Analysis")
        
        viz_options = [
            "Interactive Global Air Quality Map", "Interactive Global Weather Map", "Monthly Pollutant Heatmap", "Country Comparison", "Monthly Pollutant Trends"
        ]
        viz_option = st.selectbox("Select Visualization:", viz_options)
        
        if final_filtered_data.empty:
            st.info("No data available for the selected filters.")
        else:
            if viz_option == "Interactive Global Air Quality Map":
                st.subheader("🗺️ Interactive Global Air Quality Map")
                air_map_data = final_filtered_data.groupby('country')[pollutants].mean().reset_index()
                fig = go.Figure()
                fig.add_trace(go.Choropleth(locations=air_map_data['country'], locationmode='country names', z=air_map_data['air_quality_PM2.5'], colorscale='Viridis', colorbar_title="PM2.5 (µg/m³)"))
                fig.update_layout(updatemenus=[dict(buttons=[dict(label=pollutant.replace("air_quality_", "").replace("_", " "), method='update', args=[{'z': [air_map_data[pollutant]]}, {'colorbar.title.text': pollutant.replace("air_quality_", "").replace("_", " ")}]) for pollutant in pollutants], direction='down', showactive=True, x=0.05, y=1.15, xanchor='left', yanchor='top')], title=dict(text="Global Air Quality Levels (Select Pollutant)", x=0.5, font=dict(size=22, color='#0A043C')), geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'), template='plotly_white', height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Interactive Global Weather Map":
                st.subheader("🗺️ Interactive Global Weather Conditions Map")
                weather_params = ['temperature_celsius', 'humidity', 'pressure_mb', 'precip_mm']
                weather_data = final_filtered_data.groupby('country')[weather_params].mean().reset_index()
                fig = go.Figure()
                fig.add_trace(go.Choropleth(locations=weather_data['country'], locationmode='country names', z=weather_data['temperature_celsius'], colorscale='Plasma', colorbar_title="Temperature (°C)"))
                fig.update_layout(updatemenus=[dict(buttons=[dict(label=param.replace("_", " ").title(), method='update', args=[{'z': [weather_data[param]]}, {'colorbar.title.text': param.replace("_", " ").title()}]) for param in weather_params], direction='down', showactive=True, x=0.05, y=1.15, xanchor='left', yanchor='top')], title=dict(text="Global Weather Conditions (Select Parameter)", x=0.5, font=dict(size=22, color='#0A043C')), geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'), template='plotly_white', height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Monthly Pollutant Heatmap":
                st.subheader("🗓️ Monthly Air Pollutant Concentrations Heatmap")
                monthly_air_matrix = final_filtered_data.groupby('month')[pollutants].mean().T
                fig = ff.create_annotated_heatmap(z=monthly_air_matrix.values, x=list(monthly_air_matrix.columns), y=[p.replace("air_quality_", "").replace("_", " ").title() for p in monthly_air_matrix.index], colorscale='Plasma', annotation_text=monthly_air_matrix.round(1).values, showscale=True)
                fig.update_layout(title="Monthly Air Pollutant Concentrations (Heatmap)", template='plotly_white', title_font=dict(size=22, color='#0A043C'), width=900, height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Country Comparison":
                st.subheader("🌍 Comparison of Air Pollutants Across Top Countries")
                country_air = final_filtered_data.groupby('country')[pollutants].mean().reset_index()
                top_countries_air = final_filtered_data['country'].value_counts().nlargest(5).index
                country_air = country_air[country_air['country'].isin(top_countries_air)]
                country_air_melted = country_air.melt(id_vars='country', var_name='Pollutant', value_name='Concentration')
                country_air_melted['Pollutant'] = country_air_melted['Pollutant'].apply(lambda x: x.replace("air_quality_", "").replace("_", " ").title())
                fig = px.bar(country_air_melted, x='country', y='Concentration', color='Pollutant', barmode='group', color_discrete_sequence=['#FB923C', '#3B82F6', '#14B8A6', '#6366F1', '#A855F7', '#EF4444'])
                fig.update_layout(xaxis_title="Country", yaxis_title="Average Concentration", template='plotly_white', title_font=dict(size=22, color='#0A043C'), height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Monthly Pollutant Trends":
                st.subheader("📈 Monthly Trends of All Air Pollutants")
                monthly_air = final_filtered_data.groupby('month')[pollutants].mean().reset_index()
                air_melted = monthly_air.melt(id_vars='month', var_name='Pollutant', value_name='Concentration')
                air_melted['Pollutant'] = air_melted['Pollutant'].apply(lambda x: x.replace("air_quality_", "").replace("_", " ").title())
                fig = px.line(air_melted, x='month', y='Concentration', color='Pollutant', markers=True, line_shape='spline', color_discrete_sequence=['#FB923C', '#3B82F6', '#14B8A6', '#6366F1', '#A855F7', '#EF4444'])
                fig.update_layout(template='plotly_white', height=600)
                st.plotly_chart(fig, use_container_width=True)


    # GEOGRAPHIC ANALYSIS (Retained logic)
    elif analysis_type == "🗺️ Geographic Analysis":
        # ... (EXISTING CODE) ...
        st.header("🗺️ Geographic Analysis")
        
        viz_options = [
            "Top 10 Countries by Rainfall", "Top Hottest Days", "Top Coldest Days", "Regional Comparison Slider" 
        ]
        viz_option = st.selectbox("Select Visualization:", viz_options)
        
        if final_filtered_data.empty:
            st.info("No data available for the selected filters.")
        else:
            if viz_option == "Top 10 Countries by Rainfall":
                st.subheader("🌧️ Top 10 Countries by Average Rainfall")
                avg_rainfall = final_filtered_data.groupby('country')['precip_mm'].mean().reset_index()
                top10_rainfall = avg_rainfall.sort_values(by='precip_mm', ascending=False).head(10)
                fig = px.bar(top10_rainfall, x='country', y='precip_mm', color='precip_mm', color_continuous_scale='Blues')
                fig.update_layout(template='plotly_white', height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Top Hottest Days":
                st.subheader("🔥 Top 5 Hottest Recorded Days by Country")
                top5_hottest = (final_filtered_data.loc[final_filtered_data.groupby('country')['temperature_celsius'].idxmax()].sort_values(by='temperature_celsius', ascending=False).head(5))
                fig = px.bar(top5_hottest, x='country', y='temperature_celsius', color='temperature_celsius', text='location_name', hover_data=['date', 'location_name'], color_continuous_scale='Reds')
                fig.update_traces(textposition='outside')
                fig.update_layout(template='plotly_white', height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Top Coldest Days":
                st.subheader("❄️ Top 5 Coldest Recorded Days by Country")
                top5_coldest = (final_filtered_data.loc[final_filtered_data.groupby('country')['temperature_celsius'].idxmin()].sort_values(by='temperature_celsius', ascending=True).head(5))
                fig = px.bar(top5_coldest, x='country', y='temperature_celsius', color='temperature_celsius', text='location_name', hover_data=['date', 'location_name'], color_continuous_scale='Blues_r')
                fig.update_traces(textposition='outside')
                fig.update_layout(template='plotly_white', height=600)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Regional Comparison Slider": 
                st.subheader("📍 Interactive Regional Weather Condition Comparison (Size=Precipitation, Color=Temperature)")
                st.markdown("This map shows average conditions based on the **Current Filters**. Use the sidebar filters to dynamically change the data shown.")
                geo_comp_data = final_filtered_data.groupby(['location_name', 'country', 'latitude', 'longitude']).agg(
                    avg_temp=('temperature_celsius', 'mean'), avg_precip=('precip_mm', 'mean'), avg_humidity=('humidity', 'mean')
                ).reset_index()
                fig = px.scatter_geo(geo_comp_data, lat='latitude', lon='longitude', hover_name="location_name", color='avg_temp', size='avg_precip', projection="natural earth", color_continuous_scale='RdYlBu_r')
                fig.update_layout(template='plotly_white', height=650)
                st.plotly_chart(fig, use_container_width=True)


    # EXTREME WEATHER EVENTS (Retained logic)
    elif analysis_type == "🌡️ Extreme Weather Events":
        # ... (EXISTING CODE) ...
        st.header("🌡️ Extreme Weather Events")
        
        viz_option = st.selectbox(
            "Select Visualization:",
            ["Global Extreme Event Map", "Extreme Events Frequency Dashboard"] 
        )
        
        if final_filtered_data.empty:
            st.info("No data available for the selected filters.")
        else:
            # FIX: Robust calculation for extreme thresholds
            if len(final_filtered_data) > 1 and final_filtered_data[['temperature_celsius', 'precip_mm', 'wind_kph']].std().isnull().sum() == 0:
                temp_high = final_filtered_data['temperature_celsius'].mean() + 2 * final_filtered_data['temperature_celsius'].std()
                rain_high = final_filtered_data['precip_mm'].mean() + 2 * final_filtered_data['precip_mm'].std()
                wind_high = final_filtered_data['wind_kph'].mean() + 2 * final_filtered_data['wind_kph'].std()
                threshold_type = "2 STD above mean"
            else:
                Q1_temp, Q3_temp = final_filtered_data['temperature_celsius'].quantile([0.25, 0.75])
                IQR_temp = Q3_temp - Q1_temp
                temp_high = Q3_temp + 1.5 * IQR_temp
                Q1_rain, Q3_rain = final_filtered_data['precip_mm'].quantile([0.25, 0.75])
                IQR_rain = Q3_rain - Q1_rain
                rain_high = Q3_rain + 1.5 * IQR_rain
                Q1_wind, Q3_wind = final_filtered_data['wind_kph'].quantile([0.25, 0.75])
                IQR_wind = Q3_wind - Q1_wind
                wind_high = Q3_wind + 1.5 * IQR_wind
                threshold_type = "1.5 IQR (fallback)"

            # Filter extremes
            extreme_temps = final_filtered_data[final_filtered_data['temperature_celsius'] > temp_high].assign(event='High Temperature')
            extreme_rain = final_filtered_data[final_filtered_data['precip_mm'] > rain_high].assign(event='Heavy Rainfall')
            extreme_wind = final_filtered_data[final_filtered_data['wind_kph'] > wind_high].assign(event='High Wind')
            extreme_events = pd.concat([extreme_temps, extreme_rain, extreme_wind])


            if viz_option == "Global Extreme Event Map":
                st.subheader(f"⚠️ Extreme Weather Events Around the World (Threshold: {threshold_type})")
                
                if extreme_events.empty:
                    st.info("No extreme events detected based on the current filters.")
                else:
                    fig = px.scatter_geo(extreme_events, lat='latitude', lon='longitude', color='event', hover_name='location_name', color_discrete_map={'High Temperature': '#FB923C', 'Heavy Rainfall': '#3B82F6', 'High Wind': '#14B8A6'})
                    fig.update_traces(marker=dict(size=8, opacity=0.8, symbol='circle'))
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

            elif viz_option == "Extreme Events Frequency Dashboard":
                st.subheader(f"🚨 Extreme Events Frequency & Distribution (Threshold: {threshold_type})")
                
                if extreme_events.empty:
                    st.info("No extreme events detected based on the current filters.")
                else:
                    event_counts = extreme_events.groupby(['country', 'event']).size().reset_index(name='Count')
                    event_pivot = event_counts.pivot_table(index='country', columns='event', values='Count', fill_value=0)
                    event_pivot['Total'] = event_pivot.sum(axis=1)
                    event_pivot = event_pivot.sort_values(by='Total', ascending=False).head(10).drop(columns='Total')
                    fig = px.bar(event_pivot, x=event_pivot.index, y=event_pivot.columns, color_discrete_map={'High Temperature': '#EF4444', 'Heavy Rainfall': '#3B82F6', 'High Wind': '#10B981'})
                    fig.update_layout(xaxis_title="Country", yaxis_title="Total Count of Extreme Events (Days)", barmode='stack', template='plotly_white', height=600)
                    st.plotly_chart(fig, use_container_width=True)


else:
    st.error("❌ Unable to load data files. Please make sure 'CleanedWeatherRepository.csv' and 'CleanedWeatherRepositoryMonthly.csv' are in the same directory as this script.")

# Footer (Fixed Syntax Error)
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%); 
             border-radius: 10px; color: #4B5563;'>
    📊 Dataset: Global Climate Data 2024 | Last Updated: **{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}** | 
    <span style='color: #7C3AED; font-weight: 600;'>ClimateScope Analytics Platform</span>
</div>
""", unsafe_allow_html=True)