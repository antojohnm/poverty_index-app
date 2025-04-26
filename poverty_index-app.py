import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import folium
import numpy as np
from sklearn.linear_model import LinearRegression
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

def get_state_coordinates(state_name):
    geolocator = Nominatim(user_agent="india_map")
    location = geolocator.geocode(f"{state_name}, India")
    if location:
        return location.latitude, location.longitude
    return None

st.title("India's Poverty Data Visualization")
st.header("About our Goal (United Nations)")
st.write("The first Sustainable Development Goal aims to “End poverty in all its forms everywhere”. Its seven associated targets aims, among others, to eradicate extreme poverty for all people everywhere, reduce at least by half the proportion of men, women and children of all ages living in poverty, and implement nationally appropriate social protection systems and measures for all, including floors, and by 2030 achieve substantial coverage of the poor and the vulnerable. This webpage shows the analysed data of the poverty rates in different state of India for the past decade and displays the predicting graph of the future of poverty rates.")
# Load data
data = {
    'State/UT': [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 
        'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 
        'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'Chandigarh', 'Delhi', 'Puducherry', 
        'Lakshadweep', 'Andaman and Nicobar Islands', 'Dadra and Nagar Haveli', 'Daman and Diu', 'Ladakh'
    ],
    'Headcount Ratio (%) - 2013': [
        10.8, 29.11, 36.97, 56.34, 35.42, 5.15, 21.53, 11.5, 10.14, 47.13, 16.55, 1.24, 41.57, 18.06, 20.53, 37.08, 12.47, 79.82, 34.28, 8.1, 
        33.86, 6.07, 7.16, 12.5, 20.94, 42.59, 20.85, 25.98, 2.5, 3.1, 6.5, 5.64, 7.2, 4.8, 3.6, 2.9
    ],
    'Headcount Ratio (%) - 2014': [
        10.5, 26.99, 34.47, 53.03, 32.78, 4.619, 20.14, 11.2, 9.44, 44.48, 15.34, 1.15, 38.61, 16.66, 18.84, 35.78, 11.50, 72.25, 31.70, 7.9, 
        31.29, 5.62, 6.52, 12.2, 19.89, 39.79, 19.25, 24.04, 2.3, 2.9, 6.2, 5.34, 6.9, 4.5, 3.3, 2.7
    ],
    'Headcount Ratio (%) - 2015': [
        10.0, 24.87, 31.97, 49.73, 30.15, 4.09, 18.75, 10.8, 8.75, 41.84, 14.13, 1.07, 35.67, 15.26, 17.16, 34.50, 10.54, 64.70, 29.12, 7.5, 
        28.73, 5.19, 5.89, 11.8, 18.84, 36.99, 17.67, 22.12, 2.1, 2.7, 5.9, 5.05, 6.6, 4.2, 3.0, 2.5
    ],
    'Headcount Ratio (%) - 2016': [
        9.5, 22.76, 29.47, 46.42, 27.52, 3.56, 17.36, 10.3, 8.05, 39.20, 12.92, 0.99, 32.72, 13.87, 15.48, 33.21, 9.57, 57.13, 26.54, 7.1, 
        26.16, 4.75, 5.25, 11.3, 17.79, 34.19, 16.07, 20.19, 1.9, 2.5, 5.6, 4.75, 6.3, 3.9, 2.7, 2.3
    ],
    'Headcount Ratio (%) - 2017': [
        8.9, 20.64, 26.97, 43.12, 24.88, 3.03, 15.97, 9.8, 7.36, 36.56, 11.71, 0.90, 29.77, 12.47, 13.79, 31.92, 8.60, 49.57, 23.96, 6.7, 
        23.60, 4.31, 4.61, 10.8, 16.74, 31.39, 14.48, 18.26, 1.7, 2.3, 5.3, 4.45, 6.0, 3.6, 2.4, 2.1
    ],
    'Headcount Ratio (%) - 2018': [
        8.2, 18.52, 24.47, 39.81, 22.25, 2.49, 14.59, 9.2, 6.66, 33.91, 10.51, 0.82, 26.81, 11.07, 12.11, 30.62, 7.64, 42.01, 21.39, 6.2, 
        21.03, 3.86, 3.98, 10.1, 15.69, 28.60, 12.89, 16.32, 1.5, 2.1, 5.0, 4.16, 5.7, 3.3, 2.1, 1.9
    ],
    'Headcount Ratio (%) - 2019': [
        7.5, 16.40, 21.97, 36.51, 19.61, 1.96, 13.20, 8.5, 5.97, 31.27, 9.30, 0.73, 23.86, 9.67, 10.42, 29.33, 6.67, 34.45, 18.81, 5.8, 
        18.47, 3.42, 3.34, 9.3, 14.64, 25.80, 11.30, 14.39, 1.3, 1.9, 4.7, 3.86, 5.4, 3.0, 1.8, 1.6
    ],
    'Headcount Ratio (%) - 2020': [
        6.9, 14.28, 19.47, 33.21, 17.98, 1.44, 11.81, 7.9, 5.28, 28.62, 8.09, 0.64, 20.91, 8.27, 8.73, 28.04, 5.70, 26.89, 16.23, 5.4, 
        15.91, 2.98, 2.70, 8.5, 13.59, 23.00, 9.71, 12.46, 1.1, 1.7, 4.4, 3.56, 5.1, 2.7, 1.5, 1.3
    ], 
    'Headcount Ratio (%) - 2021': [
        6.3, 12.16, 16.97, 29.91, 16.34, 1.02, 10.42, 7.3, 4.59, 25.98, 6.88, 0.55, 17.96, 6.87, 7.04, 26.75, 4.74, 19.33, 13.65, 5.0, 
        13.35, 2.54, 2.06, 7.7, 12.54, 20.20, 8.12, 10.53, 0.9, 1.5, 4.1, 3.26, 4.8, 2.4, 1.2, 1.0
    ],
    'Headcount Ratio (%) - 2022': [
        5.7, 10.04, 14.47, 26.61, 14.71, 0.59, 9.03, 6.7, 3.90, 23.34, 5.67, 0.46, 15.01, 5.47, 5.35, 25.46, 3.77, 11.77, 11.07, 4.6,
        10.79, 2.10, 1.42, 6.9, 11.49, 17.40, 6.53, 8.60, 0.7, 1.3, 3.8, 2.96, 4.5, 2.1, 0.9, 0.7
    ],
    'Headcount Ratio (%) - 2023': [
        5.1, 7.92, 11.97, 23.31, 13.07, 0.17, 7.64, 6.1, 3.21, 20.69, 4.46, 0.37, 12.06, 4.07, 3.66, 24.17, 2.81, 4.21, 8.49, 4.2, 
        8.23, 1.66, 0.78, 6.1, 10.44, 14.60, 4.94, 6.67, 0.5, 1.1, 3.5, 2.66, 4.2, 1.8, 0.6, 0.4
    ]
}



df = pd.DataFrame(data)

# Sidebar selection
st.sidebar.header('Select Options')
states = df['State/UT'].tolist()
selected_state = st.sidebar.selectbox('Select a state:', states)

graph_types = [' ','Line Chart', 'Bar Chart', 'Pie Chart']
selected_graph = st.sidebar.selectbox('Select the type of graph:', graph_types)

# Extract data for the selected state
years = [str(year) for year in range(2013, 2024)]
columns = [f'Headcount Ratio (%) - {year}' for year in years]
state_data = df[df['State/UT'] == selected_state][columns].values.flatten()

# Convert to DataFrame
plot_df = pd.DataFrame({'Year': years, 'Headcount Ratio (%)': state_data})

# Graphs


if selected_graph == 'Bar Chart':
    st.subheader('Bar Chart: Headcount Ratio Comparison')
    fig, ax = plt.subplots()
    ax.bar(plot_df['Year'], plot_df['Headcount Ratio (%)'], color='skyblue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Headcount Ratio (%)')
    ax.set_title('Headcount Ratio Comparison')
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif selected_graph == 'Line Chart':
    st.subheader('Interactive Line Chart: Headcount Ratio Over Years')
    fig = px.line(plot_df, x='Year', y='Headcount Ratio (%)', markers=True)
    st.plotly_chart(fig)

elif selected_graph == 'Pie Chart':
    st.subheader('Pie Chart: Percentage of Poverty Distribution Over the Years')
    fig = px.pie(plot_df, names='Year', values='Headcount Ratio (%)', title=f'Poverty Distribution in {selected_state}')
    st.plotly_chart(fig)

# Interactive Map
st.subheader("Interactive Map")
m = folium.Map(location=[20.5937, 78.9629], zoom_start=4)
coords = get_state_coordinates(selected_state)
if coords:
    folium.Marker(coords, popup=selected_state, icon=folium.Icon(color='blue')).add_to(m)
    m.location = coords
    m.zoom_start = 7
else:
    st.error("Could not find coordinates for the selected region.")
folium_static(m)

data1 = {
    'State/UT': [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 
        'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 
        'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'Chandigarh', 'Delhi', 'Puducherry', 
        'Lakshadweep', 'Andaman and Nicobar Islands', 'Dadra and Nagar Haveli', 'Daman and Diu', 'Ladakh'
    ],
    '2013': [10.8, 29.11, 36.97, 56.34, 35.42, 5.15, 21.53, 11.5, 10.14, 47.13, 16.55, 1.24, 41.57, 18.06, 20.53, 37.08, 12.47, 79.82, 34.28, 8.1, 
        33.86, 6.07, 7.16, 12.5, 20.94, 42.59, 20.85, 25.98, 2.5, 3.1, 6.5, 5.64, 7.2, 4.8, 3.6, 2.9],
    '2014': [10.5, 26.99, 34.47, 53.03, 32.78, 4.619, 20.14, 11.2, 9.44, 44.48, 15.34, 1.15, 38.61, 16.66, 18.84, 35.78, 11.50, 72.25, 31.70, 7.9, 
        31.29, 5.62, 6.52, 12.2, 19.89, 39.79, 19.25, 24.04, 2.3, 2.9, 6.2, 5.34, 6.9, 4.5, 3.3, 2.7],
    '2015': [10.0, 24.87, 31.97, 49.73, 30.15, 4.09, 18.75, 10.8, 8.75, 41.84, 14.13, 1.07, 35.67, 15.26, 17.16, 34.50, 10.54, 64.70, 29.12, 7.5, 
        28.73, 5.19, 5.89, 11.8, 18.84, 36.99, 17.67, 22.12, 2.1, 2.7, 5.9, 5.05, 6.6, 4.2, 3.0, 2.5],
    '2016': [9.5, 22.76, 29.47, 46.42, 27.52, 3.56, 17.36, 10.3, 8.05, 39.20, 12.92, 0.99, 32.72, 13.87, 15.48, 33.21, 9.57, 57.13, 26.54, 7.1, 
        26.16, 4.75, 5.25, 11.3, 17.79, 34.19, 16.07, 20.19, 1.9, 2.5, 5.6, 4.75, 6.3, 3.9, 2.7, 2.3],
    '2017': [8.9, 20.64, 26.97, 43.12, 24.88, 3.03, 15.97, 9.8, 7.36, 36.56, 11.71, 0.90, 29.77, 12.47, 13.79, 31.92, 8.60, 49.57, 23.96, 6.7, 
        23.60, 4.31, 4.61, 10.8, 16.74, 31.39, 14.48, 18.26, 1.7, 2.3, 5.3, 4.45, 6.0, 3.6, 2.4, 2.1],
    '2018': [
        8.2, 18.52, 24.47, 39.81, 22.25, 2.49, 14.59, 9.2, 6.66, 33.91, 10.51, 0.82, 26.81, 11.07, 12.11, 30.62, 7.64, 42.01, 21.39, 6.2, 
        21.03, 3.86, 3.98, 10.1, 15.69, 28.60, 12.89, 16.32, 1.5, 2.1, 5.0, 4.16, 5.7, 3.3, 2.1, 1.9
    ],
    '2019': [
        7.5, 16.40, 21.97, 36.51, 19.61, 1.96, 13.20, 8.5, 5.97, 31.27, 9.30, 0.73, 23.86, 9.67, 10.42, 29.33, 6.67, 34.45, 18.81, 5.8, 
        18.47, 3.42, 3.34, 9.3, 14.64, 25.80, 11.30, 14.39, 1.3, 1.9, 4.7, 3.86, 5.4, 3.0, 1.8, 1.6
    ],
    '2020': [
        6.9, 14.28, 19.47, 33.21, 17.98, 1.44, 11.81, 7.9, 5.28, 28.62, 8.09, 0.64, 20.91, 8.27, 8.73, 28.04, 5.70, 26.89, 16.23, 5.4, 
        15.91, 2.98, 2.70, 8.5, 13.59, 23.00, 9.71, 12.46, 1.1, 1.7, 4.4, 3.56, 5.1, 2.7, 1.5, 1.3
    ], 
    '2021': [
        6.3, 12.16, 16.97, 29.91, 16.34, 1.02, 10.42, 7.3, 4.59, 25.98, 6.88, 0.55, 17.96, 6.87, 7.04, 26.75, 4.74, 19.33, 13.65, 5.0, 
        13.35, 2.54, 2.06, 7.7, 12.54, 20.20, 8.12, 10.53, 0.9, 1.5, 4.1, 3.26, 4.8, 2.4, 1.2, 1.0
    ],
    '2022': [
        5.7, 10.04, 14.47, 26.61, 14.71, 0.59, 9.03, 6.7, 3.90, 23.34, 5.67, 0.46, 15.01, 5.47, 5.35, 25.46, 3.77, 11.77, 11.07, 4.6,
        10.79, 2.10, 1.42, 6.9, 11.49, 17.40, 6.53, 8.60, 0.7, 1.3, 3.8, 2.96, 4.5, 2.1, 0.9, 0.7
    ],
    '2023': [
        5.1, 7.92, 11.97, 23.31, 13.07, 0.17, 7.64, 6.1, 3.21, 20.69, 4.46, 0.37, 12.06, 4.07, 3.66, 24.17, 2.81, 4.21, 8.49, 4.2, 
        8.23, 1.66, 0.78, 6.1, 10.44, 14.60, 4.94, 6.67, 0.5, 1.1, 3.5, 2.66, 4.2, 1.8, 0.6, 0.4
    ]
}

df = pd.DataFrame(data1)

def predict_poverty(state, years):
    state_data = df[df['State/UT'] == state].iloc[:, 1:].values.flatten()
    valid_years = np.array([int(year) for year in df.columns[1:] if not np.isnan(state_data[df.columns.get_loc(year)-1])]).reshape(-1, 1)
    valid_rates = np.array([rate for rate in state_data if not np.isnan(rate)]).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(valid_years, valid_rates)
    future_years = np.array([int(df.columns[-1]) + i for i in range(1, years+1)]).reshape(-1, 1)
    future_predictions = model.predict(future_years)

    # Ensure predictions stay within 0.1% to 0.5%
    future_predictions = np.clip(future_predictions, 0.1, 100)
    
    return future_years.flatten(), future_predictions.flatten()


st.title("Poverty Rate Prediction in Indian States")
selected_state = st.selectbox("Select a State/UT", df['State/UT'])

if selected_state:
    past_years = np.array([int(year) for year in df.columns[1:] if not np.isnan(df[df['State/UT'] == selected_state].iloc[:, 1:].values.flatten()[df.columns.get_loc(year)-1])])
    past_rates = np.array([rate for rate in df[df['State/UT'] == selected_state].iloc[:, 1:].values.flatten() if not np.isnan(rate)])
    
    future_5_years, predictions_5 = predict_poverty(selected_state, 5)
    future_10_years, predictions_10 = predict_poverty(selected_state, 10)
    
    plt.figure(figsize=(10, 5))
    plt.plot(past_years, past_rates, marker='o', label='Past Data')
    plt.plot(future_5_years, predictions_5, marker='o', linestyle='dashed', label='5-Year Prediction')
    plt.plot(future_10_years, predictions_10, marker='o', linestyle='dotted', label='10-Year Prediction')
    plt.xlabel('Year')
    plt.ylabel('Poverty Rate (%)')
    plt.title(f'Poverty Rate Prediction for {selected_state}')
    plt.legend()
    plt.grid()
    st.pyplot(plt)
