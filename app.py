import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
from prophet.utilities import regressor_coefficients
import plotly.graph_objects as go
import plotly.express as px
import calendar
import base64
import time
import json
from datetime import datetime
from prophet.plot import plot_plotly, plot_components_plotly

st.title('📈 DIMA AI Predictor')
st.info(f"""
Running data analysis and forecasting on your data.
""")
# Step 1: Import Data
df = st.file_uploader('Import .cvs files and assure one column contains the date. How can you forecast time if you do not track it xD.', type='csv')

if df is not None:
    data = pd.read_csv(df)
    data_columns = list(data.columns)  # Get the list of column names in the dataset

    st.write(data)

# Step 2: Select Forecast Horizon
periods_input = st.number_input('How many days would you like to forecast into the future?', min_value=1, max_value=365)

# Step 2: Specify column names dynamically
ds_column = st.text_input('Enter the column name containing (time)', value='ds')
y_column = st.text_input('Enter the column name you want to forecast (target variable)', value='y')
base_price_column = st.text_input('Enter the column name for new parameter', value='base_price')
gender_column = st.text_input('Enter the column name for second text parameter', value='gender')

# Run button
if st.button('RUN'):
  # Create a new DataFrame with updated column names
 new_data = pd.DataFrame()
new_data['ds'] = pd.to_datetime(data[ds_column]).dt.date  # Extract only the date portion
new_data['y'] = data[y_column]
if base_price_column in data_columns:
    new_data['base_price'] = data[base_price_column]
    new_data = new_data.rename(columns={base_price_column: 'base_price'})
if gender_column in data_columns:
    new_data['gender'] = data[gender_column]

# Print the new_data DataFrame to verify column names
st.write(new_data)

# Initialize the progress bar
progress_bar = st.progress(0)

# Perform the training and forecasting
m = Prophet()

if 'base_price' in new_data.columns:
    m.add_regressor('base_price')
if 'gender' in new_data.columns:
    m.add_regressor('gender')

m.fit(new_data)



future = m.make_future_dataframe(periods=periods_input)

if 'base_price' in new_data.columns:
    future['base_price'] = new_data['base_price']
if 'gender' in new_data.columns:
    future['gender'] = new_data['gender']

forecast = m.predict(future)

# Update the progress bar
progress_bar.progress(1.0)

# Add a short delay to show the progress bar completion
time.sleep(0.5)
# Display the training details as a CSV
# st.write(m.stan_backend.log_csv, format="csv")
# Calculate and display the model accuracy

# Display the forecast result
st.write(forecast)
# Add the code specific to Page 1 here
# Display the plot components
# Extract the predicted values from the forecast DataFrame
predicted_values = forecast['yhat']

# Get the actual values from the 'y' column in new_data
actual_values = new_data['y']

# Calculate the correlation between the predicted and actual values
correlation = predicted_values.corr(actual_values)

# Display the correlation coefficient
st.write("Model Accuracy (Correlation):", correlation)

# Display the model parameters

# Display the model parameters as collapsible JSON
with st.expander("Model Parameters"):
       st.write(m.params)

coefficients = regressor_coefficients(m)
# Display the correlation coefficient
st.write("Regressor Coefficients:", coefficients)
# print(coefficients)

# Display the changepoints
st.write("Changepoints:")
with st.expander("Changepoints:"):
        st.write(m.changepoints)
# Step 3: Visualize Forecast Data
st.write("The below visual shows future predicted values. 'yhat' is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.")

if 'y' in forecast.columns and 'yhat' in forecast.columns:
    fcst_filtered = forecast[forecast[ds_column] > new_data['ds'].max()]
    st.write(fcst_filtered)
    # st.write(fcst_filtered)

    # Calculate MAPE if both 'y' and 'yhat' columns exist in the forecast dataframe
    forecast['y_combined'] = forecast['y'].fillna(forecast['yhat'])
    df_m = forecast[[ds_column, base_price_column, 'y_combined']].join(new_data.set_index(ds_column)[y_column]).reset_index()
    df_m['abs_err'] = np.abs(df_m[y_column] - df_m['y_combined'])
    df_m['pct_err'] = df_m['abs_err'] / df_m[y_column]
    mape = df_m['pct_err'].mean()
    mape_formatted = f'{mape * 100:.2f}%'

    st.write(f"Current Accuracy: {mape_formatted}")
else:
    st.write("Cannot calculate accuracy. 'y' or 'yhat' column is missing in the forecast.")

# Plot the forecast using plotly
fig1 = go.Figure()

# Plot the subsampled actual values as scatter points
subsampled_new_data = new_data[::9]  # Replace 'n' with the subsampling factor you desire

fig1.add_trace(go.Scatter(
    x=subsampled_new_data['ds'],
    y=subsampled_new_data['y'],
    mode='markers',
    name='Actual',
    marker=dict(color='black')
))
# Plot the predicted values as a line
fig1.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Predicted',
    line=dict(color='blue')
))

# Plot the uncertainty intervals as faceted area plots with half opacity
fig1.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_upper'],
    fill=None,
    mode='lines',
    line=dict(color='blue'),
    showlegend=False
))

fig1.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_lower'],
    fill='tonexty',
    mode='lines',
    line=dict(color='blue'),
    fillcolor='rgba(0, 0, 255, 0.5)',
    name='Uncertainty Interval'
))

# Plot additional columns if present in the dataset
if base_price_column in data_columns:
    fig1.add_trace(go.Scatter(
        x=new_data['ds'],
        y=new_data['base_price'],
        mode='lines',
        name='Base Price',
        line=dict(color='green')
    ))
if gender_column in data_columns:
    fig1.add_trace(go.Scatter(
        x=new_data['ds'],
        y=new_data['gender'],
        mode='lines',
        name='Gender',
        line=dict(color='red')
    ))

fig1.update_layout(
    title='Actual vs Predicted Values',
    xaxis_title='Date',
    yaxis_title='Value',
    hovermode='x',
    showlegend=True
)
fig1.update_xaxes(rangeselector=dict(buttons=list([
    dict(count=7, label='1w', step='day', stepmode='backward'),
    dict(count=1, label='1m', step='month', stepmode='backward'),
    dict(count=6, label='6m', step='month', stepmode='backward'),
    dict(count=1, label='YTD', step='year', stepmode='todate'),
    dict(count=1, label='1y', step='year', stepmode='backward'),
    dict(step='all')
])))

st.plotly_chart(fig1)

#only dots on intersection
plot_plotly(m, forecast)

# The next few visuals show a high-level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years).
fig1 = m.plot(forecast)
st.write(fig1)
    
fig2 = m.plot_components(forecast)

st.write(fig2)

# Filter the forecast data for the next predicted week
# Convert the max date to datetime object
# Get the first 7 days of predicted data from the forecast
next_week_forecast = forecast.loc[:6, ['ds', 'yhat']]
next_week_forecast['DayOfWeek'] = next_week_forecast['ds'].dt.day_name()

# Create the heatmap
fig = go.Figure(data=go.Heatmap(
    z=[next_week_forecast['yhat']],
    x=next_week_forecast['ds'],
    y=[next_week_forecast['DayOfWeek']],
    colorscale='Blues',
    colorbar=dict(title='yhat'),
))

# Set the plot layout
fig.update_layout(
    title='Heatmap - Next Predicted Week',
    xaxis_title='Day',
    yaxis_title=None,  # Remove the y-axis title
    yaxis_showticklabels=False,  # Remove the y-axis tick labels
)

# Display the plot
st.plotly_chart(fig)

# Find the date with the highest predicted number of "y" items
max_date = forecast.loc[forecast['yhat'].idxmax(), 'ds']
max_item_number = forecast.loc[forecast['yhat'].idxmax(), 'yhat']

min_date = forecast.loc[forecast['yhat'].idxmin(), 'ds']
min_item_number = forecast.loc[forecast['yhat'].idxmin(), 'yhat']

max_date_str = max_date.strftime('%Y-%m-%d')
min_date_str = min_date.strftime('%Y-%m-%d')

max_phrase = f"<h4><b>Top predicted date is on</b> {max_date_str} <b>with</b> {max_item_number}  <b>items.</b></h4>"
min_phrase = f"<h4><b>Lowest predicted date is on</b> {min_date_str} <b>with</b> {min_item_number} <b>items.</b></h4>"


# Display the phrases
st.markdown(max_phrase, unsafe_allow_html=True)
st.markdown(min_phrase, unsafe_allow_html=True)
"""
### Step 4: Real data analysis
Some analysis on your data.
"""
if df is not None:
    # data = pd.read_csv(df)
    data['ds'] = pd.to_datetime(data['ds'], errors='coerce')
    data['base_price'] = pd.to_numeric(data['base_price'], errors='coerce')

    # Display the dataframe
    # st.write(data)
    
    # Calculate and display the pie chart
    gender_counts = data['Gender'].value_counts()
    fig_pie = px.pie(names=gender_counts.index, values=gender_counts.values)
    st.plotly_chart(fig_pie)

# Set up the layout with three columns
col1, col2 = st.beta_columns(2)

# Column 1
with col1:
    if df is not None:
        # data = pd.read_csv(df)
        data['ds'] = pd.to_datetime(data['ds'], errors='coerce')
        data['base_price'] = pd.to_numeric(data['base_price'], errors='coerce')

        # Display the dataframe
        # st.write(data)

        # Calculate and display the pie chart
        gender_counts = data['Gender'].value_counts()
        fig_pie = px.pie(names=gender_counts.index, values=gender_counts.values)
        st.plotly_chart(fig_pie, use_container_width=True)

# Column 2
with col2:
    # Group the data by product line and date to calculate the sales count
    grouped_data = data.groupby(['Product_line', 'ds']).size().reset_index(name='y')

    # Create a pivot table to reshape the data for the stacked bar chart
    pivot_table = grouped_data.pivot(index='ds', columns='Product_line', values='y').fillna(0)

    # Get the list of product lines
    product_lines = pivot_table.columns.tolist()

    # Filter options
    filter_options = ['Week', 'Month', 'Year', 'All']
    selected_filter = st.selectbox('Select time interval', filter_options)

    if selected_filter == 'Week':
        pivot_table = pivot_table.resample('W').sum()
    elif selected_filter == 'Month':
        pivot_table = pivot_table.resample('M').sum()
    elif selected_filter == 'Year':
        pivot_table = pivot_table.resample('Y').sum()

    # Create the stacked bar chart
    fig_bar = go.Figure()

    for product_line in product_lines:
        fig_bar.add_trace(go.Bar(
            x=pivot_table.index,
            y=pivot_table[product_line],
            name=product_line
        ))

    # Update layout and labels
    fig_bar.update_layout(
        title=f'Sales Performance by Product Line ({selected_filter})',
        xaxis_title='Date',
        yaxis_title='Sales Count',
        hovermode='x',
        barmode='stack',
        showlegend=True
    )

    # Display the stacked bar chart
    st.plotly_chart(fig_bar, use_container_width=True)


"""
### Step 4: Download the Forecast Data

The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)
    
