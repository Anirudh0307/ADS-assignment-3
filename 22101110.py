import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import FuncFormatter

df_countries=pd.read_csv('E:\ZK05_P2\API_NE.EXP.GNFS.ZS_DS2_en_csv_v2_6299802.csv',skiprows=4)

def plot_exports_prediction(country_name, indicator_name, df_countries):
    # Extract years and Exports of Goods and Services data for the specified country and indicator
    country_data = df_countries[
        (df_countries['Country Name'] == country_name) & (df_countries['Indicator Name'] == indicator_name)
    ]
    years = country_data.columns[4:]  # Assuming the years start from the 5th column
    exports_data = country_data.iloc[:, 4:].values.flatten()

    # Convert years to numeric values
    years_numeric = pd.to_numeric(years, errors='coerce')
    exports_data = pd.to_numeric(exports_data, errors='coerce')

    # Remove rows with NaN or inf values
    valid_data_mask = np.isfinite(years_numeric) & np.isfinite(exports_data)
    years_numeric = years_numeric[valid_data_mask]
    exports_data = exports_data[valid_data_mask]

    # Define the model function
    def export_and_gs_model(year, a, b, c):
        return a * np.exp(b * (year - 1990)) + c

    # Curve fitting with increased maxfev
    params, covariance = curve_fit(
        export_and_gs_model, years_numeric, exports_data, p0=[1, -0.1, 90], maxfev=10000
    )

    # Optimal parameters
    a_opt, b_opt, c_opt = params

    # Generate model predictions for the year 2040
    year_2040 = 2040
    exports_2040 = export_and_gs_model(year_2040, a_opt, b_opt, c_opt)

    # Plot the original data and the fitted curve
    plt.figure(figsize=(10, 6))
    plt.scatter(
        years_numeric, exports_data, label='Actual Data', color='lightcoral', alpha=0.8, edgecolors='darkred', linewidths=0.7, marker='o'
    )
    plt.plot(
        years_numeric,
        export_and_gs_model(years_numeric, a_opt, b_opt, c_opt),
        label='Fitted Curve',
        color='deepskyblue',
        linewidth=2,
    )

    # Highlight the prediction for 2040
    plt.scatter(year_2040, exports_2040, color='limegreen', marker='*', label='Prediction for 2040', s=100, edgecolors='black')

    # Add labels and legend
    plt.title(f'Exports of Goods and Services Prediction for {country_name}', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Exports of Goods and Services (% of GDP)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Beautify the plot
    plt.style.use('classic')
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage:
countries = ['Bahrain', 'Australia', 'Brazil']
indicator = 'Exports of goods and services (% of GDP)'

for country in countries:
    plot_exports_prediction(country, indicator, df_countries)

# CLUTERING
# Extract data for the years 1999 and 2022
years = ['1970', '2020']
exports_data = df_countries[['Country Name'] + years]

# Drop rows with missing values
exports_data = exports_data.dropna()

# Set 'Country Name' as the index
exports_data.set_index('Country Name', inplace=True)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(exports_data)

# Perform KMeans clustering
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(normalized_data)

# Add cluster labels to the DataFrame
exports_data['Cluster'] = labels

# Visualize the clusters
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Cluster for 1999
scatter1 = axs[0].scatter(exports_data[years[0]], exports_data.index, c=labels, cmap='Set1', edgecolor='black', linewidth=1, alpha=0.8)
axs[0].set_title(f'Exports of Goods and Services in {years[0]}', color='darkblue', fontsize=14)
axs[0].set_xlabel('Exports of goods and services (% of GDP)', color='darkblue', fontsize=12)
axs[0].set_ylabel('Countries', color='darkblue', fontsize=12)

# Cluster for 2022
scatter2 = axs[1].scatter(exports_data[years[1]], exports_data.index, c=labels, cmap='Set1', edgecolor='black', linewidth=1, alpha=0.8)
axs[1].set_title(f'Exports of Goods and Services in {years[1]}', color='darkgreen', fontsize=14)
axs[1].set_xlabel('Exports of goods and services (% of GDP)', color='darkgreen', fontsize=12)
axs[1].set_ylabel('Countries', color='darkgreen', fontsize=12)

# Manually set y-axis label
for ax in axs:
    ax.set_yticks([])
    ax.set_yticklabels([])

# Set a light gray background
fig.patch.set_facecolor('#F0F0F0')
axs[0].set_facecolor('#F0F0F0')
axs[1].set_facecolor('#F0F0F0')

# Customize the legend
legend_labels = ['Cluster 0', 'Cluster 1']
legend = axs[0].legend(handles=[scatter1.legend_elements()[0][i] for i in range(len(legend_labels))], title='Clusters', labels=legend_labels, loc='upper left', facecolor='#F0F0F0', edgecolor='darkgray')
plt.setp(legend.get_texts(), color='darkgray')

plt.tight_layout()
plt.show()