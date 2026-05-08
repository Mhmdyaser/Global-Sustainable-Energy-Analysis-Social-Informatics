import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

# Read the data
print("--- Loading Data ---")
df=pd.read_csv('global-data-on-sustainable-energy (1).csv')

# 2. تحديد الترتيب الجديد للاعمدة
new_column_order = [
    'Entity', 'Year', 'Latitude', 'Longitude', 'Land Area(Km2)', 'Density\\n(P/Km2)',
    'Access to electricity (% of population)', 'Access to clean fuels for cooking',
    'Electricity from fossil fuels (TWh)', 'Electricity from nuclear (TWh)', 
    'Electricity from renewables (TWh)', 'Low-carbon electricity (% electricity)',
    'Renewable-electricity-generating-capacity-per-capita', 
    'Renewable energy share in the total final energy consumption (%)',
    'Renewables (% equivalent primary energy)', 'Financial flows to developing countries (US $)',
    'Primary energy consumption per capita (kWh/person)', 
    'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
    'Value_co2_emissions_kt_by_country',
    'gdp_growth', 'gdp_per_capita'
]
df = df[new_column_order]

df=pd.DataFrame(df)
print("\n--- Data Head ---")
print(df.head(10))

# Data Exploration
print("\n--- Data Exploration ---")
print("Columns:", df.columns)
print("Shape:", df.shape)
print("\nInfo:")
df.info()
print("\nDescribe Numeric:\n", df.describe().T)
print("\nDescribe Objects:\n", df.describe(include = 'object').T)
unique_values = df.nunique()
print("\nUnique Values:\n", unique_values)

# Searching for null values & duplicate (Data Cleaning)
print("\n--- Data Cleaning ---")
missing_values = df.isna().sum()
print("Missing Values per column:\n", missing_values)

# Creating a bar plot for missing values
fig = px.bar(x=missing_values.index, y=missing_values.values, labels={'x': 'Columns', 'y': 'Missing Values Count'},
             title='Count of Missing Values in Each Column')
fig.write_image('01_missing_values.png') # Changed from show()
print("Saved: 01_missing_values.png")

# Drop columns with a high number of missing values
df.drop(columns=['Financial flows to developing countries (US $)','Renewables (% equivalent primary energy)',
                 'Renewable-electricity-generating-capacity-per-capita'], inplace=True)

# Calculate mean
Mean_Access = df['Access to clean fuels for cooking'].mean()
Mean_Renewable = df['Renewable energy share in the total final energy consumption (%)'].mean()
Mean_Electricity = df['Electricity from nuclear (TWh)'].mean()
Mean_Energy = df['Energy intensity level of primary energy (MJ/$2017 PPP GDP)'].mean()
Mean_Value_co2 = df['Value_co2_emissions_kt_by_country'].mean()
Mean_gdp_growth = df['gdp_growth'].mean()
Mean_gdp_per_capita = df['gdp_per_capita'].mean()

# Fill missing values
df['Access to clean fuels for cooking'].fillna(Mean_Access, inplace=True)
df['Renewable energy share in the total final energy consumption (%)'].fillna(Mean_Renewable, inplace=True)
df['Electricity from nuclear (TWh)'].fillna(Mean_Electricity, inplace=True)
df['Energy intensity level of primary energy (MJ/$2017 PPP GDP)'].fillna(Mean_Energy, inplace=True)
df['Value_co2_emissions_kt_by_country'].fillna(Mean_Value_co2, inplace=True)
df['gdp_growth'].fillna(Mean_gdp_growth, inplace=True)
df['gdp_per_capita'].fillna(Mean_gdp_per_capita, inplace=True)
# Drop rows with any remaining missing values
df = df.dropna()

print("\nShape after cleaning:", df.shape)
missing_values_after = df.isna().sum()
print("Missing values after:\n", missing_values_after)

# Duplicate Row
duplicate_rows=df.duplicated().sum()
print("Number of duplicate rows:", duplicate_rows)

# Feature Engineering
print("\n--- Feature Engineering ---")
df.rename(columns={"Value_co2_emissions_kt_by_country":"CO2" , 'Land Area(Km2)':'Land'} , inplace=True)
df.rename(columns={'Density\\n(P/Km2)': 'Density'}, inplace=True)
df['Density'] = df['Density'].str.replace(',', '').astype(int)

energy_land = df[['Entity', 'Land']].dropna()
countries = energy_land['Entity'].unique()
land = energy_land['Land'].unique()

# Clean the land area values by converting to integers
land_int = []
for num in land:
    if isinstance(num, float):
        land_int.append(int(num))
    else:
        land_int.append(int(str(num).replace(',', '')))

print("\nHead before scaling:\n", df.head())

# Columns to be scaled
columns_to_scale = ['Electricity from fossil fuels (TWh)','CO2', 'Land','Electricity from nuclear (TWh)','Electricity from renewables (TWh)','Density']
data_to_scale = df[columns_to_scale]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_to_scale)
df_scaled = df.copy()
df_scaled[columns_to_scale] = scaled_data

print("\nScaled DataFrame head:\n", df_scaled.head())

# correlation_matrix
print("\n--- Correlation Heatmap ---")
correlation_matrix = df.corr(numeric_only=True)
fig_corr = px.imshow(
    correlation_matrix,
    labels=dict(x="Features", y="Features", color="Correlation"),
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    color_continuous_scale='blues',
    title='Correlation Heatmap',
    height=1200
)
fig_corr.write_image('02_correlation_heatmap.png') 
print("Saved: 02_correlation_heatmap.png")

print('Top 5 Most Positively Correlated to CO2:\n', correlation_matrix['CO2'].sort_values(ascending=False).head(5))
print('Top 5 Most Negatively Correlated to CO2:\n', correlation_matrix['CO2'].sort_values(ascending=True).head(5))

# Feature Selection
protected_columns = ['gdp_per_capita', 'Primary energy consumption per capita (kWh/person)', 'Year']
columns_to_drop = [
    col for col in correlation_matrix.columns 
    if abs(correlation_matrix.loc['CO2', col]) < 0.5 
    and col not in protected_columns 
    and col != 'CO2'
]
print('\nThe number of columns to drop is:', len(columns_to_drop))
print('Columns to drop:', columns_to_drop)
df = df.drop(columns_to_drop, axis=1)

print("\nColumns remaining after dropping:")
print(df.columns.tolist())
print(df.head(10))

# Final Scaling
scaler = MinMaxScaler()
columns_to_scale = ['Land', 'Primary energy consumption per capita (kWh/person)', 'gdp_per_capita']
data_to_scale = df[columns_to_scale]
scaled_data = scaler.fit_transform(data_to_scale)
df_scaled = df.copy()
df_scaled[columns_to_scale] = scaled_data
print("\nFinal Scaled Head:\n", df_scaled.head(10))

# --- Data visualisation ---
print("\n--- Generating Visualisations ---")

# The Target Column 'CO2' Bar Chart
top_CO2 = df['CO2'].nlargest(15)
locations = df.loc[top_CO2.index]['Entity']
# Plotting the top 15 prices using Matplotlib
plt.figure(figsize=(10, 6))  
plt.bar(range(len(top_CO2)), top_CO2, color='#7B66FF')  
plt.xlabel('Country')  
plt.ylabel('CO2') 
plt.legend(['CO2'])
plt.title('Top 15 CO2') 
plt.xticks(range(len(top_CO2)), locations, rotation=45)  
plt.tight_layout()  
plt.savefig('03_top_15_co2.png') 
plt.close()
print("Saved: 03_top_15_co2.png")


# Calculate the maximum 'CO2' emissions for each 'Country' category and sort in descending order
max_co2 = df.groupby('Entity')['CO2'].max().reset_index()
max_co2 = max_co2.sort_values(by='CO2', ascending=False)

# Select the top 10 'Country' categories with the highest maximum 'CO2' emissions
top_10_high_co2 = max_co2.head(10)

# Create a bar plot using Plotly Express
fig = px.bar(
    top_10_high_co2,  # DataFrame containing the data
    x='Entity',  # x-values: 'Country' categories
    y='CO2',  # y-values: maximum 'CO2' emissions
    color='CO2',  # Color the bars based on the indices
    title='Top 10 Countries by Maximum CO2 Emissions',  # Set the title of the plot
    labels={'Country': 'Country', 'CO2': 'CO2 Emissions'},  # Customize labels
    template='plotly_white'  # Use a white template for the plot
)
fig.update_layout(height=550)
fig.write_image('04_top_10_countries_co2.png') # Changed from show()
print("Saved: 04_top_10_countries_co2.png")

# Calculate the median 'CO2' emissions for each 'Year'
CO2_By_Year = df.groupby('Year')['CO2'].max().reset_index()

# Create a line plot using Plotly Express
fig_CO2_By_Year = px.line(
    CO2_By_Year,  # DataFrame containing the data
    x='Year',   # x-values: Year
    y='CO2',  # y-values: median CO2
    labels={'Year': 'Year'},  # Customize label for the x-axis
    title='Maxmum CO2 Emissions by Year',  # Set the title of the plot
    height=500  # Set the height of the plot
)
fig_CO2_By_Year.write_image('05_co2_by_year.png') # Changed from show()
print("Saved: 05_co2_by_year.png")

# CO2 BoxPlot
fig3 = px.box(df_scaled, y='CO2', template='plotly_white', title='CO2 emission (BoxPlot)')
fig3.update_layout(font=dict(size=17, family="Franklin Gothic"))
fig3.write_image('06_co2_boxplot.png') 
print("Saved: 06_co2_boxplot.png")

# Scatter Subplots
columns_to_plot = [
    ('Electricity from fossil fuels (TWh)', 'CO2', 'CO2 vs Fossil Fuels'),
    ('Electricity from renewables (TWh)', 'CO2', 'CO2 vs Renewables'),
    ('Land','CO2','CO2 vs Land'),
    ('Electricity from nuclear (TWh)','CO2','CO2 vs Nuclear')
]
fig4 = make_subplots(rows=2, cols=2, subplot_titles=[title for _, _, title in columns_to_plot])
for i, (column, y_label, title) in enumerate(columns_to_plot, start=1):
    data = df_scaled.groupby(column)[y_label].sum().reset_index()
    fig4.add_trace(go.Scatter(x=data[column], y=data[y_label], mode='markers', name=title), row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)
fig4.update_layout(height=1000, width=1000, showlegend=False, title='CO2 Emissions by Various Factors')
fig4.write_image('07_co2_scatter_subplots.png') 
print("Saved: 07_co2_scatter_subplots.png")

# Top 10 Years Bar Chart
Max_CO2_Year = df.groupby('Year')['CO2'].max().reset_index().sort_values(by='CO2', ascending=False).head(10)
fig5 = px.bar(Max_CO2_Year, x='Year', y='CO2', color='CO2', title='Top 10 Years by Max CO2 Emissions', template='plotly_white')
fig5.update_traces(textfont_color='black')
fig5.update_layout(height=650)
fig5.write_image('08_top_10_years_co2.png') 
print("Saved: 08_top_10_years_co2.png")

# Fossil Fuels Line Plot
Entity_By_Fossil = df.groupby('Entity')['Electricity from fossil fuels (TWh)'].max().reset_index()
fig6 = px.line(Entity_By_Fossil, x='Entity', y='Electricity from fossil fuels (TWh)', title='Entity by Electricity from fossil fuels (TWh)', height=650)
fig6.write_image('09_fossil_fuels_line.png') 
print("Saved: 09_fossil_fuels_line.png")

# Top 10 Fossil Fuels (Matplotlib)
plt.figure(figsize=(10, 6))  
top_fossil = df['Electricity from fossil fuels (TWh)'].nlargest(10)
locs_fossil = df.loc[top_fossil.index]['Entity']
plt.bar(range(len(top_fossil)), top_fossil, color='#7B66FF')  
plt.title('Top 10 Electricity from fossil fuels (TWh)') 
plt.xticks(range(len(top_fossil)), locs_fossil, rotation=45)  
plt.tight_layout()  
plt.savefig('10_top_10_fossil_bar.png') 
plt.close()
print("Saved: 10_top_10_fossil_bar.png")

# Fossil Fuels BoxPlot
fig7 = px.box(df_scaled, y='Electricity from fossil fuels (TWh)', template='plotly_white', title='Electricity from fossil fuels (TWh)')
fig7.update_layout(font=dict(size=17, family="Franklin Gothic"))
fig7.write_image('11_fossil_boxplot.png') 
print("Saved: 11_fossil_boxplot.png")

# Renewables Line Plot
Entity_By_Renewables = df.groupby('Entity')['Electricity from renewables (TWh)'].max().reset_index()
fig8 = px.line(Entity_By_Renewables, x='Entity', y='Electricity from renewables (TWh)', title='Electricity from renewables (TWh)', height=650)
fig8.write_image('12_renewables_line.png') 
print("Saved: 12_renewables_line.png")

# Renewables BoxPlot
fig9 = px.box(df_scaled, y='Electricity from renewables (TWh)', template='plotly_white', title='Electricity from renewables (TWh)')
fig9.update_layout(font=dict(size=17, family="Franklin Gothic"))
fig9.write_image('13_renewables_boxplot.png') 
print("Saved: 13_renewables_boxplot.png")

# Land Area Bar Chart
energy_land_data_use_df = pd.DataFrame({'Country': countries, 'Land': land_int})
fig10 = px.bar(energy_land_data_use_df, x='Country', y='Land', labels={'Land': 'Land Area - km2'}, title='Countries Land Area - in km2')
fig10.update_layout(title={'x': 0.5})
fig10.write_image('14_land_area_bar.png') 
print("Saved: 14_land_area_bar.png")

# Top 10 Land
max_land = df.groupby('Entity')['Land'].max().reset_index().sort_values(by='Land', ascending=False).head(10)
fig11 = px.bar(max_land, x='Entity', y='Land', color='Land', title='Top 10 Countries by Land', template='plotly_white')
fig11.update_layout(height=650)
fig11.write_image('15_top_10_land.png') 
print("Saved: 15_top_10_land.png")

# Entity & Year Comparison
energy_co2_data = df[['Entity', 'Year', 'CO2']]
print("\nEnergy CO2 Data Sample:\n", energy_co2_data.head())

countries_list = ['Canada', 'United States', 'China', 'Brazil', 'Australia']
fig12 = make_subplots(rows=5, cols=1, subplot_titles=countries_list)
for i, country in enumerate(countries_list, start=1):
    c_data = energy_co2_data[energy_co2_data['Entity'] == country]
    fig12.add_trace(go.Bar(x=c_data['Year'], y=c_data['CO2']), row=i, col=1)

fig12.update_layout(height=1200, width=1200, showlegend=False, title='CO2 emission - Five biggest countries')
fig12.write_image('16_big_countries_co2.png') # Changed from show()
print("Saved: 16_big_countries_co2.png")

# Encoding
print("\n--- Encoding ---")
le = LabelEncoder()
df.Entity = le.fit_transform(df.Entity)
print("Entity head after encoding:\n", df.head())

# Splitting Dataset
X = df.drop(columns=['CO2'])
y = df['CO2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Model Building and Analysis
print("\n--- Model Building and Analysis ---")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
}

best_model = None
best_r2 = 0

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Track best model
    if r2 > best_r2:
        best_r2 = r2
        best_model = model_name

    # Results table
    submit = pd.DataFrame({
        'Actual CO2': y_test,
        'Predict_CO2': y_pred
    }).reset_index(drop=True)

    print(f'\n{model_name}:')
    print(f'R2 Score: {r2:.2f}')
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    print(submit.head(5))
    print('----------------------------------------')

print("\nBest Model:", best_model)
print("Best R2 Score:", best_r2)

# Feature Importance
print("\n--- Feature Importance ---")
importances = model.feature_importances_
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, importances))
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
top_feature_names, top_importances = zip(*sorted_feature_importance[:5])

fig13 = px.bar(x=top_importances, y=top_feature_names, orientation='h', title='Top 5 Feature Importance', labels={'x': 'Importance', 'y': 'Feature'}, color=top_importances, color_continuous_scale='reds')
fig13.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig13.write_image('17_feature_importance.png') 
print("Saved: 17_feature_importance.png")

# Forward Selection
x = df.drop(columns=['CO2'])
y = df['CO2']

def forward_selection(X, y, model, threshold=0.01):
    selected_features = []
    remaining_features = list(X.columns)
    best_score = 0

    while remaining_features:
        scores = []
        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            X_train, X_test, y_train, y_test = train_test_split(X[features_to_test], y, test_size=0.3, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            scores.append((feature, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_feature, best_feature_score = scores[0]

        if best_feature_score - best_score > threshold:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_score = best_feature_score
        else:
            break

    return selected_features
x=df.drop('CO2', axis=1)
y=df['CO2']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Display the shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Model after Forward Selection
y_train = y_train.loc[X_train.index]
y_test = y_test.loc[X_test.index]

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
}

best_model = None
best_r2 = 0

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    submit = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    }).reset_index(drop=True)

    if r2 > best_r2:
        best_r2 = r2
        best_model = model_name 

    print(f'{model_name}:')
    print(f'R2 Score: {r2:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(submit.head(5))
    print('----------------------------------------')



print("\n--- Process Finished ---")
