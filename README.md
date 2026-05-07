# 🌍 Global Sustainable Energy Analysis - Social Informatics

An end-to-end data science pipeline designed to explore the interaction between **Sustainable Energy indicators** and **Socio-economic growth**. This project analyzes global trends (2000-2020) to predict environmental impact and GDP growth, fully containerized using **Docker** to ensure consistency across all environments.

---

## 🚀 Features
- **Automated Cleaning:** Robust handling of missing values (Mean Imputation) and data reordering for optimized analysis.
- **Deep Insights:** Generation of **17 analytical plots** covering energy access, carbon emissions, and country-specific comparisons.
- **Machine Learning:** Integrated predictive modeling using Linear Regression, Random Forest, and Gradient Boosting with an **R2 Score of 0.96**.
- **Feature Selection:** Automated feature selection using **Forward Selection** to identify the most significant socio-economic drivers.
- **Production Ready:** Fully Dockerized with optimized system dependencies for automated reporting.

---

## 🛠️ Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

---

## 💻 How to Run (Option 1: Build from Source)

If you have the source code and want to build the environment locally:

1. **Build the image:**
   ```bash
   docker build -t social-informatics-project .
Run the container (Volume Mapping):
(This will generate the 17 result images in your current folder)

Bash
docker run --rm -v "%cd%:/app" social-informatics-project
📦 Portable Version (Option 2: Pre-built Image)
Use this option if you want to run the project without building it from the Dockerfile.

1. Download the Image
Download social-informatics.tar - Size: ~250 MB

2. Load and Execution
Open your CMD in the folder where the .tar file is located and run:

Bash
# Load the image into Docker
docker load -i social-informatics.tar

# Run the container (Mapping results to your folder)
docker run --rm -v "%cd%:/app" social-informatics-project
📊 Generated Insights
After running the container, you will find 17 PNG images in your directory:

01_missing_values.png

02_correlation_heatmap.png

03_top_15_co2.png

04_top_10_countries_co2.png

05_co2_by_year.png

06_co2_boxplot.png

07_co2_scatter_subplots.png

08_top_10_years_co2.png

09_fossil_fuels_line.png

10_top_10_fossil_bar.png

11_fossil_boxplot.png

12_renewables_line.png

13_renewables_boxplot.png

14_land_area_bar.png

15_top_10_land.png

16_big_countries_co2.png

17_feature_importance.png

Developed by: Team Code X

Field: Faculty of Computers and Artificial Intelligence (FCAI) 🚀
