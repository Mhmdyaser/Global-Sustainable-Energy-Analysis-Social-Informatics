 
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

```
3. **Run the container (Volume Mapping):**

```bash
docker run -v "%cd%:/app" social-informatics-project

```

---

## 📦 Portable Version (Option 2: Pre-built Image)

Use this option if you want to run the project without building it from the Dockerfile.

### 1. Download the Image

* **[Download social-informatics.tar](https://drive.google.com/file/d/1VEwmUuBaBHCPEGySlT940chZb6R8je8e/view?usp=drivesdk)** - **Size:** ~327MB

### 2. Load and Execution

Open your CMD in the folder where the `.tar` file is located and run:




1. **Load the image into Docker**
 
```bash
docker load -i social-informatics.tar

```
3.  **Run the container**

```bash
docker run -v "%cd%:/app" social-informatics-project

```



---

## 📊 Generated Insights

After running the container, you will find 17 PNG images in your directory:

1. `01_missing_values.png`
2. `02_correlation_heatmap.png`
3. `03_top_15_co2.png`
4. `04_top_10_countries_co2.png`
5. `05_co2_by_year.png`
6. `06_co2_boxplot.png`
7. `07_co2_scatter_subplots.png`
8. `08_top_10_years_co2.png`
9. `09_fossil_fuels_line.png`
10. `10_top_10_fossil_bar.png`
11. `11_fossil_boxplot.png`
12. `12_renewables_line.png`
13. `13_renewables_boxplot.png`
14. `14_land_area_bar.png`
15. `15_top_10_land.png`
16. `16_big_countries_co2.png`
17. `17_feature_importance.png`

**Developed by:** Team Code X

**Field:** Faculty of Computers and Artificial Intelligence (FCAI) 🚀
