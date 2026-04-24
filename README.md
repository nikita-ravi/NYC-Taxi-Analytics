# NYC Taxi Trip Analytics

A comprehensive big data analytics project leveraging **Apache Spark** and **PySpark ML** to analyze millions of NYC yellow taxi trip records. This project demonstrates end-to-end data engineering and machine learning workflows including demand forecasting, fare prediction, customer segmentation, and traffic congestion analysis.

## Project Overview

This project processes NYC Taxi & Limousine Commission (TLC) trip record data to extract actionable insights for transportation optimization, revenue forecasting, and urban planning. The analysis pipeline handles large-scale data processing using Spark's distributed computing capabilities.

### Key Objectives

| Analysis Area | Description | Model Performance |
|---------------|-------------|-------------------|
| **Demand Hotspot Analysis** | Identify high-demand pickup locations and temporal patterns | Peak Hour: 18:00, Peak Day: Friday |
| **Trip Duration Prediction** | Predict trip duration based on pickup time and locations | RMSE: 5.80 min, R²: 0.767 |
| **Passenger Segmentation** | Cluster passengers by trip characteristics using K-Means | Silhouette Score: 0.999 |
| **Payment Type Prediction** | Classify card vs. cash payments | Accuracy: 72.5%, AUC-ROC: 0.607 |
| **Fare Prediction** | Predict fare amounts based on trip features | RMSE: $4.20, R²: 0.874 |
| **Tip Amount Prediction** | Estimate tip amounts from trip characteristics | RMSE: $1.70, R²: 0.618 |
| **Fare Classification** | Predict high vs. low fare trips | Accuracy: 94.5%, AUC-ROC: 0.975 |
| **Congestion Analysis** | Identify traffic congestion hotspots | Avg Rush Hour Speed: 12.3 mph |

## Tech Stack

- **Apache Spark** - Distributed data processing
- **PySpark SQL** - Data manipulation and aggregation
- **PySpark ML** - Machine learning pipelines
- **Python** - Core programming language
- **Pandas / NumPy** - Data analysis
- **Matplotlib / Seaborn** - Data visualization
- **Google Colab** - Development environment

## Dataset

The project uses NYC TLC Yellow Taxi Trip Records in Parquet format, containing:

```
Schema:
├── VendorID: long
├── tpep_pickup_datetime: timestamp
├── tpep_dropoff_datetime: timestamp
├── passenger_count: double
├── trip_distance: double
├── RatecodeID: double
├── store_and_fwd_flag: string
├── PULocationID: long (pickup location)
├── DOLocationID: long (dropoff location)
├── payment_type: long
├── fare_amount: double
├── extra: double
├── mta_tax: double
├── tip_amount: double
├── tolls_amount: double
├── improvement_surcharge: double
├── total_amount: double
├── congestion_surcharge: double
└── airport_fee: integer
```

## Analysis Modules

### 1. Demand Hotspot Analysis

Identifies spatial and temporal patterns in taxi demand across NYC.

**Key Findings:**
- Peak demand hour: **18:00** (6 PM)
- Busiest day: **Friday**
- Top pickup location: **Zone 237** with 1.69M trips
- Top 5 locations: Zones 237, 161, 236, 186, 162

**Visualizations:**
- Hourly and daily demand bar charts
- Location-based trip count analysis
- Demand vs. average fare scatter plot
- Hourly demand heatmap for top 15 locations

### 2. Trip Duration Prediction

Predicts trip duration using a Random Forest Regressor based on early trip information.

**Features Used:**
- Hour of day, Day of week
- Pickup/Dropoff location IDs
- Trip distance

**Results:**
| Metric | Value |
|--------|-------|
| RMSE | 5.80 minutes |
| R² Score | 0.767 |

**Feature Importance:**
- `trip_distance`: 86.7%
- `hour_of_day`: 7.0%
- `PULocationID`: 2.9%

### 3. Passenger Segmentation

Uses K-Means clustering to segment passengers based on trip behavior.

**Clustering Features:**
- Trip distance
- Fare amount
- Hour of day
- Payment type

**Results:**
- **Silhouette Score: 0.999** (excellent cluster separation)
- 3 distinct passenger segments identified

**Segment Profiles:**
- **Cluster 0**: Short trips, lower fares, off-peak hours (regular commuters)
- **Cluster 1**: Long trips, higher fares, peak hours (airport/business travelers)
- **Cluster 2**: Medium trips, moderate fares (tourists/explorers)

### 4. Payment Type Prediction

Binary classification to predict card vs. cash payments.

**Features Used:**
- Fare amount, trip distance
- Time features (hour, day, rush hour, weekend)
- Pickup/Dropoff locations
- Trip duration, passenger count

**Results:**
| Metric | Value |
|--------|-------|
| Accuracy | 72.5% |
| AUC-ROC | 0.607 |
| Precision | 70.3% |
| Recall | 72.5% |
| F1 Score | 61.3% |

### 5. Fare Prediction

Regression model to predict fare amounts based on trip characteristics.

**Results:**
| Metric | Value |
|--------|-------|
| RMSE | $4.20 |
| R² Score | 0.874 |

**Top Feature Importance:**
1. `trip_distance`: 86.6%
2. `PULocationID`: 7.2%
3. `DOLocationID`: 5.0%

### 6. Tip Amount Prediction

Predicts tip amounts for credit card transactions using Gradient Boosted Trees.

**Results:**
| Metric | Value |
|--------|-------|
| RMSE | $1.70 |
| R² Score | 0.618 |

**Key Insights:**
- Best tipping hour: **16:00** (4 PM) with 34.1% average tip
- Worst tipping hour: **08:00** (8 AM) with 24.5% average tip

**Top Feature Importance:**
1. `fare_amount`: 50.9%
2. `hour_of_day`: 9.9%
3. `PULocationID`: 8.3%

### 7. High/Low Fare Classification

Binary classification to predict whether a trip will result in a high fare (>$15).

**Results:**
| Metric | Value |
|--------|-------|
| Accuracy | 94.5% |
| AUC-ROC | 0.975 |
| Precision | 94.5% |
| Recall | 94.5% |
| F1 Score | 94.4% |

**Top Feature Importance:**
- `trip_distance`: 93.2%
- `PULocationID`: 3.5%

### 8. Traffic Congestion Analysis

Identifies congestion hotspots based on average trip speeds.

**Key Findings:**
- Most congested locations: Zones 186, 161, 234, 100, 163
- Slowest hour: **15:00** (3 PM)
- Average rush hour speed: **12.3 mph**

**Prediction Model:**
| Metric | Value |
|--------|-------|
| RMSE | 3.81 mph |
| R² Score | 0.637 |

## Data Pipeline

```
Raw Parquet Files
       │
       ▼
┌──────────────────┐
│  Data Cleaning   │  • Remove outliers
│                  │  • Filter invalid records
│                  │  • Handle missing values
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Feature Eng.     │  • Time-based features
│                  │  • Derived metrics
│                  │  • Trip duration calc
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  ML Pipeline     │  • VectorAssembler
│                  │  • StandardScaler
│                  │  • Model Training
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Evaluation      │  • RMSE, R², AUC-ROC
│                  │  • Feature Importance
│                  │  • Visualizations
└──────────────────┘
```

## Machine Learning Models Used

| Task | Algorithm |
|------|-----------|
| Duration Prediction | Random Forest Regressor |
| Fare Prediction | Random Forest Regressor |
| Tip Prediction | Gradient Boosted Trees |
| Payment Classification | Random Forest Classifier |
| Fare Classification | Random Forest Classifier |
| Passenger Segmentation | K-Means Clustering |

## Project Structure

```
NYC-Taxi-Analytics/
├── NYC_Taxi_Analysis.ipynb    # Main analysis notebook
├── README.md                   # Project documentation
└── data/                       # Parquet data files (not included)
```

## Getting Started

### Prerequisites

- Python 3.7+
- Apache Spark 3.x
- Google Colab (or local Spark installation)

### Running the Analysis

1. Clone this repository
2. Upload the notebook to Google Colab
3. Mount your Google Drive containing the Parquet data files
4. Run all cells sequentially

```python
from google.colab import drive
drive.mount('/content/drive')

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("NYC Taxi Trip Analysis").getOrCreate()
```

## Key Insights Summary

| Category | Insight |
|----------|---------|
| **Peak Demand** | Friday at 6 PM, Midtown Manhattan (Zone 237) |
| **Trip Duration** | Distance is the strongest predictor (86.7% importance) |
| **Tipping Behavior** | Afternoon trips (4 PM) receive highest tips |
| **Payment Trends** | Card payments dominate, location strongly influences method |
| **Congestion** | Midtown locations slowest during afternoon rush (3 PM) |
| **Fare Prediction** | Model achieves 87.4% accuracy (R²) |

## Future Enhancements

- Integration with real-time streaming data using Spark Structured Streaming
- Geographic visualization with Folium/Kepler.gl maps
- Weather data integration for improved predictions
- Deep learning models (LSTM) for time-series forecasting
- Deployment as REST API using Flask/FastAPI

## License

This project is for educational and portfolio demonstration purposes.

---

*Built with Apache Spark and PySpark ML*
