# NutriGuide Personalized AI Meal Recommender 

## Description

NutriGuide is a personalized AI meal optimizer that recommends meals based on user profiles, recent activities, and dietary preferences. The project includes data collection, processing, and recommendation modules.

## Data Collection

The `Data collection` directory contains the `data.csv` file, which includes information about various meals, their ingredients, and categories.

## Data Processing

The `Data processing` directory contains the `dataset_processing.ipynb` notebook, which processes the collected data to extract relevant features such as nutrients, diseases, and dietary preferences. The processed data is saved in `dataset.csv`, `recent_activity.csv`, and `user_Profiles.csv`.

## Recommendation

The `Recommendation` directory contains the `final.ipynb` notebook, which implements various recommendation algorithms, including K-Nearest Neighbors (KNN), Content-Based Filtering, and Singular Value Decomposition (SVD). The `MealRecommender-KNNScoring.py` script provides additional scoring functionality for the KNN recommender.

## Usage

1. **Data Processing**: Run the `dataset_processing.ipynb` notebook to process the collected data and generate the necessary datasets.
2. **Recommendation**: Run the `final.ipynb` notebook to generate meal recommendations based on user profiles and recent activities.

## Dependencies

- Python 3.11.7
- pandas
- numpy
- scikit-learn
- scikit-surprise
- tabulate
- BeautifulSoup
- requests
- seaborn
- matplotlib
- nltk

## Installation

To install the required dependencies, run the following commands:

```sh
pip install pandas numpy scikit-learn scikit-surprise tabulate beautifulsoup4 requests seaborn matplotlib nltk
