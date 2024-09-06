import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def train_regression_models(X, y):
    """
    Train and evaluate regression models.
    Returns models and a dictionary with MSE results.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'K-Nearest Neighbors': KNeighborsRegressor()
    }

    results = {}
    
    for name, model in models.items():
        model.fit(X, y)
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        results[name] = mse
        
    return models, results

def analyze_data(df, target_column):
    """
    Analyze the dataset and return a summary and insights about the data.
    The analysis includes a description of the dataset and insights on the target column.
    """
    try:
        # Summary of the dataset
        summary = df.describe(include='all')

        # Generate insights based on the target column
        insights = {}
        top_5 = df.nlargest(5, target_column)
        bottom_5 = df.nsmallest(5, target_column)

        insights['top_5_percent'] = top_5
        insights['bottom_5_percent'] = bottom_5

        return summary, insights
    except Exception as e:
        print(f"Error analyzing data: {e}")
        return None, None

def get_insights(df):
    """
    Generate insights: Top 5 highest and lowest values based on the 'Total' column.
    """
    try:
        if 'Total' not in df.columns:
            raise ValueError("'Total' column not found in the dataset.")
        
        top_5 = df.nlargest(5, 'Total')
        bottom_5 = df.nsmallest(5, 'Total')
        
        insights = {
            'top_5': top_5,
            'bottom_5': bottom_5
        }
        return insights
    except Exception as e:
        print(f"Error generating insights: {e}")
        return None








import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_statistics(df):
    """
    Generates statistical graphs for the dataset:
    - Histograms for individual numeric columns
    - Pie chart showing the ratio of numeric column sums
    """
    try:
        # Check for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number])
        
        if numeric_columns.empty:
            print("No numeric columns found for statistics.")
            return

        # Plot histograms for each numeric column
        plt.figure(figsize=(14, 6))
        for i, column in enumerate(numeric_columns.columns):
            plt.subplot(1, len(numeric_columns.columns), i+1)
            plt.hist(numeric_columns[column].dropna(), bins=20, alpha=0.7)
            plt.title(column)
            plt.xlabel('Values')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

        # Pie chart for the sum of values in each numeric column
        column_sums = numeric_columns.sum()
        if column_sums.sum() == 0:
            print("Sum of all numeric columns is zero, pie chart cannot be generated.")
            return

        plt.figure(figsize=(10, 7))
        plt.pie(column_sums, labels=column_sums.index, autopct='%1.1f%%', startangle=140)
        plt.title('Proportion of Numeric Column Sums')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()

    except Exception as e:
        print(f"Error generating statistics: {e}")






def generate_overall_summary(df):
    """
    Generate and return a comprehensive summary of the entire dataset.
    """
    try:
        summary = df.describe(include='all')  # Generates summary including categorical columns
        return summary.to_string()  # Convert the summary to a string format for the report
    except Exception as e:
        print(f"Error generating overall summary: {e}")
        return None
