import pandas as pd
import matplotlib.pyplot as plt
from analysis_engine import analyze_data, generate_overall_summary, train_regression_models
import os

def load_data(file_path):
    """Load and clean data from the provided file path."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV, JSON, or Excel file.")
        
        # Basic cleaning
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        print("Data successfully loaded and cleaned.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df, target_column):
    """Preprocess data by encoding categorical variables and separating features and target."""
    try:
        df = pd.get_dummies(df, drop_first=True)  # One-hot encoding
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        print("Data successfully preprocessed.")
        return X, y
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None, None

def generate_report(df, X, y, target_column):
    """
    Generate a comprehensive report on the dataset and save it to a file.
    The report includes dataset summary, insights, model evaluation, and visualizations.
    """
    try:
        # Prepare report content
        report_content = ""

        # Overall dataset summary
        report_content += "Overall Dataset Summary:\n"
        summary = generate_overall_summary(df)
        if summary:
            report_content += summary + "\n\n"
        else:
            report_content += "No summary available.\n\n"

        # Insights
        dataset_summary, insights = analyze_data(df, target_column)
        report_content += "Top 5 Data Points with Highest Expected Values:\n"
        report_content += insights['top_5_percent'].to_string() + "\n\n"
        report_content += "Top 5 Data Points with Lowest Expected Values:\n"
        report_content += insights['bottom_5_percent'].to_string() + "\n\n"

        # Model evaluation
        report_content += "Model Evaluation (Mean Squared Error):\n"
        models, results = train_regression_models(X, y)
        for model, mse in results.items():
            report_content += f"{model}: MSE = {mse}\n"

        # Save report to a file
        report_directory = "reports"  # Directory where you want to save the report
        if not os.path.exists(report_directory):
            os.makedirs(report_directory)
        report_file_name = "dataset_report.txt"
        file_path = os.path.join(report_directory, report_file_name)

        with open(file_path, 'w') as file:
            file.write(report_content)
        
        # Save the statistics plot
        plt.figure(figsize=(12, 6))
        plt.title("Overall Dataset Distribution")
        df.hist(figsize=(12, 6))
        plt.tight_layout()
        plt.savefig(os.path.join(report_directory, "dataset_statistics.png"))

        # Inform the user about the saved files
        print(f"Report successfully generated and saved as '{file_path}'.")
        print(f"Statistics plot saved as 'dataset_statistics.png' in the '{report_directory}' directory.")

    except Exception as e:
        print(f"Error generating report: {e}")


def main():
    """Main function to interact with the user."""
    df = None
    X, y = None, None
    target_column = None

    while True:
        print("\nWelcome to the AI Employee!")
        print("1. Load data")
        print("2. Analyze data")
        print("3. Generate report")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            file_path = input("Enter the full path to the dataset (CSV, JSON, Excel format): ").strip()
            df = load_data(file_path)
            if df is not None:
                target_column = input("Enter the name of the target column: ").strip()
                X, y = preprocess_data(df, target_column)

        elif choice == '2':
            if df is not None:
                query = input("Enter your query about the data (type 'exit' to quit): ").strip().lower()
                if query == 'summary':
                    summary, insights = analyze_data(df, target_column)
                    print("\nSummary:\n", summary)
                    print("\nInsights:\n", insights)
                elif query == 'overall summary':
                    summary = generate_overall_summary(df)
                    print("\nOverall Summary:\n", summary)
                elif query == 'statistics':
                    plt.figure(figsize=(12, 6))
                    plt.title("Overall Dataset Distribution")
                    df.hist(figsize=(12, 6))
                    plt.tight_layout()
                    plt.show()
                elif query == 'evaluation':
                    if X is not None and y is not None:
                        models, results = train_regression_models(X, y)
                        print("\nEvaluation results:")
                        for model, mse in results.items():
                            print(f"{model}: MSE = {mse}")
                    else:
                        print("Data not loaded or preprocessed.")
                elif query == 'insights':
                    summary, insights = analyze_data(df, target_column)
                    print("\nTop 5 Data Points with Highest Expected Values:\n", insights['top_5_percent'])
                    print("\nTop 5 Data Points with Lowest Expected Values:\n", insights['bottom_5_percent'])
                elif query == 'exit':
                    break
                else:
                    print("Sorry, I didn't understand the query. Please ask for summary, overall summary, statistics, evaluation, or insights.")
            else:
                print("Data not loaded yet.")

        elif choice == '3':
            if df is not None and X is not None and y is not None:
                generate_report(df, X, y, target_column)
            else:
                print("Data not loaded or preprocessed yet.")

        elif choice == '4':
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
