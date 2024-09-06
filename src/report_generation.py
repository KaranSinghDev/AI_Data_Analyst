def generate_report(df, X, y, target_column):
    """
    Generate a comprehensive report on the dataset and save it to a file.
    The report includes dataset summary, insights, model evaluation, and written summaries.
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
        report_directory = "reports"  # Directory where the report will be saved
        if not os.path.exists(report_directory):
            os.makedirs(report_directory)
        report_file_name = "dataset_report.txt"
        file_path = os.path.join(report_directory, report_file_name)

        with open(file_path, 'w') as file:
            file.write(report_content)

        # Inform the user about the saved file
        print(f"Report successfully generated and saved as '{file_path}'.")

    except Exception as e:
        print(f"Error generating report: {e}")
