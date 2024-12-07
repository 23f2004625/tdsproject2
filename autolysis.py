import os
import subprocess
import sys
import warnings

# Suppress the specific warnings related to missing glyphs
warnings.filterwarnings("ignore", message="Glyph.*missing from font.*")

# Function to check and install required packages
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError:
        print(f"Error installing package: {package}")
        sys.exit(1)

# List of required packages
required_packages = [
    'pandas', 
    'matplotlib', 
    'seaborn', 
    'requests', 
    'chardet'
]

# Install missing packages
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Package {package} not found. Installing...")
        install_package(package)

# Import the packages now that we are sure they are installed
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import chardet
from datetime import datetime
from matplotlib import rcParams

# Configuration Section
CONFIG = {
    "AI_PROXY_URL": "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
    "AIPROXY_TOKEN": os.getenv("AIPROXY_TOKEN"),
    "OUTPUT_DIR": os.path.dirname(os.path.abspath(__file__))  # Current directory of autolysis.py
}

HEADERS = {"Authorization": f"Bearer {CONFIG['AIPROXY_TOKEN']}", "Content-Type": "application/json"}

# Function to interact with LLM via AI Proxy
def ask_llm(question, context):
    try:
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": f"{question}\nContext:\n{context}"}]
        }
        response = requests.post(CONFIG["AI_PROXY_URL"], headers=HEADERS, json=payload)
        response.raise_for_status()
        response_json = response.json()
        return response_json['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with AI Proxy: {e}")
        sys.exit(1)

# Function to detect encoding
def detect_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']
    except Exception as e:
        print(f"Error detecting file encoding: {e}")
        sys.exit(1)

# Function to save visualizations
def save_visualization(plt, file_name):
    try:
        plt.tight_layout()  # Avoid overlapping elements
        plt.savefig(os.path.join(CONFIG["OUTPUT_DIR"], file_name), bbox_inches='tight')  # Ensure nothing is cropped
        plt.close()
    except Exception as e:
        print(f"Error saving visualization {file_name}: {e}")

# Function to create correlation matrix
def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        print("Warning: No numeric columns found for correlation analysis.")
        return None
    correlation = numeric_df.corr()
    print("Correlation Matrix (Text Table):")
    print(correlation)
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    save_visualization(plt, "correlation_heatmap.png")
    return correlation

# Function to visualize outliers
def plot_outliers(df):
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        print("No numeric data for outlier analysis.")
        return
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=numeric_df)
    plt.title("Outlier Detection")
    
    # Rotate the x-axis labels to be vertical
    plt.xticks(rotation=90)
    
    # Save the visualization
    save_visualization(plt, "outliers.png")

# Function for time series analysis
def plot_time_series(df):
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            numeric_columns = df.select_dtypes(include=['number']).columns
            if numeric_columns.empty:
                print("No numeric columns for time series analysis.")
                return
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=df, x='Date', y=numeric_columns[0])
            plt.title(f"Time Series Analysis for {numeric_columns[0]}")
            save_visualization(plt, "time_series.png")
        except Exception as e:
            print(f"Error in Time Series Analysis: {e}")
    else:
        print("Date column not found for time series analysis.")

# Function for geographic analysis
def plot_geographic_analysis(df):
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x='Longitude', y='Latitude', hue=df.select_dtypes(include=['number']).columns[0])
        plt.title("Geographic Analysis")
        save_visualization(plt, "geographic_analysis.png")
    else:
        print("Latitude and Longitude columns are missing.")

# Function for categorical data analysis
def plot_categorical_data(df):
    non_numeric_df = df.select_dtypes(exclude=['number'])
    for col in non_numeric_df.columns:
        # Count the unique values in the column
        value_counts = df[col].value_counts()
        num_unique_values = len(value_counts)

        def adjust_labels(ax, labels, max_chars_per_line=10, rotate=False):
            """
            Helper function to adjust labels by splitting long ones into multiple lines
            and optionally rotating them.
            """
            new_labels = []
            for label in labels:
                split_label = "\n".join(
                    [label[i:i + max_chars_per_line] for i in range(0, len(label), max_chars_per_line)]
                )
                new_labels.append(split_label)
            ax.set_xticks(range(len(new_labels)))
            ax.set_xticklabels(new_labels, rotation=90 if rotate else 0, ha='center' if rotate else 'right')

        if num_unique_values <= 15:
            # Case 1: Bars are up to 15, don't rotate, split long labels into multiple lines
            plt.figure(figsize=(12, 8))
            ax = sns.countplot(x=col, data=df, order=value_counts.index)
            plt.title(f"Distribution of {col}")
            adjust_labels(ax, value_counts.index, max_chars_per_line=10, rotate=False)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.subplots_adjust(bottom=0.25)
            save_visualization(plt, f"{col}_distribution.png")

        elif 15 < num_unique_values <= 30:
            # Case 2: Bars are more than 15 but up to 30, rotate labels, split into 15-character lines
            plt.figure(figsize=(12, 8))
            ax = sns.countplot(x=col, data=df, order=value_counts.index)
            plt.title(f"Distribution of {col}")
            adjust_labels(ax, value_counts.index, max_chars_per_line=15, rotate=True)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.subplots_adjust(bottom=0.35)
            save_visualization(plt, f"{col}_distribution.png")

        else:
            # Case 3: Bars are more than 30, create two graphs (top 15 and bottom 15 value-wise)
            top_15 = value_counts.head(15)
            bottom_15 = value_counts.tail(15)

            # Top 15 graph
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x=top_15.index, y=top_15.values)
            plt.title(f"Top 15 Distribution of {col}")
            adjust_labels(ax, top_15.index, max_chars_per_line=15, rotate=True)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.subplots_adjust(bottom=0.35)
            save_visualization(plt, f"{col}_top_15_distribution.png")

            # Bottom 15 graph
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x=bottom_15.index, y=bottom_15.values)
            plt.title(f"Bottom 15 Distribution of {col}")
            adjust_labels(ax, bottom_15.index, max_chars_per_line=15, rotate=True)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.subplots_adjust(bottom=0.35)
            save_visualization(plt, f"{col}_bottom_15_distribution.png")

# Main function to analyze dataset
def analyze_data(file_path):
    try:
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    summary = df.describe(include='all')
    missing_values = df.isna().sum()

    analysis_context = f"Dataset summary statistics:\n{summary}\nMissing values:\n{missing_values}"
    analysis_response = ask_llm("Analyze the dataset and provide insights.", analysis_context)

    plot_correlation_matrix(df)
    plot_outliers(df)
    plot_time_series(df)
    plot_geographic_analysis(df)
    plot_categorical_data(df)

    numeric_context = f"Numeric columns summary:\n{df.select_dtypes(include=['number']).describe()}"
    numeric_response = ask_llm("Provide insights about numeric columns.", numeric_context)

    story_context = f"Dataset Analysis:\nSummary statistics:\n{summary}\nMissing Values:\n{missing_values}\n\nInsights:\n{analysis_response}\n\nNumeric Insights:\n{numeric_response}"
    story = ask_llm("Generate a story based on the analysis.", story_context)

    try:
        with open(os.path.join(CONFIG["OUTPUT_DIR"], "README.md"), "w") as f:
            f.write("# Data Analysis Report\n\n")
            f.write("## Overview\n")
            f.write(f"File: {file_path}\n\n")
            f.write("## Summary Statistics\n")
            f.write(f"{summary}\n\n")
            f.write("## Missing Values\n")
            f.write(f"{missing_values}\n\n")
            f.write("## Insights\n")
            f.write(f"{analysis_response}\n\n")
            f.write("## Numeric Insights\n")
            f.write(f"{numeric_response}\n\n")
            f.write("## Story\n")
            f.write(f"{story}\n")
    except Exception as e:
        print(f"Error writing to README.md: {e}")

    print("Analysis complete! Results saved to README.md.")

# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    analyze_data(dataset_path)