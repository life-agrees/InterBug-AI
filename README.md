# InterBug AI

**InterBug AI** is a Streamlit-powered web application designed for cross-chain bug detection in the Sei Network. It cleans, preprocesses, and analyzes blockchain transaction data using machine learning to identify potential anomalies (bugs) in real-time. The tool is interactive and user-friendly—perfect for both tech enthusiasts and newcomers.

## Live Demo

Check out the live app here: [https://seibugs.streamlit.app/](https://seibugs.streamlit.app/)

## Table of Contents

- [Features](#features)
- [Architecture & Workflow](#architecture--workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- **Data Ingestion & Preprocessing:**  
  Upload your CSV data or use sample data. The app cleans and processes blockchain data, generating key features required for anomaly detection.
  
- **Bug Detection:**  
  A custom ML pipeline calculates anomaly scores and flags potential bugs by analyzing cross-chain data from both EVM and Cosmos chains.

- **Interactive Dashboard:**  
  Visualize key metrics, distributions, and feature importances through dynamic charts and graphs using Plotly and Matplotlib.

- **Model Training & Live Monitoring:**  
  (Planned) Supports training ML models and live monitoring of new data for continuous blockchain auditing.

## Architecture & Workflow

1. **Data Upload & Cleaning:**  
   Users can upload a CSV file or fetch sample data. The raw data is cleaned (if a cleaning module is available) and saved for further processing.
   
2. **Preprocessing:**  
   The cleaned data is processed by the `preprocessing` module (from `Sei_Module.data_preprocess`) to generate essential features such as transaction mismatch and fee ratios.
   
3. **Bug Detection:**  
   Processed data is fed into the bug detection pipeline within the `framework` module (from `Sei_Module.testing`), which computes anomaly scores and flags bugs.
   
4. **Visualization:**  
   The dashboard displays key metrics, recent anomalies, and interactive charts, making it easy to interpret the results.
   
5. **Model Training & Monitoring:**  
   (Planned) Additional functionality for training models and real-time monitoring of blockchain data.

## Installation

### Requirements

- Python 3.11.5
- [Streamlit](https://streamlit.io/)
- Pandas, NumPy
- Plotly, Matplotlib, Seaborn
- Scikit-learn
- imbalanced-learn
- Custom modules from `Sei_Module`

### Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/life-agrees/InterBug-AI.git
   cd InterBug-AI


2. **Install Dependencies:**
    ```bash
   pip install -r requirements.txt

4. **Run the App:**
    ```bash
   streamlit run app.py

 ### Usage 

1. **Data Upload & Processing:**
   Navigate to the "Data Upload & Processing" page to upload your CSV file or use sample data. The app cleans and preprocesses the data using the custom preprocessing module, and then saves the processed output.

2. **Bug Detection:**
   Go to the "Bug Detection" page to run the bug detection pipeline. The app utilizes the framework module to compute anomaly scores and flag potential bugs, with interactive visualizations to help you analyze the results.

3. **Model Training & Live Monitoring:**
   Use these sections to train machine learning models and monitor blockchain data in real-time.

### Customization

- Styling:
  The app uses custom CSS to provide a modern, polished look. You can adjust the CSS in app.py to match your branding—for example, changing the background colors, sidebar gradient, or button styles.

- Modules:
  Core functionalities are implemented in:
  - `Sei_Module.data_preprocess.preprocessing` – handles data cleaning and preprocessing.
  - `Sei_Module.testing.framework` – contains the bug detection logic.
 These modules can be customized to tweak the data processing and detection algorithms as needed.

### Troubleshooting

**Module Import Issues:**
Verify that your custom modules (`Sei_Module.data_preprocess` and `Sei_Module.testing`) are in the correct directory and that the import paths in `app.py` match your project structure.

**Missing Dependencies:**
If you encounter errors related to missing packages, install them using `pip install <package>`.

**CSV File Format:**
Ensure that the CSV files you upload are correctly formatted and include the necessary columns for processing.

### Conclusion

InterBug AI is a comprehensive tool designed to simplify cross-chain bug detection for the Sei Network. It integrates data processing, ML-driven anomaly detection, and interactive visualizations into one intuitive dashboard—making advanced blockchain analytics accessible to everyone.


### License
This project is licensed under the [MIT License](LICENSE).
