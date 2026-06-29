# ECG Heartbeat Classification 🫀📊

**Domain:** Signal Processing and Intelligent Systems  
**Program:** ePBL Internship Program 2026  

## Project Overview
This project applies machine learning techniques to biomedical signal data to classify ECG heartbeats. Using the preprocessed MIT-BIH Arrhythmia dataset, the system extracts time-series signal patterns to categorize heartbeats into Normal or Abnormal classes (such as Supraventricular premature beats, Premature ventricular contractions, etc.).

## Dataset
The dataset utilized is the MIT-BIH Arrhythmia Database, consisting of 1D ECG signals sampled at 125Hz. Each signal is segmented, normalized, and padded to a fixed length of 187 features. 

## Technology Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Signal Visualization:** Matplotlib, Seaborn
* **Intelligent System / ML:** Scikit-Learn (Random Forest Classifier)

## How to Run the Project
1. Clone the repository.
2. Download the `mitbih_train.csv` and `mitbih_test.csv` files from Kaggle and place them in the root directory.
3. Install dependencies: `pip install -r requirements.txt` (or manually install pandas, numpy, matplotlib, seaborn, scikit-learn).
4. Execute the script: `python main.py`

## Results
The Random Forest classifier achieved a high degree of accuracy on the test set. 
*(Note: View the `Screenshots` folder for visual outputs of the waveform signals and the confusion matrix).*

## Project Directory Structure
```text
├── main.py                          # Complete source code
├── README.md                        # Project documentation
├── Project_Report.pdf               # Final submitted report
├── Screenshots/                     # Folder containing output images
│   ├── ecg_waveform_comparison.png
│   └── confusion_matrix.png
