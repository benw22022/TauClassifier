import uproot
import glob
from source2 import plotting_functions as pf

def visualise():

    all_data = uproot.concatenate(glob.glob("results/*.root"), library='np')
    y_true = all_data["TauClassifier_TruthScores"]
    y_pred = all_data["TauClassifier_Scores"]
    weights = all_data["TauClassifier_Weight"]
    
    pf.plot_ROC(y_true[:, 0], y_pred[:, 0], weights)
    pf.plot_confusion_matrix(y_true, y_pred, weights=weights)

if __name__ == "__main__":
    visualise()