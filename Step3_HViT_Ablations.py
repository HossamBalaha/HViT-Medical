import os, gc, torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from utils.Utils import *
from utils.Handler import ViTPredictFnWrapper
from utils.DatasetHelpers import DataLoader
from utils.ProposalHelpers import ViTForClassification
from HMB.Utils import LoadYaml, fprint
from HMB.Initializations import SeedEverything
from HMB.PyTorchHelper import EvaluateModelOnPerturbations

COLOR_MAPPING = {
  "Standard"      : "blue",
  "Q-Conditioning": "green",
  "QC"            : "green",
  "Q-Gating"      : "orange",
  "QG"            : "orange",
  "QC_QG_S"       : "red",
  "QC_QG"         : "purple",
  "QC_S"          : "brown",
  "QG_S"          : "pink",
}


# Define the main function to demonstrate the usage of the ViT model with metadata integration.
def main():
  # Validate the command-line arguments.
  args = Step3ParseArgs()  # Parse command-line arguments.
  Step3ValidateArgs(args)

  # Seed for reproducibility.
  randomNumber = np.random.randint(0, 10000)
  SeedEverything(seed=randomNumber)

  experimentsFolder = str(args.experimentsFolder)  # Base folder containing all experiments.
  projectKeyword = str(args.projectKeyword)  # Define the project keyword.
  dpi = args.dpi  # DPI for saved figures.
  maxSamples = args.maxSamples
  datasetPath = args.datasetPath  # Path to the dataset.
  embeddedPath = args.embeddedPath  # Path to the embedded metadata.
  doBestOnly = args.doBestOnly  # Whether to evaluate only the best model.
  projectDir = os.path.join(experimentsFolder, projectKeyword)
  step1Dir = os.path.join(projectDir, "Step1-Training")
  step2Dir = os.path.join(projectDir, "Step2-Evaluation")
  step3Dir = os.path.join(projectDir, "Step3-Ablations")
  os.makedirs(step3Dir, exist_ok=True)

  summaryCSVPath = os.path.join(step2Dir, "Evaluation_Summary.csv")
  df = pd.read_csv(summaryCSVPath)

  # Skip those models:
  # LateFusion
  # EarlyFusion
  # PhikonV2LinearProbe
  # ValueConditioning
  # KeyConditioning
  df = df[~df["Approach"].isin(
    [
      "LateFusion", "EarlyFusion", "PhikonV2LinearProbe",
      "ValueConditioning", "KeyConditioning"
    ])
  ]
  df = df.reset_index(drop=True)  # Reset the index after filtering.
  fprint("Filtered DataFrame after removing unrequired approaches:")
  fprint(df)
  fprint("-" * 50)

  indexOfBestModelPath = list(df.columns).index("BestModelPath")
  indexOfConfigPath = list(df.columns).index("ConfigPath")
  indexOfApproach = list(df.columns).index("Approach")
  indexOfExperiment = list(df.columns).index("Experiment")
  indexOfTrial = list(df.columns).index("Trial")

  fprint("BestModelPath Index:", indexOfBestModelPath)
  fprint("BestModelPath:", df.columns[indexOfBestModelPath])
  fprint("ConfigPath Index:", indexOfConfigPath)
  fprint("ConfigPath:", df.columns[indexOfConfigPath])
  fprint("Approach Index:", indexOfApproach)
  fprint("Approach:", df.columns[indexOfApproach])
  fprint("Experiment Index:", indexOfExperiment)
  fprint("Experiment:", df.columns[indexOfExperiment])
  fprint("Trial Index:", indexOfTrial)
  fprint("Trial:", df.columns[indexOfTrial])
  fprint("-" * 50)

  # Find the best record index based on the highest accuracy.
  bestRecordIndex = df["Weighted Average"].idxmax()
  fprint("Best Record Index:", bestRecordIndex)
  fprint("Best Record Details:")
  fprint(df.iloc[bestRecordIndex])
  fprint("-" * 50)

  if (doBestOnly):
    data = df.iloc[bestRecordIndex]
    df = pd.DataFrame([data], columns=df.columns)  # Create a new DataFrame with only the best record.
  else:
    fprint("Evaluating all records in the summary CSV.")

  for record in df.iterrows():
    bestModelPath = record[1].iloc[indexOfBestModelPath]
    configPath = record[1].iloc[indexOfConfigPath]
    # Replace the first folder name with the --experimentsFolder value to ensure correct path resolution.
    bestModelPath = bestModelPath.replace("Results", experimentsFolder)
    configPath = configPath.replace("Results", experimentsFolder)
    approach = record[1].iloc[indexOfApproach]
    experiment = record[1].iloc[indexOfExperiment]
    trial = record[1].iloc[indexOfTrial]
    runDir = os.path.join(step1Dir, experiment, trial)
    storeDir = os.path.join(step3Dir, experiment, trial)

    hparam = LoadYaml(configPath)
    # Define the architecture lookup using the exact Step1 architecture keys.
    archsDict = {
      "Standard": [["Standard"] * 6] * hparam["NumHiddenLayers"],
      # "QConditioning"      : [["Q-Conditioning"] * 6] * hparam["NumHiddenLayers"],
      "QC"      : [["Q-Conditioning"] * 6] * hparam["NumHiddenLayers"],
      # "QGating"            : [["Q-Gating"] * 6] * hparam["NumHiddenLayers"],
      "QG"      : [["Q-Gating"] * 6] * hparam["NumHiddenLayers"],
      # "KeyConditioning"  : [["Key-Conditioning"] * 6] * hparam["NumHiddenLayers"],
      # "ValueConditioning": [["Value-Conditioning"] * 6] * hparam["NumHiddenLayers"],
      # "LateFusion"         : "LateFusionViT",
      # "EarlyFusion"        : "EarlyFusionViT",
      # "PhikonV2LinearProbe": "PhikonV2LinearProbe",
      "QCQGS"   : [["Q-Conditioning", "Q-Gating", "Standard"] * 2] * hparam["NumHiddenLayers"],
      "QCQG"    : [["Q-Conditioning", "Q-Gating"] * 3] * hparam["NumHiddenLayers"],
      "QCS"     : [["Q-Conditioning", "Standard"] * 3] * hparam["NumHiddenLayers"],
      "QGS"     : [["Q-Gating", "Standard"] * 3] * hparam["NumHiddenLayers"],
    }

    if (approach not in archsDict):
      fprint(f"Skipping unsupported architecture in Step3: {approach}")
      continue

    headTypesPerBlock = archsDict[approach]

    # Check whether the architecture is a named non-ViT model.
    if (isinstance(headTypesPerBlock, str)):
      # Print a message for unsupported ablation models.
      fprint(f"Skipping non-ViT architecture in Step3: {approach}")

      # Continue to the next record.
      continue

    # Build the ViT model for head-list architectures.
    model = ViTForClassification(hparam, headTypesPerBlock)
    model.load_state_dict(torch.load(bestModelPath))  # Load the saved model state.
    model.eval()  # Set the model to evaluation mode.

    # Load the dataset and embedded metadata.
    trainDataloader, testDataloader, combinedDataloader, classesMap, rawDataset, embeddedMetadata = DataLoader(
      hparam,  # Configuration dictionary.
      datasetPath,  # Path to the dataset.
      embeddedPath,  # Path to the embedded metadata.
      hparam["BatchSize"],  # Batch size for training.
    )
    ViTPredictFn = ViTPredictFnWrapper(hparam, model, rawDataset, embeddedMetadata)

    EvaluateModelOnPerturbations(
      model=ViTPredictFn,
      run=runDir,
      datasetDir=datasetPath,
      storeDir=storeDir,
      perturbations=["gaussian", "jpeg", "speckle", "saltPepper"],
      levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
      maxSamples=maxSamples,
      preprocessFn=None,
      subset=None,
      eps=1e-10,
      dpi=dpi,
    )

    # Copy the best model to the store directory for reference.
    os.makedirs(storeDir, exist_ok=True)
    bestModelDestPath = os.path.join(storeDir, os.path.basename(bestModelPath))
    torch.save(model.state_dict(), bestModelDestPath)
    fprint(f"Saved best model to: {bestModelDestPath}")


if __name__ == "__main__":
  main()
