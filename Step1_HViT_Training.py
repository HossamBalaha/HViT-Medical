# projectKeyword = r"PH2"  # Define the project keyword.
# datasetPath = r"PH2 Dataset/Images"
# embeddedPath = r"PH2 Dataset/Metadata.csv"
# metadataDim = 6

# projectKeyword = r"NDB-UFES"  # Define the project keyword.
# datasetPath = r"NDB-UFES/OrganizedImages"  # Path to the dataset.
# embeddedPath = r"NDB-UFES/Clinical Data.csv"  # Path to the embedded metadata.
# metadataDim = 23

# projectKeyword = r"UterineCervixPhikonV2"  # Define the project keyword.
# datasetPath = r"TissueNet/Tiles"  # Path to the dataset.
# embeddedPath = r"TissueNet/UterineCervixPhikonV2.p"  # Path to the embedded metadata.
# metadataDim = 1024

# projectKeyword = r"PAD-UFES-20"  # Define the project keyword.
# datasetPath = r"Skin Cancer(PAD-UFES-20)/OrganizedImages"
# embeddedPath = r"Skin Cancer(PAD-UFES-20)/Metadata.csv"
# metadataDim = 22

# Import the operating system module.
import os
# Import the garbage collection module.
import gc
# Import the torch module.
import torch
# Import the shutil module.
import shutil
# Import the numpy module.
import numpy as np
# Import the Path class from the pathlib module.
from pathlib import Path
# Import the Step1Handler function from the Handler module.
from utils.Handler import Step1Handler
# Import all utilities from the Utils module.
from utils.Utils import *
# Import the PrintHyperParamsList function from the HMB Utils module.
from HMB.Utils import PrintHyperParamsList
# Import the SeedEverything function from the HMB Initializations module.
from HMB.Initializations import SeedEverything
from HMB.Utils import fprint


# Define the main function to execute the training pipeline.
def Main():
  # Validate the command-line arguments.
  args = Step1ParseArgs()
  # Validate the parsed arguments.
  Step1ValidateArgs(args)

  # Generate a base random number for the experiment session.
  baseRandomNumber = np.random.randint(0, 10000)
  # Seed all random number generators for reproducibility.
  SeedEverything(seed=baseRandomNumber)

  # Assign the hyperparameters file path.
  hparamsFile = args.hparamsFile
  # Assign the base folder containing all experiments.
  experimentsFolder = str(args.experimentsFolder)
  # Assign the project keyword for naming.
  projectKeyword = str(args.projectKeyword)
  # Assign the path to the dataset.
  datasetPath = args.datasetPath
  # Assign the path to the embedded metadata.
  embeddedPath = args.embeddedPath
  # Assign the dimension of the metadata.
  metadataDim = args.metadataDim
  # Assign the number of epochs.
  noOfEpochs = args.noOfEpochs
  # Assign the number of trials per architecture.
  noOfTrials = args.noOfTrials
  # Assign the size of input images.
  imageSize = args.imageSize
  # Assign the number of image channels.
  numChannels = args.numChannels
  # Assign the batch size for training.
  batchSize = args.batchSize
  # Assign the DPI for saved figures.
  dpi = args.dpi
  # Assign the flag to clean previous outputs.
  clean = args.clean

  # Read metadata missingness evaluation rates from arguments.
  metadataMissingEvalRates = args.metadataMissingEvalRates

  # Read metadata missingness mode from arguments.
  metadataMissingMode = args.metadataMissingMode

  # Read metadata missingness scope from arguments.
  metadataMissingScope = args.metadataMissingScope

  # Read metadata missingness noise standard deviation from arguments.
  metadataMissingNoiseStd = args.metadataMissingNoiseStd

  # Read metadata missingness training rate from arguments.
  metadataMissingTrainRate = args.metadataMissingTrainRate

  # Calculate the number of classes from the dataset directory.
  numClasses = len([name for name in os.listdir(datasetPath) if (os.path.isdir(os.path.join(datasetPath, name)))])

  # Define the configuration dictionary with CamelCase keys.
  config = {
    "Epochs"                  : noOfEpochs,
    "BatchSize"               : batchSize,
    "NoOfTrials"              : noOfTrials,
    "ImageSize"               : imageSize,
    "NumChannels"             : numChannels,
    "MetadataDim"             : metadataDim,
    "NumClasses"              : numClasses,
    "MetadataMissingEvalRates": metadataMissingEvalRates,
    "MetadataMissingMode"     : metadataMissingMode,
    "MetadataMissingScope"    : metadataMissingScope,
    "MetadataMissingNoiseStd" : metadataMissingNoiseStd,
    "MetadataMissingTrainRate": metadataMissingTrainRate,
  }

  # Retrieve the list of hyperparameter presets.
  hparamsList = PrintHyperParamsList(hparamsFile, returnList=True)

  # Construct the base path for the current project.
  basePath = f"{experimentsFolder}/{projectKeyword}"

  # Check if the clean flag is enabled.
  if (clean):
    # Print a message indicating the cleaning process.
    fprint(f"Cleaning previous results at: {basePath}")
    # Check if the base path exists.
    if (os.path.exists(basePath)):
      # Remove the existing directory tree.
      shutil.rmtree(basePath)

  # Iterate over each hyperparameter preset.
  for hparam in hparamsList:
    # Skip inactive hyperparameter presets.
    if (not hparam["Active"]):
      # Read the preset name from the hyperparameter dictionary.
      presetName = hparam["Name"]
      # Print a message indicating the preset is skipped.
      fprint(f"Skipping inactive hyperparameter set: {presetName}")
      # Continue to the next preset.
      continue

    # Merge the preset with the runtime configuration.
    hparam = {**hparam, **config}

    # Check whether the scheduler parameters contain a maximum-time value.
    if (hparam["SchedulerParams"].get("T_max", None) is not None):
      # Check whether the maximum-time value is the epoch placeholder.
      if (hparam["SchedulerParams"]["T_max"] == "EPOCHS"):
        # Replace the epoch placeholder with the actual epoch count.
        hparam["SchedulerParams"]["T_max"] = noOfEpochs

    # Define the architecture configurations including the ablation heads and baseline models.
    archsDict = {
      "Standard"           : [["Standard"] * 6] * hparam["NumHiddenLayers"],
      # "QConditioning"      : [["Q-Conditioning"] * 6] * hparam["NumHiddenLayers"],
      "QC"                 : [["Q-Conditioning"] * 6] * hparam["NumHiddenLayers"],
      # "QGating"            : [["Q-Gating"] * 6] * hparam["NumHiddenLayers"],
      "QG"                 : [["Q-Gating"] * 6] * hparam["NumHiddenLayers"],
      "KeyConditioning"    : [["Key-Conditioning"] * 6] * hparam["NumHiddenLayers"],
      "ValueConditioning"  : [["Value-Conditioning"] * 6] * hparam["NumHiddenLayers"],
      "LateFusion"         : "LateFusionViT",
      "EarlyFusion"        : "EarlyFusionViT",
      "PhikonV2LinearProbe": "PhikonV2LinearProbe",
      "QCQGS"              : [["Q-Conditioning", "Q-Gating", "Standard"] * 2] * hparam["NumHiddenLayers"],
      "QCQG"               : [["Q-Conditioning", "Q-Gating"] * 3] * hparam["NumHiddenLayers"],
      "QCS"                : [["Q-Conditioning", "Standard"] * 3] * hparam["NumHiddenLayers"],
      "QGS"                : [["Q-Gating", "Standard"] * 3] * hparam["NumHiddenLayers"]
    }

    # Iterate over each architecture key and its corresponding head types.
    for key, headTypesPerBlock in archsDict.items():
      # Iterate over the number of trials.
      for i in range(noOfTrials):
        # Compute a unique seed for the current trial.
        trialSeed = baseRandomNumber + i
        # Initialize the global random seeds.
        SeedEverything(seed=trialSeed)
        # Print a message indicating the start of the current architecture trial.
        fprint(f"Training {key} architecture, Trial {i + 1}/{noOfTrials} with seed {trialSeed}...")

        # Retrieve the name of the current preset using the exact JSON key.
        presetName = hparam["Name"]
        # Define the storage path for results.
        storagePath = f"{basePath}/Step1-Training/{key}_{presetName}/Trial_{i + 1}"
        # Assign the storage path to the hyperparameters dictionary using a CamelCase key.
        hparam["StoragePath"] = str(storagePath)
        # Assign the parent storage path to the hyperparameters dictionary using a CamelCase key.
        hparam["StorageParentPath"] = str(Path(storagePath).parent.parent)

        # Determine the model type based on the architecture configuration.
        modelType = headTypesPerBlock if isinstance(headTypesPerBlock, str) else "ViTForClassification"

        # Call the handler function to train the model.
        Step1Handler(
          hparam,
          datasetPath,
          embeddedPath,
          storagePath,
          headTypesPerBlock,
          dpi,
          modelType
        )
        # Print a message indicating completion.
        fprint(f"Completed {key} architecture, Trial {i + 1}/{noOfTrials}.")
        # Clear the GPU memory.
        torch.cuda.empty_cache()
        # Collect garbage to free memory.
        gc.collect()


# Check if the script is executed as the main program.
if (__name__ == "__main__"):
  # Call the main function.
  Main()
