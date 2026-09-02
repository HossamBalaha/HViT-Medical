import os, argparse, torch
from HMB.Initializations import EnsureCUDAAvailable
from HMB.Utils import fprint


def LoadHParams(hparamsFile, hparamName=None):
  import json

  # Open the hyperparameters JSON file.
  with open(hparamsFile, "r") as f:
    # Load the list of hyperparameter presets from JSON.
    hparamsList = json.load(f)
  # If a named preset was requested, search for it.
  if (hparamName is not None):
    for hp in hparamsList:
      if (hp.get("Name") == hparamName):
        return hp
    # Raise an error if the named preset was not found.
    raise RuntimeError(f"Hyperparameter preset named '{hparamName}' not found in {hparamsFile}")
  # Otherwise pick the first active preset or the first preset in the list.
  for hp in hparamsList:
    if (hp.get("active")):
      return hp
  # Return the first preset as a fallback.
  return hparamsList[0]


def PrintArgs(args):
  fprint("\n==============================")
  fprint("Parsed arguments:")
  for arg, value in vars(args).items():
    fprint(f"  {arg}: {value}")
  fprint("==============================\n")


def Step1ParseArgs():
  parser = argparse.ArgumentParser(description="Train ViT models with metadata integration.")
  parser.add_argument(
    "--experimentsFolder", type=str, required=False,
    default="Results", help="Folder to save experiment outputs."
  )
  parser.add_argument(
    "--projectKeyword", type=str, required=False,
    default="PH2", help="Project keyword for naming."
  )
  parser.add_argument(
    "--hparamsFile", type=str, required=False,
    default="assets/hparams.json",
    help="Path to the hyperparameters file."
  )
  parser.add_argument(
    "--datasetPath", type=str, required=False,
    default="PH2 Dataset/Images",
    help="Path to the dataset."
  )
  parser.add_argument(
    "--embeddedPath", type=str, required=False,
    default="PH2 Dataset/Metadata.csv",
    help="Path to the embedded metadata."
  )
  parser.add_argument(
    "--metadataDim", type=int, required=False, default=6,
    help="Dimension of the metadata."
  )
  parser.add_argument(
    "--batchSize", type=int, required=False, default=16,
    help="Batch size for training."
  )
  parser.add_argument(
    "--imageSize", type=int, required=False, default=128,
    help="Size of input images."
  )
  parser.add_argument(
    "--numChannels", type=int, required=False, default=3,
    help="Number of image channels."
  )
  parser.add_argument(
    "--noOfEpochs", type=int, required=False, default=128,
    help="Number of epochs for training."
  )
  parser.add_argument(
    "--noOfTrials", type=int, required=False, default=1,
    help="Number of trials per architecture."
  )
  parser.add_argument(
    "--mustCuda", action="store_true",
    help="Flag to enforce CUDA usage."
  )
  parser.add_argument(
    "--clean", action="store_true",
    help="Flag to clean previous outputs and start fresh."
  )
  # Add the DPI argument for saved figures.
  parser.add_argument(
    "--dpi", type=int, required=False, default=720,
    help="DPI for saved figures."
  )
  # Add metadata missingness evaluation rates for the ablation.
  parser.add_argument(
    "--metadataMissingEvalRates", type=float, nargs="+", required=False, default=[],
    help="Metadata missingness rates evaluated after training."
  )
  # Add metadata missingness mode for the ablation.
  parser.add_argument(
    "--metadataMissingMode", type=str, required=False, default="Zero",
    help="Replacement strategy for missing metadata entries."
  )
  # Add metadata missingness scope for the ablation.
  parser.add_argument(
    "--metadataMissingScope", type=str, required=False, default="Feature",
    help="Scope of metadata missingness."
  )
  # Add Gaussian noise standard deviation for missingness mode.
  parser.add_argument(
    "--metadataMissingNoiseStd", type=float, required=False, default=0.05,
    help="Noise standard deviation for Gaussian missingness mode."
  )
  # Add metadata missingness training rate for optional training-time corruption.
  parser.add_argument(
    "--metadataMissingTrainRate", type=float, required=False, default=0.0,
    help="Metadata missingness rate applied during training."
  )
  # Parse the command line arguments.
  args = parser.parse_args()
  PrintArgs(args)
  return args


def Step1ValidateArgs(args):
  # Check that the experiments folder path is a non-empty string.
  if ((not isinstance(args.experimentsFolder, str)) or (len(args.experimentsFolder.strip()) == 0)):
    # Raise a value error if experiments folder path is invalid.
    raise ValueError("Experiments folder path must be a non-empty string.")

  # Check that the project keyword is a non-empty string.
  if ((not isinstance(args.projectKeyword, str)) or (len(args.projectKeyword.strip()) == 0)):
    # Raise a value error if project keyword is invalid.
    raise ValueError("Project keyword must be a non-empty string.")

  # Check that the hyperparameters file exists and is a file.
  if (not os.path.isfile(args.hparamsFile)):
    # Raise a runtime error if the hyperparameters file is missing or not a regular file.
    raise RuntimeError(f"Hyperparameters file does not exist or is not a file: {args.hparamsFile}")

  # Check that the dataset path exists and is a directory.
  if (not os.path.isdir(args.datasetPath)):
    # Raise a runtime error if the dataset path is invalid.
    raise RuntimeError(f"Dataset path does not exist or is not a directory: {args.datasetPath}")

  # Check that the embedded metadata path exists and is a file.
  if (not os.path.isfile(args.embeddedPath)):
    # Raise a runtime error if the embedded metadata file is missing.
    raise RuntimeError(f"Embedded metadata file does not exist or is not a file: {args.embeddedPath}")

  # Validate that metadata dimension is a positive integer.
  if ((not isinstance(args.metadataDim, int)) or (args.metadataDim <= 0)):
    # Raise a value error if metadata dimension is invalid.
    raise ValueError("Metadata dimension must be a positive integer.")

  # Validate that batch size is a positive integer.
  if ((not isinstance(args.batchSize, int)) or (args.batchSize <= 0)):
    # Raise a value error if batch size is invalid.
    raise ValueError("Batch size must be a positive integer.")

  # Validate that image size is a positive integer.
  if ((not isinstance(args.imageSize, int)) or (args.imageSize <= 0)):
    # Raise a value error if image size is invalid.
    raise ValueError("Image size must be a positive integer.")

  # Validate that number of channels is either 1 or 3 (standard for grayscale or RGB).
  if ((args.numChannels not in (1, 3))):
    # Raise a value error if number of channels is unsupported.
    raise ValueError("Number of channels must be 1 (grayscale) or 3 (RGB).")

  # Validate that number of epochs is a positive integer.
  if ((not isinstance(args.noOfEpochs, int)) or (args.noOfEpochs <= 0)):
    # Raise a value error if number of epochs is invalid.
    raise ValueError("Number of epochs must be a positive integer.")

  # Validate that number of trials is a positive integer.
  if ((not isinstance(args.noOfTrials, int)) or (args.noOfTrials <= 0)):
    # Raise a value error if number of trials is invalid.
    raise ValueError("Number of trials must be a positive integer.")

  # If CUDA enforcement is requested, verify that PyTorch detects a CUDA device.
  if ((args.mustCuda) and (not EnsureCUDAAvailable())):
    # Raise a runtime error if CUDA is required but unavailable.
    raise RuntimeError("CUDA is required (--mustCuda) but not available on this system.")

  # Validate that DPI is a positive integer.
  if ((not isinstance(args.dpi, int)) or (args.dpi <= 0)):
    # Raise a value error if DPI is invalid.
    raise ValueError("DPI must be a positive integer.")

  # Validate metadata missingness evaluation rates.
  for metadataMissingRate in args.metadataMissingEvalRates:
    # Raise an error when a missingness rate is outside the valid range.
    if ((metadataMissingRate < 0.0) or (metadataMissingRate > 1.0)):
      raise ValueError("Metadata missingness rates must be between 0.0 and 1.0.")

  # Validate metadata missingness mode.
  if (args.metadataMissingMode not in ("Zero", "Mean", "Gaussian")):
    raise ValueError("Metadata missingness mode must be Zero, Mean, or Gaussian.")

  # Validate metadata missingness scope.
  if (args.metadataMissingScope not in ("Feature", "Sample")):
    raise ValueError("Metadata missingness scope must be Feature or Sample.")

  # Validate metadata missingness noise standard deviation.
  if (args.metadataMissingNoiseStd < 0.0):
    raise ValueError("Metadata missingness noise standard deviation must be non-negative.")

  # Validate metadata missingness training rate.
  if ((args.metadataMissingTrainRate < 0.0) or (args.metadataMissingTrainRate > 1.0)):
    raise ValueError("Metadata missingness training rate must be between 0.0 and 1.0.")


def Step2ParseArgs():
  parser = argparse.ArgumentParser(description="Train ViT models with metadata integration.")
  parser.add_argument(
    "--experimentsFolder", type=str, required=False,
    default="Results", help="Folder to save experiment outputs."
  )
  parser.add_argument(
    "--projectKeyword", type=str, required=False,
    default="PH2", help="Project keyword for naming."
  )
  parser.add_argument(
    "--dpi", type=int, required=False, default=720,
    help="DPI for saved figures."
  )
  args = parser.parse_args()
  PrintArgs(args)
  return args


def Step2ValidateArgs(args):
  # Check that the experiments folder path is a non-empty string.
  if ((not isinstance(args.experimentsFolder, str)) or (len(args.experimentsFolder.strip()) == 0)):
    # Raise a value error if experiments folder path is invalid.
    raise ValueError("Experiments folder path must be a non-empty string.")

  # Check that the project keyword is a non-empty string.
  if ((not isinstance(args.projectKeyword, str)) or (len(args.projectKeyword.strip()) == 0)):
    # Raise a value error if project keyword is invalid.
    raise ValueError("Project keyword must be a non-empty string.")

  # Validate that DPI is a positive integer.
  if ((not isinstance(args.dpi, int)) or (args.dpi <= 0)):
    # Raise a value error if DPI is invalid.
    raise ValueError("DPI must be a positive integer.")


def Step3ParseArgs():
  parser = argparse.ArgumentParser(description="Train ViT models with metadata integration.")
  parser.add_argument(
    "--experimentsFolder", type=str, required=False,
    default="Results", help="Folder to save experiment outputs."
  )
  parser.add_argument(
    "--projectKeyword", type=str, required=False,
    default="PH2", help="Project keyword for naming."
  )
  parser.add_argument(
    "--datasetPath", type=str, required=False,
    default="PH2 Dataset/Images",
    help="Path to the dataset."
  )
  parser.add_argument(
    "--embeddedPath", type=str, required=False,
    default="PH2 Dataset/Metadata.csv",
    help="Path to the embedded metadata."
  )
  parser.add_argument(
    "--maxSamples",
    type=int,
    default=200,
    help="Maximum number of test images to use (helps keep runtime bounded)."
  )
  parser.add_argument(
    "--mustCuda", action="store_true",
    help="Flag to enforce CUDA usage."
  )
  parser.add_argument(
    "--doBestOnly", action="store_true",
    help="Flag to evaluate only the best model (instead of all models)."
  )
  parser.add_argument(
    "--dpi", type=int, required=False, default=720,
    help="DPI for saved figures."
  )
  args = parser.parse_args()
  PrintArgs(args)
  return args


def Step3ValidateArgs(args):
  # Check that the experiments folder path is a non-empty string.
  if ((not isinstance(args.experimentsFolder, str)) or (len(args.experimentsFolder.strip()) == 0)):
    # Raise a value error if experiments folder path is invalid.
    raise ValueError("Experiments folder path must be a non-empty string.")

  # Check that the project keyword is a non-empty string.
  if ((not isinstance(args.projectKeyword, str)) or (len(args.projectKeyword.strip()) == 0)):
    # Raise a value error if project keyword is invalid.
    raise ValueError("Project keyword must be a non-empty string.")

  # Check that the dataset path exists and is a directory.
  if (not os.path.isdir(args.datasetPath)):
    # Raise a runtime error if the dataset path is invalid.
    raise RuntimeError(f"Dataset path does not exist or is not a directory: {args.datasetPath}")

  # Check that the embedded metadata path exists and is a file.
  if (not os.path.isfile(args.embeddedPath)):
    # Raise a runtime error if the embedded metadata file is missing.
    raise RuntimeError(f"Embedded metadata file does not exist or is not a file: {args.embeddedPath}")

  # If CUDA enforcement is requested, verify that PyTorch detects a CUDA device.
  if ((args.mustCuda) and (not EnsureCUDAAvailable())):
    # Raise a runtime error if CUDA is required but unavailable.
    raise RuntimeError("CUDA is required (--mustCuda) but not available on this system.")

  # Validate that DPI is a positive integer.
  if ((not isinstance(args.dpi, int)) or (args.dpi <= 0)):
    # Raise a value error if DPI is invalid.
    raise ValueError("DPI must be a positive integer.")

  if (args.doBestOnly):
    fprint("Evaluating only the best model (instead of all models) due to --doBestOnly flag.")

  try:
    maxSamples = int(args.maxSamples)
    if (maxSamples <= 0):
      fprint(f"ERROR: maxSamples must be a positive integer; defaulting to 1000.")
      args.maxSamples = 1000
  except Exception as e:
    fprint(f"ERROR: Unable to parse maxSamples '{args.maxSamples}': {e}; defaulting to 1000.")
    args.maxSamples = 1000
  fprint(f"Using max samples: {args.maxSamples}")


def Step4ParseArgs():
  # Create argument parser instance.
  parser = argparse.ArgumentParser(description="Step4: Computational complexity & latency analysis for H-ViT variants")
  # Add experiments folder argument.
  parser.add_argument(
    "--experimentsFolder", type=str, required=False,
    default="Results", help="Folder to save experiment outputs."
  )
  # Add hyperparameters file argument.
  parser.add_argument("--hparamsFile", type=str, default="assets/hparams.json")
  # Add named hyperparameter preset argument.
  parser.add_argument(
    "--hparamName", type=str, default=None,
    help="Pick a named preset from hparams (defaults to first active)"
  )
  # Add project keyword argument.
  parser.add_argument("--projectKeyword", type=str, default="PH2-v2")
  # Add image size argument.
  parser.add_argument("--imageSize", type=int, default=128)
  # Add number of image channels argument.
  parser.add_argument("--numChannels", type=int, default=3)
  # Add metadata dimensionality argument.
  parser.add_argument("--metadataDim", type=int, default=6)
  # Add number of runs for latency averaging.
  parser.add_argument("--runs", type=int, default=100, help="Number of forward passes to average latency")
  # Add warmup runs argument.
  parser.add_argument("--warmup", type=int, default=10, help="Warmup forward passes")
  # Add flag to require CUDA.
  parser.add_argument("--mustCuda", action="store_true")
  # Add flag for DPI for saved figures.
  parser.add_argument("--dpi", type=int, default=720, help="DPI for saved figures")
  # Parse arguments and return.
  args = parser.parse_args()
  return args


def Step4ValidateArgs(args):
  # Check that the experiments folder path is a non-empty string.
  if ((not isinstance(args.experimentsFolder, str)) or (len(args.experimentsFolder.strip()) == 0)):
    # Raise a value error if experiments folder path is invalid.
    raise ValueError("Experiments folder path must be a non-empty string.")

  # Validate that the hyperparameters file exists and is a file.
  if (not os.path.isfile(args.hparamsFile)):
    raise RuntimeError(f"Hyperparameters file does not exist or is not a file: {args.hparamsFile}")

  # Validate that image size is a positive integer.
  if ((not isinstance(args.imageSize, int)) or (args.imageSize <= 0)):
    raise ValueError("Image size must be a positive integer.")

  # Validate that number of channels is either 1 or 3 (standard for grayscale or RGB).
  if ((args.numChannels not in (1, 3))):
    raise ValueError("Number of channels must be 1 (grayscale) or 3 (RGB).")

  # Validate that metadata dimension is a positive integer.
  if ((not isinstance(args.metadataDim, int)) or (args.metadataDim <= 0)):
    raise ValueError("Metadata dimension must be a positive integer.")

  # Validate that number of runs is a positive integer.
  if ((not isinstance(args.runs, int)) or (args.runs <= 0)):
    raise ValueError("Number of runs must be a positive integer.")

  # Validate that number of warmup runs is a non-negative integer.
  if ((not isinstance(args.warmup, int)) or (args.warmup < 0)):
    raise ValueError("Number of warmup runs must be a non-negative integer.")

  # If CUDA enforcement is requested, verify that PyTorch detects a CUDA device.
  if ((args.mustCuda) and (not EnsureCUDAAvailable())):
    raise RuntimeError("CUDA is required (--mustCuda) but not available on this system.")

  # Validate that DPI is a positive integer.
  if ((not isinstance(args.dpi, int)) or (args.dpi <= 0)):
    raise ValueError("DPI must be a positive integer.")


def Step5ParseArgs():
  # Create an argument parser instance.
  parser = argparse.ArgumentParser(description="Step5: Interpretability analysis for H-ViT models on PH2")
  # Add experiments folder argument.
  parser.add_argument(
    "--experimentsFolder", type=str, required=False,
    default="Results", help="Folder to save experiment outputs."
  )
  parser.add_argument(
    "--projectKeyword", type=str, required=False,
    default="PH2", help="Project keyword for naming."
  )
  parser.add_argument(
    "--datasetPath", type=str, required=False,
    default="PH2 Dataset/Images",
    help="Path to the dataset."
  )
  parser.add_argument(
    "--embeddedPath", type=str, required=False,
    default="PH2 Dataset/Metadata.csv",
    help="Path to the embedded metadata."
  )
  # Add argument for model hparams file.
  parser.add_argument(
    "--hparamsFile", type=str, default="assets/hparams.json",
    help="hparams.json used for model construction"
  )
  # Add optional argument to select a named hparam preset.
  parser.add_argument(
    "--hparamName", type=str, default=None, help="Optional named hparam preset from file"
  )
  # Add optional argument for model checkpoint path.
  parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Optional model checkpoint (.pt/.pth) to load `state_dict`"
  )
  # Add argument for target image size.
  parser.add_argument(
    "--imageSize", type=int, default=128, help="Image size to resize for model if needed"
  )
  # Add argument for patch size override.
  parser.add_argument("--patchSize", type=int, default=None, help="Patch size; if None read from hparams")
  # Add argument for top-k IoU values.
  parser.add_argument(
    "--topK", type=int, nargs="+", default=[1, 5, 10, 20], help="k values for IoU@k"
  )
  # Add argument for maximum samples to process.
  parser.add_argument(
    "--maxSamples", type=int, default=200, help="Max images to process (keeps run short)."
  )
  # Add flag to require mask presence for processing.
  parser.add_argument(
    "--requireMask", action="store_true", help="If set, skip images without ground-truth mask."
  )
  # Add the ground-truth mask directory argument for quantitative IoU evaluation.
  parser.add_argument(
    "--maskDir",
    type=str,
    required=False,
    default=None,
    help="Directory containing PH2 ground-truth lesion masks."
  )
  # Add flag to require CUDA.
  parser.add_argument("--mustCuda", action="store_true")
  # Add flag for DPI for saved figures.
  parser.add_argument("--dpi", type=int, default=720, help="DPI for saved figures")
  # Parse the arguments now.
  args = parser.parse_args()
  # Print parsed arguments for logging.
  PrintArgs(args)
  # Return the parsed arguments object.
  return args


def Step5ValidateArgs(args):
  # Check that the experiments folder path is a non-empty string.
  if ((not isinstance(args.experimentsFolder, str)) or (len(args.experimentsFolder.strip()) == 0)):
    raise ValueError("Experiments folder path must be a non-empty string.")

  # Check that the project keyword is a non-empty string.
  if ((not isinstance(args.projectKeyword, str)) or (len(args.projectKeyword.strip()) == 0)):
    raise ValueError("Project keyword must be a non-empty string.")

  # Check that the dataset path exists and is a directory.
  if (not os.path.isdir(args.datasetPath)):
    raise RuntimeError(f"Dataset path does not exist or is not a directory: {args.datasetPath}")

  # Check that the metadata CSV file exists and is a file.
  if (not os.path.isfile(args.embeddedPath)):
    raise RuntimeError(f"Metadata CSV file does not exist or is not a file: {args.embeddedPath}")

  # Validate that the model hyperparameters file exists and is a file.
  if (not os.path.isfile(args.hparamsFile)):
    raise RuntimeError(f"Model hyperparameters file does not exist or is not a file: {args.hparamsFile}")

  # If checkpoint path is provided, validate that it exists and is a file.
  if (args.checkpoint is not None):
    if (not os.path.isfile(args.checkpoint)):
      raise RuntimeError(f"Model checkpoint file does not exist or is not a file: {args.checkpoint}")

  # Validate that image size is a positive integer.
  if ((not isinstance(args.imageSize, int)) or (args.imageSize <= 0)):
    raise ValueError("Image size must be a positive integer.")

  # Validate that patch size, if provided, is a positive integer.
  if (args.patchSize is not None):
    if ((not isinstance(args.patchSize, int)) or (args.patchSize <= 0)):
      raise ValueError("Patch size must be a positive integer.")

  # Validate that top-k values are positive integers.
  for k in args.topK:
    if ((not isinstance(k, int)) or (k <= 0)):
      raise ValueError(f"Top-k values must be positive integers. Invalid value: {k}")

  # Validate the mask directory when it is provided.
  if (args.maskDir is not None):
    # Check whether the mask directory exists and is a directory.
    if (not os.path.isdir(args.maskDir)):
      # Raise an error when the mask directory is invalid.
      raise RuntimeError(f"Mask directory does not exist or is not a directory: {args.maskDir}")


def ApplyMetadataMissingness(metadata, missingnessRate, missingnessMode, missingnessScope, missingnessNoiseStd):
  # Return the original metadata when no missingness is requested.
  if ((missingnessRate is None) or (missingnessRate <= 0.0)):
    return metadata

  # Clone the metadata tensor to avoid modifying the original tensor.
  corruptedMetadata = metadata.clone()

  # Create a sample-level missingness mask when requested.
  if (missingnessScope == "Sample"):
    # Draw one random value per sample.
    missingnessMask = torch.rand(corruptedMetadata.size(0), 1, device=corruptedMetadata.device) < missingnessRate
    # Expand the sample mask across all metadata features.
    missingnessMask = missingnessMask.expand_as(corruptedMetadata)
  # Create a feature-level missingness mask when requested.
  else:
    # Draw one random value per metadata feature.
    missingnessMask = torch.rand(corruptedMetadata.size(), device=corruptedMetadata.device) < missingnessRate

  # Replace missing entries with zeros when requested.
  if (missingnessMode == "Zero"):
    # Fill masked metadata entries with zero.
    corruptedMetadata = corruptedMetadata.masked_fill(missingnessMask, 0.0)
  # Replace missing entries with batch means when requested.
  elif (missingnessMode == "Mean"):
    # Compute the mean metadata value for each feature in the current batch.
    batchMean = corruptedMetadata.mean(dim=0, keepdim=True)
    # Replace masked entries with the batch mean values.
    corruptedMetadata = torch.where(missingnessMask, batchMean.expand_as(corruptedMetadata), corruptedMetadata)
  # Replace missing entries with Gaussian noise when requested.
  elif (missingnessMode == "Gaussian"):
    # Generate Gaussian noise with the requested standard deviation.
    noiseTensor = torch.normal(
      mean=0.0,
      std=missingnessNoiseStd,
      size=corruptedMetadata.size(),
      device=corruptedMetadata.device
    )
    # Replace masked entries with the generated noise values.
    corruptedMetadata = torch.where(missingnessMask, noiseTensor, corruptedMetadata)
    # Clamp corrupted values to the normalized metadata range.
    corruptedMetadata = torch.clamp(corruptedMetadata, 0.0, 1.0)

  # Return the corrupted metadata tensor.
  return corruptedMetadata
