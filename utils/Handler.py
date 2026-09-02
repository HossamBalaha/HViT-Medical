# Import the necessary libraries.
import torch, tqdm, gc, os, cv2
# Import typing modules for type hinting.
from typing import Any, Dict, List, Optional, Tuple, Callable
# Import Pandas for data manipulation and analysis.
import pandas as pd
# Import the neural network module from PyTorch.
import torch.nn as nn
# Import Matplotlib for plotting.
import matplotlib.pyplot as plt
# Import metrics from scikit-learn.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Import learning rate schedulers from PyTorch.
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
# Import optimizers from PyTorch.
from torch.optim import AdamW, SGD, Adam, RMSprop
# Import torchvision transforms for image preprocessing.
from torchvision import transforms
# Import helper functions for the Vision Transformer.
from utils.ProposalHelpers import *
# Import the ProposalHelpers module explicitly.
import utils.ProposalHelpers as ProposalHelpers
# Import the metadata missingness helper.
from utils.Utils import ApplyMetadataMissingness
# Import dataset helper functions for loading and processing datasets.
from utils.DatasetHelpers import *
# Import performance metrics functions.
from HMB.PerformanceMetrics import *
# Import utility functions for CSV and YAML operations.
from HMB.Utils import AppendOrCreateNewDataFrameCSV, SaveYaml, fprint
# Import the generic evaluation pipeline.
from HMB.PyTorchTrainingPipeline import GenericImageryEvaluatePredictPlotSubset


def EvaluateMetadataMissingness(model, testDataloader, device, config):
  # Set the model to evaluation mode.
  model.eval()

  # Initialize a list to store summary records for each missingness rate.
  missingnessRecords = []

  # Read the missingness evaluation rates from the configuration.
  missingnessRates = config.get("MetadataMissingEvalRates", [])

  # Read the missingness mode from the configuration.
  missingnessMode = config.get("MetadataMissingMode", "Zero")

  # Read the missingness scope from the configuration.
  missingnessScope = config.get("MetadataMissingScope", "Feature")

  # Read the missingness noise standard deviation from the configuration.
  missingnessNoiseStd = config.get("MetadataMissingNoiseStd", 0.05)

  # Iterate over each requested missingness rate.
  for missingnessRate in missingnessRates:
    # Initialize a list for all test labels.
    allTestLabels = []

    # Initialize a list for all test predictions.
    allTestPredictions = []

    # Disable gradient computation for evaluation.
    with torch.no_grad():
      # Iterate through the test dataloader.
      for images, metadata, labels in testDataloader:
        # Move images to the device.
        images = images.to(device)

        # Move metadata to the device.
        metadata = metadata.to(device)

        # Move labels to the device.
        labels = labels.to(device)

        # Apply the requested metadata missingness corruption.
        metadata = ApplyMetadataMissingness(
          metadata,
          missingnessRate,
          missingnessMode,
          missingnessScope,
          missingnessNoiseStd
        )

        # Perform a forward pass through the model.
        outputs, _ = model(images, metadata=metadata, outputAttentions=False)

        # Extend the list of test labels.
        allTestLabels.extend(labels.cpu().numpy())

        # Extend the list of test predictions.
        allTestPredictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    # Compute the confusion matrix for the current missingness rate.
    cmTest = confusion_matrix(allTestLabels, allTestPredictions)

    # Calculate performance metrics for the current missingness rate.
    metrics = CalculatePerformanceMetrics(
      cmTest,
      eps=1e-10,
      addWeightedAverage=True,
      addPerClass=True,
    )

    # Create an integer tag for the current missingness rate.
    missingnessTag = int(round(missingnessRate * 100.0))

    # Save detailed metrics for the current missingness rate.
    for key in list(metrics.keys()):
      # Save each metric value to a CSV file.
      AppendOrCreateNewDataFrameCSV(
        os.path.join(config["StoragePath"], "Testing", f"Missingness_{missingnessTag:03d}_Metrics.csv"),
        data=[[f"{key}", metrics[key]]],
        header=["Key", "Value"],
      )

    # Append a summary record for the current missingness rate.
    missingnessRecords.append([
      missingnessRate,
      metrics.get("Weighted Accuracy", 0.0),
      metrics.get("Weighted Recall", 0.0),
      metrics.get("Weighted Precision", 0.0),
      metrics.get("Weighted F1", 0.0),
      metrics.get("Weighted Specificity", 0.0),
      metrics.get("Weighted Average", 0.0),
    ])

  # Save the missingness summary records to a single CSV file.
  AppendOrCreateNewDataFrameCSV(
    os.path.join(config["StoragePath"], "Testing", "Missingness_Summary.csv"),
    data=missingnessRecords,
    header=[
      "MissingnessRate",
      "WeightedAccuracy",
      "WeightedRecall",
      "WeightedPrecision",
      "WeightedF1",
      "WeightedSpecificity",
      "WeightedAverage"
    ],
  )


# Define the main handler function to train and evaluate the model.
def Step1Handler(
  config,
  datasetPath,
  embeddedPath,
  storagePath,
  headTypesPerBlock,
  dpi=720,
  modelType="ViTForClassification"
):
  # Assert that the storage path does not already exist.
  assert not os.path.exists(storagePath), f"Model storage path {storagePath} already exists!"
  # Create the main directory for storing the model.
  os.makedirs(storagePath, exist_ok=True)
  # Create the training subdirectory.
  os.makedirs(storagePath + "/Training", exist_ok=True)
  # Create the testing subdirectory.
  os.makedirs(storagePath + "/Testing", exist_ok=True)
  # Define the path to store the best model.
  modelStoragePath = os.path.join(storagePath, "BestModel.pth")
  # Define the path to store the last model.
  lastModelStoragePath = os.path.join(storagePath, "LastModel.pth")
  # Define the path to store the training history.
  historyFilePath = os.path.join(storagePath, "History.csv")
  # Define the path to store the hyperparameter configuration.
  hparamsConfigsPath = os.path.join(storagePath, "Config.yaml")

  # Set the device to GPU if available, otherwise CPU.
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Read the batch size from the configuration.
  batchSize = config["BatchSize"]
  # Read the number of epochs from the configuration.
  epochs = config["Epochs"]
  # Read the optimizer name from the configuration.
  optimizerStr = config["Optimizer"]
  # Read the optimizer parameters from the configuration.
  optimizerParams = config["OptimizerParams"]
  # Read the scheduler name from the configuration.
  schedulerStr = config["Scheduler"]
  # Read the scheduler parameters from the configuration.
  schedulerParams = config["SchedulerParams"]

  # Check if CUDA is available.
  if (torch.cuda.is_available()):
    # Set the device to CUDA.
    device = torch.device("cuda")
  else:
    # Set the device to CPU.
    device = torch.device("cpu")

  # # Retrieve the model class dynamically using the provided model type string.
  # modelClass = getattr(ProposalHelpers, modelType)
  # if (modelType == "PhikonV2LinearProbe"):
  #   # Instantiate the linear probe using only the configuration dictionary.
  #   model = modelClass(config)
  # else:
  #   # Check whether the architecture supplied a model name instead of head types.
  #   if (isinstance(headTypesPerBlock, str)):
  #     # Build one standard head list for every hidden layer.
  #     headTypesPerBlock = [
  #       ["Standard"] * config["NumAttentionHeads"]
  #       for layerIndex in range(config["NumHiddenLayers"])
  #     ]
  #   # Instantiate standard transformer models with both config and head types.
  #   model = modelClass(config, headTypesPerBlock)
  
  # Retrieve the model class dynamically using the provided model type string.
  modelClass = getattr(ProposalHelpers, modelType)
  # Instantiate the selected model with both the configuration and architecture definition.
  model = modelClass(config, headTypesPerBlock)

  # Check if the dataset path exists.
  if (not os.path.exists(datasetPath)):
    # Raise an error if the dataset path is missing.
    raise FileNotFoundError(f"Dataset path {datasetPath} does not exist.")
  # Check if the embedded metadata path exists.
  if (not os.path.exists(embeddedPath)):
    # Raise an error if the embedded metadata path is missing.
    raise FileNotFoundError(f"Embedded metadata path {embeddedPath} does not exist.")

  # Load the dataset and embedded metadata.
  trainDataloader, testDataloader, combinedDataloader, classesMap, rawDataset, embeddedMetadata = DataLoader(
    config,
    datasetPath,
    embeddedPath,
    batchSize,
  )

  # Print the dataset specifications header.
  fprint("Dataset specifications:")
  # Print the number of classes.
  fprint("Number of classes:", len(rawDataset.classes))
  # Print the class mapping.
  fprint("Classes map:", rawDataset.classes)
  # Print the total number of samples.
  fprint("Number of samples:", len(rawDataset))
  # Print the number of training samples.
  fprint("Number of training samples:", len(trainDataloader.dataset))
  # Print the number of testing samples.
  fprint("Number of testing samples:", len(testDataloader.dataset))
  # Print the number of combined samples.
  fprint("Number of combined samples:", len(combinedDataloader.dataset))

  # Print a message indicating the start of training.
  fprint("Training the model...")
  # Move the model to the selected device.
  model.to(device)
  # Set the model to training mode.
  model.train()

  # Initialize the AdamW optimizer if specified.
  if (optimizerStr == "AdamW"):
    # Define the AdamW optimizer.
    optimizer = AdamW(model.parameters(), **optimizerParams)
  # Initialize the SGD optimizer if specified.
  elif (optimizerStr == "SGD"):
    # Define the SGD optimizer.
    optimizer = SGD(model.parameters(), **optimizerParams)
  # Initialize the Adam optimizer if specified.
  elif (optimizerStr == "Adam"):
    # Define the Adam optimizer.
    optimizer = Adam(model.parameters(), **optimizerParams)
  # Initialize the RMSprop optimizer if specified.
  elif (optimizerStr == "RMSprop"):
    # Define the RMSprop optimizer.
    optimizer = RMSprop(model.parameters(), **optimizerParams)
  # Raise an error for unsupported optimizers.
  else:
    # Raise a ValueError for unsupported optimizers.
    raise ValueError(f"Unsupported optimizer: {optimizerStr}")

  # Initialize the CosineAnnealingLR scheduler if specified.
  if (schedulerStr == "CosineAnnealingLR"):
    # Define the CosineAnnealingLR scheduler.
    scheduler = CosineAnnealingLR(optimizer, **schedulerParams)
  # Initialize the StepLR scheduler if specified.
  elif (schedulerStr == "StepLR"):
    # Define the StepLR scheduler.
    scheduler = StepLR(optimizer, **schedulerParams)
  # Initialize the ExponentialLR scheduler if specified.
  elif (schedulerStr == "ExponentialLR"):
    # Define the ExponentialLR scheduler.
    scheduler = ExponentialLR(optimizer, **schedulerParams)
  # Raise an error for unsupported schedulers.
  else:
    # Raise a ValueError for unsupported schedulers.
    raise ValueError(f"Unsupported scheduler: {schedulerStr}")

  # Extract all training labels to compute class frequencies.
  allTrainLabels = [label for label in trainDataloader.dataset.labels]
  # Calculate the count of each class in the training set.
  classCounts = np.bincount(allTrainLabels, minlength=len(rawDataset.classes))
  # Compute inverse frequency weights to mitigate class imbalance.
  classWeights = 1.0 / (classCounts + 1e-8)
  # Normalize the weights so they sum to the number of classes.
  classWeights = classWeights / classWeights.sum() * len(rawDataset.classes)
  # Convert the numpy weight array to a PyTorch tensor.
  weightTensor = torch.tensor(classWeights, dtype=torch.float32).to(device)
  # Define the weighted cross-entropy loss function.
  criterion = nn.CrossEntropyLoss(weight=weightTensor)

  # Initialize the optimal loss to infinity.
  optimalLoss = math.inf
  # Initialize the optimal score to zero.
  optimalScore = 0.0
  # Initialize lists to store loss and score history.
  lossHistory, scoreHistory = [], []

  # Start the training loop for the specified number of epochs.
  for epoch in range(epochs):
    # Create a progress bar for the dataloader.
    loaderIter = tqdm.tqdm(trainDataloader, desc=f"Epoch {epoch + 1}/{epochs}")
    # Initialize the average loss for the epoch.
    avgLoss = 0.0
    # Initialize the average score for the epoch.
    avgScore = 0.0
    # Initialize a list to collect all labels.
    allLabels = []
    # Initialize a list to collect all predictions.
    allPredictions = []

    # Iterate through the training batches.
    for images, metadata, labels in loaderIter:
      # Move images to the device.
      images = images.to(device)

      # Move metadata to the device.
      metadata = metadata.to(device)

      # Move labels to the device.
      labels = labels.to(device)

      # Apply training-time metadata missingness when requested.
      if (config.get("MetadataMissingTrainRate", 0.0) > 0.0):
        # Corrupt the metadata tensor using the configured missingness settings.
        metadata = ApplyMetadataMissingness(
          metadata,
          config.get("MetadataMissingTrainRate", 0.0),
          config.get("MetadataMissingMode", "Zero"),
          config.get("MetadataMissingScope", "Feature"),
          config.get("MetadataMissingNoiseStd", 0.05),
        )

      # Zero the gradients.
      optimizer.zero_grad()

      # Perform a forward pass through the model.
      outputs, _ = model.forward(images, metadata=metadata)
      # Extend the list of labels with the current batch.
      allLabels.extend(labels.cpu().numpy())
      # Extend the list of predictions with the current batch.
      allPredictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
      # Compute the loss.
      loss = criterion(outputs, labels)
      # Perform a backward pass to compute gradients.
      loss.backward()
      # Update the model parameters.
      optimizer.step()
      # Accumulate the loss.
      avgLoss += loss.item()

    # Step the learning rate scheduler.
    scheduler.step()
    # Compute the average loss for the epoch.
    avgLoss /= len(trainDataloader)
    # Compute the average score for the epoch.
    avgScore /= len(trainDataloader)
    # Get the current learning rate.
    currentLR = scheduler.get_last_lr()[0]

    # Compute the confusion matrix.
    cm = confusion_matrix(allLabels, allPredictions)
    # Calculate the performance metrics.
    metrics = CalculatePerformanceMetrics(
      cm,
      eps=1e-10,
      addWeightedAverage=True,
      addPerClass=False,
    )
    # Update the average score from the metrics.
    avgScore = metrics["Weighted Average"]
    # Print the epoch details.
    fprint(
      f"Epoch {epoch + 1}, Avg. Loss: {avgLoss:.4f}, "
      f"Avg. Score: {avgScore:.4f}, Current LR: {currentLR:.6f}"
    )
    # Save the epoch details to a CSV file.
    AppendOrCreateNewDataFrameCSV(
      historyFilePath,
      data=[
        [
          f"{epoch + 1}",
          f"{avgLoss:.6f}",
          f"{avgScore:.6f}",
          f"{currentLR:.6f}"
        ] + [f"{metrics[key]}" for key in list(metrics.keys())]
      ],
      header=["Epoch", "AvgLoss", "AvgScore", "CurrentLR"] + list(metrics.keys()),
    )

    # Check if the current model is the best so far.
    if ((avgScore > optimalScore) and (avgLoss < optimalLoss)):
      # Update the optimal score.
      optimalScore = avgScore
      # Update the optimal loss.
      optimalLoss = avgLoss
      # Print a message indicating model saving.
      fprint("Saving the model...")
      # Save the model state.
      torch.save(model.state_dict(), modelStoragePath)
      # Append the average loss to the history.
      lossHistory.append(avgLoss)
      # Append the average score to the history.
      scoreHistory.append(avgScore)

  # Save the last model state.
  torch.save(model.state_dict(), lastModelStoragePath)

  # Plot the training loss curve.
  PlotMetricCurve(
    lossHistory,
    xData=None,
    title="Training Loss Curve",
    xLabel="Epochs",
    yLabel="Loss",
    fontSize=15,
    xTicks=None,
    yTicks=None,
    xTicksRotation=0,
    yTicksRotation=0,
    save=True,
    savePath=os.path.join(storagePath, "Training", "LossCurve.pdf"),
    dpi=dpi,
    display=False,
    figSize=(5, 5),
    returnFig=False,
  )

  # Plot the training performance curve.
  PlotMetricCurve(
    scoreHistory,
    xData=None,
    title="Training Performance Curve",
    xLabel="Epochs",
    yLabel="Score",
    fontSize=15,
    xTicks=None,
    yTicks=None,
    xTicksRotation=0,
    yTicksRotation=0,
    save=True,
    savePath=os.path.join(storagePath, "Training", "ScoreCurve.pdf"),
    dpi=dpi,
    display=False,
    figSize=(5, 5),
    returnFig=False,
  )

  # Print a message indicating the end of training.
  fprint("Training completed.")
  # Print the optimal loss.
  fprint("Optimal Loss:", optimalLoss)
  # Print the optimal score.
  fprint("Optimal Score:", optimalScore)

  # Load the saved best model state.
  model.load_state_dict(torch.load(modelStoragePath))

  # Print a message indicating evaluation.
  fprint("Evaluating the model...")
  # Set the model to evaluation mode.
  model.eval()
  # Initialize a list for all test labels.
  allTestLabels = []
  # Initialize a list for all test predictions.
  allTestPredictions = []
  # Initialize a list for all test attentions.
  allTestAttentions = []

  # Disable gradient computation for evaluation.
  with torch.no_grad():
    # Iterate through the test dataloader.
    for images, metadata, labels in testDataloader:
      # Move images to the device.
      images = images.to(device)
      # Move metadata to the device.
      metadata = metadata.to(device)
      # Move labels to the device.
      labels = labels.to(device)
      # Perform a forward pass through the model.
      outputs, attentions = model(images, metadata=metadata, outputAttentions=True)

      # Extend the list of test labels.
      allTestLabels.extend(labels.cpu().numpy())

      # Extend the list of test predictions.
      allTestPredictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())

      # Extend the list of test attentions only if the model returns them.
      if (attentions is not None):
        allTestAttentions.extend(attentions)

  # Compute the test confusion matrix.
  cmTest = confusion_matrix(allTestLabels, allTestPredictions)
  # Calculate the test performance metrics.
  metrics = CalculatePerformanceMetrics(
    cmTest,
    eps=1e-10,
    addWeightedAverage=True,
    addPerClass=True,
  )
  # Print a header for the metrics.
  fprint("Testing Performance Metrics:")
  # Iterate through the metrics dictionary.
  for key, value in metrics.items():
    # Print each metric.
    fprint(f"{key}: {value}")

  # Iterate through the metrics keys to save them.
  for key in list(metrics.keys()):
    # Save the metrics to a CSV file.
    AppendOrCreateNewDataFrameCSV(
      os.path.join(storagePath, "Testing", "Metrics.csv"),
      data=[[f"{key}", metrics[key]]],
      header=["Key", "Value"],
    )

  # Add the head types to the configuration.
  config["HeadTypes"] = headTypesPerBlock
  # Add the class mapping to the configuration.
  config["ClassesMap"] = rawDataset.classes
  # Update the number of classes in the configuration.
  config["NumClasses"] = len(rawDataset.classes)
  # Add the number of samples to the configuration.
  config["NumSamples"] = len(rawDataset)
  # Add the number of training samples to the configuration.
  config["NumTrainSamples"] = len(trainDataloader.dataset)
  # Add the number of testing samples to the configuration.
  config["NumTestSamples"] = len(testDataloader.dataset)
  # Add the batch size to the configuration.
  config["BatchSize"] = batchSize
  # Add the epochs to the configuration.
  config["Epochs"] = epochs
  # Add the optimizer string to the configuration.
  config["Optimizer"] = optimizerStr
  # Add the scheduler string to the configuration.
  config["Scheduler"] = schedulerStr
  # Add the optimizer parameters to the configuration.
  config["OptimizerParams"] = optimizerParams
  # Add the scheduler parameters to the configuration.
  config["SchedulerParams"] = schedulerParams
  # Add the optimal loss to the configuration.
  config["OptimalLoss"] = float(optimalLoss)
  # Add the optimal score to the configuration.
  config["OptimalScore"] = float(optimalScore)
  # Add the device information to the configuration.
  config["Device"] = str(device)
  # Add the model storage path to the configuration.
  config["ModelStoragePath"] = str(modelStoragePath)
  # Add the last model storage path to the configuration.
  config["LastModelStoragePath"] = str(lastModelStoragePath)
  # Add the history file path to the configuration.
  config["HistoryFilePath"] = str(historyFilePath)
  # Add the length of the training dataloader to the configuration.
  config["TrainDataloaderLength"] = len(trainDataloader)
  # Add the length of the testing dataloader to the configuration.
  config["TestDataloaderLength"] = len(testDataloader)
  # Indicate that training has been completed.
  config["TrainingCompleted"] = True
  # Indicate that evaluation has been completed.
  config["EvaluationCompleted"] = True
  # Add the storage path to the configuration.
  config["StoragePath"] = str(storagePath)
  # Add the head types per block to the configuration.
  config["HeadTypesPerBlock"] = headTypesPerBlock
  # Add the device type to the configuration.
  config["DeviceType"] = "GPU" if torch.cuda.is_available() else "CPU"

  # Save the configuration dictionary to a YAML file.
  SaveYaml(
    hparamsConfigsPath,
    config
  )

  # Plot the ROC and AUC curves.
  PlotROCAUCCurve(
    allTestLabels,
    allTestPredictions,
    rawDataset.classes,
    areProbabilities=False,
    title="ROC Curve & AUC",
    figSize=(5, 5),
    cmap=None,
    display=False,
    save=True,
    fileName=os.path.join(storagePath, "Testing", "ROCCurve.pdf"),
    fontSize=15,
    plotDiagonal=True,
    annotateAUC=True,
    showLegend=True,
    returnFig=False,
    dpi=dpi,
  )

  # Plot the Precision-Recall curves.
  PlotPRCCurve(
    allTestLabels,
    allTestPredictions,
    rawDataset.classes,
    areProbabilities=False,
    title="PRC Curve",
    figSize=(5, 5),
    cmap=None,
    display=False,
    save=True,
    fileName=os.path.join(storagePath, "Testing", "PRCCurve.pdf"),
    fontSize=15,
    annotateAvg=True,
    showLegend=True,
    returnFig=False,
    dpi=dpi,
  )

  # Plot the confusion matrix.
  PlotConfusionMatrix(
    cmTest,
    rawDataset.classes,
    normalize=False,
    roundDigits=3,
    title="Confusion Matrix",
    cmap=plt.cm.Blues,
    display=False,
    save=True,
    fileName=os.path.join(storagePath, "Testing", "ConfusionMatrix.pdf"),
    fontSize=15,
    annotate=True,
    figSize=(8, 8),
    colorbar=True,
    returnFig=False,
    dpi=dpi,
  )

  # Run the metadata missingness ablation when evaluation rates are provided.
  if (len(config.get("MetadataMissingEvalRates", [])) > 0):
    # Evaluate the trained model under each requested missingness rate.
    EvaluateMetadataMissingness(model, testDataloader, device, config)

  # Wrap the model prediction function.
  ViTPredictFn = ViTPredictFnWrapper(config, model, rawDataset, embeddedMetadata)

  # Print a message indicating the start of full evaluation.
  fprint("Running full evaluation with `GenericImageryEvaluatePredictPlotSubset`...")
  # Run the comprehensive evaluation pipeline.
  GenericImageryEvaluatePredictPlotSubset(
    datasetDir=datasetPath,
    model=ViTPredictFn,
    subset=None,
    prefix="ViT",
    storageDir=storagePath + "/Artifacts",
    heavy=True,
    computeECE=True,
    exportFailureCases=True,
    saveArtifacts=True,
    maxSamples=None,
    preprocessFn=None,
  )

  # Iterate through local variables to free memory.
  for var in list(locals().keys()):
    # Delete the current local variable.
    del locals()[var]
  # Collect garbage to free memory.
  gc.collect()


# Define the wrapper function for the ViT prediction logic.
def ViTPredictFnWrapper(config, model, rawDataset, embeddedMetadata) -> tuple[Callable[..., Any], Any]:
  # Define the inner prediction function.
  def ViTPredictFn(
    image: np.ndarray,
    imgPath: Optional[str] = None,
    cls: Optional[int | str] = None,
  ) -> np.ndarray:
    r'''
    Predict class probabilities for a single image using the trained ViT model.

    This function replicates the exact preprocessing used by `allDataloader`,
    which applies only deterministic validation transforms (no augmentation).

    Input assumptions:
      - HWC format (height, width, channels)
      - Supported dtypes: uint8 (0–255) or float32/64 (0–1 or 0–255)
      - Grayscale or BGR inputs are automatically converted to RGB

    Parameters:
      image (numpy.ndarray): Input image in HWC format.

    Returns:
      numpy.ndarray: 1D array of class probabilities (length = num_classes).
    '''

    # Import OpenCV for image processing.
    import cv2
    # Import PyTorch for tensor operations.
    import torch
    # Import NumPy for array operations.
    import numpy as np
    # Import PIL for image handling.
    from PIL import Image
    # Import torchvision transforms for preprocessing.
    import torchvision.transforms as transforms

    # Check if the input is a NumPy array.
    if (not isinstance(image, np.ndarray)):
      # Raise an error if the input format is incorrect.
      raise ValueError("Input must be a numpy.ndarray in HWC format.")

    # Set the model to evaluation mode.
    model.eval()
    # Get the device of the model parameters.
    device = next(model.parameters()).device

    # Convert the NumPy array to a PIL image.
    pilImg = Image.fromarray(image)
    # Define the validation transformation pipeline.
    valTransform = transforms.Compose([
      transforms.Resize((config["ImageSize"], config["ImageSize"])),
      transforms.ToTensor(),
    ])
    # Apply the transformations and add a batch dimension.
    imgTensor = valTransform(pilImg).unsqueeze(0).to(device)

    # Extract the filename from the image path.
    filename = os.path.basename(imgPath)
    # Determine the label index from the class argument.
    labelIndex = cls if isinstance(cls, int) else rawDataset.classes.index(cls)
    # Construct the metadata dictionary key.
    key = f"{labelIndex}_{filename}"
    # Check if the key exists in the embedded metadata.
    if (key not in embeddedMetadata):
      className = rawDataset.classes[labelIndex]
      # Try the class name instead of the label index.
      key = f"{className}_{filename}"
    # Check if the key exists in the embedded metadata.
    if (key not in embeddedMetadata):
      # Raise an error if the metadata is missing.
      raise KeyError(f"Metadata for key {key} not found in embedded metadata.")
    # Retrieve the metadata array.
    metadataArray = embeddedMetadata[key]
    # Convert the metadata to a tensor and add a batch dimension.
    metadataTensor = torch.tensor(metadataArray, dtype=torch.float32).unsqueeze(0).to(device)

    # Disable gradient computation for inference.
    with torch.no_grad():
      # Perform a forward pass through the model.
      outputs, _ = model(imgTensor, metadata=metadataTensor)
      # Compute the softmax probabilities and convert to NumPy.
      probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    # Return the probabilities as a float32 array.
    return probs.astype(np.float32)

  # Return the inner prediction function.
  return ViTPredictFn
