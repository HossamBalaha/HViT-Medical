import os, argparse, json, math, torch, cv2
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import List, Tuple, Optional
from utils.Utils import *
from utils.DatasetHelpers import *
from utils.ProposalHelpers import ViTForClassification
from HMB.Initializations import UpdateMatplotlibSettings
from HMB.Utils import fprint, LoadYaml


# Extract CLS-to-patches attention vector from the model's attention outputs.
def ExtractClsToPatchesAttention(
  allAttentions: List[np.ndarray], layerIdx: int = -1,
  aggregateHeads: str = "mean"
) -> np.ndarray:
  # Select the requested layer attentions from the provided list.
  att = allAttentions[layerIdx]
  # Convert numpy arrays to torch tensors when needed.
  if (isinstance(att, np.ndarray)):
    # Convert to torch tensor for subsequent indexing and operations.
    att = torch.from_numpy(att)
  # Extract CLS token row to patch tokens (token 0 is CLS, tokens 1..N are patches).
  clsToPatches = att[:, :, 0, 1:]
  # Aggregate across heads according to the chosen method.
  if (aggregateHeads == "mean"):
    # Compute mean across head dimension.
    vec = clsToPatches.mean(dim=1)
  elif (aggregateHeads == "sum"):
    # Compute sum across head dimension.
    vec = clsToPatches.sum(dim=1)
  else:
    # Raise an error for unknown aggregation options.
    raise ValueError("Unknown aggregateHeads.")
  # Convert tensor to numpy array on CPU.
  vec = vec.cpu().numpy()
  # Normalize each sample attention vector to sum to 1.
  vec = vec / (vec.sum(axis=1, keepdims=True) + 1e-12)
  # Return the normalized attention vectors.
  return vec


# Validate and normalize a single attention vector (1D numpy array).
def ValidateAndNormalizeAttention(attnVec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
  # Convert to numpy array of float type.
  v = np.asarray(attnVec, dtype=np.float64).copy()
  # Replace negative values with zero (attention should be non-negative).
  v[v < 0] = 0.0
  s = v.sum()
  if (s <= 0):
    # If the vector sums to zero, fall back to uniform distribution.
    v = np.ones_like(v, dtype=np.float64) / float(v.shape[0])
    return v
  # Normalize to sum to 1 with numerical stability.
  return v / (s + eps)


def ComputeEntropy(p: np.ndarray, base: str = "natural") -> float:
  # Convert the input probability vector to a numpy array.
  pArr = np.asarray(p)
  # Clip values to prevent logarithmic evaluation of zero.
  pArr = np.clip(pArr, 1e-12, 1.0)
  # Compute the Shannon entropy in nats using the natural logarithm.
  ent = -np.sum(pArr * np.log(pArr))
  # Convert the entropy to bits only if explicitly requested.
  if (base == "bits"):
    # Scale the natural logarithm result by the natural log of two.
    ent = ent / np.log(2.0)
  # Return the final computed entropy value.
  return ent


def ComputeAttentionIoU(
  AttentionVector: np.ndarray,
  MaskImage: Image.Image,
  ImageSize: int,
  PatchSize: int,
  TopK: int
) -> float:
  # Calculate the number of patches per side to form the spatial grid.
  GridSize = ImageSize // PatchSize
  # Reshape the flat attention vector into a 2D grid of patch weights.
  AttentionGrid = AttentionVector.reshape((GridSize, GridSize))
  # Extract the flat indices of the top-K highest attention weights.
  TopKIndices = np.argsort(AttentionGrid.flatten())[::-1][:TopK]
  # Initialize a binary prediction mask on the patch grid.
  PredictedGrid = np.zeros((GridSize, GridSize), dtype=np.float32)
  # Iterate over the top-K indices to mark the attended patches.
  for FlatIdx in TopKIndices:
    # Compute the 2D row coordinate from the flat index.
    RowIdx = FlatIdx // GridSize
    # Compute the 2D column coordinate from the flat index.
    ColIdx = FlatIdx % GridSize
    # Activate the corresponding patch in the prediction grid.
    PredictedGrid[RowIdx, ColIdx] = 1.0
  # Resize the ground-truth mask to match the patch grid dimensions using nearest-neighbor interpolation.
  ResizedMask = MaskImage.resize((GridSize, GridSize), Image.Resampling.NEAREST)
  # Convert the resized PIL mask into a binary numpy array.
  GroundTruthGrid = (np.array(ResizedMask) > 0).astype(np.float32)
  # Calculate the intersection area between the predicted and ground-truth grids.
  Intersection = np.sum(PredictedGrid * GroundTruthGrid)
  # Calculate the union area of the two grids.
  Union = np.sum(PredictedGrid) + np.sum(GroundTruthGrid) - Intersection
  # Return the Intersection over Union score, handling division by zero.
  return float(Intersection / Union) if (Union > 0) else 0.0


# Resolve the ground-truth mask path corresponding to an image path.
def ResolveMaskPath(imgPath: str, maskDir: Optional[str]) -> Optional[Path]:
  # Return None when no mask directory is provided.
  if (maskDir is None):
    # Indicate that mask-based evaluation is disabled.
    return None

  # Extract the image file stem from the image path.
  fileStem = Path(imgPath).stem

  # Define candidate mask suffixes.
  suffixCandidates = ["", "_mask", "_segmentation", "_lesion"]

  # Define candidate mask extensions.
  extensionCandidates = [".png", ".PNG", ".tif", ".TIF", ".bmp", ".BMP", ".jpg", ".JPG"]

  # Iterate over suffix candidates.
  for suffix in suffixCandidates:
    # Iterate over extension candidates.
    for extension in extensionCandidates:
      # Build a candidate mask file path.
      candidatePath = Path(maskDir) / f"{fileStem}{suffix}{extension}"

      # Return the candidate path when the file exists.
      if (candidatePath.is_file()):
        # Return the located mask path.
        return candidatePath

  # Return None when no matching mask file is found.
  return None


# Build patch grid coordinates for a square image divided by patchSize.
def PatchGridCoords(imageSize: int, patchSize: int) -> List[Tuple[int, int, int, int]]:
  # Initialize an empty list to hold patch boxes.
  coords: List[Tuple[int, int, int, int]] = []
  # Compute number of patches per side (integer division).
  n = imageSize // patchSize
  # Iterate rows and columns to populate patch coordinates.
  for r in range(n):
    for c in range(n):
      # Compute top-left x coordinate.
      x0 = c * patchSize
      # Compute top-left y coordinate.
      y0 = r * patchSize
      # Append the patch box as (x0, y0, w, h).
      coords.append((x0, y0, patchSize, patchSize))
  # Return the list of patch coordinates.
  return coords


def SaveHeatmapOverlay(
  origImg,
  attnVec: np.ndarray,
  patchCoords: List[Tuple[int, int, int, int]],
  outPath: str,
  cmap: str = "summer",
  alpha: float = 0.5,
) -> None:
  # Convert the PIL image to a numpy array in RGB format.
  origNumpy = np.array(origImg.convert("RGB"))
  # Determine the image pixel dimensions (height, width) for array operations.
  imgHeight, imgWidth = origNumpy.shape[:2]
  # Create an empty numpy array for the heatmap values.
  hmArr = np.zeros((imgHeight, imgWidth), dtype=np.float32)
  # Validate that the attention vector length matches patchCoords (or use minimum).
  nPatches = min(len(attnVec), len(patchCoords))
  # Paint each patch value into the heatmap array.
  for i in range(nPatches):
    # Get the attention value for the current patch.
    v = float(attnVec[i])
    # Extract patch coordinates and dimensions.
    x0, y0, patchW, patchH = patchCoords[i]
    # Calculate the end coordinates clamped to image boundaries.
    x1 = min(x0 + patchW, imgWidth)
    y1 = min(y0 + patchH, imgHeight)
    # Clamp the start coordinates to be non-negative.
    x0Clamped = max(0, x0)
    y0Clamped = max(0, y0)
    # Assign the attention value to the patch region in the heatmap.
    hmArr[y0Clamped:y1, x0Clamped:x1] = v
  # Normalize heatmap to [0, 1] range for colormap mapping.
  if (hmArr.max() - hmArr.min() > 0):
    # Apply min-max normalization to the heatmap array.
    hmArr = (hmArr - hmArr.min()) / (hmArr.max() - hmArr.min() + 1e-12)
  # Otherwise set all values to zero if normalization is not possible.
  else:
    # Fill the heatmap array with zeros.
    hmArr = np.zeros_like(hmArr)
  # Scale the normalized heatmap to 0-255 range for OpenCV colormap.
  hmScaled = (hmArr * 255).astype(np.uint8)
  # Map the OpenCV colormap name to the corresponding constant.
  cmapMap = {
    "jet"    : cv2.COLORMAP_JET,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma" : cv2.COLORMAP_PLASMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma"  : cv2.COLORMAP_MAGMA,
    "hot"    : cv2.COLORMAP_HOT,
    "cool"   : cv2.COLORMAP_COOL,
    "spring" : cv2.COLORMAP_SPRING,
    "summer" : cv2.COLORMAP_SUMMER,
    "autumn" : cv2.COLORMAP_AUTUMN,
    "winter" : cv2.COLORMAP_WINTER,
    "rainbow": cv2.COLORMAP_RAINBOW,
    "turbo"  : cv2.COLORMAP_TURBO,
  }
  # Get the OpenCV colormap constant or default to JET.
  cmapConst = cmapMap.get(cmap.lower(), cv2.COLORMAP_JET)
  # Apply the colormap to the scaled heatmap to get a BGR image.
  heatBgr = cv2.applyColorMap(hmScaled, cmapConst)
  # Convert the original image from RGB to BGR for OpenCV compatibility.
  origBgr = cv2.cvtColor(origNumpy, cv2.COLOR_RGB2BGR)
  # Blend the original image with the heatmap using weighted addition.
  # Formula: dst = src1 * (1-alpha) + src2 * alpha + gamma.
  blended = cv2.addWeighted(origBgr, 1.0 - alpha, heatBgr, alpha, 0)
  # Ensure the output directory exists before saving.
  outDir = os.path.dirname(outPath)
  # Create the output directory if it does not exist and is specified.
  if (outDir and not os.path.exists(outDir)):
    # Make the directory with exist_ok to avoid errors if already present.
    os.makedirs(outDir, exist_ok=True)
  # Save the blended image to the specified output path using OpenCV.
  # Note: cv2.imwrite does not preserve DPI metadata use PIL if DPI is required.
  cv2.imwrite(outPath, blended)


def main():
  # Validate the command-line arguments.
  args = Step5ParseArgs()  # Parse command-line arguments.
  Step5ValidateArgs(args)
  UpdateMatplotlibSettings()

  # Choose device based on request and availability.
  device = torch.device("cuda" if (args.mustCuda and torch.cuda.is_available()) else "cpu")

  hparam = LoadHParams(args.hparamsFile, hparamName=args.hparamName)

  # Ensure required keys exist and override with CLI parameters.
  hparam = {**hparam}

  # Set image size using the Step1 hyperparameter key.
  hparam["ImageSize"] = args.imageSize

  # Override patch size when specified.
  if (args.patchSize is not None):
    # Store patch size using the Step1 hyperparameter key.
    hparam["PatchSize"] = args.patchSize

  # Resolve checkpoint path early to read trained configuration.
  chkPointPath = Path(args.checkpoint) if (args.checkpoint) else None

  # Load the configuration from the checkpoint directory to ensure matching dimensions.
  if (chkPointPath and chkPointPath.parent.is_dir()):
    configPath = chkPointPath.parent / "Config.yaml"
    if (not configPath.is_file()):
      configPath = chkPointPath.parent / "Testing" / "Config.yaml"
    if (configPath.is_file()):
      trainedConfig = LoadYaml(str(configPath))
      # Override hparam with trained config to ensure architecture matches the checkpoint.
      if ("MetadataDim" in trainedConfig):
        hparam["MetadataDim"] = trainedConfig["MetadataDim"]
      if ("ImageSize" in trainedConfig):
        hparam["ImageSize"] = trainedConfig["ImageSize"]
      if ("NumClasses" in trainedConfig):
        hparam["NumClasses"] = trainedConfig["NumClasses"]
      if ("NumChannels" in trainedConfig):
        hparam["NumChannels"] = trainedConfig["NumChannels"]

  # Ensure metadata dimension uses the Step1 key as a fallback.
  if ("MetadataDim" not in hparam):
    # Set the default metadata dimension.
    hparam["MetadataDim"] = 6

  # Ensure number of classes uses the Step1 key.
  hparam.setdefault("NumClasses", 3)

  # Ensure number of input channels uses the Step1 key.
  hparam.setdefault("NumChannels", 3)

  # Prepare output directories for results.
  outRoot = Path(f"{args.experimentsFolder}/{args.projectKeyword}/Step5-Interpretability")
  (outRoot / "Heatmaps").mkdir(parents=True, exist_ok=True)
  (outRoot / "RawAttention").mkdir(parents=True, exist_ok=True)

  # Load the dataset and embedded metadata.
  trainDataloader, testDataloader, combinedDataloader, classesMap, rawDataset, embeddedMetadata = DataLoader(
    hparam,  # Configuration dictionary.
    args.datasetPath,  # Path to the dataset.
    args.embeddedPath,  # Path to the embedded metadata.
    batchSize=1,
  )

  # Define architecture configurations using the exact Step1 architecture keys.
  archsDict = {
    "Standard"         : [["Standard"] * 6] * hparam["NumHiddenLayers"],
    # "QConditioning"    : [["Q-Conditioning"] * 6] * hparam["NumHiddenLayers"],
    "QC"               : [["Q-Conditioning"] * 6] * hparam["NumHiddenLayers"],
    # "QGating"          : [["Q-Gating"] * 6] * hparam["NumHiddenLayers"],
    "QG"               : [["Q-Gating"] * 6] * hparam["NumHiddenLayers"],
    "KeyConditioning"  : [["Key-Conditioning"] * 6] * hparam["NumHiddenLayers"],
    "ValueConditioning": [["Value-Conditioning"] * 6] * hparam["NumHiddenLayers"],
    "QCQGS"            : [["Q-Conditioning", "Q-Gating", "Standard"] * 2] * hparam["NumHiddenLayers"],
    "QCQG"             : [["Q-Conditioning", "Q-Gating"] * 3] * hparam["NumHiddenLayers"],
    "QCS"              : [["Q-Conditioning", "Standard"] * 3] * hparam["NumHiddenLayers"],
    "QGS"              : [["Q-Gating", "Standard"] * 3] * hparam["NumHiddenLayers"],
  }

  # chkPointPath = Path(args.checkpoint) if (args.checkpoint) else None
  # parent = chkPointPath.parent.parent if (chkPointPath) else None
  # Determine the parent directory of the checkpoint to extract the architecture key.
  parent = chkPointPath.parent.parent if (chkPointPath) else None
  if (not parent):
    raise ValueError("Checkpoint path is required to determine architecture for interpretability analysis.")
  else:
    # # archKey = parent.name
    # # archKey = archKey[:archKey.find("_Preset_")] if ("_Preset_" in archKey) else archKey
    # # Extract the architecture key by slicing up to the preset name and removing trailing underscores.
    # archKey = parent.name
    # if ("Preset" in archKey):
    #   archKey = archKey[:archKey.find("Preset")].rstrip("_")
    # headTypes = archsDict.get(archKey)
    # if (headTypes is None):
    #   raise ValueError(f"Unknown architecture key derived from checkpoint path: {archKey}.")

    # Read the experiment folder name from the checkpoint path.
    archKey = parent.name

    # Keep only the architecture key before the first underscore.
    if ("_" in archKey):
      # Split the folder name and retain the architecture prefix.
      archKey = archKey.split("_", 1)[0]

    # Lookup the architecture definition using the Step1 key.
    headTypes = archsDict.get(archKey)

    # Raise an error when the architecture key is unknown.
    if (headTypes is None):
      # Provide a clear error message.
      raise ValueError(f"Unknown architecture key derived from checkpoint path: {archKey}.")
  fprint(f"Using architecture '{archKey}' with head types: {headTypes} for interpretability analysis.")

  # Instantiate the ViT model using the repository helper class.
  model = ViTForClassification(hparam, headTypes)
  # Load checkpoint weights when provided and compatible.
  if (args.checkpoint and os.path.isfile(args.checkpoint)):
    sd = torch.load(args.checkpoint, map_location="cpu")
    try:
      model.load_state_dict(sd)
    except Exception:
      # Print a warning and continue without loading if incompatible.
      fprint("Warning: checkpoint appears incompatible with model architecture skipping load.")
  # Move model to selected device.
  model.to(device)
  # Put model into evaluation mode.
  model.eval()

  # Read the training dataset from the dataloader.
  trainDataset = trainDataloader.dataset

  # Read the number of available training samples.
  numAvailable = len(trainDataset)

  # Compute the number of samples to process.
  numSamples = min(args.maxSamples, numAvailable)

  # Sample unique dataset indices for interpretability analysis.
  sampleIndices = np.random.choice(numAvailable, size=numSamples, replace=False)

  # Initialize a list for the selected examples.
  randomExamples = []

  # Iterate over the selected sample indices.
  for sampleIndex in sampleIndices:
    # Convert the NumPy index to an integer.
    datasetIndex = int(sampleIndex)

    # Read the transformed image, metadata, and label from the dataset.
    imageTensor, metadataTensor, labelTensor = trainDataset[datasetIndex]

    # Read the original image path from the dataset.
    imgPath = trainDataset.imagePaths[datasetIndex]

    # Store the example using CamelCase keys.
    randomExamples.append({
      "Index"    : datasetIndex,
      "Image"    : imageTensor,
      "Metadata" : metadataTensor,
      "Label"    : labelTensor,
      "ImagePath": imgPath,
    })

  # Print the number of selected examples.
  fprint(f"Loaded {len(randomExamples)} random examples from the training set for sanity checks and visualization.")

  # randomExamples = []
  # for i in range(args.maxSamples):
  #   nextBatch = next(iter(trainDataloader))
  #   rndExample = {
  #     "index"   : i,
  #     "image"   : nextBatch[0][0],
  #     "metadata": nextBatch[1][0],
  #     "label"   : nextBatch[2][0],
  #   }
  #   randomExamples.append(rndExample)
  # fprint(f"Loaded {len(randomExamples)} random examples from the training set for sanity checks and visualization.")

  # results = []
  # for record in randomExamples:
  #   img = record["image"].unsqueeze(0).to(device)
  #   metadata = record["metadata"].unsqueeze(0).to(device)
  #   with torch.no_grad():
  #     logits, attentions = model(img, metadata=metadata, outputAttentions=True)
  #   attnVecs = ExtractClsToPatchesAttention(attentions, layerIdx=-1, aggregateHeads="mean")
  #   attnVec = attnVecs[0]
  #   attnVec = ValidateAndNormalizeAttention(attnVec)
  #   fprint(
  #     f"Sanity check - attention vector sum: {attnVec.sum():.4f}, min: {attnVec.min():.6f}, max: {attnVec.max():.6f}"
  #   )
  #   # Save the raw attention vector as a numpy file for reproducibility.
  #   np.save(outRoot / "RawAttention" / f"{record['index']}_Attention.npy", attnVec)
  #   # For a sanity check, we can visualize the attention vector as a heatmap overlay on the input image.
  #   patchCoords = PatchGridCoords(args.imageSize, hparam["PatchSize"])
  #   heatmapPath = outRoot / "Heatmaps" / f"{record['index']}_SanityCheck_Heatmap.png"
  #   SaveHeatmapOverlay(
  #     origImg=transforms.ToPILImage()(record["image"]),
  #     attnVec=attnVec,
  #     patchCoords=patchCoords,
  #     outPath=str(heatmapPath),
  #     cmap="viridis",
  #     alpha=0.5,
  #   )
  #   # Build a per-image result row dictionary with CamelCase keys.
  #   resultRow = {
  #     "Index"      : record["index"],
  #     "Label"      : int(record["label"].cpu().numpy()),
  #     "Entropy"    : ComputeEntropy(attnVec, base="natural"),
  #     "HeatmapPath": str(heatmapPath),
  #   }
  #   results.append(resultRow)

  # Initialize a list for per-image result rows.
  results = []

  # Iterate over the selected examples.
  for record in randomExamples:
    # Move the image tensor to the selected device and add a batch dimension.
    img = record["Image"].unsqueeze(0).to(device)

    # Move the metadata tensor to the selected device and add a batch dimension.
    metadata = record["Metadata"].unsqueeze(0).to(device)

    # Read the mask directory from arguments when provided.
    maskDir = getattr(args, "maskDir", None)

    # Resolve the ground-truth mask path for the current image.
    maskPath = ResolveMaskPath(record["ImagePath"], maskDir)

    # Skip the sample when masks are required but unavailable.
    if ((args.requireMask) and ((maskPath is None) or (not maskPath.is_file()))):
      # Continue to the next sample.
      continue

    # Disable gradient computation for inference.
    with torch.no_grad():
      # Perform a forward pass through the model.
      logits, attentions = model(img, metadata=metadata, outputAttentions=True)

    # Extract CLS-to-patch attention vectors from the final layer.
    attnVecs = ExtractClsToPatchesAttention(attentions, layerIdx=-1, aggregateHeads="mean")

    # Select the first sample attention vector.
    attnVec = attnVecs[0]

    # Normalize the attention vector.
    attnVec = ValidateAndNormalizeAttention(attnVec)

    # Print a sanity check for the attention vector.
    fprint(
      f"Sanity check - attention vector sum: {attnVec.sum():.4f}, "
      f"min: {attnVec.min():.6f}, max: {attnVec.max():.6f}"
    )

    # Save the raw attention vector as a NumPy file for reproducibility.
    np.save(outRoot / "RawAttention" / f"{record['Index']}_Attention.npy", attnVec)

    # Build patch coordinates for heatmap overlay.
    patchCoords = PatchGridCoords(args.imageSize, hparam["PatchSize"])

    # Define the heatmap output path.
    heatmapPath = outRoot / "Heatmaps" / f"{record['Index']}_SanityCheck_Heatmap.png"

    # Save the attention heatmap overlay.
    SaveHeatmapOverlay(
      origImg=transforms.ToPILImage()(record["Image"]),
      attnVec=attnVec,
      patchCoords=patchCoords,
      outPath=str(heatmapPath),
      cmap="viridis",
      alpha=0.5,
    )

    # Build a per-image result row dictionary with CamelCase keys.
    resultRow = {
      "Index"      : record["Index"],
      # .cpu().numpy()
      "Label"      : int(record["Label"]),
      "Entropy"    : ComputeEntropy(attnVec, base="natural"),
      "HeatmapPath": str(heatmapPath),
    }

    # Compute quantitative IoU metrics when a ground-truth mask is available.
    if ((maskPath is not None) and (maskPath.is_file())):
      # Open the mask image and convert it to grayscale.
      maskImage = Image.open(maskPath).convert("L")

      # Iterate over the requested Top-K values.
      for topK in args.topK:
        # Compute the IoU between the Top-K attention patches and the ground-truth mask.
        iouScore = ComputeAttentionIoU(
          AttentionVector=attnVec,
          MaskImage=maskImage,
          ImageSize=args.imageSize,
          PatchSize=hparam["PatchSize"],
          TopK=topK,
        )

        # Store the IoU score using a CamelCase column name.
        resultRow[f"IoUAt{topK}"] = iouScore

    # Append the result row to the results list.
    results.append(resultRow)

  # Convert results to a DataFrame and save as CSV.
  resultsDf = pd.DataFrame(results)
  resultsDf.to_csv(outRoot / "RandomExamples_AttentionMetrics.csv", index=False)

  # Identify IoU columns in the results DataFrame.
  iouColumns = [col for col in resultsDf.columns if col.startswith("IoUAt")]

  # Save the mean IoU summary when masks were evaluated.
  if (len(iouColumns) > 0):
    # Compute the mean IoU for each Top-K value.
    iouSummary = resultsDf[iouColumns].mean(axis=0)

    # Save the IoU summary to a CSV file.
    iouSummary.to_csv(outRoot / "IoUSummary.csv", header=["MeanIoU"])

    # Print the IoU summary.
    fprint("IoU Summary:")

    # Print each mean IoU value.
    fprint(iouSummary)
  else:
    # Inform the user that no IoU metrics were computed.
    fprint("No IoU metrics were computed because no ground-truth masks were found.")

  # Create a new figure with specified dimensions for the 2x8 grid layout.
  plt.figure(figsize=(16, 10))
  # Retrieve and sort heatmap file paths, selecting the first 16 samples.
  sampleHeatmaps = sorted((outRoot / "Heatmaps").glob("*_Heatmap.png"))[:32]
  # Iterate over the selected heatmap paths with their indices.
  for i, heatmapPath in enumerate(sampleHeatmaps):
    # Open the heatmap image file using PIL.
    img = Image.open(heatmapPath)
    # Create a subplot at the current grid position.
    plt.subplot(4, 8, i + 1)
    # Display the heatmap image in the current axes.
    plt.imshow(img)
    # Hide the axis ticks and labels for cleaner visualization.
    plt.axis("off")
    # Set a descriptive title for the current sample.
    plt.title(f"Sample {i + 1}")
  # Adjust subplot parameters to prevent overlap between elements.
  # plt.tight_layout()
  # Add a vertical colorbar legend positioned above the image grid.
  cbarAx = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height].
  # Create a normalization object mapping attention weights to the 0:1 range.
  norm = plt.Normalize(vmin=0.0, vmax=1.0)
  # Create a ScalarMappable object for the viridis colormap with the defined normalization.
  sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
  # Set an empty array to satisfy the ScalarMappable interface requirements.
  sm.set_array([])
  # Add the colorbar to the specified axes with horizontal orientation and label.
  cbar = plt.colorbar(sm, cax=cbarAx, label="Attention Weight", orientation="vertical")
  cbar.ax.set_facecolor("black")  # Set the colorbar background to light gray for contrast.
  cbar.ax.yaxis.label.set_color("yellow")  # Set the colorbar label text color to red for visibility.
  # Save the complete figure to the specified output path with the requested DPI.
  plt.savefig(outRoot / "SampleHeatmaps.png", dpi=args.dpi)
  # Close the figure to release memory resources.
  plt.close()

  # Generate an entropy histogram plot and save it.
  entropies = resultsDf["Entropy"].values
  plt.figure(figsize=(6, 4))
  plt.hist(entropies, bins=30)
  plt.title(r"Attention entropy distribution (CLS$\rightarrow$Patch)")
  plt.xlabel("Entropy (nats)")
  plt.ylabel("Count")
  plt.tight_layout()
  plt.savefig(outRoot / "EntropyHist.png", dpi=args.dpi)
  plt.close()

  # Print final status message with results directory.
  fprint("Done. Results saved to", outRoot)


# Execute main when script is run as program.
if (__name__ == "__main__"):
  main()
