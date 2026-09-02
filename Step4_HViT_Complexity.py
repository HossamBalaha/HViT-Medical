import os
import copy, torch
import numpy as np
import torch.nn as nn
from utils.Utils import *
from utils.ProposalHelpers import ViTForClassification
from HMB.PyTorchHelper import (
  MeasureLatency, ComputeProfileFLOPs, GetPyTorchDeviceName, PyTorchCrossAttentionHead, GetParamCount
)
from HMB.Initializations import UpdateMatplotlibSettings
from HMB.Utils import FormatNumericWithDelta, fprint
from HMB.PlotsHelper import PlotApproachesComplexityComparison


# Define the function to modify hyperparameters using Step1 keys.
def ModifyHparamsForApproach(hparam, approach):
  # Make a deep copy of the hyperparameters to avoid mutation.
  hp = copy.deepcopy(hparam)

  # Increase patch size for hierarchical or pyramid models.
  if (approach in ("SwinTransformer", "T2TViT", "PVT", "PVTv2", "NesT", "MobileViT")):
    # Read the old patch size using the Step1 key.
    old = hp.get("PatchSize", 16)

    # Update the patch size using the Step1 key.
    hp["PatchSize"] = min(hp.get("ImageSize", 128), max(old * 2, 1))

  # Reduce hidden size and attention heads for lightweight models.
  if (approach in ("LeViT", "PiT", "EfficientFormer")):
    # Reduce hidden size using the Step1 key.
    hp["HiddenSize"] = max(32, hp.get("HiddenSize", 768) // 2)

    # Adjust attention heads using the Step1 key.
    hp["NumAttentionHeads"] = max(1, hp.get("NumAttentionHeads", 12) // 2)

  # Halve the number of heads for PVT-like models.
  if (approach in ("PVT", "PVTv2")):
    # Update the number of attention heads using the Step1 key.
    hp["NumAttentionHeads"] = max(1, hp.get("NumAttentionHeads", 12) // 2)

  # Return the modified hyperparameters.
  return hp


# Apply architecture-specific modifications to the model.
def ApplyModelModifications(model, approach, metadataDim):
  # Apply full cross-attention head replacement for the Cross-Attention approach.
  if (approach == "Cross-Attention"):
    # Define the nested function to replace heads.
    def ReplaceHeadsWithCrossAttention(model, metadataDim):
      # Iterate over transformer blocks in the encoder.
      for blk in model.encoder.blocks:
        # Get the multi-head attention module for the block.
        mha = blk.attention
        # Prepare a new list of heads.
        newHeads = []
        # Replace each head with a PyTorchCrossAttentionHead that matches sizes.
        for head in mha.heads:
          # Create PyTorchCrossAttentionHead with matching shapes.
          newHead = PyTorchCrossAttentionHead(
            hiddenSize=model.hiddenSize,
            attentionHeadSize=head.attentionHeadSize if hasattr(head, "attentionHeadSize") else (
              model.hiddenSize // model.config["NumAttentionHeads"]),
            dropout=model.config.get("attentionProbsDropoutProb", 0.1),
            bias=model.config.get("qkvBias", True),
            metadataDim=metadataDim,
          )
          # Append the new head to the list.
          newHeads.append(newHead)
        # Replace the heads ModuleList with the new heads.
        mha.heads = nn.ModuleList(newHeads)
      # Return the modified model.
      return model

    # Return the result of the nested replacement function.
    return ReplaceHeadsWithCrossAttention(model, metadataDim)

  # Apply CrossViT emulation by replacing alternate heads with PyTorchCrossAttentionHead.
  if (approach == "CrossViT"):
    # Iterate over transformer blocks in the encoder.
    for blk in model.encoder.blocks:
      # Get the multi-head attention module for the block.
      mha = blk.attention
      # Prepare a new list of heads.
      newHeads = []
      # Iterate over the existing heads with their indices.
      for i, head in enumerate(mha.heads):
        # Check if the current head index is even.
        if ((i % 2) == 0):
          # Create a new cross-attention head for even indices.
          newHead = PyTorchCrossAttentionHead(
            hiddenSize=model.hiddenSize,
            attentionHeadSize=getattr(head, "attentionHeadSize", model.hiddenSize // model.config["NumAttentionHeads"]),
            dropout=model.config.get("attentionProbsDropoutProb", 0.1),
            bias=model.config.get("qkvBias", True),
            metadataDim=metadataDim,
          )
          # Append the new cross-attention head.
          newHeads.append(newHead)
        # Otherwise keep the original head.
        else:
          # Append the original head for odd indices.
          newHeads.append(head)
      # Replace the heads ModuleList with the new mixed heads.
      mha.heads = nn.ModuleList(newHeads)
    # Return the modified model.
    return model

  # Emulate DeiT by expanding the classifier outputs to include a distillation head.
  if (approach == "DeiT"):
    # Replace the classifier with a linear layer outputting double the classes.
    model.classifier = nn.Linear(model.hiddenSize, model.numClasses * 2)
    # Return the modified model.
    return model

  # Emulate CvT by adding an extra 1x1 conv after the patch projection if possible.
  if (approach == "CvT"):
    # Attempt to modify the patch embedding projection.
    try:
      # Extract the patch embedding projection layer.
      proj = model.embedding.patchEmbeddings.projection
      # Check if the projection is a Conv2d layer.
      if (isinstance(proj, nn.Conv2d)):
        # Replace the projection with a sequential module containing the original and a 1x1 conv.
        model.embedding.patchEmbeddings.projection = nn.Sequential(
          proj,
          nn.Conv2d(proj.out_channels, proj.out_channels, kernel_size=1, stride=1, bias=True),
        )
    # Catch any exceptions during modification.
    except Exception:
      # Skip CvT emulation when model structure is unexpected.
      pass
    # Return the modified model.
    return model

  # Add a small pre-classifier MLP for several transformer variants.
  if (approach in ("XCiT", "ConViT", "TNT", "PiT", "LeViT", "NesT", "EfficientFormer", "MobileViT")):
    # Replace the classifier with a sequential MLP.
    model.classifier = nn.Sequential(
      nn.Linear(model.hiddenSize, model.hiddenSize),
      nn.GELU(),
      nn.Linear(model.hiddenSize, model.numClasses),
    )
    # Return the modified model.
    return model

  # Default behaviour: return model unchanged.
  return model


# Define the main function to execute the complexity measurement pipeline.
def main():
  # Validate the command-line arguments.
  args = Step4ParseArgs()
  # Perform additional argument validation.
  Step4ValidateArgs(args)
  # Update matplotlib settings for consistent plotting.
  UpdateMatplotlibSettings()

  # Choose device based on request and availability.
  device = torch.device("cuda" if (args.mustCuda and torch.cuda.is_available()) else "cpu")

  # Load hyperparameters for the run.
  hparam = LoadHParams(args.hparamsFile, args.hparamName)
  # Copy hyperparameters and add runtime overrides used by ViT constructor.
  hparam = {**hparam}
  # Set image size in hyperparameters.
  hparam["ImageSize"] = args.imageSize
  # Set number of channels in hyperparameters.
  hparam["NumChannels"] = args.numChannels
  # Set metadata dimensionality in hyperparameters.
  hparam["MetadataDim"] = args.metadataDim
  # Set number of classes for classification.
  hparam["NumClasses"] = 2

  # Prepare results list.
  results = []
  # Define architecture configurations to measure, using CamelCase keys.
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
    "Cross-Attention"  : [["Standard"] * 6] * hparam["NumHiddenLayers"],
    "DeiT"             : [["Standard"] * 6] * hparam["NumHiddenLayers"],
    "CrossViT"         : [["Standard"] * 6] * hparam["NumHiddenLayers"],
    "CvT"              : [["Standard"] * 6] * hparam["NumHiddenLayers"],
    "XCiT"             : [["Standard"] * 6] * hparam["NumHiddenLayers"],
    "ConViT"           : [["Standard"] * 6] * hparam["NumHiddenLayers"],
    "TNT"              : [["Standard"] * 6] * hparam["NumHiddenLayers"],
  }

  # Create input tensors.
  batch = 1
  # Generate random image input tensor.
  x = torch.randn(batch, args.numChannels, args.imageSize, args.imageSize)
  # Generate random metadata input tensor.
  metadata = torch.randn(batch, args.metadataDim)

  # Move inputs to CUDA when requested.
  if (args.mustCuda):
    # Move image tensor to device.
    x = x.to(device)
    # Move metadata tensor to device.
    metadata = metadata.to(device)

  # Iterate over each approach to build, modify and measure the model.
  for approach, headTypesPerBlock in archsDict.items():
    # Create a modified copy of hyperparameters for the approach so token counts/widths can be emulated.
    approachHparam = ModifyHparamsForApproach(hparam, approach)
    # Build a ViT model instance for classification using the approach hyperparameters.
    model = ViTForClassification(approachHparam, headTypesPerBlock)
    # Apply lightweight architecture-specific modifications such as head replacements or extra convs.
    model = ApplyModelModifications(model, approach, args.metadataDim)

    # Move model to CUDA when requested.
    if (args.mustCuda):
      # Move the constructed model to the selected device.
      model = model.to(device)

    # Measure latency by averaging multiple forward passes.
    (meanLatency, stdLatency, additionalStats) = MeasureLatency(
      model,
      device,
      [x, metadata],
      runs=args.runs,
      warmup=args.warmup,
      useCudaEvents=args.mustCuda,
    )

    # Append results for this approach.
    results.append({
      "Approach"       : approach,
      "FLOPs_G"        : ComputeProfileFLOPs(model, [x, metadata], doCPU=not args.mustCuda, formatGigas=True),
      "Params_M"       : GetParamCount(model, restrictGrad=True, formatMillions=True),
      "Latency_ms_mean": meanLatency,
      "Latency_ms_std" : stdLatency,
    })

    # Free memory by deleting model and clearing CUDA cache when applicable.
    del model
    # Clear the CUDA cache to prevent out-of-memory errors.
    torch.cuda.empty_cache()

  # Compute percentage deltas relative to Standard baseline for FLOPs and Params.
  base = next((r for r in results if (r["Approach"] == "Standard")), None)
  # Extract baseline FLOPs if the baseline exists.
  baseFlops = base["FLOPs_G"] if (base is not None) else None
  # Extract baseline Params if the baseline exists.
  baseParams = base["Params_M"] if (base is not None) else None

  # Define the output directory for the complexity results.
  outdir = f"{args.experimentsFolder}/{args.projectKeyword}/Step4-Complexity"

  # Create the output directory if it does not exist.
  os.makedirs(outdir, exist_ok=True)

  # Initialize a list to store the text table lines.
  textLines = []

  # Append the text table header to the list.
  textLines.append("\nComputational Complexity Results:")

  # Define the column headers for the text table.
  colApproach = "Approach"
  colFlops = "FLOPs (G)"
  colParams = "Params (M)"
  colLatency = "Latency (ms)"

  # Append the column names to the list.
  textLines.append(f"{colApproach:<20} {colFlops:>20} {colParams:>18} {colLatency:>18}")

  # Iterate over the results to format the text table rows.
  for r in results:
    # Format the FLOPs string with delta.
    fstr = FormatNumericWithDelta(r["FLOPs_G"], baseFlops, fmt="{:.2f}")

    # Format the Params string with delta.
    pstr = FormatNumericWithDelta(r["Params_M"], baseParams, fmt="{:.1f}")

    # Extract the approach name to avoid single quotes in the f-string.
    approachName = r["Approach"]

    # Extract the mean latency to avoid single quotes in the f-string.
    meanLatency = r["Latency_ms_mean"]

    # Extract the standard deviation to avoid single quotes in the f-string.
    stdLatency = r["Latency_ms_std"]

    # Append the formatted row to the list.
    textLines.append(f"{approachName:<20} {fstr:>20} {pstr:>18} {meanLatency:8.1f} \u00B1 {stdLatency:.1f}")

  # Join the text lines into a single string.
  textOutput = "\n".join(textLines)

  # Print the text table to the console.
  fprint(textOutput)

  # Define the text output file path.
  textFilePath = os.path.join(outdir, "ComplexityResults.txt")

  # Save the text table to a file.
  with open(textFilePath, "w") as f:
    # Write the text output to the file.
    f.write(textOutput + "\n")

  # Initialize a list to store the LaTeX table lines.
  latexLines = []

  # Append the LaTeX table header to the list.
  latexLines.append("\nLaTeX Table (copy-paste):")

  # Append the table environment start.
  latexLines.append("\\begin{table}[htbp]")

  # Append the centering command.
  latexLines.append("\\centering")

  # Append the tabular environment start.
  latexLines.append("\\begin{tabular}{lccc}")

  # Append the top rule.
  latexLines.append("\\toprule")

  # Append the column headers.
  latexLines.append("\\textbf{Method} & \\textbf{FLOPs (G)} & \\textbf{Params (M)} & \\textbf{Latency (ms)} \\\\ ")

  # Append the mid rule.
  latexLines.append("\\midrule")

  # Iterate over the results to format the LaTeX table rows.
  for r in results:
    # Format the FLOPs string with delta.
    flopsStr = FormatNumericWithDelta(r["FLOPs_G"], baseFlops, fmt="{:.2f}")

    # Format the Params string with delta.
    paramsStr = FormatNumericWithDelta(r["Params_M"], baseParams, fmt="{:.1f}")

    # Extract the approach name to avoid single quotes in the format string.
    approachName = r["Approach"]

    # Extract the mean latency to avoid single quotes in the f-string.
    meanLatency = r["Latency_ms_mean"]

    # Extract the standard deviation to avoid single quotes in the f-string.
    stdLatency = r["Latency_ms_std"]

    # Format the latency string with standard deviation.
    latencyStr = f"{meanLatency:.1f} $\\pm$ {stdLatency:.1f}"

    # Append the formatted row to the list.
    latexLines.append("{} & {} & {} & {} \\\\".format(approachName, flopsStr, paramsStr, latencyStr))

  # Append the bottom rule.
  latexLines.append("\\bottomrule")

  # Append the tabular environment end.
  latexLines.append("\\end{tabular}")

  # Append the table caption.
  latexLines.append(
    "\\caption{Computational complexity comparison (ViT-Base, input size %dx%d). "
    "Measurements averaged over %d forward passes on the chosen device.}" % (
      args.imageSize, args.imageSize, args.runs)
  )

  # Append the table label.
  latexLines.append("\\label{table:complexity}")

  # Append the table environment end.
  latexLines.append("\\end{table}")

  # Join the LaTeX lines into a single string.
  latexOutput = "\n".join(latexLines)

  # Print the LaTeX table to the console.
  fprint(latexOutput)

  # Define the LaTeX output file path.
  latexFilePath = os.path.join(outdir, "ComplexityTable.tex")

  # Save the LaTeX table to a file.
  with open(latexFilePath, "w") as f:
    # Write the LaTeX output to the file.
    f.write(latexOutput + "\n")

  # Extract the approach names for the plotting function.
  approaches = [r["Approach"] for r in results]

  # Extract the FLOPs values for the plotting function.
  flopsValues = [r["FLOPs_G"] if (r["FLOPs_G"] is not None) else np.nan for r in results]

  # Extract the Params values for the plotting function.
  paramsValues = [r["Params_M"] if (r["Params_M"] is not None) else np.nan for r in results]

  # Check if the device type is CUDA to include specific device name in title.
  if (device.type == "cuda"):
    # Get the specific device name for the title.
    devName = GetPyTorchDeviceName(device)

    # Format the title string with the specific device name.
    titleDevice = f"{device.type.upper()} - {devName}"
  # Otherwise use a generic title string for non-CUDA devices.
  else:
    # Format the title string with generic device details.
    titleDevice = f"{device.type.upper()}"

  # Call the plotting function with the collected results and baseline values for delta computation.
  PlotApproachesComplexityComparison(
    approaches,
    flopsValues,
    paramsValues,
    outdir,
    baseFlops=baseFlops,
    baseParams=baseParams,
    deviceName=titleDevice,
  )


# Run main when executed as a script.
if (__name__ == "__main__"):
  # Call the main execution function.
  main()

  # Example usage:
  # python Step4_HViT_Complexity.py --hparamsFile assets/hparams.json --hparamName "Preset_07_Base-v2" --projectKeyword "PH2-v2" --imageSize 128 --metadataDim 6 --runs 100 --warmup 10 --mustCuda
