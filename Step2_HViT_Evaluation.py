import os, torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
from utils.Utils import *
from HMB.Utils import LoadYaml, fprint
import HMB.StatisticalAnalysisHelper as sah
from HMB.Initializations import SeedEverything
from HMB.PlotsHelper import GenerateGenericBubblePlot
from HMB.PerformanceMetrics import CalculatePerformanceMetrics

COLOR_MAPPING = {
  "Standard"           : "blue",
  "QConditioning"      : "green",
  "QC"                 : "green",
  "QGating"            : "orange",
  "QG"                 : "orange",
  "KeyConditioning"    : "cyan",
  "ValueConditioning"  : "olive",
  "LateFusion"         : "magenta",
  "EarlyFusion"        : "teal",
  "PhikonV2LinearProbe": "gold",
  "QCQGS"              : "red",
  "QCQG"               : "purple",
  "QCS"                : "brown",
  "QGS"                : "pink",
}


# Define the main function to demonstrate the usage of the ViT model with metadata integration.
def main():
  # Validate the command-line arguments.
  args = Step2ParseArgs()  # Parse command-line arguments.
  Step2ValidateArgs(args)

  # Seed for reproducibility.
  randomNumber = np.random.randint(0, 10000)
  SeedEverything(seed=randomNumber)

  experimentsFolder = str(args.experimentsFolder)  # Base folder containing all experiments.
  projectKeyword = str(args.projectKeyword)  # Define the project keyword.

  dpi = args.dpi  # DPI for saved figures.
  projectDir = os.path.join(experimentsFolder, projectKeyword)
  step1Dir = os.path.join(projectDir, "Step1-Training")
  step2Dir = os.path.join(projectDir, "Step2-Evaluation")
  os.makedirs(step2Dir, exist_ok=True)

  history = []
  experiments = [el for el in os.listdir(step1Dir) if os.path.isdir(os.path.join(step1Dir, el))]
  for experiment in experiments:
    trials = os.listdir(os.path.join(step1Dir, experiment))
    for trial in trials:
      # Try to find Config.yaml.
      runDir = os.path.join(step1Dir, experiment, trial)
      potentialPath = os.path.join(runDir, "Config.yaml")
      if (not os.path.isfile(potentialPath)):
        potentialPath = os.path.join(runDir, "Testing", "Config.yaml")
        if (not os.path.isfile(potentialPath)):
          fprint(f"  -> WARNING: Config.yaml not found in {runDir}; skipping evaluation.")
          continue
      fprint(f"  -> Found Config.yaml at {potentialPath}; proceeding with evaluation.")
      configs = LoadYaml(potentialPath)

      fprint(f"Evaluating {experiment}, {trial}...")
      runDir = os.path.join(step1Dir, experiment, trial)
      predictionsCSVPath = os.path.join(runDir, "Artifacts", "ViT_Predictions_None.csv")
      df = pd.read_csv(predictionsCSVPath)
      trueLabels = df["trueClassName"]
      predLabels = df["predictedClassName"]
      cm = confusion_matrix(trueLabels, predLabels, normalize="true")
      metrics = CalculatePerformanceMetrics(
        cm,
        eps=1e-10,  # Small value to avoid division by zero.
        addWeightedAverage=True,  # Whether to include weighted averages in the output.
        addPerClass=False,  # Whether to include per-class metrics in the output.
      )
      metrics["Experiment"] = experiment
      metrics["Trial"] = trial

      # Read the configured preset name from the run configuration.
      name = configs["Name"] if ("Name" in configs) else "Unknown"
      # Extract the Step1 architecture key from the experiment folder name.
      approach = experiment.replace(f"_{name}", "") if (name in experiment) else "Unknown"
      # Store the extracted approach in the metrics row.
      metrics["Approach"] = approach
      # Store the configuration name in the metrics row.
      metrics["ConfigName"] = name

      # name = configs["Name"] if ("Name" in configs) else "Unknown"
      # nameIdx = experiment.find(name)
      # approach = experiment[:nameIdx - 1] if (nameIdx > 0) else "Unknown"

      metrics["Color"] = COLOR_MAPPING.get(metrics["Approach"], "gray")
      metrics["SizeColumn"] = metrics["Weighted Recall"] * float(dpi)
      metrics["RunPath"] = runDir
      metrics["BestModelPath"] = os.path.join(runDir, "BestModel.pth")
      metrics["LastModelPath"] = os.path.join(runDir, "LastModel.pth")
      metrics["ConfigPath"] = os.path.join(runDir, "Config.yaml")
      history.append(metrics)
      fprint(f"  -> Completed evaluation for {experiment}, {trial}.")

  historyDF = pd.DataFrame(history)
  historyDF = historyDF[
    ["Experiment", "Trial"] +
    [col for col in historyDF.columns if (col not in ["Experiment", "Trial"])]
    ]
  outputCSVPath = os.path.join(step2Dir, "Evaluation_Summary.csv")
  historyDF.to_csv(outputCSVPath, index=False)
  fprint(f"Saved evaluation summary to: {outputCSVPath}")

  # Generate another summary dataframe grouped by Approach with mean metrics by keeping other columns.
  objCols = [col for col in historyDF.columns if (historyDF[col].dtype == "object")]
  numericCols = [col for col in historyDF.columns if col not in objCols]
  fprint(f"Generating summary by `Experiment` with mean of columns: {numericCols} (object columns: {objCols})")
  summaryDF = historyDF.copy()
  summaryDF = summaryDF.drop(columns=["TP", "TN", "FP", "FN", "Weights"], axis=1)
  summaryDF = summaryDF.groupby("Experiment").agg({
    "Approach"  : "first",
    "Color"     : "first",
    "ConfigName": "first",
    **{col: "mean" for col in numericCols},
  }).reset_index()
  outputSummaryCSVPath = os.path.join(step2Dir, "Evaluation_Summary_By_Experiment.csv")
  summaryDF.to_csv(outputSummaryCSVPath, index=False)
  fprint(f"Saved evaluation summary by `Experiment` to: {outputSummaryCSVPath}")

  # Call generic plotter.
  outputPath = os.path.join(step2Dir, "Evaluation_BubblePlot.png")
  success = GenerateGenericBubblePlot(
    data=summaryDF,
    xColumn="Weighted Average",
    yColumn="Approach",
    sizeColumn="SizeColumn",
    colorColumn="ConfigName",
    outputPath=outputPath,
    figureSize=(12, 8),
    dpiValue=dpi,
    title="Model Performance Analysis\n(Bubble Size = Weighted Recall | Color = Config | Zones = Performance Tiers)",
    xlabel="Weighted Average Metric",
    ylabel="Approach",
    performanceZones=True,
    concentrationPeak=True,
  )
  fprint(f"Saved evaluation bubble plot to: {outputPath}")

  def GenerateWeightedMetricsCSV(historyDF, outputDir: Path):
    '''Generate a CSV file containing weighted metrics from all trials.'''
    allTrialResults = historyDF.to_dict(orient="records")
    expCol = "Experiment"
    uniqueExp = historyDF[expCol].unique()
    uniqueExpFiltered = [
      exp[:exp.find("_Preset_")] if ("_Preset_" in exp) else exp
      for exp in uniqueExp
    ]
    fprint(f"\nUnique experiments found: {uniqueExpFiltered}")
    metricColumns = [
      "Weighted Accuracy",
      "Weighted Recall",
      "Weighted Precision",
      "Weighted F1",
      "Weighted Specificity",
      "Weighted Average",
    ]
    columnNames = [col.replace("Weighted ", "") for col in metricColumns]
    csvData = []
    # Initialize a dictionary to store the number of trials for each experiment.
    trialCounts = {}
    fprint(f"\nGenerating weighted metrics CSV with columns: {columnNames} for experiments: {uniqueExpFiltered}")
    for exp in uniqueExp:
      # Filter the history DataFrame for the current experiment.
      expData = historyDF[historyDF[expCol] == exp]
      if (expData.empty):
        # Print a warning if no data is found for the experiment.
        fprint(f"  -> WARNING: No data found for experiment {exp}; skipping.")
        continue
      # Reset the index of the filtered DataFrame.
      expData = expData.reset_index(drop=True)
      # Store the number of trials for the current experiment.
      trialCounts[exp] = len(expData)
      for metric in metricColumns:
        if (metric not in expData.columns):
          # Print a warning if the metric is not found in the data.
          fprint(f"  -> WARNING: Metric {metric} not found in data for experiment {exp}; skipping this metric.")
          continue
        # Extract the metric values as a numpy array.
        values = expData[metric].values
        # Append the metric values for this experiment to the CSV data.
        csvData.append(values)
    # Find the minimum number of trials across all experiments.
    minTrials = min(trialCounts.values())
    # Identify the experiments that have the lowest number of trials.
    badRecords = [exp for exp, count in trialCounts.items() if (count == minTrials)]
    # Print the bad records to the console.
    fprint(f"  -> WARNING: Experiments with the lowest number of trials ({minTrials}): {badRecords}")
    # Determine the maximum number of trials to pad the data uniformly.
    maxTrials = max([len(values) for values in csvData] + [0])
    # Pad each metric array with NaNs to ensure they all have the same length.
    paddedCsvData = [np.pad(values, (0, maxTrials - len(values)), constant_values=np.nan) for values in csvData]
    # Transpose the padded CSV data to have metrics as columns and experiments as rows.
    csvData = np.array(paddedCsvData).T
    metricsDF = pd.DataFrame(csvData, columns=columnNames * len(uniqueExp))
    # Add a second header row with experiment names.
    secondHeader = []
    for exp in uniqueExp:
      expName = exp[:exp.find("_Preset_")] if ("_Preset_" in exp) else exp
      secondHeader.extend([expName] + [""] * (len(metricColumns) - 1))
    metricsDF.columns = pd.MultiIndex.from_tuples(zip(secondHeader, metricsDF.columns))
    csvPath = outputDir / "WeightedMetrics.csv"
    metricsDF.to_csv(csvPath, index=False)
    fprint("\nWeighted Metrics Report:")
    fprint(f"{metricsDF.to_string(index=False)}")
    return csvPath

  # Use the DataFrame we already created above.
  metricsCSVPath = GenerateWeightedMetricsCSV(
    historyDF,
    Path(step2Dir)
  )
  fprint(f"\u2713 Weighted metrics CSV saved: {metricsCSVPath}")

  # Run Statistical Analysis Helper (sah) on the generated CSV.
  dataSah, namesSah, metricsSah = sah.ExtractDataFromSummaryFile(metricsCSVPath)
  fprint("\nWeighted Metrics Summary Table:")
  fprint("Names:", namesSah)
  fprint("Metrics:", metricsSah)
  for record in dataSah:
    fprint(record)

  fprint("\nGenerating performance plots...")

  whichToPlot = [
    # --- Distribution & Single Metric Analysis ---
    # "Histograms",  # Frequency distribution of a single metric.
    # "DensityPlots",  # Smoothed probability density (often with RugPlots).
    # "BoxPlots",  # Summarize distribution (median, quartiles, outliers).
    # "ViolinPlots",  # Combine density shape with box plot summary.
    # "QQPlots",  # Compare distribution to a theoretical one (e.g., Normal).
    # "CDFPlots",  # Cumulative distribution function.
    # "ECDFPlots",  # Empirical cumulative distribution function.
    # "SwarmPlots",  # Show individual data points, especially for small datasets.
    # "StripPlots",  # Like swarm but allows overlap.
    # "DotPlots",  # Dot plot for small counts.
    # "StackedBarPlots",  # Stacked bar plot for group comparison.
    # "StackedAreaPlots",  # Stacked area plot for cumulative trends.
    # "Histogram2DPlots",  # 2D histogram for joint distribution of two metrics.
    # "StepPlots",  # Step plot for discrete changes.

    # --- Comparative Analysis (Multiple Datasets/Metrics) ---
    # "BarPlots",  # Compare aggregated values (e.g., means) across datasets.
    # "LinePlots",  # Show trends over trials/iterations for each dataset.

    # --- Relationships & Correlations (Between Metrics) ---
    # "ScatterPlots",  # Show relationship between two metrics.
    # "HexbinPlots",  # 2D density plot for large datasets in scatter plots.
    # "PairPlots",  # Matrix of scatter plots for multiple metrics (often includes CorrelationHeatmaps).
    # "CorrelationHeatmaps",  # Standalone heatmap of correlation matrix (can be part of PairPlots).
    # "BlandAltmanPlots",  # Compare agreement between two measurement methods/metrics.

    # --- Advanced Diagnostics (often related to others) ---
    # "ResidualPlots",  # Diagnostics for regression (ScatterPlot related)
    # "QQResidualPlots",  # Diagnostics for regression normality (ScatterPlot related)

    # --- Other/Advanced ---
    # "ContourPlots",  # Show 3D relationships in 2D (requires specific data structure).
    # "PieCharts",  # Show proportions (use sparingly, often better replaced by BarPlots).
    "RaincloudPlots",  # Raincloud plot: distribution + box/violin + raw data.
    # "AndrewsCurves",  # Andrews curves for high-dimensional data.
    # "ParallelCoordinates",  # Parallel coordinates for multi-metric comparison.
    # "RadarPlots",  # Radar (spider) plots for profile comparison.
    # "BoxenPlots",  # Boxen (letter value) plots for large data.
    # "LollipopPlots",  # Lollipop plots for mean/median comparison.
    # "SlopeCharts",  # Slope charts for before/after or paired data.
    # "DumbbellPlots",  # Dumbbell plots for paired difference visualization.
    # "TreemapPlots",  # Treemap for hierarchical metric visualization.
    # "SunburstPlots",  # Sunburst for hierarchical metric visualization.
  ]

  sah.PlotMetrics(
    dataSah, namesSah, metricsSah,
    factor=6,
    keyword="ModelPerformance",
    dpi=dpi,
    xTicksRotation=45,
    fontSize=14,
    showFigures=False,
    storeInsideNewFolder=True,
    newFolderName=os.path.join(step2Dir, "PerformancePlots"),
    noOfPlotsPerRow=3,
    cmap="tab20",
    differentColors=True,
    fixedTicksColors=True,
    fixedTicksColor="black",
    extension=".pdf",
    whichToPlot=whichToPlot,
    xLabelAlignment="right",
  )
  fprint("\u2713 Performance plots generated.")
  fprint("\nGenerating statistical analysis report...")
  overallReport = []
  for metric in metricsSah:
    for index, data in enumerate(dataSah):
      report = sah.StatisticalAnalysis(
        data[metric]["Trials"],
        hypothesizedMean=data[metric]["Mean"],
        secondMetricList=None,
      )
      report["Type"] = namesSah[index]
      report["Metric"] = metric
      overallReport.append(report)
  reportDF = pd.DataFrame(overallReport)
  reportCsvPath = Path(step2Dir) / "StatisticalAnalysisReport.csv"
  reportDF.to_csv(reportCsvPath, index=False)
  fprint(f"\u2713 Statistical analysis report saved: {reportCsvPath}")
  fprint(f"\n{'=' * 80}")

  # Prepare latex table content for the report.
  fprint("\nLaTeX Table Content:")
  cols = ["Type", "Accuracy", "Recall", "Precision", "F1", "Specificity", "Average"]
  table = []
  for type in reportDF["Type"].unique():
    typeData = reportDF[reportDF["Type"] == type]
    record = []
    for metric in typeData["Metric"].unique():
      metricData = typeData[typeData["Metric"] == metric]
      if (metricData.empty):
        continue
      row = metricData.iloc[0]
      std = row["Standard Deviation (Population)"]
      mean = row["Mean"]
      ci = row["Confidence Interval (Mean)"]
      # ciLow = dict(ci)["Lower Bound"]
      # ciHigh = dict(ci)["Upper Bound"]
      rowStr = f"{mean:.4f} $\\pm$ {std:.4f}"  # (CI: {ciLow:.4f} to {ciHigh:.4f})
      record.append((metric, rowStr))

    record = sorted(record, key=lambda x: cols.index(x[0]))
    record = [value for metric, value in record]
    record = f"{type} & " + " & ".join(record)

    table.append(record)

  colsBold = [f"\\textbf{{{col}}}" for col in cols]
  data2Latex = " & ".join(colsBold) + " \\\\\n\\midrule\n" + " \\\\\n".join(table) + " \\\\"
  data2Latex = data2Latex.replace("_", "\\_")
  fprint(data2Latex)
  # Store the LaTeX table content in a text file for later use in the report.
  latexTablePath = Path(step2Dir) / "LatexTableContent.txt"
  with open(latexTablePath, "w") as f:
    f.write(data2Latex)
  fprint(f"\u2713 LaTeX table content saved: {latexTablePath}")


if __name__ == "__main__":
  main()
