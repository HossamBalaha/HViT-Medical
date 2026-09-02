# Import necessary libraries.
import pickle, torch, os
import numpy as np
import pandas as pd
import PIL.Image as Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from HMB.DatasetsHelper import RawImageFolder


class TransformDataset(Dataset):
  def __init__(self, imagePaths, labels, transform, embeddedMetadata, classToIdx):
    self.imagePaths = imagePaths
    self.labels = labels
    self.transform = transform
    self.embeddedMetadata = embeddedMetadata
    self.class_to_idx = classToIdx

  def __len__(self):
    return len(self.imagePaths)

  def __getitem__(self, idx):
    imgPath = self.imagePaths[idx]
    label = self.labels[idx]
    image = Image.open(imgPath).convert("RGB")
    if (self.transform):
      image = self.transform(image)

    # Reconstruct metadata key: need filename and label.
    filename = os.path.basename(imgPath)
    key = f"{label}_{filename}"
    metadata = self.embeddedMetadata.get(key)
    # print(f"Retrieving metadata for key: {key}, found: {metadata is not None}")
    # Print first 5 keys for debugging.
    # print("Example keys:", list(self.embeddedMetadata.keys())[:5])
    if (metadata is None):
      # Try the class name instead of label index if the key is not found.
      className = [k for k, v in self.class_to_idx.items() if (v == label)]
      if (className):
        key = f"{className[0]}_{filename}"
        metadata = self.embeddedMetadata.get(key)
        # print(f"Trying with class name: {key}, found: {metadata is not None}")
    if (metadata is None):
      raise KeyError(f"Metadata not found for key: `{key}`.")
    metadata = torch.tensor(metadata, dtype=torch.float32)

    return image, metadata, label


# Define the CustomDataset class to handle images and metadata together.
class CustomDataset(Dataset):
  def __init__(self, dataset, embeddedMetadata, indices=None):
    self.dataset = dataset  # Store the dataset.
    self.embeddedMetadata = embeddedMetadata  # Store the embedded metadata.
    self.indices = indices if (indices is not None) else list(range(len(dataset)))  # Store the indices.

  def __len__(self):
    return len(self.indices)  # Return the length of the dataset.

  def __getitem__(self, idx):
    # Get the index of the image in the dataset.
    actualIDx = self.indices[idx]  # Get the actual index.
    image, label = self.dataset[actualIDx]  # Get the image and label.
    filenameWithPath = self.dataset.samples[actualIDx][0]  # Get the file path.
    filename = filenameWithPath.split("\\")[-1]  # Extract the filename.
    metadata = self.embeddedMetadata[f"{label}_{filename}"]  # Retrieve metadata using the label and filename.
    metadata = torch.tensor(metadata, dtype=torch.float32)  # Convert metadata to a tensor.
    return image, metadata, label  # Return the image, metadata, and label.


# Define the DataLoader function to load the dataset and embedded metadata.
def DataLoader(
  config, datasetPath, embeddedPath, batchSize,
  trainRatio=0.8, doAugmentation=True
):
  '''
  Load the dataset and embedded metadata.
  '''

  # Load the raw dataset to get paths and labels.
  rawDataset = RawImageFolder(datasetPath)
  classesMap = rawDataset.classToIdx  # Get the mapping of class names to indices.
  print("Classes Map:", classesMap)  # Print the class mapping.

  # Load the embedded metadata.
  if (embeddedPath.endswith(".p")):
    with open(embeddedPath, "rb") as f:
      embeddedMetadata = pickle.load(f)  # Load metadata from a pickle file.
  elif (embeddedPath.endswith(".csv")):
    # Load the metadata from a CSV file.
    embeddedMetadata = {}  # Initialize an empty dictionary for metadata.
    df = pd.read_csv(embeddedPath)  # Load the CSV file into a DataFrame.
    # Encode the labels that are not numeric using LabelEncoder.
    for col in df.columns[1:-1]:
      if (df[col].dtype == "object"):
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col])  # Encode non-numeric labels.
      # Normalize the metadata using MinMaxScaler.
      scaler = MinMaxScaler()
      df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))  # Normalize the metadata.
    # Encode the labels in the "Class" column using classesMap.
    df["Class"] = df["Class"].map(classesMap)  # Map class names to indices.
    print(df.head(5))  # Print the first few rows of the DataFrame.

    # Iterate through the DataFrame and create a dictionary of embedded metadata.
    for _, row in df.iterrows():
      label = row["Class"]  # Get the class label.
      filename = row["Filename"]  # Get the filename.
      metadata = row[1:-1].values  # Get the metadata values.
      # Create a unique key for each image using its label and filename.
      key = f"{label}_{filename}"
      embeddedMetadata[key] = metadata.astype(np.float32)  # Store metadata as float32.
    print("Key Example:", list(embeddedMetadata.keys())[0])  # Print an example key.
    print("Metadata Example:", list(embeddedMetadata.values())[0])  # Print an example metadata value.

  # Split the dataset into training and testing sets.
  allPaths = [path for path in rawDataset.GetPaths()]  # Get all image paths.
  allLabels = [label for label in rawDataset.GetLabels()]  # Get all labels.
  indices = list(range(len(rawDataset)))  # Get all indices.
  trainIndices, testIndices = train_test_split(
    indices,
    test_size=1.0 - trainRatio,
    stratify=allLabels,
    random_state=np.random.randint(0, 1000),
  )
  trainPaths = [allPaths[i] for i in trainIndices]
  trainLabels = [allLabels[i] for i in trainIndices]
  testPaths = [allPaths[i] for i in testIndices]
  testLabels = [allLabels[i] for i in testIndices]

  if (doAugmentation):
    print("Data augmentation is enabled.")
    trainTransform = transforms.Compose([
      transforms.Resize((config["ImageSize"], config["ImageSize"])),  # Resize the images.
      transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally.
      transforms.RandomVerticalFlip(),  # Randomly flip images vertically.
      transforms.RandomRotation(10),  # Randomly rotate images by up to 10 degrees.
      transforms.ToTensor(),  # Convert images to tensors.
    ])  # Define the transformation pipeline.
  else:
    print("Data augmentation is disabled.")
    trainTransform = transforms.Compose([
      transforms.Resize((config["ImageSize"], config["ImageSize"])),  # Resize the images.
      transforms.ToTensor(),  # Convert images to tensors.
    ])  # Define the transformation pipeline.
  testTransform = transforms.Compose([
    transforms.Resize((config["ImageSize"], config["ImageSize"])),  # Resize the images.
    transforms.ToTensor(),  # Convert images to tensors.
  ])  # Define the transformation pipeline for testing.
  allTransform = transforms.Compose([
    transforms.Resize((config["ImageSize"], config["ImageSize"])),  # Resize the images.
    transforms.ToTensor(),  # Convert images to tensors.
  ])  # Define the transformation pipeline for all data.

  # Create datasets for training, testing, and combined data.
  trainDataset = TransformDataset(trainPaths, trainLabels, trainTransform, embeddedMetadata, classesMap)
  testDataset = TransformDataset(testPaths, testLabels, testTransform, embeddedMetadata, classesMap)
  combinedDataset = TransformDataset(allPaths, allLabels, allTransform, embeddedMetadata, classesMap)

  # Create DataLoaders for training, testing, and combined data.
  trainDataloader = torch.utils.data.DataLoader(
    trainDataset,
    batch_size=batchSize,
    shuffle=True,
    # num_workers=1,
    # pin_memory=True,
  )

  testDataloader = torch.utils.data.DataLoader(
    testDataset,
    batch_size=batchSize,
    shuffle=False,
    # num_workers=1,
    # pin_memory=True,
  )

  combinedDataloader = torch.utils.data.DataLoader(
    combinedDataset,
    batch_size=batchSize,
    shuffle=False,
    # num_workers=1,
    # pin_memory=True,
  )

  return trainDataloader, testDataloader, combinedDataloader, classesMap, rawDataset, embeddedMetadata
