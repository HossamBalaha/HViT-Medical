# Import necessary libraries.
import math  # Standard library for mathematical operations.
import torch  # PyTorch library for deep learning operations.
import torch.nn as nn  # PyTorch's neural network module.
import torch.nn.functional as F  # Functional interface for neural network operations.
from HMB.Utils import fprint


class PatchEmbeddings(nn.Module):
  '''
  Converts images into patches and projects them into a vector space using convolution.

  This class divides an image into fixed-size patches and uses a convolutional layer
  to project each patch into a hidden dimension space.

  Attributes:
    imageSize (int): Size of the input image (assumed square).
    patchSize (int): Size of each patch (assumed square).
    numChannels (int): Number of channels in the input image.
    hiddenSize (int): Dimensionality of the output embedding.
    numPatches (int): Total number of patches computed from image and patch sizes.
    projection (nn.Conv2d): Convolutional layer used for projecting patches.
  '''

  def __init__(self, config):
    '''
    Initializes PatchEmbeddings with configuration parameters.

    Parameters:
      config (dict): Dictionary containing hyperparameters:
        - imageSize: size of input image (e.g., 224).
        - patchSize: size of each patch (e.g., 16).
        - numChannels: number of input channels (e.g., 3).
        - hiddenSize: embedding dimension (e.g., 768).
    '''

    super().__init__()
    # Read the image size from the configuration.
    self.imageSize = config["ImageSize"]
    # Read the patch size from the configuration.
    self.patchSize = config["PatchSize"]
    # Read the number of input channels from the configuration.
    self.numChannels = config.get("NumChannels", 3)
    # Read the hidden embedding size from the configuration.
    self.hiddenSize = config["HiddenSize"]

    # Calculate number of patches per side and total patches.
    self.numPatches = (self.imageSize // self.patchSize) ** 2

    # Projection layer: converts image patches into embeddings.
    self.projection = nn.Conv2d(
      self.numChannels,  # Number of input channels.
      self.hiddenSize,  # Output embedding dimension.
      kernel_size=self.patchSize,  # Size of the patch.
      stride=self.patchSize,  # Stride equal to patch size to avoid overlap.
    )

  def forward(self, x):
    '''
    Forward pass through the patch embedding layer.

    Parameters:
      x (Tensor): Input tensor of shape (batchSize, numChannels, imageSize, imageSize).

    Returns:
      Tensor: Output tensor of shape (batchSize, numPatches, hiddenSize).
    '''

    # Apply convolution to extract patches.
    x = self.projection(x)

    # Flatten spatial dimensions and transpose.
    x = x.flatten(2).transpose(1, 2)

    return x


class StandardEmbeddings(nn.Module):
  '''
  Combines patch embeddings with learnable [CLS] token and position embeddings.

  This class adds a [CLS] token to the sequence of patch embeddings and applies
  positional encodings before feeding into the transformer.

  Attributes:
    clsToken (nn.Parameter): Learnable [CLS] token added to the beginning of sequence.
    positionEmbeddings (nn.Parameter): Positional embeddings for all tokens.
    dropout (nn.Dropout): Dropout layer applied after adding embeddings.
    patchEmbeddings (PatchEmbeddings): Patch embedding module.
  '''

  def __init__(self, config):
    '''
    Initializes the StandardEmbeddings module.

    Parameters:
      config (dict): Configuration dictionary with keys:
        - hiddenSize: Embedding dimension.
        - hiddenDropoutProb: Dropout probability.
    '''

    super().__init__()

    # Store configuration parameters.
    self.config = config
    self.patchEmbeddings = PatchEmbeddings(config)

    # Create the learnable classification token.
    self.clsToken = nn.Parameter(torch.randn(1, 1, config["HiddenSize"]))
    # Create the positional embeddings for all tokens.
    self.positionEmbeddings = nn.Parameter(
      torch.randn(1, self.patchEmbeddings.numPatches + 1, config["HiddenSize"])
    )
    # Create the dropout layer for the embedding output.
    self.dropout = nn.Dropout(config["HiddenDropoutProb"])

  def forward(self, x):
    '''
    Forward pass through the standard embedding layer.

    Parameters:
      x (Tensor): Input tensor of shape (batchSize, numChannels, imageSize, imageSize).

    Returns:
      Tensor: Embedded tokens of shape (batchSize, numTokens, hiddenSize).
    '''

    # Convert image to patches using the patch embedding layer.
    x = self.patchEmbeddings(x)

    # Get batch size and number of tokens.
    batchSize, _, _ = x.size()

    # Expand [CLS] token to match batch size.
    clsTokens = self.clsToken.expand(batchSize, -1, -1)

    # Concatenate [CLS] token to the start of patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)

    # Add positional embeddings.
    x = x + self.positionEmbeddings

    # Apply dropout.
    x = self.dropout(x)

    return x


class AttentionHeadParent(nn.Module):
  '''
  Base class for all attention head types.
  Provides shared functionality like query/key/value projections and attention score computation.
  '''

  def __init__(self, config, hiddenSize, attentionHeadSize, dropout, bias=True, metadataDim=None):
    '''
    Initializes the base attention head with query, key, value projections.

    Parameters:
      config (dict): Configuration dictionary containing model hyperparameters.
      hiddenSize (int): Size of the input embeddings.
      attentionHeadSize (int): Size of the attention head.
      dropout (float): Dropout probability.
      bias (bool): Whether to include bias in linear layers.
      metadataDim (int or None): Dimension of metadata input if used.
    '''

    super().__init__()
    self.attentionHeadSize = attentionHeadSize  # Store attention head size.

    # Define QKV projections.
    self.query = nn.Linear(hiddenSize, attentionHeadSize, bias=bias)  # Query projection layer.
    self.key = nn.Linear(hiddenSize, attentionHeadSize, bias=bias)  # Key projection layer.
    self.value = nn.Linear(hiddenSize, attentionHeadSize, bias=bias)  # Value projection layer.

    # Dropout layer for attention scores.
    self.dropout = nn.Dropout(dropout)

    # Default head type.
    self.headType = "Base"

  def forwardHelper(self, attentionScores, value, metadata=None):
    '''
    Helper method to compute attention output from attention scores and values.

    Parameters:
      attentionScores (Tensor): Scaled dot-product attention scores.
      value (Tensor): Projected value tensor.
      metadata (Tensor or None): Optional metadata input.

    Returns:
      Tensor: Output of the attention operation.
      Tensor: Normalized attention probabilities.
    '''

    # Compute attention probabilities using softmax.
    attentionProbs = F.softmax(attentionScores, dim=-1)  # Normalize attention scores.
    attentionProbs = self.dropout(attentionProbs)  # Apply dropout to attention probs.
    contextLayer = torch.matmul(attentionProbs, value)  # Weighted sum using attention.

    # Return output and attention weights.
    return contextLayer, attentionProbs


class StandardAttentionHead(AttentionHeadParent):
  '''
  A standard attention head without metadata integration.
  '''

  def __init__(self, config, hiddenSize, attentionHeadSize, dropout, bias=True, metadataDim=None):
    '''
    Initializes the standard attention head.

    Parameters:
      config (dict): Configuration dictionary.
      hiddenSize (int): Input embedding size.
      attentionHeadSize (int): Size of this attention head.
      dropout (float): Dropout rate.
      bias (bool): Whether to use bias in projections.
      metadataDim (int or None): Dimension of metadata inputs.
    '''

    super().__init__(
      config,  # Configuration dictionary.
      hiddenSize,  # Input embedding size.
      attentionHeadSize,  # Size of this attention head.
      dropout,  # Dropout rate.
      bias=bias,  # Whether to use bias in projections.
      metadataDim=None,  # Dimension of metadata inputs.
    )
    self.headType = "Standard"  # Set the head type.

  def forward(self, x, metadata=None):
    '''
    Forward pass for standard attention head.

    Parameters:
      x (Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
      metadata (Tensor): Metadata tensor (not used in standard head).

    Returns:
      Tensor: Contextualized output tensor.
      Tensor: Attention probabilities.
    '''

    query = self.query(x)
    key = self.key(x)
    value = self.value(x)

    # Compute scaled dot-product attention scores.
    attentionScores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attentionHeadSize)
    return self.forwardHelper(attentionScores, value, metadata=metadata)


class QConditioningAttentionHead(AttentionHeadParent):
  '''
  An attention head that applies Q-Conditioning, where metadata is added directly to the query.
  '''

  def __init__(self, config, hiddenSize, attentionHeadSize, dropout, bias=True, metadataDim=None):
    '''
    Initializes the Q-Conditioning attention head.

    Parameters:
      config (dict): Configuration dictionary.
      hiddenSize (int): Input embedding size.
      attentionHeadSize (int): Size of this attention head.
      dropout (float): Dropout rate.
      bias (bool): Whether to use bias in projections.
      metadataDim (int or None): Dimension of metadata inputs.
    '''
    super().__init__(
      config,  # Configuration dictionary.
      hiddenSize,  # Input embedding size.
      attentionHeadSize,  # Size of this attention head.
      dropout,  # Dropout rate.
      bias=bias,  # Whether to use bias in projections.
      metadataDim=None,  # Dimension of metadata inputs.
    )

    self.headType = "Q-Conditioning"  # Set the head type.

    self.metadataProjection = nn.Linear(
      metadataDim, attentionHeadSize, bias=bias
    )  # Linear projection of metadata into query space.
    self.gateProjection = None  # No gate in Q-Conditioning.

  def forward(self, x, metadata=None):
    '''
    Forward pass for Q-Conditioning attention head.

    Parameters:
      x (Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
      metadata (Tensor): Metadata tensor of shape [batch_size, metadata_dim].

    Returns:
      Tensor: Contextualized output tensor.
      Tensor: Attention probabilities.
    '''
    query = self.query(x)
    key = self.key(x)
    value = self.value(x)

    # Add sequence dim.
    metadataConditioning = self.metadataProjection(metadata).unsqueeze(1)

    # Inject metadata into queries.
    newQuery = query + metadataConditioning

    # Compute scaled dot-product attention scores.
    attentionScores = torch.matmul(newQuery, key.transpose(-1, -2)) / math.sqrt(self.attentionHeadSize)

    return self.forwardHelper(attentionScores, value, metadata=metadata)


class KeyConditioningAttentionHead(AttentionHeadParent):
  '''
  An attention head that applies Key-Conditioning, where metadata is added directly to the key.
  '''

  def __init__(self, config, hiddenSize, attentionHeadSize, dropout, bias=True, metadataDim=None):
    '''
    Initializes the Key-Conditioning attention head.
    '''
    super().__init__(config, hiddenSize, attentionHeadSize, dropout, bias=bias, metadataDim=None)
    # Set the head type identifier for the hybrid framework.
    self.headType = "Key-Conditioning"
    # Define the linear projection to map metadata into the key space.
    self.metadataProjection = nn.Linear(metadataDim, attentionHeadSize, bias=bias)

  def forward(self, x, metadata=None):
    '''
    Forward pass for Key-Conditioning attention head.
    '''
    # Compute the standard query projection.
    query = self.query(x)
    # Compute the standard key projection.
    key = self.key(x)
    # Compute the standard value projection.
    value = self.value(x)
    # Project the metadata and add a sequence dimension for broadcasting.
    metadataConditioning = self.metadataProjection(metadata).unsqueeze(1)
    # Inject the metadata directly into the key representations.
    newKey = key + metadataConditioning
    # Compute the scaled dot-product attention scores using the modified keys.
    attentionScores = torch.matmul(query, newKey.transpose(-1, -2)) / math.sqrt(self.attentionHeadSize)
    # Return the context layer and attention probabilities via the helper method.
    return self.forwardHelper(attentionScores, value, metadata=metadata)


class ValueConditioningAttentionHead(AttentionHeadParent):
  '''
  An attention head that applies Value-Conditioning, where metadata is added directly to the value.
  '''

  def __init__(self, config, hiddenSize, attentionHeadSize, dropout, bias=True, metadataDim=None):
    '''
    Initializes the Value-Conditioning attention head.
    '''
    super().__init__(config, hiddenSize, attentionHeadSize, dropout, bias=bias, metadataDim=None)
    # Set the head type identifier for the hybrid framework.
    self.headType = "Value-Conditioning"
    # Define the linear projection to map metadata into the value space.
    self.metadataProjection = nn.Linear(metadataDim, attentionHeadSize, bias=bias)

  def forward(self, x, metadata=None):
    '''
    Forward pass for Value-Conditioning attention head.
    '''
    # Compute the standard query projection.
    query = self.query(x)
    # Compute the standard key projection.
    key = self.key(x)
    # Compute the standard value projection.
    value = self.value(x)
    # Project the metadata and add a sequence dimension for broadcasting.
    metadataConditioning = self.metadataProjection(metadata).unsqueeze(1)
    # Inject the metadata directly into the value representations.
    newValue = value + metadataConditioning
    # Compute the standard scaled dot-product attention scores.
    attentionScores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attentionHeadSize)
    # Return the context layer using the modified values and standard attention probabilities.
    return self.forwardHelper(attentionScores, newValue, metadata=metadata)


class QGatingAttentionHead(AttentionHeadParent):
  '''
  An attention head that applies Q-Gating, where metadata gates the attention scores.
  '''

  def __init__(self, config, hiddenSize, attentionHeadSize, dropout, bias=True, metadataDim=None):
    '''
    Initializes the Q-Gating attention head.

    Parameters:
      config (dict): Configuration dictionary.
      hiddenSize (int): Input embedding size.
      attentionHeadSize (int): Size of this attention head.
      dropout (float): Dropout rate.
      bias (bool): Whether to use bias in projections.
      metadataDim (int or None): Dimension of metadata inputs.
    '''

    super().__init__(
      config,  # Configuration dictionary.
      hiddenSize,  # Input embedding size.
      attentionHeadSize,  # Size of this attention head.
      dropout,  # Dropout rate.
      bias=bias,  # Whether to use bias in projections.
      metadataDim=None,  # Dimension of metadata inputs.
    )
    # Set the head type.
    self.headType = "Q-Gating"

    # Number of patches.
    # N = (config["imageSize"] // config["patchSize"]) ** 2
    # Define projections for metadata and gating.
    # self.metadataProjection = nn.Linear(metadataDim, attentionHeadSize, bias=bias)
    # self.gateProjection = nn.Linear(attentionHeadSize * 2, N + 1, bias=bias)

    # Define the linear projection to map metadata into the attention head space.
    self.metadataProjection = nn.Linear(metadataDim, attentionHeadSize, bias=bias)
    # Define the gate projection to output a single scalar per query token for sequence-length agnosticism.
    self.gateProjection = nn.Linear(attentionHeadSize * 2, 1, bias=bias)

  def forward(self, x, metadata=None):
    '''
    Forward pass for Q-Gating attention head.

    Parameters:
      x (Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
      metadata (Tensor): Metadata tensor of shape [batch_size, metadata_dim].

    Returns:
      Tensor: Contextualized output tensor.
      Tensor: Gated attention probabilities.
    '''

    # Project query, key, and value.
    query = self.query(x)
    key = self.key(x)
    value = self.value(x)

    # metadataProjection = self.metadataProjection(metadata).unsqueeze(1).expand(-1, query.size(1), -1)
    # gateInput = torch.cat([query, metadataProjection], dim=-1)  # Combine query & metadata.
    # gate = torch.sigmoid(self.gateProjection(gateInput))  # Gate via sigmoid.
    #
    # attentionScores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attentionHeadSize)
    # attentionScores = attentionScores * gate  # Apply gating mask.

    # Project the metadata into the query space and expand it to match the sequence length.
    metadataProjection = self.metadataProjection(metadata).unsqueeze(1).expand(-1, query.size(1), -1)
    # Concatenate the query and metadata projections along the feature dimension.
    gateInput = torch.cat([query, metadataProjection], dim=-1)
    # Compute the scalar gate values and apply the sigmoid activation function.
    gate = torch.sigmoid(self.gateProjection(gateInput))
    # Calculate the raw scaled dot-product attention scores.
    attentionScores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attentionHeadSize)
    # Multiply the attention scores by the broadcasted gate mask.
    attentionScores = attentionScores * gate

    return self.forwardHelper(attentionScores, value, metadata=metadata)


class MultiHeadAttention(nn.Module):
  '''
  Multi-head attention module supporting multiple head types.
  '''

  def __init__(self, config, headTypes):
    super().__init__()  # Initialize the parent class.

    # Read the hidden size from the configuration.
    self.hiddenSize = config["HiddenSize"]
    # Read the number of attention heads from the configuration.
    self.numAttentionHeads = config["NumAttentionHeads"]
    # Assert that the hidden size is perfectly divisible by the number of heads.
    assert (self.hiddenSize % self.numAttentionHeads == 0), \
      "Hidden size must be divisible by the number of attention heads."

    # Compute the exact dimension size for each individual attention head.
    self.attentionHeadSize = self.hiddenSize // self.numAttentionHeads

    self.allHeadSize = self.numAttentionHeads * self.attentionHeadSize  # Compute the total size of all heads.
    # Read the query-key-value bias flag from the configuration.
    self.qkvBias = config["QkvBias"]
    # Read the metadata dimension from the configuration.
    self.metadataDim = config.get("MetadataDim", None)
    # Validate headTypes and ensure they match the number of heads.
    if (len(headTypes) != self.numAttentionHeads):
      # Raise an error if the number of head types does not match.
      raise ValueError(
        f"Number of head types ({len(headTypes)}) "
        f"must match `numAttentionHeads` ({self.numAttentionHeads})!"
      )
    self.headTypes = headTypes  # Store the head types.
    # Create a list of attention heads with specified types.
    attentionHeadList = []  # Initialize an empty list for attention heads.
    for headType in headTypes:
      if (headType == "Standard"):
        attentionHeadList.append(
          StandardAttentionHead(
            config,
            self.hiddenSize,  # Input hidden size.
            self.attentionHeadSize,  # Size of this attention head.
            config["AttentionProbsDropoutProb"],  # Dropout probability.
            bias=self.qkvBias,  # Whether to use bias in projections.
            metadataDim=self.metadataDim,  # Dimension of metadata inputs.
          )
        )  # Add a StandardAttentionHead to the list.
      elif (headType == "Q-Conditioning"):
        attentionHeadList.append(
          QConditioningAttentionHead(
            config,
            self.hiddenSize,  # Input hidden size.
            self.attentionHeadSize,  # Size of this attention head.
            config["AttentionProbsDropoutProb"],  # Dropout probability.
            bias=self.qkvBias,  # Whether to use bias in projections.
            metadataDim=self.metadataDim,  # Dimension of metadata inputs.
          )
        )  # Add a QConditioningAttentionHead to the list.
      elif (headType == "Q-Gating"):
        # Append a QGatingAttentionHead to the list.
        attentionHeadList.append(
          QGatingAttentionHead(
            config,
            self.hiddenSize,
            self.attentionHeadSize,
            config["AttentionProbsDropoutProb"],
            bias=self.qkvBias,
            metadataDim=self.metadataDim,
          )
        )  # Add a QGatingAttentionHead to the list.
      elif (headType == "Key-Conditioning"):
        # Append a KeyConditioningAttentionHead to the list for ablation studies.
        attentionHeadList.append(
          KeyConditioningAttentionHead(
            config,
            self.hiddenSize,
            self.attentionHeadSize,
            config["AttentionProbsDropoutProb"],
            bias=self.qkvBias,
            metadataDim=self.metadataDim,
          )
        )  # Add a KeyConditioningAttentionHead to the list.
      elif (headType == "Value-Conditioning"):
        # Append a ValueConditioningAttentionHead to the list for ablation studies.
        attentionHeadList.append(
          ValueConditioningAttentionHead(
            config,
            self.hiddenSize,
            self.attentionHeadSize,
            config["AttentionProbsDropoutProb"],
            bias=self.qkvBias,
            metadataDim=self.metadataDim,
          )
        )  # Add a ValueConditioningAttentionHead to the list.
      else:
        # Raise an error for unknown head types.
        raise ValueError(f"Unknown attention head type: {headType}")
    self.heads = nn.ModuleList(attentionHeadList)  # Convert the list of heads to a ModuleList.
    # Linear layer to project the attention output back to the hidden size.
    self.outputProjection = nn.Linear(self.allHeadSize, self.hiddenSize)  # Define the output projection layer.
    self.outputDropout = nn.Dropout(config["HiddenDropoutProb"])  # Define a dropout layer.

  def forward(self, x, metadata=None, outputAttentions=False):
    attentionOutputs = [
      head(x, metadata=metadata)
      for head in self.heads
    ]  # Compute attention outputs for all heads.
    attentionOutput = torch.cat([output for output, _ in attentionOutputs], dim=-1)  # Concatenate attention outputs.
    attentionOutput = self.outputProjection(attentionOutput)  # Project the concatenated output.
    attentionOutput = self.outputDropout(attentionOutput)  # Apply dropout to the projected output.
    if (not outputAttentions):
      return attentionOutput, None  # Return only the attention output if attentions are not requested.
    else:
      attentionProbs = torch.stack([probs for _, probs in attentionOutputs], dim=1)  # Stack attention probabilities.
      return attentionOutput, attentionProbs  # Return both the attention output and probabilities.


# Define the MLP class for a multi-layer perceptron.
class MLP(nn.Module):
  '''
  A multi-layer perceptron module.
  '''

  def __init__(self, config):
    super().__init__()  # Initialize the parent class.
    # Define the first dense layer.
    self.dense1 = nn.Linear(config["HiddenSize"], config["IntermediateSize"])
    # Define the activation function.
    self.activation = nn.GELU()
    # Define the second dense layer.
    self.dense2 = nn.Linear(config["IntermediateSize"], config["HiddenSize"])
    # Define the dropout layer.
    self.dropout = nn.Dropout(config["HiddenDropoutProb"])

  def forward(self, x):
    x = self.dense1(x)  # Apply the first dense layer.
    x = self.activation(x)  # Apply the activation function.
    x = self.dense2(x)  # Apply the second dense layer.
    x = self.dropout(x)  # Apply dropout to the output.
    return x  # Return the processed tensor.


# Define the Block class for a single transformer block.
class Block(nn.Module):
  '''
  A single transformer block.
  '''

  def __init__(self, config, headTypes):
    super().__init__()  # Initialize the parent class.
    self.attention = MultiHeadAttention(config, headTypes)  # Initialize the MultiHeadAttention module.
    self.layerNorm1 = nn.LayerNorm(config["HiddenSize"])  # Define the first layer normalization.
    self.mlp = MLP(config)  # Initialize the MLP module.
    self.layerNorm2 = nn.LayerNorm(config["HiddenSize"])  # Define the second layer normalization.

  def forward(self, x, metadata=None, outputAttentions=False):
    attentionOutput, attentionProbs = self.attention(
      self.layerNorm1(x),
      metadata=metadata,
      outputAttentions=outputAttentions,
    )  # Compute attention output.
    x = x + attentionOutput  # Add the attention output to the input (residual connection).
    mlpOutput = self.mlp(self.layerNorm2(x))  # Compute the MLP output.
    x = x + mlpOutput  # Add the MLP output to the input (residual connection).
    if (not outputAttentions):
      return x, None  # Return only the output if attentions are not requested.
    else:
      return x, attentionProbs  # Return both the output and attention probabilities.


# Define the Encoder class for the transformer encoder module.
class Encoder(nn.Module):
  '''
  The transformer encoder module.
  '''

  def __init__(self, config, headTypesPerBlock):
    super().__init__()  # Initialize the parent class.
    # Read the number of hidden layers from the configuration.
    numHiddenLayers = config["NumHiddenLayers"]
    # Check whether the head configuration is a model-name string.
    if (isinstance(headTypesPerBlock, str)):
      # Build one standard head list for every hidden layer.
      headTypesPerBlock = [["Standard"] * config["NumAttentionHeads"] for layerIndex in range(numHiddenLayers)]
    # Validate the number of head-type lists.
    if (len(headTypesPerBlock) != numHiddenLayers):
      # Raise an error when the counts do not match.
      raise ValueError(
        f"Number of head types per block ({len(headTypesPerBlock)}) "
        f"must match NumHiddenLayers ({numHiddenLayers})!"
      )
    self.blocks = nn.ModuleList([
      Block(config, headTypes)
      for headTypes in headTypesPerBlock
    ])  # Create a list of transformer blocks.

  def forward(self, x, metadata=None, outputAttentions=False):
    allAttentions = []  # Initialize an empty list for attention probabilities.
    for block in self.blocks:
      x, attentionProbs = block(x, metadata=metadata, outputAttentions=outputAttentions)  # Process each block.
      if (outputAttentions):
        allAttentions.append(attentionProbs)  # Append attention probabilities if requested.
    if (not outputAttentions):
      return x, None  # Return only the output if attentions are not requested.
    else:
      return x, allAttentions  # Return both the output and attention probabilities.


# Define the ViTForClassification class for classification tasks with metadata integration.
class ViTForClassification(nn.Module):
  '''
  The ViT model for classification with metadata integration.
  '''

  def __init__(self, config, headTypesPerBlock):
    super().__init__()  # Initialize the parent class.
    self.config = config  # Store the configuration.
    # Read the image size from the configuration.
    self.imageSize = config["ImageSize"]
    # Read the hidden size from the configuration.
    self.hiddenSize = config["HiddenSize"]
    # Read the number of classes from the configuration.
    self.numClasses = config["NumClasses"]
    # fprint(f"Number of classes: {self.numClasses}")  # Print the number of classes.
    # fprint(f"Hidden size: {self.hiddenSize}")  # Print the hidden size.
    # fprint(f"Image size: {self.imageSize}")  # Print the image size.
    self.embedding = StandardEmbeddings(config)  # Initialize the StandardEmbeddings module.
    self.encoder = Encoder(config, headTypesPerBlock)  # Initialize the Encoder module.
    self.classifier = nn.Linear(self.hiddenSize, self.numClasses)  # Define the classification layer.
    self.apply(self._init_weights)  # Apply weight initialization.

  def forward(self, x, metadata=None, outputAttentions=False):
    embeddingOutput = self.embedding(x)  # Generate embeddings.
    encoderOutput, allAttentions = self.encoder(
      embeddingOutput,  # Input the embeddings to the encoder.
      metadata=metadata,  # Pass metadata to the encoder for attention conditioning.
      outputAttentions=outputAttentions,  # Control whether to output attention probabilities.
    )  # Process through the encoder.
    logits = self.classifier(encoderOutput[:, 0])  # Compute logits using the [CLS] token.
    if (not outputAttentions):
      return logits, None  # Return only the logits if attentions are not requested.
    else:
      return logits, allAttentions  # Return both the logits and attention probabilities.

  def _init_weights(self, module):
    if (isinstance(module, nn.Linear)):
      # Initialize weights using a normal distribution.
      nn.init.normal_(module.weight, std=self.config.get("InitializerRange", 0.02))
      if (module.bias is not None):
        nn.init.zeros_(module.bias)  # Initialize biases to zero.


# Define a Vision Transformer that uses late fusion for metadata integration.
class LateFusionViT(nn.Module):
  # Initialize the late fusion model with configuration and head types.
  def __init__(self, config, headTypesPerBlock):
    # Call the parent class constructor to initialize the module.
    super().__init__()
    # Store the configuration dictionary for future reference.
    self.config = config
    # Initialize the standard patch and position embeddings.
    self.embedding = StandardEmbeddings(config)
    # Initialize the transformer encoder without metadata awareness.
    self.encoder = Encoder(config, headTypesPerBlock)
    # Define the late fusion classification head using CamelCase dictionary keys.
    self.classifier = nn.Linear(config["HiddenSize"] + config["MetadataDim"], config["NumClasses"])

  # Define the forward pass for the late fusion model.
  def forward(self, x, metadata=None, outputAttentions=False):
    # Generate the initial patch and position embeddings from the input image.
    embeddingOutput = self.embedding(x)
    # Pass the embeddings through the standard transformer encoder without metadata.
    encoderOutput, allAttentions = self.encoder(embeddingOutput, metadata=None, outputAttentions=outputAttentions)
    # Extract the aggregated CLS token representation from the encoder output.
    clsToken = encoderOutput[:, 0]
    # Concatenate the visual CLS token with the raw metadata vector along the feature dimension.
    combinedFeatures = torch.cat([clsToken, metadata], dim=-1)
    # Compute the final classification logits using the combined features.
    logits = self.classifier(combinedFeatures)
    # Return the logits and attention weights if requested by the caller.
    return logits, allAttentions


# Define a Vision Transformer that uses early fusion by injecting metadata into the input sequence.
class EarlyFusionViT(nn.Module):
  # Initialize the early fusion model with configuration and head types.
  def __init__(self, config, headTypesPerBlock):
    # Call the parent class constructor to initialize the module.
    super().__init__()
    # Store the configuration dictionary for future reference.
    self.config = config
    # Initialize the standard patch and position embeddings.
    self.embedding = StandardEmbeddings(config)
    # Define a linear layer to project metadata into the hidden embedding space using CamelCase keys.
    self.metadataProjection = nn.Linear(config["MetadataDim"], config["HiddenSize"])
    # Initialize the standard transformer encoder.
    self.encoder = Encoder(config, headTypesPerBlock)
    # Define the standard classification head using CamelCase keys.
    self.classifier = nn.Linear(config["HiddenSize"], config["NumClasses"])

  # Define the forward pass for the early fusion model.
  def forward(self, x, metadata=None, outputAttentions=False):
    # Generate the initial patch and position embeddings from the input image.
    embeddingOutput = self.embedding(x)
    # Check if metadata is provided for early injection into the sequence.
    if (metadata is not None):
      # Project the metadata into the visual embedding dimension.
      metaEmbedding = self.metadataProjection(metadata).unsqueeze(1)
      # Add the projected metadata directly to the CLS token at index zero.
      modifiedCls = embeddingOutput[:, :1] + metaEmbedding
      # Reconstruct the sequence with the modified CLS token and original patches.
      embeddingOutput = torch.cat([modifiedCls, embeddingOutput[:, 1:]], dim=1)
    # Pass the modified embeddings through the transformer encoder without passing metadata again.
    encoderOutput, allAttentions = self.encoder(embeddingOutput, metadata=None, outputAttentions=outputAttentions)
    # Compute the final classification logits using the aggregated CLS token.
    logits = self.classifier(encoderOutput[:, 0])
    # Return the logits and attention weights if requested by the caller.
    return logits, allAttentions


# Define a robust linear probe for frozen PhikonV2 embeddings.
class PhikonV2LinearProbe(nn.Module):
  # Initialize the PhikonV2 linear probe.
  def __init__(self, config, headTypesPerBlock=None):
    # Call the parent constructor.
    super().__init__()

    # Store the configuration dictionary.
    self.config = config

    # Ignore the head-type argument because the probe has no transformer heads.
    self.headTypesPerBlock = headTypesPerBlock

    # Read the metadata dimension using CamelCase configuration keys.
    metadataDim = config.get("MetadataDim", config.get("metadataDim", None))

    # Read the number of classes using CamelCase configuration keys.
    numClasses = config.get("NumClasses", config.get("numClasses", None))

    # Validate the metadata dimension.
    if (metadataDim is None):
      # Raise an informative error when the metadata dimension is missing.
      raise ValueError("PhikonV2LinearProbe requires MetadataDim in the config dictionary.")

    # Validate the number of classes.
    if (numClasses is None):
      # Raise an informative error when the number of classes is missing.
      raise ValueError("PhikonV2LinearProbe requires NumClasses in the config dictionary.")

    # Store the metadata dimension as an instance attribute.
    self.metadataDim = metadataDim

    # Store the number of classes as an instance attribute.
    self.numClasses = numClasses

    # Read the dropout probability from the configuration.
    dropoutProb = config.get("HiddenDropoutProb", config.get("hiddenDropoutProb", 0.1))

    # Define a dropout layer for regularization.
    self.dropout = nn.Dropout(dropoutProb)

    # Define the linear classification layer.
    self.classifier = nn.Linear(metadataDim, numClasses)

  # Define the forward pass for the PhikonV2 linear probe.
  def forward(self, x=None, metadata=None, outputAttentions=False):
    # Validate that metadata is provided.
    if (metadata is None):
      # Raise an error because the probe operates only on PhikonV2 embeddings.
      raise ValueError("PhikonV2LinearProbe requires the metadata tensor containing PhikonV2 embeddings.")

    # Read the device of the classifier parameters.
    targetDevice = next(self.classifier.parameters()).device

    # Move the metadata tensor to the classifier device and cast it to float32.
    metadataTensor = metadata.to(device=targetDevice, dtype=torch.float32)

    # Add a batch dimension when a single embedding vector is provided.
    if (metadataTensor.dim() == 1):
      # Add the missing batch dimension.
      metadataTensor = metadataTensor.unsqueeze(0)

    # Flatten all feature dimensions after the batch dimension.
    metadataTensor = metadataTensor.reshape(metadataTensor.size(0), -1)

    # Validate the final feature dimension.
    if (metadataTensor.size(-1) != self.metadataDim):
      # Raise an informative dimension mismatch error.
      raise ValueError(
        f"Expected metadata dimension {self.metadataDim}, "
        f"but received {metadataTensor.size(-1)}."
      )

    # Apply dropout to the frozen embeddings.
    droppedMetadata = self.dropout(metadataTensor)

    # Compute the classification logits.
    logits = self.classifier(droppedMetadata)

    # Return an empty attention list when attentions are requested.
    if (outputAttentions):
      # Return the logits and an empty attention list for interface compatibility.
      return logits, []

    # Return the logits and None when attentions are not requested.
    return logits, None
