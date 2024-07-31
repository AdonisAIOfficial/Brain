class Tokenizer {
  constructor(initialVocabulary = {}) {
    this.vocabulary = { ...initialVocabulary, "<stop>": 0 }; // Ensure stop token is included
    this.reverseVocabulary = Object.fromEntries(
      Object.entries(this.vocabulary).map(([k, v]) => [v, k]),
    );
    this.nextId = Object.keys(this.vocabulary).length;
  }

  tokenize(text) {
    const tokens = text.toLowerCase().match(/\w+|[^\w\s]/g) || [];
    // console.log("Tokenizing:", tokens); // Debug statement
    return tokens.map((token) => {
      if (!this.vocabulary.hasOwnProperty(token)) {
        this.vocabulary[token] = this.nextId++;
        this.reverseVocabulary[this.vocabulary[token]] = token;
      }
      return this.vocabulary[token];
    });
  }

  decode(ids) {
    // console.log("Decoding IDs:", ids); // Debug statement
    // console.log(`Vocab: ${Object.keys(this.vocabulary).length}`);
    return ids.map((id) => this.reverseVocabulary[id] || "<unk>").join(" ");
  }
}

class EmbeddingLayer {
  constructor(vocabSize, embeddingDim, plasticity) {
    this.vocabSize = vocabSize;
    this.embeddingDim = embeddingDim;
    this.plasticity = plasticity;
    this.embeddings = this.initEmbeddings();
  }

  initEmbeddings() {
    return Array.from({ length: this.vocabSize }, () =>
      Array.from({ length: this.embeddingDim }, () => Math.random() * 0.1),
    );
  }

  apply(tokens) {
    // console.log("Applying embeddings to tokens:", tokens); // Debug statement
    return tokens.map((tokenId) => {
      if (tokenId >= this.embeddings.length) {
        this.addNewEmbedding(tokenId);
      }
      return this.embeddings[tokenId].map(
        (val) => val + (Math.random() - 0.5) * this.plasticity,
      );
    });
  }

  addNewEmbedding(tokenId) {
    while (this.embeddings.length <= tokenId) {
      this.embeddings.push(
        Array.from({ length: this.embeddingDim }, () => Math.random() * 0.1),
      );
    }
    if (this.embeddings[tokenId] === undefined) {
      this.embeddings[tokenId] = Array.from(
        { length: this.embeddingDim },
        () => Math.random() * 0.1,
      );
    }
  }

  adjustWeightsBasedOnReward(reward) {
    this.embeddings = this.embeddings.map((embedding) =>
      embedding.map((val) => val + (Math.random() - 0.5) * reward),
    );
  }
}

class PositionalEncoder {
  constructor(maxSeqLength, embeddingDim) {
    this.maxSeqLength = maxSeqLength;
    this.embeddingDim = embeddingDim;
    this.positionalEncodings = this.initPositionalEncodings();
  }

  initPositionalEncodings() {
    const posEncodings = Array.from({ length: this.maxSeqLength }, (_, pos) =>
      Array.from(
        { length: this.embeddingDim },
        (_, i) => pos / Math.pow(10000, (2 * (i / 2)) / this.embeddingDim),
      ),
    );
    return posEncodings.map((encoding, pos) =>
      encoding.map((val, i) => (i % 2 === 0 ? Math.sin(val) : Math.cos(val))),
    );
  }

  apply(embeddings) {
    // console.log("Applying positional encoding:", embeddings); // Debug statement
    return embeddings.map((embedding, pos) =>
      embedding.map(
        (val, i) => val + (this.positionalEncodings[pos] || [])[i] || 0,
      ),
    );
  }
}

class TransformerLayer {
  constructor(numHeads, embeddingDim, plasticity) {
    this.numHeads = numHeads;
    this.embeddingDim = embeddingDim;
    this.plasticity = plasticity;
    this.multiHeadAttention = new MultiHeadAttention(numHeads, embeddingDim);
    this.feedForward = new FeedForwardNetwork(embeddingDim, plasticity);
    this.layerNorm1 = new LayerNormalization(embeddingDim);
    this.layerNorm2 = new LayerNormalization(embeddingDim);
  }

  apply(inputs) {
    // console.log("Applying transformer layer:", inputs); // Debug statement
    const attentionOutput = this.multiHeadAttention.apply(inputs);
    const attentionResidual = this.layerNorm1.apply(inputs, attentionOutput);
    const feedForwardOutput = this.feedForward.apply(attentionResidual);
    return this.layerNorm2.apply(attentionResidual, feedForwardOutput);
  }
}

class MultiHeadAttention {
  constructor(numHeads, embeddingDim) {
    this.numHeads = numHeads;
    this.embeddingDim = embeddingDim;
    this.attentionHeads = Array.from(
      { length: numHeads },
      () => new AttentionHead(embeddingDim / numHeads),
    );
    this.linear = new LinearTransformation(embeddingDim, embeddingDim);
  }

  apply(inputs) {
    const headOutputs = this.attentionHeads.map((head) => head.apply(inputs));
    const concatenated = this.concatenate(headOutputs);
    return this.linear.apply(concatenated);
  }

  concatenate(headOutputs) {
    const headOutputLength = headOutputs[0][0].length;
    return Array.from({ length: headOutputs[0].length }, (_, rowIdx) =>
      headOutputs.map((headOutput) => headOutput[rowIdx]),
    );
  }
}

class AttentionHead {
  constructor(dim) {
    this.dim = dim;
    this.queryWeights = this.initWeights(dim);
    this.keyWeights = this.initWeights(dim);
    this.valueWeights = this.initWeights(dim);
    this.outputWeights = this.initWeights(dim);
  }

  initWeights(dim) {
    return Array.from({ length: dim }, () => Math.random() * 0.1);
  }

  apply(inputs) {
    const queries = this.transform(inputs, this.queryWeights);
    const keys = this.transform(inputs, this.keyWeights);
    const values = this.transform(inputs, this.valueWeights);
    return this.transform(values, this.outputWeights);
  }

  transform(inputs, weights) {
    return inputs.map((input) =>
      Array.from({ length: weights.length }, (_, col) =>
        input.reduce((sum, val, rowIdx) => sum + val * weights[rowIdx], 0),
      ),
    );
  }
}

class FeedForwardNetwork {
  constructor(embeddingDim, plasticity) {
    this.embeddingDim = embeddingDim;
    this.plasticity = plasticity;
    this.fc1 = new LinearTransformation(
      embeddingDim,
      embeddingDim * 4,
      plasticity,
    );
    this.fc2 = new LinearTransformation(
      embeddingDim * 4,
      embeddingDim,
      plasticity,
    );
  }

  apply(inputs) {
    const hidden = this.fc1.apply(inputs);
    const activated = hidden.map((row) => row.map((val) => Math.max(0, val)));
    return this.fc2.apply(activated);
  }
}

class LinearTransformation {
  constructor(inputDim, outputDim, plasticity = 0) {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.plasticity = plasticity;
    this.weights = this.initWeights(inputDim, outputDim);
    this.biases = Array.from({ length: outputDim }, () => 0);
  }

  initWeights(inputDim, outputDim) {
    return Array.from({ length: inputDim }, () =>
      Array.from({ length: outputDim }, () => Math.random() * 0.1),
    );
  }

  apply(inputs) {
    return inputs
      .map((row) =>
        Array.from(
          { length: this.outputDim },
          (_, col) =>
            row.reduce(
              (sum, val, rowIdx) => sum + val * this.weights[rowIdx][col],
              0,
            ) + this.biases[col],
        ),
      )
      .map((row) =>
        row.map((val) => val + (Math.random() - 0.5) * this.plasticity),
      );
  }
}

class OutputLayer {
  constructor(embeddingDim, vocabSize) {
    this.embeddingDim = embeddingDim;
    this.vocabSize = vocabSize;
    this.linear = new LinearTransformation(embeddingDim, vocabSize);
  }

  generate(inputs) {
    const logits = this.linear.apply(inputs);
    return this.softmax(logits);
  }

  softmax(logits) {
    return logits.map((row) => {
      const maxLogit = Math.max(...row);
      const expScores = row.map((score) => Math.exp(score - maxLogit));
      const sumExpScores = expScores.reduce((a, b) => a + b, 0);
      return expScores.map((score) => score / sumExpScores);
    });
  }
}

class LayerNormalization {
  constructor(embeddingDim) {
    this.embeddingDim = embeddingDim;
    this.gamma = Array.from({ length: embeddingDim }, () => 1);
    this.beta = Array.from({ length: embeddingDim }, () => 0);
  }

  apply(inputs, residual) {
    const mean = inputs[0].map(
      (_, j) => inputs.reduce((sum, row) => sum + row[j], 0) / inputs.length,
    );
    const variance = inputs[0].map(
      (_, j) =>
        inputs.reduce((sum, row) => sum + (row[j] - mean[j]) ** 2, 0) /
        inputs.length,
    );

    const normalized = inputs.map((row) =>
      row.map((val, j) => (val - mean[j]) / Math.sqrt(variance[j] + 1e-6)),
    );

    return normalized.map((row, i) =>
      row.map((val, j) => val * this.gamma[j] + this.beta[j]),
    );
  }
}

const fs = require("fs");

class Adonis {
  constructor(
    vocabSize,
    embeddingDim,
    numHeads,
    numLayers,
    maxSeqLength,
    plasticity,
  ) {
    this.vocabSize = vocabSize;
    this.embeddingDim = embeddingDim;
    this.numHeads = numHeads;
    this.numLayers = numLayers;
    this.maxSeqLength = maxSeqLength;
    this.plasticity = plasticity;
    this.vocabulary = {}; // Initialize empty vocabulary
    this.tokenizer = new Tokenizer(this.vocabulary);
    this.embeddingLayer = new EmbeddingLayer(
      vocabSize,
      embeddingDim,
      plasticity,
    );
    this.positionalEncoder = new PositionalEncoder(maxSeqLength, embeddingDim);
    this.transformerLayers = this.initTransformerLayers(
      numLayers,
      numHeads,
      embeddingDim,
      plasticity,
    );
    this.outputLayer = new OutputLayer(embeddingDim, vocabSize);
  }

  initTransformerLayers(numLayers, numHeads, embeddingDim, plasticity) {
    return Array.from(
      { length: numLayers },
      () => new TransformerLayer(numHeads, embeddingDim, plasticity),
    );
  }

  inPut(text, reward = 0) {
    const tokens = this.tokenizer.tokenize(text);
    const embeddedTokens = this.embeddingLayer.apply(tokens);
    const positionalEncoded = this.positionalEncoder.apply(embeddedTokens);

    let transformerOutput = positionalEncoded;
    let outputTokens = [];

    while (true) {
      // Apply transformer layers
      for (const layer of this.transformerLayers) {
        transformerOutput = layer.apply(transformerOutput);
      }

      // Generate output logits
      const logits = this.outputLayer.generate(transformerOutput);

      // Get the token with the highest probability
      const predictedTokenId = logits[0].indexOf(Math.max(...logits[0]));

      // Check if the stop token is generated
      if (predictedTokenId === this.tokenizer.vocabulary["<stop>"]) {
        break;
      }

      outputTokens.push(predictedTokenId);

      // Update embeddings based on reward (example)
      this.embeddingLayer.adjustWeightsBasedOnReward(reward);

      // Prepare for the next iteration
      transformerOutput = this.embeddingLayer.apply([predictedTokenId]);
      transformerOutput = this.positionalEncoder.apply(transformerOutput);
    }

    return this.tokenizer.decode(outputTokens);
  }

  // Save the state to a JSON file
  save(filePath) {
    const state = {
      vocabSize: this.vocabSize,
      embeddingDim: this.embeddingDim,
      numHeads: this.numHeads,
      numLayers: this.numLayers,
      maxSeqLength: this.maxSeqLength,
      plasticity: this.plasticity,
      vocabulary: this.tokenizer.vocabulary,
      embeddings: this.embeddingLayer.embeddings,
      positionalEncodings: this.positionalEncoder.positionalEncodings,
      transformerLayers: this.transformerLayers.map((layer) => ({
        multiHeadAttention: {
          numHeads: layer.multiHeadAttention.numHeads,
          embeddingDim: layer.multiHeadAttention.embeddingDim,
        },
        feedForward: {
          embeddingDim: layer.feedForward.embeddingDim,
          plasticity: layer.feedForward.plasticity,
        },
        layerNorm1: {
          gamma: layer.layerNorm1.gamma,
          beta: layer.layerNorm1.beta,
        },
        layerNorm2: {
          gamma: layer.layerNorm2.gamma,
          beta: layer.layerNorm2.beta,
        },
      })),
      outputLayer: {
        embeddingDim: this.outputLayer.embeddingDim,
        vocabSize: this.outputLayer.vocabSize,
      },
    };

    fs.writeFileSync(filePath, JSON.stringify(state, null, 2));
  }

  // Load the state from a JSON file
  load(filePath) {
    const state = JSON.parse(fs.readFileSync(filePath));

    this.vocabSize = state.vocabSize;
    this.embeddingDim = state.embeddingDim;
    this.numHeads = state.numHeads;
    this.numLayers = state.numLayers;
    this.maxSeqLength = state.maxSeqLength;
    this.plasticity = state.plasticity;
    this.tokenizer = new Tokenizer(state.vocabulary);
    this.embeddingLayer = new EmbeddingLayer(
      this.vocabSize,
      this.embeddingDim,
      this.plasticity,
    );
    this.positionalEncoder = new PositionalEncoder(
      this.maxSeqLength,
      this.embeddingDim,
    );
    this.transformerLayers = state.transformerLayers.map((layerData) => {
      const transformerLayer = new TransformerLayer(
        layerData.multiHeadAttention.numHeads,
        layerData.multiHeadAttention.embeddingDim,
        layerData.feedForward.plasticity,
      );
      transformerLayer.multiHeadAttention.attentionHeads.forEach((head, i) => {
        head.queryWeights =
          state.transformerLayers[i].multiHeadAttention.queryWeights;
        head.keyWeights =
          state.transformerLayers[i].multiHeadAttention.keyWeights;
        head.valueWeights =
          state.transformerLayers[i].multiHeadAttention.valueWeights;
        head.outputWeights =
          state.transformerLayers[i].multiHeadAttention.outputWeights;
      });
      transformerLayer.feedForward.fc1.weights =
        state.transformerLayers[i].feedForward.fc1.weights;
      transformerLayer.feedForward.fc1.biases =
        state.transformerLayers[i].feedForward.fc1.biases;
      transformerLayer.feedForward.fc2.weights =
        state.transformerLayers[i].feedForward.fc2.weights;
      transformerLayer.feedForward.fc2.biases =
        state.transformerLayers[i].feedForward.fc2.biases;
      transformerLayer.layerNorm1.gamma =
        state.transformerLayers[i].layerNorm1.gamma;
      transformerLayer.layerNorm1.beta =
        state.transformerLayers[i].layerNorm1.beta;
      transformerLayer.layerNorm2.gamma =
        state.transformerLayers[i].layerNorm2.gamma;
      transformerLayer.layerNorm2.beta =
        state.transformerLayers[i].layerNorm2.beta;
      return transformerLayer;
    });
    this.outputLayer = new OutputLayer(
      state.outputLayer.embeddingDim,
      state.outputLayer.vocabSize,
    );
    this.embeddingLayer.embeddings = state.embeddings;
    this.positionalEncoder.positionalEncodings = state.positionalEncodings;
  }
}

module.exports = Adonis;

module.exports = Adonis;
