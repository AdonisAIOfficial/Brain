const fs = require("fs");

// Helper function to initialize weights
function initWeights(inputDim, outputDim) {
  return Array.from({ length: inputDim }, () =>
    Array.from({ length: outputDim }, () => Math.random() * 0.1),
  );
}

// Helper function to initialize biases
function initBiases(dim) {
  return Array.from({ length: dim }, () => 0);
}

class Tokenizer {
  constructor(initialVocabulary = {}) {
    this.vocabulary = { ...initialVocabulary, "<stop>": 0 };
    this.reverseVocabulary = Object.fromEntries(
      Object.entries(this.vocabulary).map(([k, v]) => [v, k]),
    );
    this.nextId = Object.keys(this.vocabulary).length;
  }

  tokenize(text) {
    const tokens = text.toLowerCase().match(/\w+|[^\w\s]/g) || [];
    return tokens.map((token) => {
      if (!(token in this.vocabulary)) {
        this.vocabulary[token] = this.nextId++;
        this.reverseVocabulary[this.vocabulary[token]] = token;
      }
      return this.vocabulary[token];
    });
  }

  decode(ids) {
    return ids.map((id) => this.reverseVocabulary[id] || "<unk>").join(" ");
  }
}

class EmbeddingLayer {
  constructor(vocabSize, embeddingDim) {
    this.embeddingDim = embeddingDim;
    this.embeddings = Array.from({ length: vocabSize }, () =>
      Array.from({ length: embeddingDim }, () => Math.random() * 0.1),
    );
  }

  apply(tokens) {
    return tokens.map(
      (tokenId) =>
        this.embeddings[tokenId] ||
        Array.from({ length: this.embeddingDim }, () => Math.random() * 0.1),
    );
  }
}

class PositionalEncoder {
  constructor(maxSeqLength, embeddingDim) {
    this.positionalEncodings = Array.from({ length: maxSeqLength }, (_, pos) =>
      Array.from(
        { length: embeddingDim },
        (_, i) => pos / Math.pow(10000, (2 * (i / 2)) / embeddingDim),
      ).map((val, i) => (i % 2 === 0 ? Math.sin(val) : Math.cos(val))),
    );
  }

  apply(embeddings) {
    return embeddings.map((embedding, pos) =>
      embedding.map(
        (val, i) => val + (this.positionalEncodings[pos] || [])[i] || 0,
      ),
    );
  }
}

class LinearTransformation {
  constructor(inputDim, outputDim) {
    this.weights = initWeights(inputDim, outputDim);
    this.biases = initBiases(outputDim);
    this.weightGradients = Array.from({ length: inputDim }, () =>
      Array.from({ length: outputDim }, () => 0),
    );
    this.biasGradients = Array.from({ length: outputDim }, () => 0);
  }

  apply(inputs) {
    this.lastInputs = inputs;
    return inputs.map((row) =>
      Array.from(
        { length: this.biases.length },
        (_, col) =>
          row.reduce(
            (sum, val, rowIdx) => sum + val * this.weights[rowIdx][col],
            0,
          ) + this.biases[col],
      ),
    );
  }

  computeGradients(gradients) {
    const inputTranspose = this.lastInputs[0].map((_, colIndex) =>
      this.lastInputs.map((row) => row[colIndex]),
    );

    this.weightGradients = inputTranspose.map((inputRow, rowIndex) =>
      gradients[0].map((_, colIndex) =>
        inputRow.reduce(
          (sum, inputVal, inputIndex) =>
            sum + inputVal * gradients[inputIndex][colIndex],
          0,
        ),
      ),
    );

    this.biasGradients = gradients[0].map((_, colIndex) =>
      gradients.reduce((sum, row) => sum + row[colIndex], 0),
    );
  }

  updateWeights(learningRate) {
    this.weights = this.weights.map((row, rowIndex) =>
      row.map(
        (val, colIndex) =>
          val - learningRate * this.weightGradients[rowIndex][colIndex],
      ),
    );

    this.biases = this.biases.map(
      (val, colIndex) => val - learningRate * this.biasGradients[colIndex],
    );
  }
}
class LayerNormalization {
  constructor(embeddingDim, epsilon = 1e-6) {
    this.embeddingDim = embeddingDim;
    this.epsilon = epsilon; // Small constant to avoid division by zero
    this.gamma = Array.from({ length: embeddingDim }, () => 1); // Scale parameter
    this.beta = Array.from({ length: embeddingDim }, () => 0); // Shift parameter
    this.inputMean = Array.from({ length: embeddingDim }, () => 0);
    this.inputVariance = Array.from({ length: embeddingDim }, () => 0);
  }

  apply(inputs, residual) {
    const normalizedInputs = inputs.map((row) => {
      const mean = row.reduce((sum, val) => sum + val, 0) / this.embeddingDim;
      const variance =
        row.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
        this.embeddingDim;

      this.inputMean = mean;
      this.inputVariance = variance;

      return row.map(
        (val) =>
          ((val - mean) / Math.sqrt(variance + this.epsilon)) *
            this.gamma[row.indexOf(val)] +
          this.beta[row.indexOf(val)],
      );
    });

    // Add the residual connection
    return normalizedInputs.map((row, i) =>
      row.map((val, j) => val + (residual[i] ? residual[i][j] : 0)),
    );
  }

  computeGradients(gradients, inputs) {
    // Gradients for gamma and beta
    const gradGamma = inputs.map((row, rowIdx) =>
      row.map(
        (val, colIdx) =>
          (gradients[rowIdx][colIdx] * (val - this.inputMean)) /
          Math.sqrt(this.inputVariance + this.epsilon),
      ),
    );

    const gradBeta = gradients.map((row, rowIdx) => row.map((val) => val));

    // Update gamma and beta
    this.gamma = this.gamma.map(
      (val, idx) => val - 0.01 * gradGamma.flat()[idx],
    );
    this.beta = this.beta.map((val, idx) => val - 0.01 * gradBeta.flat()[idx]);
  }
}

class AttentionHead {
  constructor(dim) {
    this.dim = dim;
    this.queryWeights = initWeights(dim, dim);
    this.keyWeights = initWeights(dim, dim);
    this.valueWeights = initWeights(dim, dim);
    this.outputWeights = initWeights(dim, dim);

    // Gradients for weights
    this.queryGradients = initWeights(dim, dim);
    this.keyGradients = initWeights(dim, dim);
    this.valueGradients = initWeights(dim, dim);
    this.outputGradients = initWeights(dim, dim);
  }

  apply(inputs) {
    const queries = this.transform(inputs, this.queryWeights);
    const keys = this.transform(inputs, this.keyWeights);
    const values = this.transform(inputs, this.valueWeights);

    const attentionScores = this.computeAttentionScores(queries, keys);
    const weightedValues = this.applyAttentionToValues(attentionScores, values);
    return this.transform(weightedValues, this.outputWeights);
  }

  transform(inputs, weights) {
    return inputs.map((input) =>
      Array.from({ length: weights[0].length }, (_, col) =>
        input.reduce((sum, val, rowIdx) => sum + val * weights[rowIdx][col], 0),
      ),
    );
  }

  computeAttentionScores(queries, keys) {
    return queries.map((query) =>
      keys.map((key) => query.reduce((sum, q, i) => sum + q * key[i], 0)),
    );
  }

  applyAttentionToValues(attentionScores, values) {
    const attentionWeights = this.softmax(attentionScores);
    return attentionWeights.map((weights, i) =>
      weights.reduce(
        (weightedSum, weight, j) =>
          weightedSum.map((val, k) => val + weight * values[j][k]),
        Array.from({ length: values[0].length }, () => 0),
      ),
    );
  }

  softmax(scores) {
    return scores.map((row) => {
      const maxScore = Math.max(...row);
      const expScores = row.map((score) => Math.exp(score - maxScore));
      const sumExpScores = expScores.reduce((a, b) => a + b, 0);
      return expScores.map((score) => score / sumExpScores);
    });
  }

  computeGradients(attentionGradients) {
    // Compute gradients for attention head weights
    const weightGradients = this.transformGradient(
      attentionGradients,
      this.queryWeights,
    );
    this.queryGradients = weightGradients.query;
    this.keyGradients = weightGradients.key;
    this.valueGradients = weightGradients.value;
    this.outputGradients = weightGradients.output;
  }

  transformGradient(attentionGradients, weights) {
    const inputGradients = attentionGradients.map((row, rowIndex) =>
      weights[0].map((_, colIndex) =>
        row.reduce(
          (sum, val, rowIdx) => sum + val * weights[rowIdx][colIndex],
          0,
        ),
      ),
    );

    return {
      query: this.computeWeightGradients(inputGradients, this.queryWeights),
      key: this.computeWeightGradients(inputGradients, this.keyWeights),
      value: this.computeWeightGradients(inputGradients, this.valueWeights),
      output: this.computeWeightGradients(inputGradients, this.outputWeights),
    };
  }

  computeWeightGradients(inputGradients, weights) {
    return weights.map((row, rowIndex) =>
      row.map((_, colIndex) =>
        inputGradients.reduce(
          (sum, inputGrad, rowIdx) =>
            sum + inputGrad[rowIdx] * weights[rowIdx][colIndex],
          0,
        ),
      ),
    );
  }

  updateWeights(learningRate) {
    this.queryWeights = this.queryWeights.map((row, rowIndex) =>
      row.map(
        (val, colIndex) =>
          val - learningRate * this.queryGradients[rowIndex][colIndex],
      ),
    );

    this.keyWeights = this.keyWeights.map((row, rowIndex) =>
      row.map(
        (val, colIndex) =>
          val - learningRate * this.keyGradients[rowIndex][colIndex],
      ),
    );

    this.valueWeights = this.valueWeights.map((row, rowIndex) =>
      row.map(
        (val, colIndex) =>
          val - learningRate * this.valueGradients[rowIndex][colIndex],
      ),
    );

    this.outputWeights = this.outputWeights.map((row, rowIndex) =>
      row.map(
        (val, colIndex) =>
          val - learningRate * this.outputGradients[rowIndex][colIndex],
      ),
    );
  }
}

class MultiHeadAttention {
  constructor(numHeads, embeddingDim) {
    this.numHeads = numHeads;
    this.embeddingDim = embeddingDim;
    this.heads = Array.from(
      { length: numHeads },
      () => new AttentionHead(embeddingDim),
    );
  }

  apply(inputs) {
    const headOutputs = this.heads.map((head) => head.apply(inputs));
    return this.concatenateHeads(headOutputs);
  }

  concatenateHeads(headOutputs) {
    return headOutputs[0].map((_, colIndex) =>
      headOutputs.map((headOutput) => headOutput.map((row) => row[colIndex])),
    );
  }

  computeGradients(attentionGradients) {
    // Compute gradients for all attention heads
    this.heads.forEach((head, index) =>
      head.computeGradients(attentionGradients[index]),
    );
  }

  updateWeights(learningRate) {
    // Update weights for all attention heads
    this.heads.forEach((head) => head.updateWeights(learningRate));
  }
}

class FeedForwardNetwork {
  constructor(embeddingDim) {
    this.fc1 = new LinearTransformation(embeddingDim, embeddingDim * 4);
    this.fc2 = new LinearTransformation(embeddingDim * 4, embeddingDim);
  }

  apply(inputs) {
    const hidden = this.fc1.apply(inputs);
    const reluHidden = hidden.map((row) => row.map((val) => Math.max(0, val)));
    return this.fc2.apply(reluHidden);
  }

  computeGradients(outputGradients) {
    // Backpropagation through feed-forward network
    const fc2Gradients = this.fc2.applyGradient(outputGradients);
    this.fc2.computeGradients(outputGradients);
    const hiddenGradients = this.fc1.applyGradient(fc2Gradients);
    this.fc1.computeGradients(hiddenGradients);
  }

  updateWeights(learningRate) {
    this.fc1.updateWeights(learningRate);
    this.fc2.updateWeights(learningRate);
  }
}

class TransformerLayer {
  constructor(numHeads, embeddingDim) {
    this.attention = new MultiHeadAttention(numHeads, embeddingDim);
    this.feedForward = new FeedForwardNetwork(embeddingDim);
    this.layerNorm1 = new LayerNormalization(embeddingDim);
    this.layerNorm2 = new LayerNormalization(embeddingDim);
  }

  apply(inputs) {
    const attentionOutput = this.attention.apply(inputs);
    const normalizedAttention = this.layerNorm1.apply(attentionOutput, inputs);
    const feedForwardOutput = this.feedForward.apply(normalizedAttention);
    return this.layerNorm2.apply(feedForwardOutput, normalizedAttention);
  }

  computeGradients() {
    // Compute gradients for attention and feed-forward layers
    this.attention.computeGradients();
    this.feedForward.computeGradients();
  }

  updateWeights(learningRate) {
    this.attention.updateWeights(learningRate);
    this.feedForward.updateWeights(learningRate);
  }
}

class OutputLayer {
  constructor(embeddingDim, vocabSize) {
    this.fc = new LinearTransformation(embeddingDim, vocabSize);
  }

  generate(inputs) {
    return this.fc.apply(inputs);
  }

  computeGradients(logits, targets) {
    // Compute gradients for the output layer using cross-entropy loss
    const logitsFlattened = logits.flat();
    const targetsFlattened = targets.flat();
    const outputGradients = logitsFlattened.map((logit, i) => {
      const target = targetsFlattened[i];
      const predicted = Math.exp(logit) / (1 + Math.exp(logit));
      return predicted - target;
    });
    this.fc.computeGradients(outputGradients);
  }

  updateWeights(learningRate) {
    this.fc.updateWeights(learningRate);
  }
}

class Adonis {
  constructor(vocabSize, embeddingDim, numHeads, numLayers, maxSeqLength) {
    this.tokenizer = new Tokenizer();
    this.embeddingLayer = new EmbeddingLayer(vocabSize, embeddingDim);
    this.positionalEncoder = new PositionalEncoder(maxSeqLength, embeddingDim);
    this.transformerLayers = Array.from(
      { length: numLayers },
      () => new TransformerLayer(numHeads, embeddingDim),
    );
    this.outputLayer = new OutputLayer(embeddingDim, vocabSize);
    this.learningRate = 0.01; // Set learning rate
  }

  train(inputs, targets, reps) {
    for (let i = 0; i < reps; i++) {
      const embeddings = this.embeddingLayer.apply(inputs);
      const positionalEncoded = this.positionalEncoder.apply(embeddings);

      let transformerOutput = positionalEncoded;
      for (const layer of this.transformerLayers) {
        transformerOutput = layer.apply(transformerOutput);
      }

      const logits = this.outputLayer.generate(transformerOutput);
      const loss = this.calculateLoss(logits, targets);

      // Compute gradients
      this.computeGradients(logits, targets);

      // Update weights
      this.updateWeights();
    }
  }

  computeGradients(logits, targets) {
    // Compute gradients for output layer
    this.outputLayer.computeGradients(logits, targets);
    // Compute gradients for transformer layers
    for (const layer of this.transformerLayers) {
      layer.computeGradients();
    }
    // Compute gradients for embedding and positional encoding layers if needed
  }

  updateWeights() {
    // Update weights for output layer
    this.outputLayer.updateWeights(this.learningRate);
    // Update weights for transformer layers
    for (const layer of this.transformerLayers) {
      layer.updateWeights(this.learningRate);
    }
    // Update weights for embedding and positional encoding layers if needed
  }

  calculateLoss(logits, targets) {
    // Implement cross-entropy loss
    const logitsFlattened = logits.flat();
    const targetsFlattened = targets.flat();
    return logitsFlattened.reduce(
      (loss, logit, i) =>
        loss - targetsFlattened[i] * Math.log(1 + Math.exp(-logit)),
      0,
    );
  }
}

module.exports = { Adonis };
