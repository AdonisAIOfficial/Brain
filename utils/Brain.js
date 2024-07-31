const fs = require("fs");

// Your existing encoding and decoding functions
const { text2FloatArray, floatArray2Text } = require("./textEncoding");

// Define the Brain class
class Brain {
  constructor(
    neuronDensity,
    sideLength,
    maxSynapseLength,
    inputSize,
    outputSize,
    plasticity,
  ) {
    this.neuronDensity = neuronDensity;
    this.sideLength = sideLength;
    this.maxSynapseLength = maxSynapseLength;
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.plasticity = plasticity;

    this.neurons = [];
    this.synapses = [];
    this.inputNeurons = [];
    this.outputNeurons = [];

    this.area = Math.pow(this.sideLength, 2);
    this.totalNeurons = Math.floor(this.neuronDensity * this.area);
    this.neuronsPerDimension = Math.floor(Math.sqrt(this.totalNeurons));
    this.spacing = this.sideLength / this.neuronsPerDimension;

    this.initBrain();
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  initBrain() {
    if (this.neurons.length === 0) {
      this.initializeNeurons();
      this.createSynapses();
      this.selectInputAndOutputNeurons();
    }
  }

  initializeNeurons() {
    const offsetRange = this.spacing * 0.1;

    for (let x = 0; x < this.neuronsPerDimension; x++) {
      for (let y = 0; y < this.neuronsPerDimension; y++) {
        const basePosition = {
          x: x * this.spacing,
          y: y * this.spacing,
        };

        const randomOffset = {
          x: (Math.random() - 0.5) * 2 * offsetRange,
          y: (Math.random() - 0.5) * 2 * offsetRange,
        };

        this.neurons.push(
          new Neuron(
            this.neurons.length,
            {
              x: basePosition.x + randomOffset.x,
              y: basePosition.y + randomOffset.y,
            },
            Math.random() * 0.5 + 0.5, // Random threshold for firing
          ),
        );
      }
    }
  }

  createSynapses() {
    for (let i = 0; i < this.neurons.length; i++) {
      for (let j = i + 1; j < this.neurons.length; j++) {
        const neuronA = this.neurons[i];
        const neuronB = this.neurons[j];

        const distance = Math.sqrt(
          Math.pow(neuronA.position.x - neuronB.position.x, 2) +
            Math.pow(neuronA.position.y - neuronB.position.y, 2),
        );

        if (distance <= this.maxSynapseLength) {
          this.synapses.push(
            new Synapse(neuronA.id, neuronB.id, Math.random()),
          );
        }
      }
    }

    console.log(`Initialized ${this.synapses.length} synapses.`);
  }

  selectInputAndOutputNeurons() {
    if (this.neurons.length < this.inputSize + this.outputSize) {
      throw new Error("Not enough neurons to select input and output neurons.");
    }

    // Select input neurons spread out in the network
    const inputInterval = Math.floor(this.neurons.length / this.inputSize);
    for (let i = 0; i < this.inputSize; i++) {
      this.inputNeurons.push(this.neurons[i * inputInterval]);
    }

    // Select output neurons spread out in the network
    const outputInterval = Math.floor(this.neurons.length / this.outputSize);
    for (let i = 0; i < this.outputSize; i++) {
      const neuronIndex = (i + 1) * outputInterval - 1;
      if (neuronIndex < this.neurons.length) {
        this.outputNeurons.push(this.neurons[neuronIndex]);
      } else {
        // Fallback if the calculated index is out of bounds
        this.outputNeurons.push(this.neurons[this.neurons.length - 1]);
      }
    }

    this.inputNeurons.forEach((neuron, index) => {
      console.log(`Input Neuron ${index} ID: ${neuron.id}`);
    });
    this.outputNeurons.forEach((neuron, index) => {
      console.log(`Output Neuron ${index} ID: ${neuron.id}`);
    });
  }

  // Tokenize and encode text input for the Brain
  encodeText(text, length) {
    const tokens = this.tokenize(text);
    return tokens.map((token) => text2FloatArray(token, length));
  }

  // Decode and reconstruct text output from the Brain
  decodeOutput(floatArrays) {
    return floatArrays
      .map((floatArray) => floatArray2Text(floatArray))
      .join("");
  }

  // Tokenize input text into manageable units
  tokenize(text) {
    // Simple space-based tokenization
    return text.split(" ").filter((token) => token.length > 0);
  }

  feed(inputText) {
    // Encode the input text
    const encodedInputs = this.encodeText(inputText, this.inputSize);

    // Check if input size matches
    if (encodedInputs.length !== this.inputNeurons.length) {
      throw new Error(
        `Input values must match the number of input neurons (${this.inputNeurons.length}).`,
      );
    }

    // Set input values to input neurons
    this.inputNeurons.forEach((neuron, index) => {
      neuron.potential = (neuron.potential || 0) + encodedInputs[index];
    });

    // Propagate signals through the network
    this.propagateSignals();

    // Retrieve outputs from output neurons
    const outputValues = this.outputNeurons.map((neuron) =>
      this.sigmoid(neuron.potential),
    );

    // Decode the output values
    const decodedOutput = this.decodeOutput(outputValues);

    return decodedOutput;
  }

  propagateSignals() {
    for (let timeStep = 0; timeStep < this.neurons.length; timeStep++) {
      let activeSynapses = new Set();

      this.neurons.forEach((neuron) => {
        if (neuron.fired) return;

        let incomingSignal = 0;
        this.synapses.forEach((synapse) => {
          if (synapse.target === neuron.id) {
            const fromNeuron = this.neurons.find(
              (n) => n.id === synapse.source,
            );
            if (fromNeuron && fromNeuron.fired) {
              incomingSignal += fromNeuron.potential * synapse.weight;
              activeSynapses.add(synapse);
            }
          }
        });

        neuron.potential += incomingSignal;
        if (neuron.potential >= neuron.threshold) {
          neuron.fired = true;
        }
      });

      this.updateSynapseWeights(activeSynapses);

      // Apply decay to potentials
      this.neurons.forEach((neuron) => {
        neuron.potential *= 0.999; // Decay factor
        neuron.fired = false;
      });

      // Debug: Log neuron potentials
      this.neurons.forEach((neuron) => {
        console.log(`Neuron ${neuron.id} Potential: ${neuron.potential}`);
      });
    }
  }

  updateSynapseWeights(activeSynapses) {
    activeSynapses.forEach((synapse) => {
      const sourceNeuron = this.neurons.find((n) => n.id === synapse.source);
      const targetNeuron = this.neurons.find((n) => n.id === synapse.target);

      if (sourceNeuron && targetNeuron) {
        // Apply plasticity to adjust weights
        synapse.weight +=
          this.plasticity * (sourceNeuron.potential - targetNeuron.potential);

        // Ensure weight stays within a reasonable range (optional)
        synapse.weight = Math.max(0, Math.min(1, synapse.weight));
      }
    });
  }

  saveModel(filePath) {
    const model = {
      neuronDensity: this.neuronDensity,
      sideLength: this.sideLength,
      maxSynapseLength: this.maxSynapseLength,
      plasticity: this.plasticity,
      neurons: this.neurons,
      synapses: this.synapses,
      inputNeurons: this.inputNeurons.map((neuron) => neuron.id),
      outputNeurons: this.outputNeurons.map((neuron) => neuron.id),
    };

    fs.writeFileSync(filePath, JSON.stringify(model, null, 2));
    console.log(`Model saved to ${filePath}`);
  }

  loadModel(filePath) {
    const model = JSON.parse(fs.readFileSync(filePath));

    this.neuronDensity = model.neuronDensity;
    this.sideLength = model.sideLength;
    this.maxSynapseLength = model.maxSynapseLength;
    this.plasticity = model.plasticity;

    this.neurons = model.neurons.map(
      (neuron) => new Neuron(neuron.id, neuron.position, neuron.threshold),
    );
    this.synapses = model.synapses.map(
      (synapse) => new Synapse(synapse.source, synapse.target, synapse.weight),
    );
    this.inputNeurons = model.inputNeurons.map((id) =>
      this.neurons.find((neuron) => neuron.id === id),
    );
    this.outputNeurons = model.outputNeurons.map((id) =>
      this.neurons.find((neuron) => neuron.id === id),
    );

    console.log(`Model loaded from ${filePath}`);
  }
}

// Define Neuron and Synapse classes
class Neuron {
  constructor(id, position, threshold) {
    this.id = id;
    this.position = position;
    this.threshold = threshold;
    this.potential = 0;
    this.fired = false;
  }
}

class Synapse {
  constructor(source, target, weight) {
    this.source = source;
    this.target = target;
    this.weight = weight;
  }
}

// Export the Brain class
module.exports = Brain;
