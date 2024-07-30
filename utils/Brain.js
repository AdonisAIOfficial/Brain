const fs = require("fs");

class Brain {
  constructor(
    neuronDensity,
    sideLength,
    maxSynapseLength,
    contextLength,
    plasticity,
  ) {
    this.neuronDensity = neuronDensity;
    this.sideLength = sideLength;
    this.maxSynapseLength = maxSynapseLength;
    this.contextLength = contextLength;
    this.plasticity = plasticity;

    this.neurons = [];
    this.synapses = [];
    this.entryNeurons = [];
    this.exitNeuron = null;

    this.volume = Math.pow(this.sideLength, 3);
    this.totalNeurons = Math.floor(this.neuronDensity * this.volume);
    this.neuronsPerDimension = Math.floor(Math.cbrt(this.totalNeurons));
    this.spacing = this.sideLength / this.neuronsPerDimension;

    this.initBrain();
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  initBrain() {
    this.initializeNeurons();
    this.createSynapses();
    this.selectEntryAndExitNeurons();
  }

  initializeNeurons() {
    const offsetRange = this.spacing * 0.1;

    for (let x = 0; x < this.neuronsPerDimension; x++) {
      for (let y = 0; y < this.neuronsPerDimension; y++) {
        for (let z = 0; z < this.neuronsPerDimension; z++) {
          const basePosition = {
            x: x * this.spacing,
            y: y * this.spacing,
            z: z * this.spacing,
          };

          const randomOffset = {
            x: (Math.random() - 0.5) * 2 * offsetRange,
            y: (Math.random() - 0.5) * 2 * offsetRange,
            z: (Math.random() - 0.5) * 2 * offsetRange,
          };

          this.neurons.push({
            id: this.neurons.length,
            position: {
              x: basePosition.x + randomOffset.x,
              y: basePosition.y + randomOffset.y,
              z: basePosition.z + randomOffset.z,
            },
            potential: 0, // Membrane potential
            threshold: Math.random() * 0.5 + 0.5, // Random threshold for firing
            fired: false,
          });
        }
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
            Math.pow(neuronA.position.y - neuronB.position.y, 2) +
            Math.pow(neuronA.position.z - neuronB.position.z, 2),
        );

        if (distance <= this.maxSynapseLength) {
          this.synapses.push({
            from: neuronA.id,
            to: neuronB.id,
            weight: Math.random(),
          });
        }
      }
    }

    console.log(`Initialized ${this.synapses.length} synapses.`);
  }

  selectEntryAndExitNeurons() {
    if (this.neurons.length < this.contextLength + 1) {
      throw new Error("Not enough neurons to select entry and exit neurons.");
    }

    // Select entry neurons spread out in the network
    const interval = Math.floor(this.neurons.length / this.contextLength);
    for (let i = 0; i < this.contextLength; i++) {
      this.entryNeurons.push(this.neurons[i * interval]);
    }

    // Select an exit neuron that is not one of the entry neurons
    let exitNeuronIndex;
    do {
      exitNeuronIndex = Math.floor(Math.random() * this.neurons.length);
    } while (this.entryNeurons.some((neuron) => neuron.id === exitNeuronIndex));

    this.exitNeuron = this.neurons[exitNeuronIndex];

    this.entryNeurons.forEach((neuron, index) => {
      console.log(`Entry Neuron ${index} ID: ${neuron.id}`);
    });
    console.log(`Exit Neuron ID: ${this.exitNeuron.id}`);
  }

  feed(inputValues) {
    if (
      !Array.isArray(inputValues) ||
      inputValues.length !== this.contextLength
    ) {
      throw new Error(
        `Input values must be an array of length ${this.contextLength}.`,
      );
    }

    // Set input values to entry neurons
    this.entryNeurons.forEach((neuron, index) => {
      neuron.potential += inputValues[index];
    });

    // Propagate signals
    this.propagateSignals();

    // Apply sigmoid function to the exit neuron's potential
    return this.sigmoid(this.exitNeuron.potential);
  }

  propagateSignals() {
    for (let timeStep = 0; timeStep < this.neurons.length; timeStep++) {
      let activeSynapses = new Set();

      this.neurons.forEach((neuron) => {
        if (neuron.fired) return;

        let incomingSignal = 0;
        this.synapses.forEach((synapse) => {
          if (synapse.to === neuron.id) {
            const fromNeuron = this.neurons.find((n) => n.id === synapse.from);
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
        neuron.potential *= 0.9; // Decay factor
        neuron.fired = false;
      });
    }
  }

  updateSynapseWeights(activeSynapses) {
    this.synapses.forEach((synapse) => {
      const fromNeuron = this.neurons.find((n) => n.id === synapse.from);
      const toNeuron = this.neurons.find((n) => n.id === synapse.to);

      if (activeSynapses.has(synapse)) {
        // Hebbian learning: strengthen synapse if both neurons are active
        if (fromNeuron.fired && toNeuron.fired) {
          synapse.weight +=
            this.plasticity * (fromNeuron.potential / toNeuron.potential);
        } else {
          synapse.weight += this.plasticity;
        }
      } else {
        // Apply long-term depression if neurons are not active together
        synapse.weight -= this.plasticity * 0.5;
      }

      // Clamp the weight to the range [0, 1]
      synapse.weight = Math.max(0, Math.min(1, synapse.weight));
    });

    console.log(`Updated ${this.synapses.length} synapses.`);
  }

  saveModel(filePath) {
    const model = {
      neuronDensity: this.neuronDensity,
      sideLength: this.sideLength,
      maxSynapseLength: this.maxSynapseLength,
      plasticity: this.plasticity,
      neurons: this.neurons,
      synapses: this.synapses,
      entryNeurons: this.entryNeurons.map((neuron) => neuron.id),
      exitNeuron: this.exitNeuron.id,
    };

    fs.writeFileSync(filePath, JSON.stringify(model, null, 2));
    console.log(`Model saved to ${filePath}`);
  }

  loadModel(filePath) {
    const model = JSON.parse(fs.readFileSync(filePath, "utf8"));

    this.neuronDensity = model.neuronDensity;
    this.sideLength = model.sideLength;
    this.maxSynapseLength = model.maxSynapseLength;
    this.plasticity = model.plasticity;
    this.neurons = model.neurons;
    this.synapses = model.synapses;
    this.entryNeurons = model.entryNeurons.map((id) =>
      this.neurons.find((neuron) => neuron.id === id),
    );
    this.exitNeuron = this.neurons.find(
      (neuron) => neuron.id === model.exitNeuron,
    );

    console.log(`Model loaded from ${filePath}`);
  }
}

module.exports = Brain;
