class Neuron {
  constructor(
    id,
    position,
    type,
    potential = Math.random() * 0.5 + 0.5,
    threshold = Math.random() * 0.5 + 0.5,
  ) {
    this.id = id;
    this.position = position;
    this.type = type; // 'sensory', 'interneuron', 'motor'
    this.potential = potential;
    this.threshold = threshold;
    this.fired = false;
  }

  activate(input) {
    // Activation function can be adjusted based on neuron type
    this.potential += input;
    if (this.potential >= this.threshold) {
      this.fired = true;
    }
  }
}
class Synapse {
  constructor(source, target, weight = Math.random() * 0.1) {
    this.source = source;
    this.target = target;
    this.weight = weight;
  }
}
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
    this.sensoryNeurons = [];
    this.interneurons = [];
    this.motorNeurons = [];
    this.stopNeuron = null;

    this.initBrain();
  }

  initBrain() {
    this.initializeNeurons();
    this.createSynapses();
    this.selectStopNeuron();
  }

  initializeNeurons() {
    for (let x = 0; x < this.neuronDensity; x++) {
      for (let y = 0; y < this.neuronDensity; y++) {
        const position = {
          x: x * (this.sideLength / this.neuronDensity),
          y: y * (this.sideLength / this.neuronDensity),
        };
        let type;

        // Randomly assign neuron types for simplicity
        if (Math.random() < 0.33) type = "sensory";
        else if (Math.random() < 0.66) type = "interneuron";
        else type = "motor";

        const neuron = new Neuron(this.neurons.length, position, type);
        this.neurons.push(neuron);
        if (type === "sensory") this.sensoryNeurons.push(neuron);
        else if (type === "interneuron") this.interneurons.push(neuron);
        else this.motorNeurons.push(neuron);
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
          this.synapses.push(new Synapse(neuronA.id, neuronB.id));
        }
      }
    }
  }

  selectStopNeuron() {
    // Randomly select a neuron as the stop neuron
    this.stopNeuron =
      this.neurons[Math.floor(Math.random() * this.neurons.length)];
  }

  feed(inputValues) {
    if (inputValues.length !== this.sensoryNeurons.length) {
      throw new Error(
        `Input values must be an array of length ${this.sensoryNeurons.length}.`,
      );
    }

    // Set input values to sensory neurons
    this.sensoryNeurons.forEach((neuron, index) =>
      neuron.activate(inputValues[index]),
    );

    // Propagate signals
    this.propagateSignals();

    // Retrieve outputs from motor neurons
    const outputValues = this.motorNeurons.map((neuron) => neuron.potential);

    // Check if stop neuron is fired
    if (this.stopNeuron.fired) {
      // Optionally reset stop neuron state for next iteration
      this.stopNeuron.fired = false;
    }

    return outputValues;
  }

  propagateSignals() {
    let activeNeurons = new Set();
    this.neurons.forEach((neuron) => {
      if (neuron.fired) {
        this.synapses.forEach((synapse) => {
          if (synapse.source === neuron.id) {
            const targetNeuron = this.neurons.find(
              (n) => n.id === synapse.target,
            );
            if (targetNeuron) {
              targetNeuron.activate(neuron.potential * synapse.weight);
              activeNeurons.add(targetNeuron);
            }
          }
        });
      }
    });

    // Update synapse weights and apply plasticity
    this.updateSynapseWeights(activeNeurons);

    // Apply decay to potentials
    this.neurons.forEach((neuron) => (neuron.potential *= 0.999));
  }

  updateSynapseWeights(activeNeurons) {
    activeNeurons.forEach((neuron) => {
      this.synapses.forEach((synapse) => {
        if (synapse.target === neuron.id) {
          const sourceNeuron = this.neurons.find(
            (n) => n.id === synapse.source,
          );
          if (sourceNeuron) {
            synapse.weight +=
              this.plasticity * (sourceNeuron.potential - neuron.potential);
            synapse.weight = Math.max(0, Math.min(1, synapse.weight));
          }
        }
      });
    });
  }
}
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
    this.sensoryNeurons = [];
    this.interneurons = [];
    this.motorNeurons = [];
    this.stopNeuron = null;

    this.initBrain();
  }

  initBrain() {
    this.initializeNeurons();
    this.createSynapses();
    this.selectStopNeuron();
  }

  initializeNeurons() {
    for (let x = 0; x < this.neuronDensity; x++) {
      for (let y = 0; y < this.neuronDensity; y++) {
        const position = {
          x: x * (this.sideLength / this.neuronDensity),
          y: y * (this.sideLength / this.neuronDensity),
        };
        let type;

        // Randomly assign neuron types for simplicity
        if (Math.random() < 0.33) type = "sensory";
        else if (Math.random() < 0.66) type = "interneuron";
        else type = "motor";

        const neuron = new Neuron(this.neurons.length, position, type);
        this.neurons.push(neuron);
        if (type === "sensory") this.sensoryNeurons.push(neuron);
        else if (type === "interneuron") this.interneurons.push(neuron);
        else this.motorNeurons.push(neuron);
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
          this.synapses.push(new Synapse(neuronA.id, neuronB.id));
        }
      }
    }
  }

  selectStopNeuron() {
    // Randomly select a neuron as the stop neuron
    this.stopNeuron =
      this.neurons[Math.floor(Math.random() * this.neurons.length)];
  }

  feed(inputValues) {
    if (inputValues.length !== this.sensoryNeurons.length) {
      throw new Error(
        `Input values must be an array of length ${this.sensoryNeurons.length}.`,
      );
    }

    // Set input values to sensory neurons
    this.sensoryNeurons.forEach((neuron, index) =>
      neuron.activate(inputValues[index]),
    );

    // Propagate signals
    this.propagateSignals();

    // Retrieve outputs from motor neurons
    const outputValues = this.motorNeurons.map((neuron) => neuron.potential);

    // Check if stop neuron is fired
    if (this.stopNeuron.fired) {
      // Optionally reset stop neuron state for next iteration
      this.stopNeuron.fired = false;
    }

    return outputValues;
  }

  propagateSignals() {
    let activeNeurons = new Set();
    this.neurons.forEach((neuron) => {
      if (neuron.fired) {
        this.synapses.forEach((synapse) => {
          if (synapse.source === neuron.id) {
            const targetNeuron = this.neurons.find(
              (n) => n.id === synapse.target,
            );
            if (targetNeuron) {
              targetNeuron.activate(neuron.potential * synapse.weight);
              activeNeurons.add(targetNeuron);
            }
          }
        });
      }
    });

    // Update synapse weights and apply plasticity
    this.updateSynapseWeights(activeNeurons);

    // Apply decay to potentials
    this.neurons.forEach((neuron) => (neuron.potential *= 0.999));
  }

  updateSynapseWeights(activeNeurons) {
    activeNeurons.forEach((neuron) => {
      this.synapses.forEach((synapse) => {
        if (synapse.target === neuron.id) {
          const sourceNeuron = this.neurons.find(
            (n) => n.id === synapse.source,
          );
          if (sourceNeuron) {
            synapse.weight +=
              this.plasticity * (sourceNeuron.potential - neuron.potential);
            synapse.weight = Math.max(0, Math.min(1, synapse.weight));
          }
        }
      });
    });
  }
}
