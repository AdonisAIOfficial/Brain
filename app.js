const readline = require("readline");
const { text2FloatArray, floatArray2Text } = require("./utils/process");
const Brain = require("./utils/Brain"); // Updated Brain class
const fs = require("fs");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function readLine(query) {
  return new Promise((resolve) => {
    rl.question(query, (answer) => {
      resolve(answer);
    });
  });
}
const divider = 10000; // devides the neuron density for performance
const neuronDensity = 10000000 / divider; // average of human brain per cm3 is 10M neurons
const sideLength = 1; // thats in cm
const maxSynapseLength = 0.1;
const contextLength = 50;
const plasticity = 0.01; // keep value low. This is how fast synapses strenghen or weaken
const brain = new Brain(
  neuronDensity,
  sideLength,
  maxSynapseLength,
  contextLength,
  plasticity,
);

try {
  brain.loadModel("./models/model");
  console.log("Loaded model");
} catch (error) {
  console.log("New model created");
  brain.saveModel("./models/model");
  console.log("Model saved");
}

async function main() {
  while (true) {
    let userMessage = await readLine("Message Adonis:\n");
    if (userMessage === "") {
      brain.saveModel("./models/model");
      process.exit();
    }

    let response = await getAdonisResponse(userMessage);
    console.log(`Adonis: ${response.text}`);

    // let guide = await readLine("Guide: ");
    // if (guide != "") {
    //   let guideFloats = text2FloatArray(guide, contextLength);
    //   for (let i = 0; i < trainReps; i++) {
    //     brain.Train([response.floats], [guideFloats], epochs);
    //   }
    // }
  }
}

async function getAdonisResponse(message) {
  const inputFloats = text2FloatArray(message, contextLength);
  const outputFloats = brain.feed(inputFloats);
  const outputText = floatArray2Text(outputFloats);
  console.log(outputFloats);
  return { text: outputText, floats: outputFloats };
}

main();
