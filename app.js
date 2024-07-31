const readline = require("readline");
const fs = require("fs");
const { text2FloatArray, floatArray2Text } = require("./utils/process");
const Adonis = require("./utils/Adonis.js"); // Import Adonis class

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

// Initialize Adonis with required parameters
const adonis = new Adonis(100, 64, 8, 6, 512, 0.01);

async function main() {
  while (true) {
    let userMessage = await readLine("Message Adonis:\n");
    if (userMessage === "") {
      process.exit();
    }

    let response = await getAdonisResponse(userMessage);
    console.log(`Adonis: ${response}`);

    // let guide = await readLine("Guide: ");
    // if (guide != "") {
    //   let guideFloats = text2FloatArray(guide, contextLength);
    //   for (let i = 0; i < trainReps; i++) {
    //     adonis.Train([response.floats], [guideFloats], epochs);
    //   }
    // }
  }
}

async function getAdonisResponse(message) {
  const response = adonis.inPut(message);
  return response; // Adonis does not return float arrays in this implementation
}

main();
