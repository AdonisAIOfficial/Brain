const readline = require("readline");
const fs = require("fs");
const { Adonis } = require("./utils/Adonis.js"); // Import Adonis class

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

// Create an instance of Adonis with appropriate parameters
// const adonis = new Adonis(10000, 512, 16, 12, 256, 2048);
const adonis = new Adonis(10000, 64, 4, 4, 128, 512);

(async () => {
  try {
    // Load training text if model is not pre-trained (no loading/saving implemented)
    const trainingText = fs.readFileSync("./trainingText.txt", "utf-8");
    await adonis.feed(trainingText);
    console.log("Model trained.");
  } catch (error) {
    console.error("Error training model:", error);
    process.exit(1);
  }

  async function main() {
    while (true) {
      let userMessage = await readLine("Message Adonis:\n");
      if (userMessage === "") {
        console.log("Exiting...");
        process.exit();
      }

      let response = await getAdonisResponse(userMessage);
      console.log(`Adonis: ${response}`);
    }
  }

  async function getAdonisResponse(message) {
    try {
      const response = await adonis.predict(message);
      return response;
    } catch (error) {
      console.error("Error generating response:", error);
      return "Sorry, I couldn't understand that.";
    }
  }

  await main();
})();
