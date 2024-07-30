const map = {
  " ": 0,
  A: 1,
  a: 2,
  B: 3,
  b: 4,
  C: 5,
  c: 6,
  D: 7,
  d: 8,
  E: 9,
  e: 10,
  F: 11,
  f: 12,
  G: 13,
  g: 14,
  H: 15,
  h: 16,
  I: 17,
  i: 18,
  J: 19,
  j: 20,
  K: 21,
  k: 22,
  L: 23,
  l: 24,
  M: 25,
  m: 26,
  N: 27,
  n: 28,
  O: 29,
  o: 30,
  P: 31,
  p: 32,
  Q: 33,
  q: 34,
  R: 35,
  r: 36,
  S: 37,
  s: 38,
  T: 39,
  t: 40,
  U: 41,
  u: 42,
  V: 43,
  v: 44,
  W: 45,
  w: 46,
  X: 47,
  x: 48,
  Y: 49,
  y: 50,
  Z: 51,
  z: 52,
  0: 53,
  1: 54,
  2: 55,
  3: 56,
  4: 57,
  5: 58,
  6: 59,
  7: 60,
  8: 61,
  9: 62,
  "!": 63,
  "@": 64,
  "#": 65,
  $: 66,
  "%": 67,
  "^": 68,
  "&": 69,
  "*": 70,
  "(": 71,
  ")": 72,
  "-": 73,
  _: 74,
  "=": 75,
  "+": 76,
  "[": 77,
  "]": 78,
  "{": 79,
  "}": 80,
  "|": 81,
  "\\": 82,
  ";": 83,
  ":": 84,
  "'": 85,
  '"': 86,
  ",": 87,
  ".": 88,
  "<": 89,
  ">": 90,
  "/": 91,
  "?": 92,
  "`": 93,
  "~": 94,
};

// Create a reverse mapping from numeric values to characters
const reverseMap = Object.fromEntries(
  Object.entries(map).map(([char, num]) => [num, char]),
);

function normalizeNumbers(numbers) {
  const min = Math.min(...Object.values(map));
  const max = Math.max(...Object.values(map));
  return numbers.map((num) => (num - min) / (max - min));
}

function denormalizeNumbers(normalizedNumbers) {
  const min = Math.min(...Object.values(map));
  const max = Math.max(...Object.values(map));
  return normalizedNumbers.map((num) => Math.round(num * (max - min) + min));
}

function text2FloatArray(text, length) {
  let numbers = [];
  // If text exceeds length, cut it till it fits
  if (text.length > length) {
    text = text.slice(0, length);
  }
  for (let char of text) {
    if (map[char] !== undefined) {
      numbers.push(map[char]);
    } else {
      throw new Error("The text contains an unsupported character");
    }
  }
  // If text is less than length, fill the FloatArray with 0 till it hits length (without replacing the actual text)
  while (numbers.length < length) {
    numbers.push(0); // Fill with zeros
  }
  numbers = normalizeNumbers(numbers);
  return numbers;
}

function findClosestNumber(value) {
  const values = Object.keys(reverseMap).map(Number);
  let closest = values[0];
  let minDiff = Math.abs(value - closest);

  for (const num of values) {
    const diff = Math.abs(value - num);
    if (diff < minDiff) {
      closest = num;
      minDiff = diff;
    }
  }

  return closest;
}
function floatArray2Text(floatArray) {
  const denormalizedNumbers = denormalizeNumbers(floatArray);
  let text = "";

  for (let num of denormalizedNumbers) {
    let closestNum = reverseMap[num] ? num : findClosestNumber(num);
    text += reverseMap[closestNum] || "🚫"; // Use a placeholder for unsupported numbers
  }

  return text;
}

module.exports = {
  text2FloatArray,
  floatArray2Text,
};
