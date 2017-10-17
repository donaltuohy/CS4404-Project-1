const ml = require('ml-regression');
const csv = require('csvtojson');
const readline = require('readline'); // For user prompt to allow predictions
const SLR = ml.SLR; // Simple Linear Regression

const csvFilePath = 'kc_house_data.csv'; // Data
let csvData = [], // parsed Data
  X = [], // Input
  y = []; // Output

let regressionModel;

csv()
  .fromFile(csvFilePath)
  .on('json', (jsonObj) => {
    csvData.push(jsonObj);
  })
  .on('done', () => {
    dressData(); // To get data points from JSON Objects
    performRegression();
  });

function dressData() {
  csvData.forEach((row) => {
    X.push(parseFloat(row.price));
    y.push(parseFloat(row.sqft_living));
  });
}

function performRegression() {
  regressionModel = new SLR(X, y); // Train the model on training data
  console.log(regressionModel.toString(3));
  predictOutput();
}

function predictOutput() {
  rl.question('Enter input X for prediction (Press CTRL+C to exit) : ', (answer) => {
    console.log(`At X = ${answer}, y =  ${regressionModel.predict(parseFloat(answer))}`);
    predictOutput();
  });
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

