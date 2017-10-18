const ml = require('ml-regression');
const csv = require('csvtojson');
const SLR = ml.SLR; // Simple Linear Regression

const MLR = require( 'ml-regression-multivariate-linear');

// const x = [[0, 0], [1, 2], [2, 3], [3, 4]];
// // Y0 = X0 * 2, Y1 = X1 * 2, Y2 = X0 + X1
// const y = [[0, 0, 0], [2, 4, 3], [4, 6, 5], [6, 8, 7]];
// const mlr = new MLR(x, y);
// console.log(mlr.predict([3, 3]));
// // [6, 6, 6]


const csvFilePath = 'kc_house_data.csv'; // Data
let csvData = [], // parsed Data
  X = [[]], // Input
  y = [[]]; // Output
let mlr;

const readline = require('readline'); // For user prompt to allow predictions

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

csv()
  .fromFile(csvFilePath)
  .on('json', (jsonObj) => {
    csvData.push(jsonObj);
  })
  .on('done', () => {
    dressData(); // To get data points from JSON Objects
    performRegression();
  });

function performRegression() {
  X.splice(0,1);
  y.splice(0,1);
  mlr = new MLR(X, y); // Train the model on training data
  console.log("Result: ");
  console.log(mlr.weights);
  predictOutput();
}

function dressData() {
  csvData.forEach((row) => {
    X.push([parseFloat(row.sqft_living)]);
    y.push([parseFloat(row.price)]);
  });
}


function predictOutput() {
  rl.question('Enter living square footage (Press CTRL+C to exit) : ', (answer) => {
    console.log(`At X = ${answer}, y =  ${mlr.predict([parseFloat(answer)])}`);
    predictOutput();
  });
}