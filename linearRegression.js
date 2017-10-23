const ml = require('ml-regression');
const csv = require('csvtojson');
const SLR = ml.SLR; // Simple Linear Regression

const MLR = require( 'ml-regression-multivariate-linear');



// Setup Globals
const testFilePath = 'test.csv';
const filePath = "sum_10k_without_noise.csv";
const readline = require('readline'); 
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});


let readFromCsv = new Promise(function (resolve, reject) {
  let csvData = [];
  csv()
  .fromFile(filePath)
  .on('json', (jsonObj) => {
    csvData.push(jsonObj);
  })
  .on('done', () => {
    console.log("Done");
    // console.log(intermediateData);
    resolve(csvData);
  })
  .on('error', (err) => {
    reject(err);
  });
});



function createDesignMatrix(features, csvData) {
  let designMatrix = [[]];
  csvData.forEach((row) => {
    let newRow = [];
    Object.keys(row).map((key,index) => {
      console.log(features);
      console.log(key);
        if(features.includes(key)) {
          console.log(key)
          newRow.push(row.key)
        }
    });
    designMatrix.push(newRow);
  });
  return designMatrix
}

function doRegression() {
  const mlr = new MLR(X, y);
  console.log(mlr.toJSON())
  console.log(mlr.predict([1, 2, 3]));

}





// Main
readFromCsv.then((readData) => {
  const X = createDesignMatrix(["x1", "x2", "x3"], readData)
  console.log(X);
}).catch((err) => {
    console.log(err);
});


// let csvData = readCSV(testFilePath);
// let X = createDesignMatrix(["x1", "x2", "x3"], csvData);
// console.log(X);












// Basic Linear Regression
// let inputs = [1, 2, 3, 4, 5];
// let outputs = [10, 20, 30, 40, 50];
// let regression = new SLR(inputs, outputs);
// console.log(regression.toString(3));




 
// const x = [
//   [0, 0], 
//   [1, 2], 
//   [2, 3], 
//   [3, 4]
// ];

// // Y = 5 + x1 + x2
// const y = [[0], [8], [10], [12]];










/*
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




function predictOutput() {
  rl.question('Enter living square footage (Press CTRL+C to exit) : ', (answer) => {
    console.log(`At X = ${answer}, y =  ${mlr.predict([parseFloat(answer)])}`);
    predictOutput();
  });
}

*/