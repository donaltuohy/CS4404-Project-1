const CONFIG = require('./config');
const util = require('./util');

// const SLR = ml.SLR; // Simple Linear Regression
const MLR = require( 'ml-regression-multivariate-linear');


const DATASET = CONFIG.ACTIVE_LINEAR_REGRESSION_DATASET;
const TRAINING_PARAMS = CONFIG.TRAINING_PARAMS;





/*
  Applies linear model to test features
*/
function predictY(xTest, weights) {
  let predictions = [];
  for(var row=0; row<xTest.length; row++) {
    var sum = 0;
    for(var col=0; col<xTest[0].length; col++) {
      sum += xTest[row][col] * weights[col][0];
    }
    predictions.push(sum);
  }

  return predictions;
}

/*
  Calculates the Mean Square and Absolute Errors (MSE, MAE) 
*/
function evaluateModel(yTest, predictions) {
  let mse = [];
  let mae = [];
  const numFeautres = yTest.length;

  // Calculate individual metrics for each test case
  for(var i=0; i<numFeautres; i++) {
    var difference = yTest[i] - predictions[i];
    mae.push(Math.abs(difference));
    mse.push(Math.pow(difference, 2))
  }

  return {
    mse: util.getMeanOfVector(mse),
    mae: util.getMeanOfVector(mae)
  }
}

function train(xTrain, yTrain) {
  const mlr = new MLR(xTrain,yTrain);
  const weights = mlr.toJSON().weights;

  return weights;
}

function evaluate() {

}



// Main function starts here
util.readFromCsv.then((readData) => {
  let X = util.createDesignMatrix(readData);
  let y = util.createLabelVector(readData);
  
  X = util.standardizeData(X);
  y = util.standardizeData(y);

  let averageMSE, averageMAE;
  if(TRAINING_PARAMS['SPLIT_METHOD'] === 'KFOLD') {
    const kFoldMSE = [], kFoldMAE = [];
    for(let i=0; i<TRAINING_PARAMS['NUM_SPLITS']; i++) {
      const { xTrain, yTrain, xTest, yTest } = util.splitKFoldCrossVal(X, y, TRAINING_PARAMS['NUM_SPLITS'], i);
      console.log("Evaluation fold " + i + " - Training with X = [" + xTrain.length + "," + xTrain[0].length + "], y = [" + yTrain.length + "]");
      
      const weights = train(xTrain, yTrain);

      // Predict the test values
      const predictions = predictY(xTest, weights);

      // Evaluate the model
      const { mse, mae } = evaluateModel(yTest, predictions);
      kFoldMSE.push(mse);
      kFoldMAE.push(mae);
      console.log("Mean Square Error: ", mse);
      console.log("Mean Absolute Error: ", mae, "\n");
    }

    averageMSE = util.getMeanOfVector(kFoldMSE);
    averageMAE = util.getMeanOfVector(kFoldMAE);
  } else {
    const { xTrain, yTrain, xTest, yTest } = util.split7030(X, y);
    console.log("Evaluation with 70/30: training with X = [" + xTrain.length + "," + xTrain[0].length + "], y = [" + yTrain.length + "]");
  
    const weights = train(xTrain, yTrain);
    const predictions = predictY(xTest, weights);
    const evaluation = evaluateModel(yTest, predictions)
    averageMAE = evaluation.mae;
    averageMSE = evaluation.mse;
  }

  console.log("Average MSE: ", averageMSE);
  console.log("AverageMAE: ", averageMAE);
  

  // console.log("Starting training with X = [" + xTrain.length + "," + xTrain[0].length + "], y = [" + yTrain.length + "]");
  // const mlr = new MLR(xTrain,yTrain);
  // const weights = mlr.toJSON().weights;
  // console.log("Weights: ", weights);

  // // Predict the test values
  // const predictions = predictY(xTest, weights);

  // // Evaluate the model
  // const { mse, mae } = evaluateModel(yTest, predictions);
  // console.log("Mean Square Error: ", mse);
  // console.log("Mean Absolute Error: ", mae);

}).catch((err) => {
    console.log(err);
});

