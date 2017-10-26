const CONFIG = require('./config');
const util = require('./util');

// const SLR = ml.SLR; // Simple Linear Regression
const KNN = require( 'ml-knn');

const DATASET = CONFIG.ACTIVE_LINEAR_REGRESSION_DATASET;
const TRAINING_PARAMS = CONFIG.TRAINING_PARAMS;


/*
  Main function
*/
util.readFromCsv.then((readData) => {
  let points = util.createDesignMatrix(readData);
  let classes = util.createLabelVector(readData);
  
//   points = util.standardizeData(points);
//   classes = util.standardizeData(classes);

  let averageMSE, averageMAE;
  if(TRAINING_PARAMS['SPLIT_METHOD'] === 'KFOLD') {
    const kFoldMSE = [], kFoldMAE = [];
    for(let i=0; i<TRAINING_PARAMS['NUM_SPLITS']; i++) {
      const splitData = util.splitKFoldCrossVal(points, classes, TRAINING_PARAMS['NUM_SPLITS'], i);
      // console.log(splitData);
      const pointsTrain = splitData.xTrain;
      const classesTrain = splitData.yTrain;
      const pointsTest = splitData.xTest;
      const classesTest = splitData.yTest;
      
      console.log("Evaluation fold " + i + " - Training with points = [" + pointsTrain.length + "," + pointsTrain[0].length + "], classes = [" + classesTrain.length + "]");
      
    
    }


  } else {
    const splitData = util.split7030(points, classes);
    const pointsTrain = splitData.xTrain;
    const classesTrain = splitData.yTrain;
    const pointsTest = splitData.xTest;
    const classesTest = splitData.yTest;

    // console.log("Evaluation with 70/30");
    // console.log("Training with pointsTrain = [" + pointsTrain.length + "," + pointsTrain[0].length + "], classesTrain = [" + classesTrain.length + "]");
    // console.log("Testing with pointsTest = [" + pointsTest.length + "," + pointsTest[0].length + "], classesTrain = [" + classesTest.length + "]");
    const knn = new KNN(pointsTrain, classesTrain, {k: 2});
    const predictions = knn.predict(pointsTest);
    console.log(predictions)
    
  }

}).catch((err) => {
    console.log(err);
});

