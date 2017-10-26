const CONFIG = require('./config');
const util = require('./util');

const IrisDataset = require('ml-dataset-iris');
const RFClassifier = require('ml-random-forest').RandomForestClassifier;;

const DATASET = CONFIG.ACTIVE_LINEAR_REGRESSION_DATASET;
const TRAINING_PARAMS = CONFIG.TRAINING_PARAMS;


/*
  Main function
*/
util.readFromCsv.then((readData) => {
  let points = util.createDesignMatrix(readData);
  let classes = util.createLabelVector(readData);

  let averageMSE, averageMAE;
  if(TRAINING_PARAMS['SPLIT_METHOD'] === 'KFOLD') {
    const kFoldMSE = [], kFoldMAE = [];
    for(let i=0; i<TRAINING_PARAMS['NUM_SPLITS']; i++) {
      const splitData = util.splitKFoldCrossVal(points, classes, TRAINING_PARAMS['NUM_SPLITS'], i);
      const pointsTrain = splitData.xTrain;
      const classesTrain = splitData.yTrain;
      const pointsTest = splitData.xTest;
      const classesTest = splitData.yTest;
      
      console.log("Evaluation fold " + i + " - Training with points = [" + pointsTrain.length + "," + pointsTrain[0].length + "], classes = [" + classesTrain.length + "]");
      
    
    }


  } 
  
  else {
    const splitData = util.split7030(points, classes);
    const pointsTrain = splitData.xTrain;
    const classesTrain = splitData.yTrain;
    const pointsTest = splitData.xTest;
    const classesTest = splitData.yTest;

    console.log("Evaluation with 70/30");
    console.log("Training with pointsTrain = [" + pointsTrain.length + "," + pointsTrain[0].length + "], classesTrain = [" + classesTrain.length + "]");

    var options = {
        seed: 3,
        maxFeatures: 0.8,
        replacement: true,
        nEstimators: 25
    };
    
    var classifier = new RFClassifier();
    classifier.train(pointsTrain, classesTrain);
    var result = classifier.predict(classesTest);

    
    console.log(result);
  }

}).catch((err) => {
    console.log(err);
});

