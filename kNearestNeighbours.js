const CONFIG = require('./config');
const util = require('./util');

const KNN = require( 'ml-knn');

const DATASET = CONFIG.ACTIVE_LINEAR_REGRESSION_DATASET;
const TRAINING_PARAMS = CONFIG.TRAINING_PARAMS;


/*
  Returns the % of classifications that were correct
*/
function getAccuracy(predictions, actual) {
  const numElements = predictions.length;
  let numCorrect = 0;
  for(let i=0; i<numElements; i++) {
    const difference = Math.abs(predictions[i] - actual[i]);
    if(difference <= TRAINING_PARAMS['KNN_THRESHOLD']) {
      numCorrect ++;
    } 
  }

  return numCorrect / parseFloat(numElements);
}



/*
  Main function
*/
util.readFromCsv.then((readData) => {
  console.log("Running KNN: K = " + TRAINING_PARAMS['K'] + ", threshold = ", TRAINING_PARAMS['KNN_THRESHOLD'])
  let points = util.createDesignMatrix(readData);
  let classes = util.createLabelVector(readData);
  
  let accuracy;
  if(TRAINING_PARAMS['SPLIT_METHOD'] === 'KFOLD') {
    const accuracies = [];
    for(let i=0; i<TRAINING_PARAMS['NUM_FOLDS']; i++) {
      const splitData = util.splitKFoldCrossVal(points, classes, TRAINING_PARAMS['NUM_FOLDS'], i);
      const pointsTrain = splitData.xTrain;
      const classesTrain = splitData.yTrain;
      const pointsTest = splitData.xTest;
      const classesTest = splitData.yTest;
            
      const knn = new KNN(pointsTrain, classesTrain, {k: TRAINING_PARAMS['K']});
      const predictions = knn.predict(pointsTest);
      const accuracy = getAccuracy(predictions, classesTest);
      console.log("Evaluation fold " + i + " - Accuracy: " + accuracy);
      accuracies.push(accuracy);
    }

    accuracy = util.getMeanOfVector(accuracies);

  } else {
    const splitData = util.split7030(points, classes);
    const pointsTrain = splitData.xTrain;
    const classesTrain = splitData.yTrain;
    const pointsTest = splitData.xTest;
    const classesTest = splitData.yTest;

    console.log("Evaluation with 70/30");
    const knn = new KNN(pointsTrain, classesTrain, {k: TRAINING_PARAMS['K']});
    const predictions = knn.predict(pointsTest);

    accuracy = getAccuracy(predictions, classesTest);    
  }

  console.log("Average Accuracy: ", accuracy);

}).catch((err) => {
    console.log(err);
});

