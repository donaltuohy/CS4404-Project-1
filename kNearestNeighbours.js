const KNN = require( 'ml-knn');

const CONFIG = require('./config');
const util = require('./util');
const TRAINING_PARAMS = CONFIG.TRAINING_PARAMS;


/*
  Returns the % of classifications that were correct
*/
function getAccuracy(predictions, actual) {
  const numElements = predictions.length;

  let numCorrect = 0;
  for(let i=0; i<numElements; i++) {
    if(typeof actual[i] === "string") {
      if(actual[i] === predictions[i]) {
        numCorrect ++;
      }
    } else {
      const difference = Math.abs(predictions[i] - actual[i]);
      if(difference <= TRAINING_PARAMS['KNN_THRESHOLD']) {
        numCorrect ++;
      } 
    }

  }

  return numCorrect / parseFloat(numElements);
}



/*
  Main function
*/
util.readFromCsv.then((readData) => {
  console.log("Running KNN: K = " + TRAINING_PARAMS['K'])
  let points = util.createDesignMatrix(readData);
  let classes = util.createLabelVector(readData);
  classes = util.multiClasstoBinaryClass(classes, util.getMeanOfVector(classes.map(classAsArr => { return classAsArr[0] })));
 
  console.log("Starting Evaluation")
  let accuracy;
  if(TRAINING_PARAMS['SPLIT_METHOD'] === 'KFOLD') {
    const accuracies = [], precisions = [];
    for(let i=0; i<TRAINING_PARAMS['NUM_FOLDS']; i++) {
      
      // Partition the data
      const splitData = util.splitKFoldCrossVal(points, classes, TRAINING_PARAMS['NUM_FOLDS'], i);
      const pointsTrain = splitData.xTrain;
      const classesTrain = splitData.yTrain;
      const pointsTest = splitData.xTest;
      const classesTest = splitData.yTest;
            
      // Predict 
      const knn = new KNN(pointsTrain, classesTrain, {k: TRAINING_PARAMS['K']});
      console.log("Predicting ", i);
      const predictions = knn.predict(pointsTest);
   
      const foldMetrics = util.getClassificationMetrics(predictions, classesTest);
      console.log("Fold " + i + " - Accuracy: " + foldMetrics.accuracy + ", Precision: " + foldMetrics.precision);
      accuracies.push(foldMetrics.accuracy);
      precisions.push(foldMetrics.precision)
    }

    accuracy = util.getMeanOfVector(accuracies);
    precision = util.getMeanOfVector(precisions);

  } else {
    const splitData = util.split7030(points, classes);
    const pointsTrain = splitData.xTrain;
    const classesTrain = splitData.yTrain;
    const pointsTest = splitData.xTest;
    const classesTest = splitData.yTest;
    console.log("Evaluation with 70/30");

    const knn = new KNN(pointsTrain, classesTrain, {k: TRAINING_PARAMS['K']});
    const predictions = knn.predict(pointsTest);
    const metrics = util.getClassificationMetrics(predictions, classesTest); 
      
    accuracy = metrics.accuracy;
    precision = metrics.precision;
  }

  console.log("Accuracy: ", accuracy, ", Precision: ", precision);

}).catch((err) => {
    console.log(err);
});

