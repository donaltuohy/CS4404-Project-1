const IrisDataset = require('ml-dataset-iris');
const RFClassifier = require('ml-random-forest').RandomForestClassifier;


const CONFIG = require('./config');
const util = require('./util');
const DATASET = CONFIG.ACTIVE_LINEAR_REGRESSION_DATASET;
const TRAINING_PARAMS = CONFIG.TRAINING_PARAMS;

// Returns all distinct elements in the classes vector
function getDistinctClasses(vector) {
  let distinct = [];
  let extracted;
  // Extract data if it is a list of lists
  if(vector[0].length) {
    extracted = vector.map(elem => {
      return elem[0];
    })
  } else {
    extracted = vector;
  }

  // Find distinct elements
  extracted.forEach(element => { 
    if(!distinct.includes(element)) {
      distinct.push(element);
    }
  })

  return distinct.sort();
}

// Returns a vector of classes as indices instead of their numeric value
function indexClasses(vector) {
  const distinct =  getDistinctClasses(vector);
  // Replace class data with index of class
  vector = vector.map(point => {
    return distinct.indexOf(point);
  })

  return vector;
}


// Returns number of times an element is in an array
function getNumOccurences(vector, val) {
  let n = 0;
  vector.forEach((elem) => {
    if(elem === val) {
      n ++;
    }
  })
  
  return n;
}

/*
  Main function
*/
util.readFromCsv.then((readData) => {
  let points = util.createDesignMatrix(readData);
  let classes = util.createLabelVector(readData);
  const mean = util.getMeanOfVector(classes.map(elem => {return elem[0]}))

  // Convert classes into indices 
  classes = util.multiClasstoBinaryClass(classes, mean)
  classes = indexClasses(classes)
 
  let accuracy, precission;
  if(TRAINING_PARAMS['SPLIT_METHOD'] === 'KFOLD') {
    const accuracies = [], precisions = [];
    for(let i=0; i<TRAINING_PARAMS['NUM_FOLDS']; i++) {
      // Partition the data
      const splitData = util.splitKFoldCrossVal(points, classes, TRAINING_PARAMS['NUM_FOLDS'], i);
      const pointsTrain = splitData.xTrain;
      const classesTrain = splitData.yTrain;
      const pointsTest = splitData.xTest;
      const classesTest = splitData.yTest;

      // Train the model
      console.log("Training Fold ", i);
      const classifier = new RFClassifier();
      classifier.train(pointsTrain, classesTrain);
      
      // Predict Test values
      console.log("Predicting fold ", i);
      const predictions = classifier.predict(pointsTest);

      // Compute the metrics for this fold
      const foldMetrics = util.getClassificationMetrics(predictions, classesTest);
      console.log("# Predicted 0 : ", getNumOccurences(predictions, 0), "Actual 0: ", getNumOccurences(classesTest, 0))
      console.log("# Predicted 1 : ", getNumOccurences(predictions, 1), "Actual 1: ", getNumOccurences(classesTest, 1))
      console.log("Fold ", i, " evaluation: Accuracy: ", foldMetrics.accuracy, ", Precision: ", foldMetrics.precision, "\n")
      accuracies.push(foldMetrics.accuracy);
      precisions.push(foldMetrics.precision);
    
    }

    // Compute mean metrics accross folds
    accuracy = util.getMeanOfVector(accuracies);
    console.log(precisions)
    precision = util.getMeanOfVector(precisions);
  } 
  
  else {
    // Partition the data
    const splitData = util.split7030(points, classes);
    const pointsTrain = splitData.xTrain;
    let classesTrain = splitData.yTrain;
    const pointsTest = splitData.xTest;
    let classesTest = splitData.yTest;

    // Train the model
    console.log("Training with 70/30");
    console.log("Training 0s : ", getNumOccurences(classesTrain, 0))
    console.log("Training 1s : ", getNumOccurences(classesTrain, 1))
    const classifier = new RFClassifier();
    classifier.train(pointsTrain, classesTrain);

    // Predict the model
    console.log("Predicting..")
    const predictions = classifier.predict(pointsTest);
    console.log("# Predicted 0 : ", getNumOccurences(predictions, 0), "Actual 0: ", getNumOccurences(classesTest, 0))
    console.log("# Predicted 1 : ", getNumOccurences(predictions, 1), "Actual 1: ", getNumOccurences(classesTest, 1))

    // Evaluate metrics
    const metrics = util.getClassificationMetrics(predictions, classesTest);  
    accuracy = metrics.accuracy;
    precission = metrics.precision;
  }

console.log("Accuracy: ", accuracy, ", Precision: ", precission);

}).catch((err) => {
    console.log(err);
});

