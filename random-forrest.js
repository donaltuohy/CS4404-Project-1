const CONFIG = require('./config');
const util = require('./util');

const IrisDataset = require('ml-dataset-iris');
const RFClassifier = require('ml-random-forest').RandomForestClassifier;;

const DATASET = CONFIG.ACTIVE_LINEAR_REGRESSION_DATASET;
const TRAINING_PARAMS = CONFIG.TRAINING_PARAMS;


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

  return distinct;
}


function indexClasses(vector) {
  const distinct =  getDistinctClasses(vector);
  
  // Replace class data with index of class
  vector = vector.map(point => {
      return distinct.indexOf(point[0]);
  })

  return vector;
}


function getAccuracy(predictions, actual) {
  let numCorrect = 0;

  for(let i=0; i<predictions.length; i++) {
    if(predictions[i] === actual[i]) {
      numCorrect ++;
    }
  }

  return parseFloat(numCorrect) / predictions.length
}


/*
  Main function
*/
util.readFromCsv.then((readData) => {
  let points = util.createDesignMatrix(readData);
  let classes = util.createLabelVector(readData);
  classes = indexClasses(classes);

  let accuracy;
  if(TRAINING_PARAMS['SPLIT_METHOD'] === 'KFOLD') {
    const accuracies = [];
    for(let i=0; i<TRAINING_PARAMS['NUM_FOLDS']; i++) {
      const splitData = util.splitKFoldCrossVal(points, classes, TRAINING_PARAMS['NUM_FOLDS'], i);
      const pointsTrain = splitData.xTrain;
      const classesTrain = splitData.yTrain;
      const pointsTest = splitData.xTest;
      const classesTest = splitData.yTest;
      const classifier = new RFClassifier();
      classifier.train(pointsTrain, classesTrain);
      
      const predictions = classifier.predict(pointsTest);
      currentAccuracy = getAccuracy(predictions, classesTest);
      console.log("Evaluation fold " + i + ": Accuracy = " + currentAccuracy);
      
      accuracies.push(currentAccuracy)
    
    }

    accuracy = util.getMeanOfVector(accuracies);
  } 
  
  else {
    const splitData = util.split7030(points, classes);
    const pointsTrain = splitData.xTrain;
    let classesTrain = splitData.yTrain;
    const pointsTest = splitData.xTest;
    let classesTest = splitData.yTest;

    console.log("Evaluation with 70/30");
    console.log("Training with pointsTrain = [" + pointsTrain.length + "," + pointsTrain[0].length + "], classesTrain = [" + classesTrain.length + "]");
    const classifier = new RFClassifier();
    classifier.train(pointsTrain, classesTrain);
    console.log("Finished Training");

    
    const predictions = classifier.predict(pointsTest);
    accuracy = getAccuracy(predictions, classesTest)  
  }

console.log(accuracy)

}).catch((err) => {
    console.log(err);
});

