const CONFIG = require('./config');
const util = require('./util');

const IrisDataset = require('ml-dataset-iris');
const RFClassifier = require('ml-random-forest').RandomForestClassifier;;

const DATASET = CONFIG.ACTIVE_LINEAR_REGRESSION_DATASET;
const TRAINING_PARAMS = CONFIG.TRAINING_PARAMS;


function getDistinctClasses(vector) {
  let distinct = [];
  vector.forEach(element => {
    if(!distinct.includes(element)) {
      distinct.push(element);
    }
  })

  return distinct;
}

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
    let classesTrain = splitData.yTrain;
    const pointsTest = splitData.xTest;
    let classesTest = splitData.yTest;

    const distinct =  getDistinctClasses(classesTrain);
    classesTrain = classesTrain.map(point => {
        return distinct.indexOf(point);
    })

    classesTest = classesTest.map(point => {
        return distinct.indexOf(point);
    })



    console.log("Evaluation with 70/30");
    console.log("Training with pointsTrain = [" + pointsTrain.length + "," + pointsTrain[0].length + "], classesTrain = [" + classesTrain.length + "]");


    const classifier = new RFClassifier();
    classifier.train(pointsTrain, classesTrain);

    console.log("Finished Training");
    const predictions = classifier.predict(pointsTest);

    let numCorrect = 0;

    for(let i=0; i<predictions.length; i++) {
      if(predictions[i] === classesTest[i]) {
        numCorrect ++;
      }
    }

    const accuracy = parseFloat(numCorrect) / predictions.length
    console.log(accuracy)
    
  }

}).catch((err) => {
    console.log(err);
});

