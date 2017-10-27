const csv = require('csvtojson');

const CONFIG = require('./config');
const DATASET = CONFIG.ACTIVE_DATASET;
const TRAINING_PARAMS = CONFIG.TRAINING_PARAMS;

 
/*
  Reads in data from file specified by ACTIVE_DATASET (config.js)
*/
let readFromCsv = new Promise(function (resolve, reject) {
  let csvData = [];
  console.log("Reading " + DATASET.FILE_NAME)
  csv({delimiter: DATASET.DELIMETER})
  .fromFile(DATASET.FILE_NAME)
  .on('json', (jsonObj) => {
    csvData.push(jsonObj);
  })
  .on('done', () => {
    resolve(csvData);
  })
  .on('error', (err) => {
    reject(err);
  });
});
 
/*
  Gets desired features as specified in ACTIVE_DATASET (config.js)
*/
function getSelectedFeatures(csvData) {
    const allFeatures = Object.keys(csvData[0]);
    let selectedFeatures = [];
    if(DATASET.OMIT_FEATURES) {
      selectedFeatures = allFeatures.filter(feature => {
        return !DATASET.FEATURES.includes(feature)
      })
    } else {
      selectedFeatures = DATASET['FEATURES'];
    }
    return selectedFeatures;
}

/*
  Allow for a smaller number of instances than what are in file
*/
function getDesiredNumberOfInstances(csvData) {
  if(TRAINING_PARAMS['DESIRED_NUM_INSTANCES']) {
    return Math.min(csvData.length, TRAINING_PARAMS['DESIRED_NUM_INSTANCES']);
  }

  return csvData.length;
}

/*
  Builds standard design matrix from desired features
*/
function createDesignMatrix(csvData) {
  let selectedFeatures = getSelectedFeatures(csvData);
  let designMatrix = [];
  const maxNumInstances = getDesiredNumberOfInstances(csvData);
  for(let i=0; i< maxNumInstances; i++) {
    let newRow = [];
    let row = csvData[i];
    Object.keys(row).map((key,index) => {
        if(selectedFeatures.includes(key)) {
          newRow.push(parseFloat(row[key]))
        }
    });
    designMatrix.push(newRow);
  }
  return designMatrix;
}


/*
  Creates output array of arrays (for library compatability)
*/
function createLabelVector(csvData) {
  let labelVector = [];
  const maxNumInstances = getDesiredNumberOfInstances(csvData);
  for(let i=0; i<maxNumInstances; i++) {
    const parsedFloat = parseFloat(csvData[i][DATASET.LABEL]);
    if(isNaN(parsedFloat)) {
      labelVector.push(csvData[i][DATASET.LABEL]) 
    } else {
      labelVector.push([parsedFloat]); 
    }

  }

  return labelVector;   // remove empty first element
}


/*
  Applies Z score transformation to a matrix
*/
function standardizeData(matrix) {
  const numColumns = matrix[0].length;
  const colVectors = getColumnVectorsFromMatrix(matrix);
  let columnAverages = [];
  let columnStdDevs = [];

  // Get Column Vector averages
  colVectors.forEach(colVector => {
    columnAverages.push(getMeanOfVector(colVector));
  })

  // Get column vectors standard deviations
  colVectors.forEach((colVector, index) => {
    let accumulator = 0;
    colVector.forEach(rowElement => {
      accumulator += Math.pow((rowElement - columnAverages[index]), 2)
    })
    columnStdDevs.push(Math.sqrt(accumulator / matrix.length));
  });

  return matrix.map((row, rowIndex) => {
    return row.map((col, colIndex) => {
      return (col - columnAverages[colIndex]) / columnStdDevs[colIndex]
    })
  })

}


/*
  Returns an array of columns contained in a matrix
*/
function getColumnVectorsFromMatrix(matrix) {
  let colVectors = [];
  for(let columnIndex=0; columnIndex<matrix[0].length; columnIndex++) {
    colVectors.push(matrix.map((row) => { return row[columnIndex] }))
  }
  
  return colVectors;
}


/*
  Splits X and y into xTrain, yTrain, xTest, yTest in a 7:3 ratio
*/
function split7030(X, y) {
  let numTrainInstances = Math.round(X.length*0.7);

  return {
    xTrain: X.slice(0,numTrainInstances),
    yTrain: y.slice(0,numTrainInstances),
    xTest: X.slice(numTrainInstances),
    yTest: y.slice(numTrainInstances)
  }
}


/*
  Splits train/test data in a ratio specified by split factor
  Crossval index allows the test data to start at any multiple of split factor 
  from the begining of the dataset
*/
function splitKFoldCrossVal(X, y, splitFactor, crossValIndex=0) {
    testSetSize = Math.round(X.length / splitFactor);
    startTestIndex = crossValIndex * testSetSize;

    xTrain = [], yTrain = [], xTest = [], yTest = [];

    xTest = X.slice(startTestIndex, startTestIndex + testSetSize);
    yTest = y.slice(startTestIndex, startTestIndex + testSetSize);
    

    xTrain = X.slice(0,startTestIndex);
    yTrain = y.slice(0,startTestIndex);

    xTrain = xTrain.concat(X.slice(startTestIndex+testSetSize));
    yTrain = yTrain.concat(y.slice(startTestIndex+testSetSize));

    return {
        xTrain: xTrain,
        yTrain: yTrain,
        xTest: xTest,
        yTest: yTest
    }
}

/*
  Computes mean of a vector
*/
function getMeanOfVector(vector) {
  let cumulativeSum = 0;
  vector.forEach(element => {
    cumulativeSum += element;
  })

  return cumulativeSum / vector.length;
}

/*
  Converts a multi classification set into a binary classification by using the specified threshold
*/
function multiClasstoBinaryClass(labelVector, threshold) {
  for(let i=0; i<labelVector.length; i++) {
    if(labelVector[i] <= threshold) {
      labelVector[i] = 0;
    } else {
      labelVector[i] = 1
    }
  }

  return labelVector;
}

/*
  Returns an object representation of the confusion matrix
*/
function createConfusionMatrix(predicted, actual) {
  let truePositives = 0, falsePositves = 0, trueNegatives = 0, falseNegatives = 0;
  for(let i=0; i<predicted.length; i++) {
    let predictedValue = predicted[i];
    let wasCorrect = predictedValue === actual[i];
    
    if(wasCorrect && predictedValue === 1) {
      truePositives ++;
    } else if(wasCorrect && predictedValue === 0) {
      trueNegatives ++;
    } else if(!wasCorrect && predictedValue === 1) {
      falseNegatives++;
    } else {
      falsePositves++;
    }
  }
  
  return {
    truePositives,
    trueNegatives,
    falsePositves,
    falseNegatives
  }
}

/*
  Computes accuracy and precission for a given set of test results
*/
function getClassificationMetrics(predicted, actual) {
  const { truePositives, trueNegatives, falsePositves, falseNegatives } = createConfusionMatrix(predicted, actual);
  return {
    accuracy: (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositves + falseNegatives),
    precision: (truePositives) / ((truePositives + falsePositves))
  }
}

module.exports = {
    readFromCsv,
    getSelectedFeatures,
    getDesiredNumberOfInstances,
    createDesignMatrix,
    createLabelVector,
    standardizeData,
    getColumnVectorsFromMatrix,
    split7030,
    splitKFoldCrossVal,
    getMeanOfVector,
    multiClasstoBinaryClass,
    getClassificationMetrics
}