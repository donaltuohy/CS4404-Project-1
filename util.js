const csv = require('csvtojson');

const CONFIG = require('./config');
// const DATASET = CONFIG.ACTIVE_LINEAR_REGRESSION_DATASET;
const DATASET = CONFIG.ACTIVE_KNN_DATASET;
const TRAINING_PARAMS = CONFIG.TRAINING_PARAMS;

 
let readFromCsv = new Promise(function (resolve, reject) {
  let csvData = [];
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

function getDesiredNumberOfInstances(csvData) {
  if(TRAINING_PARAMS['DESIRED_NUM_INSTANCES']) {
    return Math.min(csvData.length, TRAINING_PARAMS['DESIRED_NUM_INSTANCES']);
  }

  return csvData.length;
}

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

function createLabelVector(csvData) {
  let labelVector = [];
  const maxNumInstances = getDesiredNumberOfInstances(csvData);

  for(let i=0; i<maxNumInstances; i++) {
    labelVector.push([parseFloat(csvData[i][DATASET.LABEL])]); 
  }

  return labelVector;   // remove empty first element
}

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

function getColumnVectorsFromMatrix(matrix) {
  let colVectors = [];
  for(let columnIndex=0; columnIndex<matrix[0].length; columnIndex++) {
    colVectors.push(matrix.map((row) => { return row[columnIndex] }))
  }
  
  return colVectors;
}


function split7030(X, y) {
  let numTrainInstances = Math.round(X.length*0.7);

  return {
    xTrain: X.slice(0,numTrainInstances),
    yTrain: y.slice(0,numTrainInstances),
    xTest: X.slice(numTrainInstances),
    yTest: y.slice(numTrainInstances)
  }
}

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

function getMeanOfVector(vector) {
  let cumulativeSum = 0;
  vector.forEach(element => {
    cumulativeSum += element;
  })

  return cumulativeSum / vector.length;
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
    getMeanOfVector
}