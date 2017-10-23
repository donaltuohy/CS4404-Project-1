const CONFIG = require('./config.js');
const csv = require('csvtojson');
// const SLR = ml.SLR; // Simple Linear Regression
const MLR = require( 'ml-regression-multivariate-linear');


const DATASET = CONFIG.ACTIVE_LINEAR_REGRESSION_DATASET;
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
      selectedFeatures = DATASET[FEATURES];
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
  let designMatrix = [[]];
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
  return designMatrix.splice(1);
}

function createLabelVector(csvData) {
  let labelVector = [[]];
  const maxNumInstances = getDesiredNumberOfInstances(csvData);

  for(let i=0; i<maxNumInstances; i++) {
    labelVector.push([parseFloat(csvData[i][DATASET.LABEL])]); 
  }

  return labelVector.splice(1);   // remove empty first element
}





// Main
readFromCsv.then((readData) => {
  const X = createDesignMatrix(readData);
  const y = createLabelVector(readData);
  console.log("Starting training with X = (" + X.length + "," + X[0].length + "), y = (" + y.length + ")");
  const mlr = new MLR(X,y);
  console.log(mlr.toJSON());
}).catch((err) => {
    console.log(err);
});

