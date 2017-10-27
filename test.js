const IrisDataset =  require('ml-dataset-iris');
const RFClassifier = require('ml-random-forest').RandomForestClassifier;
 
var trainingSet = IrisDataset.getNumbers();
var predictions = IrisDataset.getClasses().map(
    (elem) => IrisDataset.getDistinctClasses().indexOf(elem)
);
 

var options = {
    seed: 3,
    maxFeatures: 0.8,
    replacement: true,
    nEstimators: 25
};
//  console.log(trainingSet);
//  console.log(predictions)
var classifier = new RFClassifier();
console.log("trainig with: (", trainingSet.length, ",", trainingSet[0].length, ")")
console.log(predictions.length)
classifier.train(trainingSet, predictions);
var result = classifier.predict(trainingSet);
// console.log(result)