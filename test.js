const KNN = require('ml-knn')

var dataset = [[0,0], [1,1], [10,10], [11,11]];
var predictions = [0,0,1,1];
var knn = new KNN(dataset, predictions, {k: 1});

var dataset = [[0, 1],
               [12, 10]];
 console.log(knn)
var ans = knn.predict(dataset);
console.log(ans);