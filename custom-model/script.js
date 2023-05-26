/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *ls
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const TRAINING_DATA = [];

const PREDICT_DATA = [6, 10, 13, 15];

// Generate input numbers from 1 to 20 inclusive
const INPUTS = [];
for (let n = 1; n<=20; n++) {
  INPUTS.push(n);
  TRAINING_DATA.push({ x: n });
}

// // Current listed house prices in dollars given their features above (target output values you want to predict).

// Generate input numbers from 1 to 20 inclusive
const OUTPUTS = [];
for (let n = 1; n<=20; n++) {
  OUTPUTS.push(2*n);
  TRAINING_DATA[n-1].y = 2*n;

  // OUTPUTS.push(n*n);
  // TRAINING_DATA[n-1].y = n*n;
}

// Shuffle the two arrays to remove any order, but do so in the same way so 
// inputs still match outputs indexes.
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// Input feature Array of Arrays needs 1D tensor to store.
const INPUTS_TENSOR = tf.tensor1d(INPUTS);

// Output can stay 1 dimensional.
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

// Function to take a Tensor and normalize values
// with respect to each column of values contained in that Tensor.
function normalize(tensor, min, max) {
  const result = tf.tidy(function() {
    // Find the minimum value contained in the Tensor.
    const MIN_VALUES = min || tf.min(tensor, 0);

    // Find the maximum value contained in the Tensor.
    const MAX_VALUES = max || tf.max(tensor, 0);

    // Now calculate subtract the MIN_VALUE from every value in the Tensor
    // And store the results in a new Tensor.
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

    // Calculate the range size of possible values.
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    // Calculate the adjusted values divided by the range size as a new Tensor.
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);
    
    // Return the important tensors.
    return {NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES};
  });
  return result;
}


// Normalize all input feature arrays and then dispose of the original non normalized Tensors.
const FEATURE_RESULTS = normalize(INPUTS_TENSOR);
console.log('Normalized Values:');
FEATURE_RESULTS.NORMALIZED_VALUES.print();

console.log('Min Values:');
FEATURE_RESULTS.MIN_VALUES.print();

console.log('Max Values:');
FEATURE_RESULTS.MAX_VALUES.print();

INPUTS_TENSOR.dispose();


// Now actually create and define model architecture.
const model = tf.sequential();

// We will use one dense layer with 1 neuron (units) and an input of 
// 1 input feature values (representing input number).
model.add(tf.layers.dense({inputShape: [1], units: 1}));
model.add(tf.layers.dense({units: 1}));

// model.add(tf.layers.dense({inputShape: [1], units: 100, activation: 'relu'}));
// model.add(tf.layers.dense({units: 100, activation: 'relu'}));
// model.add(tf.layers.dense({units: 1}));

tfvis.show.modelSummary({name: 'Model Summary'}, model);

train();


async function train() {
  // Choose a learning rate that is suitable for the data we are using.
  const LEARNING_RATE = 0.01;
  
  // Compile the model with the defined learning rate and specify
  // our loss function to use.
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: 'meanSquaredError'
  });

  // Finally do the training itself 
  let results = await model.fit(FEATURE_RESULTS.NORMALIZED_VALUES, OUTPUTS_TENSOR, {
    shuffle: true,         // Ensure data is shuffled again before using each epoch.
    batchSize: 2,         // As we have a lot of training data, batch size is set to 64.
    epochs: 200,             // Go over the data 200 times!
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
  
  OUTPUTS_TENSOR.dispose();
  FEATURE_RESULTS.NORMALIZED_VALUES.dispose();
  
  console.log("Average error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1]));
  // console.log("Average validation error loss: " + Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1]));
    
  // Once trained we can evaluate the model.
  evaluate();
}

function evaluate() {
  // Predict answer for a single piece of data.
  const preds = tf.tidy(function() {
    let newInput = normalize(tf.tensor1d(PREDICT_DATA), FEATURE_RESULTS.MIN_VALUES, FEATURE_RESULTS.MAX_VALUES);
    let output = model.predict(newInput.NORMALIZED_VALUES);
  
    return output.dataSync();
  });
  
  const predictedPoints = Array.from(PREDICT_DATA).map((val, i) => {
    return {x: val, y: preds[i]}
  });

  FEATURE_RESULTS.MIN_VALUES.dispose();
  FEATURE_RESULTS.MAX_VALUES.dispose();
  model.dispose();

  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data'},
    {values: [TRAINING_DATA, predictedPoints], series: ['original', 'predicted']},
    {
      xLabel: 'X',
      yLabel: 'Y',
      height: 300
    }
  );
}
