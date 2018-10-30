import 'bootstrap/dist/css/bootstrap.css';
import * as tf from '@tensorflow/tfjs';

document.getElementById('output').innerText = "set plot";

const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
// Specify loss and optimizer for model
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
// Prepare training data
const input = tf.tensor2d([4, 5, 12, 2], [4, 1]);
const ys = tf.tensor2d([27, 20, 40, 30], [4, 1]);
// Train the model
model.fit(input, ys, {epochs: 500}).then(() => {
    // Use model to predict values
    model.predict(tf.tensor2d([2], [1,1])).print();
});