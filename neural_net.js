function NeuralNet(){
  this.input_neurons = 1182;
  this.hidden_neurons = 15;
  this.output_neurons = 4;

  this.W_0 = math.zeros([this.hidden_neurons, this.input_neurons]);
  this.W_1 = math.zeros([this.output_neurons, this.hidden_neurons]);

  this.B_0 = math.zeros([this.hidden_neurons, 1]);
  this.B_1 = math.zeros([this.output_neurons, 1]);

  this.Input = [];
  this.Output_hidden = [];
  this.Output = [];

  /** Runs one forward iteration of the neural network.
  * @param {array} input_vector A vector to be inputted into the neural network
  * @return {array} A vector representing the output of the neural network
  */
  this.normal_forward = function(input_vector){
    this.Input = [];
    this.Output_hidden = [];
    this.Output = [];
    return math.transpose(this.forward(math.transpose([input_vector])))[0];
  }

  this.forward = function(input_vector){
    this.Input.push(input_vector);

    var z_hidden = math.add(math.multiply(this.W_0, input_vector), this.B_0);
    var output_hidden = math.tanh(z_hidden);

    this.Output_hidden.push(output_hidden);

    var z_output = math.add(math.multiply(this.W_1, output_hidden), this.B_1);
    var output = math.tanh(z_output)

    this.Output.push(output);

    return output
  }

  /** Runs one backwards iteration of the neural network to tune the weights to
  *   the desired output
  * @param {array} expected A vector representing the expected output of the neural network.
  */
  this.backward = function(expected){
    var delta_output = math.subtract(expected, this.Output[0]);

    var activation_derivative = math.map(this.Output[0], function(value){
      return 1 - math.pow(value, 2);
    });
    delta_output = math.dotMultiply(delta_output, activation_derivative);


    //Ouptut Layer
    var W_1_update = math.multiply(delta_output, math.transpose(this.Output_hidden[0]));

    this.W_1 = math.add(this.W_1, W_1_update);

    //Hidden Layer
    var delta_hidden = math.multiply(math.transpose(this.W_1), delta_output);

    var W_0_update = math.multiply(delta_hidden, math.transpose(this.Input[0]));
    this.W_0 = math.add(this.W_0, W_0_update);
  }

  /** Mini-btach training by running one forward pass and one backwards pass
  *   of the neural network.
  * @param {array} word_vector A vector representing the input vector to the neural network
  * @param {array} expected_vector A vector representing the expected output of the neural network
  *
  */
  this.train = function(word_vector, expected_vector){
    this.Input = [];
    this.Output_hidden = [];
    this.Output = [];


    var word_vector = math.transpose([word_vector]);
    var expected_vector = math.transpose([expected_vector]);

    this.forward(word_vector);
    this.backward(expected_vector);
  }

}
