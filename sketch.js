var weights_0;
var weights_1;
var baises_0;
var biases_1;
var net;
var txt_file;
var word_dict;

var sentiment = [];
var time = 0;
var showOverallSentiment = false;

function preload() {
  net = new NeuralNet()

  word_dict = loadStrings("neural_net_data/word_dict.txt", loadedWordDict);

  weights_0 = loadStrings("neural_net_data/w_0.txt", loadedW_0);
  weights_1 = loadStrings("neural_net_data/w_1.txt", loadedW_1);

  biases_0 = loadStrings("neural_net_data/b_0.txt", loadedB_0);
  biases_1 = loadStrings("neural_net_data/b_1.txt", loadedB_1);

  txt_file = loadStrings("smith.txt");
}

/**
* @callback Callback function that loads the word vector file
*/
function loadedWordDict(){
  word_dict = JSON.parse(word_dict[0]);
}

/**
* @callback Load the pretrained weights for the neural network and put them into the
*           neural network
*/
function loadedW_0(){
  for(var i in weights_0){
    var weights = str(weights_0[i]).split(" ");
    for(var j in weights){
      net.W_0[i][j] = float(weights[j]);
    }
  }
}
/**
* @callback Load the pretrained weights for the neural network and put them into the
*           neural network
*/
function loadedW_1(){
  for(var i in weights_1){
    var weights = str(weights_1[i]).split(" ");
    for(var j in weights){
      net.W_1[i][j] = float(weights[j]);
    }
  }
}
/**
* @callback Load the pretrained biases for the neural network and put them into the
*           neural network
*/
function loadedB_0(){
  for(var i in biases_0){
    net.B_0[i][0] = float(biases_0[i]);
  }
}
/**
* @callback Load the pretrained biases for the neural network and put them into the
*           neural network
*/
function loadedB_1(){
  for(var i in biases_1){
    net.B_1[i][0] = float(biases_1[i]);
  }
}

/** Converts a string into a word vector that can be used by the neural network
* @param {string} text text to be converted into a vector
* @return {array} A word vector that represents the string
*/
function text2vec(text){
  var vector = math.zeros([1182])
  var words = RiTa.tokenize(text)

  for(var i = 0; i < words.length; i++){
    if(word_dict[words[i]] !== undefined){
      vector[word_dict[words[i]]] = 1;
    }
  }

  return vector;
}

/** Takes a piece of text and runs it through the neural network
* @param {string} text Text to be analyzed
* @return {array} A vector of sentiments computed by the neural network
*/
function compute_text_sentiment(text){
  return net.normal_forward(text2vec(text));
}

/** Finds the index of the largest element in an array
* @param {array} arr an array
* @return {number} Index of the largest element in the array
*/
function indexOfMaxElem(arr){
  var max = -1;
  var index = 0;
  for(var i = 0; i < arr.length; i++){
    if(arr[i] > max){
      index = i;
      max = arr[i];
    }
  }
  return index
}

function setup(){
  createSpan(" A neural network trained on 150 0000 amazon reviews; reacts to Cordwainer Smith's \"When the People Fell\" ");
  createCanvas(700, 6500);
}


function draw(){
  background(255);
  time += 1;
  if(time % 5 == 0){
    showOverallSentiment = true;
  }

  var positiveScore = 0;
  var negativeScore = 0;
  for(var i = 0; i < txt_file.length; i++){
    var sentiment_vector = compute_text_sentiment(txt_file[i]);
    var txt = txt_file[i];

    var indexOfMax = indexOfMaxElem(sentiment_vector);
    /*
    * The color of the text is dependent on the sentiment_vector
    * score provided by the neural network. A low sentiment score
    * is red, while a high sentiment score is green.
    */
    if(indexOfMax == 3){
      positiveScore += 1;
      fill(0, math.abs(sentiment_vector[indexOfMax]) * 255, 0);
    }else if(indexOfMax == 2){
      positiveScore += 1;
      fill(0, math.abs(sentiment_vector[indexOfMax]) * 255, 0);
    }else if(indexOfMax == 1){
      negativeScore += 1;
      fill(math.abs(sentiment_vector[indexOfMax]) * 255, 0, 0);
    }else if(indexOfMax == 0){
      negativeScore += 1;
      fill(math.abs(sentiment_vector[indexOfMax]) * 255, 0, 0);
    }

    noStroke();
    text(txt_file[i], 45, 20 + i * 15);

    rect(0, i * 30, 40, 30);

  }


  if(showOverallSentiment){
    fill(0);
    text("Overall positive sentiment: " + positiveScore/txt_file.length, 45, 10);
  }
}
