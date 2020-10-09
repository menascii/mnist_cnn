/* mnist training program in c++ by @menascii
     
   compile:
      g++ training_mnist.cpp
    
   run:
      ./a.out  

   input:      
      hardcoded file name for 60,000 mnist training images
      hardcoded file name for 60,000 mnist training labels
      download training data from http://yann.lecun.com/exdb/mnist/
        train-images-idx3-ubyte.gz
        train-labels-idx1-ubyte.gz
        
        unzip and rename accordingly for hardcoded filename values
   
   output:
      write updated model weights to file
      
   note:
      while this program is updating the weights you can execute testing_mnist.cpp
      to check the accuracy using the mnist test data. you can also just wait for the 
      process to complete and check the final output. 

      refer to testing_mnist.cpp for further details.
*/

#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

void read_headers(ifstream &image, ifstream &label);
void read_image(ifstream &image, ifstream &label, int image_digit[28][28], double *layer_one, double expected[]);
void get_digit(ifstream &image, int image_digit[28][28], double *layer_one);
void get_label(ifstream &label, double expected[]);
void print_digit(int image_digit[28][28]);

void init_layers(double *&layer_one, double *&layer_two, double *&layer_three);
void set_zero_pad(int image_digit[28][28], int zero_pad_image_digit[32][32]);
void print_zero_pad(int zero_pad_image_digit[32][32]);

double** set_kernel_weights(int input_layer_size, int output_layer_size);
double*** convolute_image(int zero_pad_image_digit[32][32], double** kernel_weights);
void print_covolution_layers(double*** covolution_layers);

int main()
{
  // mnist binary training digits file name                                       
  string training_images = "train-images";
  // mnist binary training labels                                                                                    
  string training_labels = "train-labels";
  // output weights file name                                                           
  string model_weights = "model-weights";

  ifstream training_image;
  ifstream training_label;

  // mnist image digit 28 x 28                              
  int image_digit[28][28];
  // expected value as output array
  // value in array is 7
  // {0, 0, 0, 0, 0, 0, 0, 1, 0, 0}
  double expected[10];

  // neural network layers
  double *layer_one, *layer_two, *layer_three;

  int zero_pad_image_digit[32][32];

  double ***first_covolution_layers;


  double **kernel_weights;
    
  // open mnist training images
  training_image.open(training_images.c_str(), ios::in | ios::binary);
  // open mnist training labels
  training_label.open(training_labels.c_str(), ios::in | ios::binary);
  // read headers from training images and labels
  read_headers(training_image, training_label);

  // initialize neural network layers
  init_layers(layer_one, layer_two, layer_three);


  // seed random number
  srand(time(0));

  cout << "............training mnist dataset" << endl;
  // training loop to iterate each mnist digit and label
  for (int image_index = 0; image_index < 10; image_index++)
  {
    cout << "mnist training image #: " << image_index << endl; 
    // read image and label
    // assign each 28*28 pixel to first layer as 784 activations in neural network
    read_image(training_image, training_label, image_digit, layer_one, expected);
    
    set_zero_pad(image_digit, zero_pad_image_digit);

    // print mnist 28x28 mnist digit and label
    print_digit(image_digit);

    print_zero_pad(zero_pad_image_digit);
    
    first_covolution_layers = convolute_image(zero_pad_image_digit, kernel_weights);
  
    print_covolution_layers(first_covolution_layers);
  }
  
  training_image.close();
  training_label.close();
  return 0;
}

void read_headers(ifstream &training_image, ifstream &training_label)
{
  // strip headers from training files
  char number;
  for (int i = 0; i < 16; i++)
  {
    training_image.read(&number, sizeof(char));
  }
  for (int i = 0; i < 8; i++)
  {
    training_label.read(&number, sizeof(char));
  }
}

void init_layers(double *&layer_one, double *&layer_two, double *&layer_three)
{
  layer_one = new double[784];
  layer_two = new double[128];
  layer_three = new double[10];
}


void set_zero_pad(int image_digit[28][28], int zero_pad_image_digit[32][32])
{
  for (int x = 0; x < 32; x++)
  {
    for (int y = 0; y < 32; y++)
    {
      zero_pad_image_digit[x][y] = 0;
    }
  }

  for (int x = 0; x < 28; x++)
  {
    for (int y = 0; y < 28; y++)
    {
      zero_pad_image_digit[x + 2][y + 2] = image_digit[x][y];
    }
  }
}

double** set_kernel_weights(int input_layer_size, int output_layer_size)
{
  double **layer_weights;


  layer_weights = new double *[input_layer_size];  
  for (int i = 0; i < input_layer_size; i++)
  {
    layer_weights[i] = new double [output_layer_size];
    for (int j = 0; j < output_layer_size; j++)
    {
      int sign = rand() % 2;
      layer_weights[i][j] = (double)(rand() % 6) / 10.0;
      if (sign == 1)
      {
        layer_weights[i][j] = - layer_weights[i][j];
      }
    }
  }
  
  return layer_weights;
}

double **set_weights(int input_layer_size, int output_layer_size)
{
  // set weights values from input layer to output layer
  double **layer_weights = new double *[input_layer_size];  
  for (int i = 0; i < input_layer_size; i++)
  {
    layer_weights[i] = new double [output_layer_size];
    for (int j = 0; j < output_layer_size; j++)
    {
      int sign = rand() % 2;
      layer_weights[i][j] = (double)(rand() % 6) / 10.0;
      if (sign == 1)
      {
        layer_weights[i][j] = - layer_weights[i][j];
      }
    }
  }
  return layer_weights;
}

double **init_deltas(int input_layer_size, int output_layer_size)
{
  double **layer_deltas = new double *[input_layer_size];
  for (int i = 0; i < input_layer_size; i++)
  {
    layer_deltas[i] = new double [output_layer_size];
    for (int j = 0; j < output_layer_size; j++)
    {
      layer_deltas[i][j] = 0.0;
    }
  }
  return layer_deltas;
}

void read_image(ifstream &image, ifstream &label, int image_digit[28][28], double *layer_one, double expected[])
{
  // read 28x28 binary mnist image
  get_digit(image, image_digit, layer_one);
  // read binary mnist label 
  get_label(label, expected);
}

void get_digit(ifstream &image, int image_digit[28][28], double *layer_one)
{
  // read 28x28 image one character at a time
  char number;
  for (int x = 0; x < 28; x++)
  {
    for (int y = 0; y < 28; y++)
    {
      int layer_index = y + x * 28;
      image.read(&number, sizeof(char));
      if (number == 0)
      {
          image_digit[x][y] = 0;
      }
      else
      {
          image_digit[x][y] = 1;
      }
      layer_one[layer_index] = image_digit[x][y];
    }
  }
}

void print_digit(int image_digit[28][28])
{
  cout << "####### training digit #######" << endl;
  for (int x = 0; x < 28; x++)
  {
    for (int y = 0; y < 28; y++)
    {
      if (image_digit[x][y] == 0)
      {
        cout << ".";
      }
      else
      {
        cout << "@";
      }
    }
    cout << endl;
  }
  cout << endl;
}

void print_zero_pad(int zero_pad_image_digit[32][32])
{
  cout << "####### training digit #######" << endl;
  for (int x = 0; x < 32; x++)
  {
    for (int y = 0; y < 32; y++)
    {
      if (zero_pad_image_digit[x][y] == 0)
      {
        cout << ".";
      }
      else
      {
        cout << "@";
      }
    }
    cout << endl;
  }
  cout << endl;
}

void get_label(ifstream &label, double expected[])
{
  // read training label value
  char number;
  label.read(&number, sizeof(char));
  for (int i = 0; i < 10; i++)
  {
    expected[i] = 0.0;
  }
  expected[number] = 1.0;
  cout << "mnist label: " << (int)(number) << endl;
}


double*** convolute_image(int zero_pad_image_digit[32][32], double** kernel_weights)
{
  double ***covolution_layers = new double**[5];
  for (int t = 0; t < 5; t++)
  {
    double** kernel_weights = set_kernel_weights(5, 5);
    // create 28 kernels or features
    double convolute = 0; // This holds the convolution results for an index.
    int x, y; // Used for input matrix index
    covolution_layers[t] = new double*[28];
    // Fill output matrix: rows and columns are i and j respectively
    for (int i = 0; i < 28; i++)
    {
      covolution_layers[t][i] = new double[28];
      for (int j = 0; j < 28; j++)
      {
        x = i;
        y = j;

        // Kernel rows and columns are k and l respectively
        for (int k = 0; k < 5; k++)
        {
          for (int l = 0; l < 5; l++)
          {
            // Convolute here.
            //cout << "{ " << x << ", " << y << "} => " << zero_pad_image_digit[x][y] << " ";
            convolute += zero_pad_image_digit[x][y] * kernel_weights[k][l];
            y++; // Move right.
          }
          x++; // Move down.
          y = j; // Restart column position
        }
        covolution_layers[t][i][j] = convolute; // Add result to output matrix.
        convolute = 0; // Needed before we move on to the next index.
      }
    }
  }
  return covolution_layers;
}

void print_covolution_layers(double*** covolution_layers)
{
  for (int k = 0; k < 5; k++)
  {
    for (int x = 0; x < 28; x++)
    {
      for (int y = 0; y < 28; y++)
      {
        if (covolution_layers[k][x][y] == 0)
          cout << ".";
        else if(covolution_layers[k][x][y] < 0.2)
          cout << ";";
        else if(covolution_layers[k][x][y] < 0.5)
          cout << "^";
        else
          cout << "@";
      }
      cout << endl;
    }
    cout << endl;
  }
}
