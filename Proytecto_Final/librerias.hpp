#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <tuple>
#include <iostream>
#include <iterator>

using namespace cv;
using namespace cv::dnn;
using namespace std;

Mat frame;
VideoCapture cap(0);

//variables para detectar la cara
vector<vector<int>> bboxes;
Mat frameFace;
int padding = 0;
bool cambiar=true;
int v = 99;

//se crea un vector para realizar el rango de edades
vector<string> ageList = {"(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
      "(38-43)", "(48-53)", "(60-100)"};

// vector para definir el genero
vector<string> genderList = {"Masculino", "Femenino"};


