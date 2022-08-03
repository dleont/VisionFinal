#include "librerias.hpp"

tuple<Mat, vector<vector<int>>> getFaceBox(Net net, Mat &frame, double conf_threshold)
{
    Mat frameOpenCVDNN = frame.clone();
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;
    double inScaleFactor = 1.0;
    Size size = Size(300, 300);
    // std::vector<int> meanVal = {104, 117, 123};
    Scalar meanVal = Scalar(104, 117, 123);

    cv::Mat inputBlob;
    inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, size, meanVal, true, false);

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    vector<vector<int>> bboxes;

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > conf_threshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
            vector<int> box = {x1, y1, x2, y2};
            bboxes.push_back(box);
            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
        }
    }

    return make_tuple(frameOpenCVDNN, bboxes);
}


int main(int argc, char** argv)
{

//carga las redes neuronales
string faceProto = "/home/darwin/Documentos/UPS/8VO/VisionComputador/Proytecto_Final/red-neuronal/opencv_face_detector.pbtxt.txt";
string faceModel = "/home/darwin/Documentos/UPS/8VO/VisionComputador/Proytecto_Final/red-neuronal/opencv_face_detector_uint8.pb";

string ageProto = "/home/darwin/Documentos/UPS/8VO/VisionComputador/Proytecto_Final/red-neuronal/age_deploy.prototxt";
string ageModel = "/home/darwin/Documentos/UPS/8VO/VisionComputador/Proytecto_Final/red-neuronal/age_net.caffemodel";

string genderProto = "/home/darwin/Documentos/UPS/8VO/VisionComputador/Proytecto_Final/red-neuronal/gender_deploy.prototxt";
string genderModel = "/home/darwin/Documentos/UPS/8VO/VisionComputador/Proytecto_Final/red-neuronal/gender_net.caffemodel";

//pinta los recuadros 
Scalar MODEL_MEAN_VALUES = Scalar(78.4263377603, 87.7689143744, 114.895847746);

//se carga la red neuronal
Net ageNet = readNet(ageModel, ageProto);
Net genderNet = readNet(genderModel, genderProto);
Net faceNet = readNet(faceModel, faceProto);

ageNet.setPreferableBackend(DNN_TARGET_CPU);
genderNet.setPreferableBackend(DNN_TARGET_CPU);
faceNet.setPreferableBackend(DNN_TARGET_CPU);

cout << "Red cargada existosamente" << endl;

VideoWriter video("videoCaptura.avi",cv::VideoWriter::fourcc('M','J','P','G'),4, Size(600,400));

while(true) {
    
    if(waitKey(50) == 99){
        v= 99;
    }
    if(waitKey(50) == 105){
        v= 105;
    }

    if(v==99){
        cap >> frame;
    }
    if(v==105){
        frame= imread("/home/darwin/Documentos/UPS/8VO/VisionComputador/Proytecto_Final/img.jpg");
    }

    flip(frame,frame,1);
    
    tie(frame, bboxes) = getFaceBox(faceNet, frame, 0.7);
       

    for (auto it = begin(bboxes); it != end(bboxes); ++it) {
        Rect rec(it->at(0) - padding, it->at(1) - padding, it->at(2) - it->at(0) + 2*padding, it->at(3) - it->at(1) + 2*padding);
        ///Mat face = frame(rec); // take the ROI of box on the frame
        rectangle(frame, rec, Scalar(255, 0, 0), 2, 8, 0);
        Mat face = frame(rec); // take the ROI of box on the frame
        
        imwrite("/home/darwin/Documentos/UPS/8VO/VisionComputador/Proytecto_Final/N.jpg",face);

        Mat blob;
        blob = blobFromImage(frame, 1, Size(227, 227), MODEL_MEAN_VALUES, false);
        genderNet.setInput(blob);
        vector<float> genderPreds = genderNet.forward();
        int max_index_gender = std::distance(genderPreds.begin(), max_element(genderPreds.begin(), genderPreds.end()));
        string gender = genderList[max_index_gender];
        //cout << "Género: " << gender << endl;

        ageNet.setInput(blob);
        vector<float> agePreds = ageNet.forward();
        //Distancia d la edad.
        int max_indice_age = std::distance(agePreds.begin(), max_element(agePreds.begin(), agePreds.end()));
        string age = ageList[max_indice_age];
        //cout << "Edad: " << age << endl;
        
        string label = gender + ", " + age; 
        cv::putText(frame, label, Point(it->at(0), it->at(1) -15), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 255), 2, cv::LINE_AA);
        
    }

    imshow("Frame", frame);

    resize(frame,frame,Size(600,400));
    video.write(frame);

    if(waitKey(50) == 27){
        break;
    } 

    }

    cap.release();
	video.release();
}