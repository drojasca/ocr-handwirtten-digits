#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include "std_msgs/String.h"
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <fstream>

const int SZ = 20;
const cv::Size Image_Size(28, 28);
const auto affineFlags = cv::WARP_INVERSE_MAP|cv::INTER_LINEAR;
cv::Ptr<cv::ml::SVM> svms;
const std::string TRAINING_PATH;

image_transport::Publisher pub;

/**
 * deskew the image, reduce tilt of image
 * code taken from https://docs.opencv.org/4.2.0/dd/d3b/tutorial_py_svm_opencv.html
*/
const cv::Mat deskew (cv::Mat & img) {
    cv::Moments m = cv::moments(img);

    // check average intensity of one of the central moments
    // Mat_ (int _rows, int _cols)
    if (abs(m.mu02) < 1e-2) {
        return img.clone();
    }

    double skew = m.mu11/m.mu02;

    // Mat and Mat_ are basically the same but later if you know the matrix type at the start;
    // calculate the afline transform (rotate, resizing without the changing the ratios)
    // this code fills out the matrix
    cv::Mat warpMat = (cv::Mat_<double> (2,3) << 1, skew, -0.5*SZ*skew, 0, 1 , 0);

    cv::Mat imgOut = cv::Mat::zeros (img.rows, img.cols, img.type());
    // apply the Affline transform to the image
    warpAffine(img, imgOut, warpMat, imgOut.size(),affineFlags);

    return imgOut;
}

/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
* Code taken from  https://github.com/ahmetozlu/vehicle_counting_hog_svm/blob/master/src/Main.cpp
*/
void convert_to_mat_data(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData)
{
	//--Convert data
	const int rows = (int)train_samples.size();
	const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
	cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = cv::Mat(rows, cols, CV_32FC1);
	std::vector<cv::Mat>::const_iterator itr = train_samples.begin();
	std::vector<cv::Mat>::const_iterator end = train_samples.end();
	for (int i = 0; itr != end; ++itr, ++i)
	{
		CV_Assert(itr->cols == 1 ||
			itr->rows == 1);
		if (itr->cols == 1)
		{
			transpose(*(itr), tmp);
			tmp.copyTo(trainData.row(i));
		}
		else if (itr->rows == 1)
		{
			itr->copyTo(trainData.row(i));
		}
	}
}

/**
 * Will read in the images and load then into vector, and their corresponding
 * label to another vector
*/
void load_images(std::vector<cv::Mat>& images, std::vector<int>& labels) {
    cv::String path;
    std::vector<cv::String> fileNames;
    ROS_INFO("Getting Training Data");
    for (int i = 0; i < 10; i++){
        std::string temp1 = TRAINING_PATH;
        std::string temp2 = "/" + std::to_string(i);
        std::string temp3 = "/*.jpg";

        path = temp1 + temp2 + temp3;
        fileNames.clear();
        cv::glob(path, fileNames, true);

        for (auto j = fileNames.begin(); j != fileNames.end(); j++) {
            cv::Mat im = cv::imread(*j);
            cv::resize( im, im, Image_Size, 0, 0, CV_INTER_LINEAR);
            cv::cvtColor(im, im, cv::COLOR_BGR2GRAY);
            deskew(im);
            images.push_back(im.clone());
            labels.push_back(i);            
        }
    }
}

/**
 * Calculate HOG for a vector of gradients
*/
const void calculateHOG(std::vector<cv::Mat>& img, std::vector<cv::Mat>& gradient) {
    cv::HOGDescriptor hog(cv::Size(28,28), cv::Size(14,14), cv::Size(7,7), cv::Size(7,7),9, 1, -1, 0, 0.2000, false, 64, false);
	std::vector< float > descriptors;

    for (auto i = img.begin(); i != img.end(); i++){
        hog.compute(*i, descriptors, cv::Size(4, 4), cv::Size(0, 0)); //saves the gradient in the matrix
        gradient.push_back(cv::Mat(descriptors).clone());
    }
}


/**
 * Train the SVM
*/
const cv::Ptr<cv::ml::SVM> train(std::vector<cv::Mat>& gradient, std::vector<int>& labels) {
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setKernel(cv::ml::SVM::RBF);
	svm->setType(cv::ml::SVM::C_SVC);
    svm->setC(12.5);
    svm->setGamma(0.138);

    cv::Mat formatted;

    convert_to_mat_data(gradient, formatted);
    ROS_INFO("Training Model");
    svm->train(formatted, cv::ml::ROW_SAMPLE, cv::Mat(labels));
    ROS_INFO("Model Trained");
    svm->save("hand_written_detector.yml");

    return svm;
}

/**
 * Main Training function that will perform steps to train a model
*/
const void run_training () {
    std::vector<cv::Mat> images;
    std::vector<int> labels;
    std::vector<cv::Mat> gradient;
    load_images(images, labels);
    calculateHOG(images, gradient);
    cv::Ptr<cv::ml::SVM> svm = train(gradient, labels);
}


/**
 * Function that convert cv::Mat to sensor_msg
*/
const void convert_to_msg(cv::Mat image) {
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    pub.publish(msg);
}

/**
 * Function that finds contours in the image
 * Code inspired from https://www.hackevolve.com/recognize-handwritten-digits-1/
*/

const void get_contour (cv::Mat& img, std::vector<cv::Rect>& contours, std::vector<cv::Mat>& digits) {
    cv::Mat blackHat, threshold, clone;

    if (img.size().width > 640) {
        double factor = (double)640/img.size().width;
        int height = img.size().height*factor;
        cv::resize(img, img, cv::Size(640, height), 0, 0, CV_INTER_LINEAR);
    }
    cv::cvtColor(img, clone, cv::COLOR_BGR2GRAY);
    std::vector<std::vector<cv::Point>> locations;

    // Find the contours of the imagesource devel/setup.bash 
    cv::Mat kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7.5,7.5));
    cv::morphologyEx(clone, blackHat, cv::MORPH_BLACKHAT, kernal); // get dark spots
    cv::threshold(blackHat, threshold, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::dilate(threshold, threshold, kernal);
    cv::findContours(threshold.clone(), locations, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (auto& point: locations) {
        cv::Rect bounding = cv::boundingRect(point);
        if (bounding.area() > 500) {
            contours.push_back(bounding);
            cv::Mat hull;
            cv::Mat mask (clone.rows, clone.cols, CV_8U);

            // isolate the Contour Better
            cv::convexHull(point, hull);
            cv::drawContours(mask, hull, -1, 255, -1);
            cv::bitwise_and(clone, clone, mask, mask);
            cv::Mat digit = mask(bounding);
            cv::resize(digit, digit, Image_Size, 0, 0, cv::INTER_AREA);
            deskew(digit);
            digits.push_back(digit);
        }
    }
}

/**
 * Function that runs detection on image
*/
const void run_detector (cv::Mat& img) {
    
    std::vector<cv::Mat> digits;
    auto font = cv::FONT_HERSHEY_SIMPLEX;
    std::vector<cv::Rect> contours;

    get_contour(img, contours, digits);

    // Get Gradients and predict
    cv::Mat prediction, formatted;
    std::vector<cv::Mat> gradients;
    calculateHOG(digits, gradients);
    convert_to_mat_data(gradients, formatted);
    svms -> predict(formatted, prediction);

    for (int i = 0; i < prediction.rows; i++) {
        int x = contours[i].tl().x;
        int y = contours[i].tl().y;
        float ans = prediction.at<float>(i, 0);
        cv::rectangle(img, contours[i], cv::Scalar(0, 0, 255), 3);
        cv::putText(img, std::to_string((int)ans), cv::Point(x,y), font, 2, cv::Scalar(0, 255, 255), 3); 
    }
    
    convert_to_msg(img);
}


// called whenever there is a new image published
void GetPath (const std_msgs::String::ConstPtr& msg) {
    ROS_INFO("RUNNING DETECTOR");
    std::string path = msg -> data.c_str();
    std::ifstream infile(path);
    
    if (!infile.good()){
        ROS_ERROR("Invalid path");
    }
    else {
        cv::Mat img = cv::imread(path);
        run_detector(img);
        ROS_INFO("DETECTOR RUN");
    }
}


int main (int argc, char **argv) {

    // check if file exists
    std::ifstream infile("hand_written_detector.yml");
    if (!infile.good()){
        run_training();
    }
    //Load SVM
    svms = cv::ml::StatModel::load<cv::ml::SVM>("hand_written_detector.yml");

    // Subscriber that gets file path
    ROS_INFO("DETECTOR READY");
    ros::init(argc, argv, "Detector");
    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe("getImage", 1000, GetPath);

    // Publish annotated Image
    image_transport::ImageTransport it(n);
    pub = it.advertise("postImage", 1);
    ros::spin();
    
    return 0;
}
