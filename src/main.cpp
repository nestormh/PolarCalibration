#include <iostream>
#include <stdio.h>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/concept_check.hpp>

#include <opencv2/nonfree/features2d.hpp>
#include "polarcalibration.h"

using namespace std;

#define BASE_PATH "/local/imaged/stixels/bahnhof"
#define IMG1_PATH "seq03-img-left"
#define FILE_STRING1 "image_%08d_0.png"
// #define IMG2_PATH "seq03-img-left"
// #define FILE_STRING2 "image_%08d_0.png"
#define IMG2_PATH "seq03-img-right"
#define FILE_STRING2 "image_%08d_1.png"
#define CALIBRATION_STRING "cam%d.cal"
#define MIN_IDX 10//138 //120
#define MAX_IDX 999

// #define BASE_PATH "/local/imaged/stixels/castlejpg"
// #define FILE_STRING "castle.%03d.jpg"
// #define MIN_IDX 26
// #define MAX_IDX 27

void getCalibrationMatrix(const boost::filesystem::path &filePath, cv::Mat & cameraMatix, cv::Mat & distCoeffs) {
    ifstream fin(filePath.c_str(), ios::in);
    
    cameraMatix = cv::Mat(3, 3, CV_64FC1);
    fin >> cameraMatix.at<double>(0, 0);
    fin >> cameraMatix.at<double>(0, 1);
    fin >> cameraMatix.at<double>(0, 2);
    fin >> cameraMatix.at<double>(1, 0);
    fin >> cameraMatix.at<double>(1, 1);
    fin >> cameraMatix.at<double>(1, 2);
    fin >> cameraMatix.at<double>(2, 0);
    fin >> cameraMatix.at<double>(2, 1);
    fin >> cameraMatix.at<double>(2, 2);
    
    distCoeffs = cv::Mat(1, 4, CV_64FC1);
    fin >> distCoeffs.at<double>(0, 0);
    fin >> distCoeffs.at<double>(0, 1);
    fin >> distCoeffs.at<double>(0, 2);
    fin >> distCoeffs.at<double>(0, 3);
    
//     cout << "cameraMatix:\n" << cameraMatix << endl;
//     cout << "distCoeffs:\n" << distCoeffs << endl;
    
    fin.close();
}

int main(int argc, char * argv[]) {
  
    cv::Mat showImg1, showImg2;
    
    cv::namedWindow("showImg1");
    cv::namedWindow("showImg2");
    
    PolarCalibration calibrator;
    calibrator.toggleShowCommonRegion(false);
    calibrator.toggleShowIterations(false);
    for (uint32_t i = MIN_IDX; i < MAX_IDX; i++) {
        boost::filesystem::path img1Path(BASE_PATH);
        boost::filesystem::path img2Path(BASE_PATH);
        
        char imageName[1024];
        sprintf(imageName, FILE_STRING1, i);
        img1Path /= IMG1_PATH;
        img1Path /= imageName;
        sprintf(imageName, FILE_STRING2, i + 1);
//         sprintf(imageName, FILE_STRING2, i);
        img2Path /= IMG2_PATH;
        img2Path /= imageName;
        
        cout << img1Path.string() << endl;
        cout << img2Path.string() << endl;
        
        cv::Mat img1distorted = cv::imread(img1Path.string(), 0);
        cv::Mat img2distorted = cv::imread(img2Path.string(), 0);
        
        // Images are dedistorted
        boost::filesystem::path calibrationPath1(BASE_PATH);
        boost::filesystem::path calibrationPath2(BASE_PATH);
        
        char calibrationName[1024];
        
        sprintf(calibrationName, CALIBRATION_STRING, 1);
        calibrationPath1 /= calibrationName;
        sprintf(calibrationName, CALIBRATION_STRING, 2);
        calibrationPath2 /= calibrationName;
        
        cout << calibrationPath1 << endl;
        cout << calibrationPath2 << endl;
        
        cv::Mat cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2;
        getCalibrationMatrix(calibrationPath1, cameraMatrix1, distCoeffs1);
        getCalibrationMatrix(calibrationPath2, cameraMatrix2, distCoeffs2);

        if (!calibrator.compute(img1distorted, img2distorted, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2))
            continue;
        
        // Visualization
        cv::Mat img1, img2, rectified1, rectified2;
        cv::undistort(img1distorted, img1, cameraMatrix1, distCoeffs1);
        cv::undistort(img2distorted, img2, cameraMatrix2, distCoeffs2);
        calibrator.getRectifiedImages(img1, img2, rectified1, rectified2);
        
        cv::Mat scaled1, scaled2;
        cv::Size newSize;
        if (rectified1.cols > rectified1.rows) {
            newSize = cv::Size(600, 600 * rectified1.rows / rectified1.cols);
        } else {
            newSize = cv::Size(600 * rectified1.cols / rectified1.rows, 600);
        }
        
        cout << "prevSize " << rectified1.size() << endl;
        cout << "newSize " << newSize << endl;

        cv::resize(rectified1, scaled1, newSize);
        cv::resize(rectified2, scaled2, newSize);

        cv::imshow("showImg1", scaled1);
        cv::imshow("showImg2", scaled2);
        cv::moveWindow("showImg2", 700, 0);
        
        uint8_t keycode = cv::waitKey(0);
        switch (keycode) {
            case 'q':
                exit(0);
                break;
            default:
                ;
        }
   }
  
  return 0;
}


