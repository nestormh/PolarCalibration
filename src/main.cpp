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
#define MIN_IDX 119
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
    
    cout << "cameraMatix:\n" << cameraMatix << endl;
    cout << "distCoeffs:\n" << distCoeffs << endl;
    
    fin.close();
}

double pseudo_atan2(const double & y, const double & x) {
//     if (x > 0) {
        return atan2(y, x);
//     } else {
//         return atan2(-y, -x);
//     }
    double angle = 0;
    if (x != 0) 
        angle = atan(fabs(y) / fabs(x));
    if (y == 0)
        angle = CV_PI;
    
    if (y >= 0) {
        if (x >= 0)
            return angle;
        else
            return CV_PI + angle;
    } else {
        if (x >= 0)
            return 2 * CV_PI - angle;
        else
            return angle + CV_PI;
    }
    
}

void computeEpilinesBasedOnCase(const cv::Point2d &epipole, const cv::Size imgDimensions, 
                                const cv::Mat & F, const uint32_t & imgIdx, 
                                vector<cv::Point2f> &externalPoints, vector<cv::Vec3f> &epilines) {
    
    if (epipole.y < 0) { // Cases 1, 2 and 3
        if (epipole.x < 0) { // Case 1
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(imgDimensions.width - 1, 0);
            externalPoints[1] = cv::Point2f(0, imgDimensions.height - 1);
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 2
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(imgDimensions.width - 1, 0);
            externalPoints[1] = cv::Point2f(0, 0);
        } else { // Case 3
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(imgDimensions.width - 1, imgDimensions.height - 1);
            externalPoints[1] = cv::Point2f(0, 0);
        }
    } else if (epipole.y <= imgDimensions.height - 1) { // Cases 4, 5 and 6
        if (epipole.x < 0) { // Case 4
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(0, 0);
            externalPoints[1] = cv::Point2f(0, imgDimensions.height - 1);
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 5
            externalPoints.resize(4);
            externalPoints[0] = cv::Point2f(0, 0);
            externalPoints[1] = cv::Point2f(imgDimensions.width - 1, 0);
            externalPoints[2] = cv::Point2f(imgDimensions.width - 1, imgDimensions.height - 1);
            externalPoints[3] = cv::Point2f(0, imgDimensions.height - 1);
        } else { // Case 6
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(imgDimensions.width - 1, imgDimensions.height - 1);
            externalPoints[1] = cv::Point2f(imgDimensions.width - 1, 0);
        }        
    } else { // Cases 7, 8 and 9
        if (epipole.x < 0) { // Case 7
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(0, 0);
            externalPoints[1] = cv::Point2f(imgDimensions.width - 1, imgDimensions.height - 1);
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 8
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(0, imgDimensions.height - 1);
            externalPoints[1] = cv::Point2f(imgDimensions.width - 1, imgDimensions.height - 1);
        } else { // Case 9
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(0, imgDimensions.height - 1);
            externalPoints[1] = cv::Point2f(imgDimensions.width - 1, 0);            
        }
    }
    
    cv::computeCorrespondEpilines(externalPoints, imgIdx, F, epilines);
}

void polarTransform(/*const*/ cv::Mat & img, /*const*/ cv::Point2d &epipole, 
                    /*const*/ vector<cv::Point2f> &externalPoints, /*const*/ vector<cv::Vec3f> &epilines, 
                    cv::Mat & output) {
    
//     output = cv::Mat::zeros(sqrt(img.rows *img.rows + img.cols * img.cols), 2 * (img.rows + img.cols), CV_8UC1);
    output = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    
//     if (epilines.size() == 4) {
//         // TODO: Include the case 5
//         cout << "SORRY: Case 5 not implemented" << endl;
//     }
    
//     epipole=cv::Point2d(-10, 20);
//     externalPoints.clear();
//     externalPoints.reserve(2);
//     externalPoints[0]=cv::Point2f(0,0);
//     externalPoints[1]=cv::Point2f(0,480);
    
    double thetaMin = DBL_MAX;
    double thetaMax = DBL_MIN;
    for (uint32_t i = 0; i < epilines.size(); i++) {
        const double &theta = atan((externalPoints[i].y - epipole.y) / (externalPoints[i].x - epipole.x));
        cout << theta << endl;
        if (theta < thetaMin)
            thetaMin = theta;
        if (theta > thetaMax)
            thetaMax = theta;
    }
    
    double rhoMax = DBL_MIN;
    vector<cv::Point2d> corners(4);
    corners[0] = cv::Point(0, 0);
    corners[1] = cv::Point(img.cols, 0);
    corners[2] = cv::Point(img.cols, img.rows);
    corners[3] = cv::Point(0,img.rows);
    
    for (uint32_t i = 0; i < corners.size(); i++) {
        double dist = sqrt((corners[i].x - epipole.x) * (corners[i].x - epipole.x) + (corners[i].y - epipole.y) * (corners[i].y - epipole.y));
        cout << dist << endl;
        
        if (dist > rhoMax) rhoMax = dist;
    }
    
    double rhoMin = 0;
    if (epipole.y < 0) { // Cases 1, 2 and 3
        if (epipole.x < 0) { // Case 1
            cout << "Case 1 " << endl;
            rhoMin = sqrt(epipole.x * epipole.x + epipole.y * epipole.y);
        } else if (epipole.x < img.cols) { // Case 2
            cout << "Case 2 " << endl;
            rhoMin = floor(epipole.y);
        } else { // Case 3
            cout << "Case 3 " << endl;
            rhoMin = sqrt((epipole.x - img.cols) * (epipole.x - img.cols) + epipole.y * epipole.y);
        }
    } else if (epipole.y < img.rows) { // Cases 4, 5 and 6
        if (epipole.x < 0) { // Case 4
            cout << "Case 4 " << endl;
            rhoMin = floor(epipole.x);
        } else if (epipole.x < img.cols) { // Case 5
            cout << "Case 5 " << endl;
            rhoMin = 0;
        } else { // Case 6
            cout << "Case 6 " << endl;
            rhoMin = floor(epipole.x - img.cols + 1);
        }        
    } else { // Cases 7, 8 and 9
        if (epipole.x < 0) { // Case 7
            cout << "Case 7 " << endl;
            rhoMin = sqrt(epipole.x * epipole.x + (epipole.y - img.rows) * (epipole.y - img.rows));
        } else if (epipole.x < img.cols) { // Case 8
            cout << "Case 8 " << endl;
            rhoMin = floor(epipole.y - img.rows + 1);
        } else { // Case 9
            cout << "Case 9 " << endl;
            rhoMin = sqrt((epipole.x - img.cols) * (epipole.x - img.cols) + (epipole.y - img.rows) * (epipole.y - img.rows));
        }
    }
    
    cout << "thetaMin " << thetaMin << endl;
    cout << "thetaMax " << thetaMax << endl;
    cout << "rhoMin " << rhoMin << endl;
    cout << "rhoMax " << rhoMax << endl;
    
    exit(0);
            
//     uint32_t x = 0;
//     uint32_t y = 20;
    for (uint32_t y = 0; y < img.rows; y++) {
        for (uint32_t x = 0; x < img.cols; x++) {
            
            double rho = sqrt((x - epipole.x) * (x - epipole.x) + (y - epipole.y) * (y - epipole.y));
            double theta = 0;
            if ((epipole.x - x) != 0)
                theta = atan((y - epipole.y) / (x - epipole.x));
//             cout << "theta " << theta << endl;

            theta = (theta - thetaMin) / (thetaMax - thetaMin) * output.rows;
            rho = (rho - rhoMin) / (rhoMax - rhoMin) * output.cols;
            
//             cout << "Theta = " << theta << endl;
//             cout << "Rho = " << rho << endl;
//             cout << "(" << x << ", " << y << ")" << " >> " << rho << ", " << theta << endl;
            
//             if ((rho >= 0) && (rho < output.cols) && (theta >= 0) && (theta < output.rows)) {
//                 if ((rho < 0) || (rho >= output.cols)) {
//                     cout << "x = " << x << endl;
//                     cout << "y = " << y << endl;
//                     cout << "epipole = " << epipole << endl;
//                     cout << "rho = " << rho << endl;
//                     cout << "rhoMin = " << rhoMin << endl;
//                     cout << "rhoMax = " << rhoMax << endl;
//                     cout << "output.cols = " << output.cols << endl;
//                 }
            
                output.at<uint8_t>(theta, rho) = img.at<uint8_t>(y, x);
//             }
        }
    }
    
    cv::imshow("output", output);
    cv::waitKey(0);
    
    exit(0);
}

bool getThetaAB(cv::Point2d & epipole, const cv::Vec3f &epiline, const cv::Size &imgDimensions, double & newTheta) {
    const double x = -epiline[2] / epiline[0];
    cout << "getThetaAB: " << cv::Point2d(x, 0) << endl;
    
    if ((x >= 0) && (x <= (imgDimensions.width - 1))) {
        newTheta = atan2(-epipole.y, x - epipole.x) + 2 * CV_PI;
        
        return true;
    }
    return false;
}

bool getThetaCD(cv::Point2d & epipole, const cv::Vec3f &epiline, const cv::Size &imgDimensions, double & newTheta) {
    const double x = -(epiline[1] * (imgDimensions.height - 1) + epiline[2]) / epiline[0];
    
    cout << "getThetaCD: " << cv::Point2d(x, imgDimensions.height - 1) << endl;
    
    if ((x >= 0) && (x <= (imgDimensions.width - 1))) {
        newTheta = atan2((imgDimensions.height - 1) - epipole.y, x - epipole.x) + 2 * CV_PI;
        
        return true;
    }
    return false;
}

bool getThetaBD(cv::Point2d & epipole, const cv::Vec3f &epiline, const cv::Size &imgDimensions, double & newTheta) {
    const double y = -(epiline[0] * (imgDimensions.width - 1) + epiline[2]) / epiline[1];
    
    cout << "getThetaBD: " << cv::Point2d(imgDimensions.width - 1, y) << endl;
    
    if ((y >= 0) && (y <= (imgDimensions.height - 1))) {
        newTheta = atan2(y - epipole.y, (imgDimensions.width - 1) - epipole.x) + 2 * CV_PI;
        
        return true;
    }
    return false;
}

bool getThetaAC(cv::Point2d & epipole, const cv::Vec3f &epiline, const cv::Size &imgDimensions, double & newTheta) {
    const double y = -epiline[2] / epiline[1];
    
    cout << "getThetaAC: " << cv::Point2d(0, y) << endl;
    
    if ((y >= 0) && (y <= (imgDimensions.height - 1))) {
        newTheta = atan2(y - epipole.y, -epipole.x) + 2 * CV_PI;
        
        return true;
    }
    return false;
}

void getThetaFromEpilines(/*const*/ cv::Point2d &epipole, const cv::Size imgDimensions, 
                         const vector<cv::Vec3f> &epilines, double & newTheta, double & minTheta, double & maxTheta) {
    if (epipole.y < 0) { // Cases 1, 2 and 3
        if (epipole.x < 0) { // Case 1
            if (getThetaAB(epipole, epilines[0], imgDimensions, newTheta))
                minTheta = newTheta;
            else if (getThetaAC(epipole, epilines[0], imgDimensions, newTheta))
                minTheta = newTheta;
            if (getThetaAC(epipole, epilines[1], imgDimensions, newTheta))
                maxTheta = newTheta;
            else if (getThetaAB(epipole, epilines[1], imgDimensions, newTheta))
                maxTheta = newTheta;
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 2
            if (getThetaAB(epipole, epilines[0], imgDimensions, newTheta))
                minTheta = newTheta;
            if (getThetaAB(epipole, epilines[1], imgDimensions, newTheta))
                maxTheta = newTheta;
        } else { // Case 3
            if (getThetaBD(epipole, epilines[0], imgDimensions, newTheta))
                minTheta = newTheta;
            else if (getThetaAB(epipole, epilines[1], imgDimensions, newTheta))
                minTheta = newTheta;
            if (getThetaAB(epipole, epilines[1], imgDimensions, newTheta))
                maxTheta = newTheta;
            else if (getThetaBD(epipole, epilines[0], imgDimensions, newTheta))
                maxTheta = newTheta;
        }
    } else if (epipole.y <= imgDimensions.height - 1) { // Cases 4, 5 and 6
        if (epipole.x < 0) { // Case 4
            if (getThetaAC(epipole, epilines[0], imgDimensions, newTheta))
                minTheta = newTheta;
            if (getThetaAC(epipole, epilines[1], imgDimensions, newTheta))
                maxTheta = newTheta;
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 5
            if (getThetaAB(epipole, epilines[0], imgDimensions, newTheta))
                minTheta = newTheta;
            else if (getThetaBD(epipole, epilines[0], imgDimensions, newTheta))
                minTheta = newTheta;
            else if (getThetaCD(epipole, epilines[0], imgDimensions, newTheta))
                minTheta = newTheta;
            else if (getThetaAC(epipole, epilines[0], imgDimensions, newTheta))
                minTheta = newTheta;
            
            if (getThetaAC(epipole, epilines[0], imgDimensions, newTheta))
                maxTheta = newTheta;
            else if (getThetaCD(epipole, epilines[0], imgDimensions, newTheta))
                maxTheta = newTheta;
            else if (getThetaBD(epipole, epilines[0], imgDimensions, newTheta))
                maxTheta = newTheta;
            else if (getThetaAB(epipole, epilines[0], imgDimensions, newTheta))
                maxTheta = newTheta;
        } else { // Case 6
            if (getThetaBD(epipole, epilines[0], imgDimensions, newTheta))
                minTheta = newTheta;
            if (getThetaBD(epipole, epilines[1], imgDimensions, newTheta))
                maxTheta = newTheta;
        }        
    } else { // Cases 7, 8 and 9
        if (epipole.x < 0) { // Case 7
            if (getThetaAC(epipole, epilines[0], imgDimensions, newTheta))
                minTheta = newTheta;
            else if (getThetaCD(epipole, epilines[0], imgDimensions, newTheta))
                minTheta = newTheta;
            
            if (getThetaCD(epipole, epilines[1], imgDimensions, newTheta))
                maxTheta = newTheta;
            else if (getThetaAC(epipole, epilines[1], imgDimensions, newTheta))
                maxTheta = newTheta;
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 8
            if (getThetaCD(epipole, epilines[0], imgDimensions, newTheta))
                minTheta = newTheta;
            if (getThetaCD(epipole, epilines[1], imgDimensions, newTheta))
                maxTheta = newTheta;
        } else { // Case 9
            if (getThetaCD(epipole, epilines[0], imgDimensions, newTheta))
                minTheta = newTheta;
            else if (getThetaBD(epipole, epilines[0], imgDimensions, newTheta))
                minTheta = newTheta;
            
            if (getThetaBD(epipole, epilines[1], imgDimensions, newTheta))
                maxTheta = newTheta;
            else if (getThetaCD(epipole, epilines[1], imgDimensions, newTheta))
                maxTheta = newTheta;
        }
    }
}

void determineCommonRegion(/*const*/ cv::Point2d &epipole, const cv::Size imgDimensions, 
                              const vector<cv::Point2f> &externalPoints, const vector<cv::Vec3f> &epilines,
                              double & minTheta, double & maxTheta) {
    
    cout << "************** determineCommonRegion **************" << endl;
    
    cout << externalPoints[0] << endl;
    cout << externalPoints[1] << endl;
    
    cout << "minTheta: " << (externalPoints[0].y - epipole.y) << " / " << (externalPoints[0].x - epipole.x) << endl;
    cout << "maxTheta: " << (externalPoints[1].y - epipole.y) << " / " << (externalPoints[1].x - epipole.x) << endl;
    
    minTheta = atan2((externalPoints[0].y - epipole.y), (externalPoints[0].x - epipole.x)) + 2 * CV_PI;
    maxTheta = atan2((externalPoints[1].y - epipole.y), (externalPoints[1].x - epipole.x)) + 2 * CV_PI;
    
    cout << "minTheta " << minTheta * 180 / CV_PI << "(" << (minTheta - 2 * CV_PI) * 180 / CV_PI << ")" << endl;
    cout << "maxTheta " << maxTheta * 180 / CV_PI << "(" << (maxTheta - 2 * CV_PI) * 180 / CV_PI << ")" << endl;
    
    double newTheta;
    getThetaFromEpilines(epipole, imgDimensions, epilines, newTheta, minTheta, maxTheta);
    
    cout << "minTheta " << minTheta * 180 / CV_PI << "(" << (minTheta - 2 * CV_PI) * 180 / CV_PI << ")" << endl;
    cout << "maxTheta " << maxTheta * 180 / CV_PI << "(" << (maxTheta - 2 * CV_PI) * 180 / CV_PI << ")" << endl;
    
    cout << "************** determineCommonRegion **************" << endl;
    
//     exit(0);
}

void determineCommonRegion_v2(const cv::Point2d &epipole, const cv::Size imgDimensions, 
                              const vector<cv::Point2f> &externalPoints, const vector<cv::Vec3f> &epilines,
                              double & minTheta, double & maxTheta) {
    
    cout << externalPoints[0] << endl;
    cout << externalPoints[1] << endl;
    
    cout << "minTheta: " << (externalPoints[0].y - epipole.y) << " / " << (externalPoints[0].x - epipole.x) << endl;
    cout << "maxTheta: " << (externalPoints[1].y - epipole.y) << " / " << (externalPoints[1].x - epipole.x) << endl;
    
    minTheta = atan2((externalPoints[0].y - epipole.y), (externalPoints[0].x - epipole.x)) + 2 * CV_PI;
    maxTheta = atan2((externalPoints[1].y - epipole.y), (externalPoints[1].x - epipole.x)) + 2 * CV_PI;

    cout << "minTheta " << minTheta * 180 / CV_PI << "(" << (minTheta - 2 * CV_PI) * 180 / CV_PI << ")" << endl;
    cout << "maxTheta " << maxTheta * 180 / CV_PI << "(" << (maxTheta - 2 * CV_PI) * 180 / CV_PI << ")" << endl;
    cout << externalPoints[0] << endl;
    cout << externalPoints[1] << endl;
    
    cv::Point2d tmpPoint1();
    cout << "Angle1 = " << atan2((externalPoints[0].y - epipole.y), (externalPoints[0].x - epipole.x)) + 2 * CV_PI;
    
    exit(0);
    
    double tmpMinTheta = std::numeric_limits<double>::max();
    double tmpMaxTheta = std::numeric_limits<double>::min();
    
    if (epipole.y < 0) { // Cases 1, 2 and 3
        if (epipole.x < 0) { // Case 1
            const double y0 = -epilines[0][2] / epilines[0][1]; // y = -(a*x + c) / b
            const double x0 = -epilines[0][2] / epilines[0][0]; // x = -(b*y + c) / a
            
            const double y1 = -epilines[1][2] / epilines[1][1]; // y = -(a*x + c) / b
            const double x1 = -epilines[1][2] / epilines[1][0]; // x = -(b*y + c) / a
            
            cout << cv::Point2d(0, y0) << " -- " << cv::Point2d(x0, 0) << endl;
            cout << cv::Point2d(0, y1) << " -- " << cv::Point2d(x1, 0) << endl;
            
            if ((y0 >= 0) && (y0 <= imgDimensions.height - 1))
                tmpMinTheta = min(atan2(y0 - epipole.y, -epipole.x) + 2 * CV_PI, tmpMinTheta);
            cout << "tmpMinTheta = " << y0 - epipole.y << " / " << -epipole.x << " = " << tmpMinTheta * 180 / CV_PI << endl;
            if ((x0 >= 0) && (x0 <= imgDimensions.width - 1))
                tmpMinTheta = min(atan2(-epipole.y, x0 - epipole.x) + 2 * CV_PI, tmpMinTheta);
            cout << "tmpMinTheta = " << -epipole.y << " / " << x0 - epipole.x << " = " << tmpMinTheta * 180 / CV_PI << endl;
            
            if ((y1 >= 0) && (y1 <= imgDimensions.height - 1))
                tmpMaxTheta = max(atan2(y1 - epipole.y, -epipole.x) + 2 * CV_PI, tmpMaxTheta);
            cout << "tmpMaxTheta = " << y1 - epipole.y << " / " << -epipole.x << " = " << tmpMaxTheta * 180 / CV_PI << endl;
            if ((x1 >= 0) && (x1 <= imgDimensions.width - 1))
                tmpMaxTheta = max(atan2(-epipole.y, x1 - epipole.x) + 2 * CV_PI, tmpMaxTheta);
            cout << "tmpMaxTheta = " << -epipole.y << " / " << -epipole.x << " = " << tmpMaxTheta * 180 / CV_PI << endl;
            
//             if (tmpMinTheta != std::numeric_limits<double >::max())
//                 minTheta = tmpMinTheta;
//             if (tmpMaxTheta != std::numeric_limits<double >::min())
//                 maxTheta = tmpMaxTheta;
            
//             if (minTheta > maxTheta) {
// //                 tmpMinTheta = minTheta;
// //                 minTheta = maxTheta;
// //                 maxTheta = tmpMinTheta;
//                 maxTheta += 2 * CV_PI;
//             }
                
            
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 2
            const double x0 = -epilines[0][2] / epilines[0][0]; // x = -(b*y + c) / a
            const double x1 = -epilines[1][2] / epilines[1][0]; // x = -(b*y + c) / a
            
            cout << cv::Point2d(0, -1) << " -- " << cv::Point2d(x0, 0) << endl;
            cout << cv::Point2d(0, -1) << " -- " << cv::Point2d(x1, 0) << endl;
            
            if ((x0 >= 0) && (x0 <= imgDimensions.width - 1))
                minTheta = atan2(-epipole.y, x0 - epipole.x) + 2 * CV_PI;
            
            if ((x1 >= 0) && (x1 <= imgDimensions.width - 1))
                maxTheta = atan2(-epipole.y, x1 - epipole.x) + 2 * CV_PI;
        } else { // Case 3
            const double y0 = -(epilines[0][0] * (imgDimensions.width - 1) + epilines[0][2]) / epilines[0][1]; // y = -(a*x + c) / b
            const double x0 = -epilines[0][2] / epilines[0][0]; // x = -(b*y + c) / a
            
            const double y1 = -(epilines[1][0] * (imgDimensions.width - 1) + epilines[1][2]) / epilines[1][1]; // y = -(a*x + c) / b
            const double x1 = -epilines[1][2] / epilines[1][0]; // x = -(b*y + c) / a
            
            cout << cv::Point2d(imgDimensions.width - 1, y0) << " -- " << cv::Point2d(x0, 0) << endl;
            cout << cv::Point2d(imgDimensions.width - 1, y1) << " -- " << cv::Point2d(x1, 0) << endl;
            
            if ((y0 >= 0) && (y0 <= imgDimensions.height - 1)) 
                minTheta = atan2(y0 - epipole.y, (imgDimensions.width - 1) - epipole.x) + 2 * CV_PI; 
            if ((x0 >= 0) && (x0 <= imgDimensions.width - 1))
                minTheta = atan2(-epipole.y, x0 - epipole.x) + 2 * CV_PI;
            
            if ((y1 >= 0) && (y1 <= imgDimensions.height - 1))
                maxTheta = atan2(y1 - epipole.y, (imgDimensions.width - 1) - epipole.x) + 2 * CV_PI;
            if ((x1 >= 0) && (x1 <= imgDimensions.width - 1))
                maxTheta = atan2(-epipole.y, x1 - epipole.x) + 2 * CV_PI;
        }
    } else if (epipole.y <= imgDimensions.height - 1) { // Cases 4, 5 and 6
        if (epipole.x < 0) { // Case 4
            const double y0 = -epilines[0][2] / epilines[0][1]; // y = -(a*x + c) / b
            const double y1 = -epilines[1][2] / epilines[1][1]; // y = -(a*x + c) / b
            
            cout << cv::Point2d(0, y0) << " -- " << cv::Point2d(-1, 0) << endl;
            cout << cv::Point2d(0, y1) << " -- " << cv::Point2d(-1, 0) << endl;
            
            if ((y0 >= 0) && (y0 <= imgDimensions.height - 1)) 
                minTheta = atan2(y0 - epipole.y, -epipole.x) + 2 * CV_PI;

            if ((y1 >= 0) && (y1 <= imgDimensions.height - 1))
                maxTheta = atan2(y1 - epipole.y, -epipole.x) + 2 * CV_PI;
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 5
            maxTheta += 2 * CV_PI;
        } else { // Case 6
            const double y0 = -(epilines[0][0] * (imgDimensions.width - 1) + epilines[0][2]) / epilines[0][1]; // y = -(a*x + c) / b
            const double y1 = -(epilines[1][0] * (imgDimensions.width - 1) + epilines[1][2]) / epilines[1][1]; // y = -(a*x + c) / b
            
            cout << cv::Point2d(imgDimensions.width - 1, y0) << " -- " << endl;
            cout << cv::Point2d(imgDimensions.width - 1, y1) << " -- " << endl;
            
            if ((y0 >= 0) && (y0 <= imgDimensions.height - 1)) 
                minTheta = atan2(y0 - epipole.y, (imgDimensions.width - 1) - epipole.x) + 2 * CV_PI; 
            if ((y1 >= 0) && (y1 <= imgDimensions.height - 1))
                maxTheta = atan2(y1 - epipole.y, (imgDimensions.width - 1) - epipole.x) + 2 * CV_PI;
        }
    } else { // Cases 7, 8 and 9
        if (epipole.x < 0) { // Case 7
            const double y0 = -epilines[0][2] / epilines[0][1]; // y = -(a*x + c) / b
            const double x0 = -(epilines[0][1] * (imgDimensions.height - 1) + epilines[0][2]) / epilines[0][0]; // x = -(b*y + c) / a
            
            const double y1 = -epilines[1][2] / epilines[1][1]; // y = -(a*x + c) / b
            const double x1 = -(epilines[1][1] * (imgDimensions.height - 1) + epilines[1][2]) / epilines[1][0]; // x = -(b*y + c) / a
            
            cout << cv::Point2d(0, y0) << " -- " << cv::Point2d(x0, 0) << endl;
            cout << cv::Point2d(0, y1) << " -- " << cv::Point2d(x1, 0) << endl;
            
            if ((y0 >= 0) && (y0 <= imgDimensions.height - 1)) 
                minTheta = atan2(y0 - epipole.y, -epipole.x) + 2 * CV_PI;
            if ((x0 >= 0) && (x0 <= imgDimensions.width - 1))
                minTheta = atan2((imgDimensions.height - 1) - epipole.y, x0 - epipole.x) + 2 * CV_PI;
            
            if ((y1 >= 0) && (y1 <= imgDimensions.height - 1))
                maxTheta = atan2(y1 - epipole.y, -epipole.x) + 2 * CV_PI;
            if ((x1 >= 0) && (x1 <= imgDimensions.width - 1))
                maxTheta = atan2((imgDimensions.height - 1) - epipole.y, x1 - epipole.x) + 2 * CV_PI;
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 8
            const double x0 = -(epilines[0][1] * (imgDimensions.height - 1) + epilines[0][2]) / epilines[0][0]; // x = -(b*y + c) / a
            const double x1 = -(epilines[1][1] * (imgDimensions.height - 1) + epilines[1][2]) / epilines[1][0]; // x = -(b*y + c) / a
            
            cout << " -- " << cv::Point2d(x0, imgDimensions.height - 1) << endl;
            cout << " -- " << cv::Point2d(x1, imgDimensions.height - 1) << endl;
            
            if ((x0 >= 0) && (x0 <= imgDimensions.width - 1))
                minTheta = atan2((imgDimensions.height - 1) - epipole.y, x0 - epipole.x) + 2 * CV_PI;
            if ((x1 >= 0) && (x1 <= imgDimensions.width - 1))
                maxTheta = atan2((imgDimensions.height - 1) - epipole.y, x1 - epipole.x) + 2 * CV_PI;
        } else { // Case 9
            const double x0 = -(epilines[0][1] * (imgDimensions.height - 1) + epilines[0][2]) / epilines[0][0]; // x = -(b*y + c) / a
            const double x1 = -(epilines[1][1] * (imgDimensions.height - 1) + epilines[1][2]) / epilines[1][0]; // x = -(b*y + c) / a
            
            const double y0 = -(epilines[0][0] * (imgDimensions.width - 1) + epilines[0][2]) / epilines[0][1]; // y = -(a*x + c) / b
            const double y1 = -(epilines[1][0] * (imgDimensions.width - 1) + epilines[1][2]) / epilines[1][1]; // y = -(a*x + c) / b
            
            cout << cv::Point2d(imgDimensions.width - 1, y0) << " -- " << cv::Point2d(x0, imgDimensions.height - 1) << endl;
            cout << cv::Point2d(imgDimensions.width - 1, y1) << " -- " << cv::Point2d(x1, imgDimensions.height - 1) << endl;
            
            if ((x0 >= 0) && (x0 <= imgDimensions.width - 1))
                minTheta = atan2((imgDimensions.height - 1) - epipole.y, x0 - epipole.x) + 2 * CV_PI;
            if ((y0 >= 0) && (y0 <= imgDimensions.height - 1)) 
                minTheta = atan2(y0 - epipole.y, (imgDimensions.width - 1) - epipole.x) + 2 * CV_PI; 
            
            if ((x1 >= 0) && (x1 <= imgDimensions.width - 1))
                maxTheta = atan2((imgDimensions.height - 1) - epipole.y, x1 - epipole.x) + 2 * CV_PI;
            if ((y1 >= 0) && (y1 <= imgDimensions.height - 1))
                maxTheta = atan2(y1 - epipole.y, (imgDimensions.width - 1) - epipole.x) + 2 * CV_PI;
        }
    }
    
    cout << "minTheta (final) " << minTheta * 180 / CV_PI << "(" << (minTheta - 2 * CV_PI) * 180 / CV_PI << ")" << endl;
    cout << "maxTheta (final) " << maxTheta * 180 / CV_PI << "(" << (maxTheta - 2 * CV_PI) * 180 / CV_PI << ")" << endl;
    
    exit(0);
}

void determineRhoRange(const cv::Point2d &epipole, const cv::Size imgDimensions, 
                           const vector<cv::Point2f> &externalPoints, const vector<cv::Vec3f> &epilines,
                           double & minRho, double & maxRho) {
    if (epipole.y < 0) { // Cases 1, 2 and 3
        if (epipole.x < 0) { // Case 1
            minRho = sqrt(epipole.x * epipole.x + epipole.y * epipole.y);         // Point A
            maxRho = sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) + 
                          ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y));        // Point D
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 2
            minRho = -epipole.y;
            maxRho = max(sqrt(epipole.x * epipole.x + 
                            ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y)),        // Point C
                         sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) + 
                            ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y))        // Point D
            );
        } else { // Case 3
            minRho = sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) + 
                            epipole.y * epipole.y);        // Point B
            maxRho = sqrt(epipole.x * epipole.x + 
                        ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y));        // Point C
        }
    } else if (epipole.y <= imgDimensions.height - 1) { // Cases 4, 5 and 6
        if (epipole.x < 0) { // Case 4
            minRho = -epipole.x;
            maxRho = max(
                sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) + 
                    ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y)),        // Point D
                sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) + 
                         epipole.y * epipole.y)        // Point B
            );
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 5
            minRho = 0;
            maxRho = max(
                max(
                    sqrt(epipole.x * epipole.x + epipole.y * epipole.y),        // Point A
                    sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) + 
                        epipole.y * epipole.y)        // Point B
                ),
                max(
                    sqrt(epipole.x * epipole.x + 
                        ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y)),        // Point C
                    sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) + 
                        ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y))        // Point D
                )
            );
        } else { // Case 6
            minRho = epipole.x - (imgDimensions.width - 1);
            maxRho = max(
                sqrt(epipole.x * epipole.x + epipole.y * epipole.y),        // Point A
                sqrt(epipole.x * epipole.x + 
                         ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y))        // Point C
            );
        }        
    } else { // Cases 7, 8 and 9
        if (epipole.x < 0) { // Case 7
            minRho = sqrt(epipole.x * epipole.x + 
                        ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y));        // Point C
            maxRho = sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) + 
                            epipole.y * epipole.y);        // Point B
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 8
            minRho = epipole.y - (imgDimensions.height - 1);
            maxRho = max(
                sqrt(epipole.x * epipole.x + epipole.y * epipole.y),        // Point A
                sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) + 
                         epipole.y * epipole.y)        // Point B
                         
            );
        } else { // Case 9
            minRho = sqrt(((imgDimensions.width - 1) - epipole.x) * ((imgDimensions.width - 1) - epipole.x) + 
                            ((imgDimensions.height - 1) - epipole.y) * ((imgDimensions.height - 1) - epipole.y));        // Point D
            maxRho = sqrt(epipole.x * epipole.x + epipole.y * epipole.y);        // Point A
        }
    }
    
    cout << "minRho = " << minRho << endl;
    cout << "maxRho = " << maxRho << endl;
}

void determineCommonRegion_v1(const cv::Point2d &epipole, const cv::Size imgDimensions, 
                           const vector<cv::Point2f> &externalPoints, const vector<cv::Vec3f> &epilines,
                           double & minTheta, double & maxTheta,
                           double & minRho, double & maxRho) {
    
    if (externalPoints.size() == 2) {
        minTheta = std::numeric_limits<double>::infinity();
        maxTheta = -std::numeric_limits<double>::infinity();
        uint32_t minIdx = externalPoints.size(), maxIdx = externalPoints.size();
        
        for (uint32_t i = 0; i < externalPoints.size(); i++) {
            double currTheta = pseudo_atan2(externalPoints[i].y - epipole.y, externalPoints[i].x - epipole.x);
            
            cout << "currTheta " << currTheta * 180 / CV_PI << endl;
            if (currTheta < minTheta) {
                minTheta = currTheta;
                minIdx = i;
                cout << "minTheta " << minTheta * 180 / CV_PI << endl;
            }

            if (currTheta > maxTheta) {
                maxTheta = currTheta;
                maxIdx = i;
                cout << "maxTheta " << maxTheta * 180 / CV_PI << endl;
            } else { 
                cout << currTheta * 180 / CV_PI << " >? " << maxTheta * 180 / CV_PI << endl;
                cout << currTheta << " >? " << maxTheta << endl;
            }
        }
        
        cout << "minTheta = " << minTheta * 180 / CV_PI << " --> " << externalPoints[minIdx] << endl;
        cout << "maxTheta = " << maxTheta * 180 / CV_PI << " --> " << externalPoints[maxIdx] << endl;
        
//         exit(0);
        
        cv::Point2d intersectionPoints;
        double tmpMinTheta = std::numeric_limits<double>::max();
        double tmpMaxTheta = std::numeric_limits<double>::min();
        for (uint32_t i = 0; i < externalPoints.size(); i++) {
            double y = -(epilines[i][0] * externalPoints[i].x + epilines[i][2]) / epilines[i][1];
            double x = -(epilines[i][1] * externalPoints[i].y + epilines[i][2]) / epilines[i][0];
            
            if ((y >= 0) && (y <= imgDimensions.height - 1)) {
                if (i == minIdx) {
                    tmpMinTheta = min(tmpMinTheta,
                                   pseudo_atan2(y - epipole.y, externalPoints[i].x - epipole.x));
                    cout << "tmpMinTheta = " << tmpMinTheta * 180 / CV_PI << " --> " << cv::Point2f(externalPoints[minIdx].x, y) << endl;
                } else if (i == maxIdx) {
                    tmpMaxTheta = max(tmpMaxTheta,
                                   pseudo_atan2(y - epipole.y, externalPoints[i].x - epipole.x));
                    cout << "tmpMaxTheta = " << tmpMaxTheta * 180 / CV_PI << " --> " << cv::Point2f(externalPoints[minIdx].x, y) << endl;
                }
            }
            if ((x >= 0) && (x <= imgDimensions.width - 1)) {
                if (i == minIdx) {
                    tmpMinTheta = min(tmpMinTheta,
                                   pseudo_atan2(externalPoints[i].y - epipole.y, x - epipole.x));
                    cout << "tmpMinTheta = " << tmpMinTheta * 180 / CV_PI << " --> " << cv::Point2f(x, externalPoints[minIdx].y) << endl;
                } else if (i == maxIdx) {
                    tmpMaxTheta = max(tmpMaxTheta,
                                   pseudo_atan2(externalPoints[i].y - epipole.y, x - epipole.x));
                    cout << "tmpMaxTheta = " << tmpMaxTheta * 180 / CV_PI << " --> " << cv::Point2f(x, externalPoints[minIdx].y) << endl;
                }
            }
        }
        
        cout << "maxThetaJustBefore = " << maxTheta * 180 / CV_PI << " --> " << externalPoints[maxIdx] << endl;
        cout << "tmpMaxThetaJustBefore = " << tmpMaxTheta * 180 / CV_PI << " --> " << externalPoints[maxIdx] << endl;
        
        if (tmpMinTheta > minTheta)
            minTheta = tmpMinTheta;
        if (tmpMaxTheta < maxTheta)
            maxTheta = tmpMaxTheta;
//         if (tmpMinTheta != std::numeric_limits<double>::max())
//             minTheta = max(minTheta, tmpMinTheta);
//         if (tmpMaxTheta != std::numeric_limits<double>::min())
//             maxTheta = min(maxTheta, tmpMaxTheta);
        
        cout << "minTheta = " << minTheta * 180 / CV_PI << " --> " << externalPoints[minIdx] << endl;
        cout << "maxTheta = " << maxTheta * 180 / CV_PI << " --> " << externalPoints[maxIdx] << endl;
    
        // Now we determine the max and min rho
        vector<cv::Point2f> possibleNearestPoints(8);
        possibleNearestPoints[0] = cv::Point2f(0, 0);
        possibleNearestPoints[1] = cv::Point2f(imgDimensions.width - 1, 0);
        possibleNearestPoints[2] = cv::Point2f(0, imgDimensions.height - 1);
        possibleNearestPoints[3] = cv::Point2f(imgDimensions.width - 1, imgDimensions.height - 1);
        possibleNearestPoints[4] = cv::Point2f(epipole.x, 0);
        possibleNearestPoints[5] = cv::Point2f(epipole.x, imgDimensions.height - 1);
        possibleNearestPoints[6] = cv::Point2f(0, epipole.y);
        possibleNearestPoints[7] = cv::Point2f(imgDimensions.width - 1, epipole.y);
        
        minRho = std::numeric_limits<double>::max();
        maxRho = std::numeric_limits<double>::min();
        
        for (vector <cv::Point2f>::iterator it = 
                possibleNearestPoints.begin(); it != possibleNearestPoints.end(); it++) {
        
            if ((it->x >= 0) && (it->y >= 0) &&
                (it->x <= imgDimensions.width - 1) && (it->y <= imgDimensions.height - 1)) {
                    minRho = min(minRho,
                                sqrt((epipole.x - it->x) * (epipole.x - it->x) + 
                                    (epipole.y - it->y) * (epipole.y - it->y)));
                    maxRho = max(maxRho,
                                sqrt((epipole.x - it->x) * (epipole.x - it->x) + 
                                    (epipole.y - it->y) * (epipole.y - it->y)));
            }
        }
        
        cout << "minRho = " << minRho << endl;
        cout << "maxRho = " << maxRho << endl;
    } else { // Case 5
        minTheta = 0;
        maxTheta = 2.1 * M_PI;
        minRho = 0;
        maxRho = std::numeric_limits<double>::min();
        
        vector<cv::Point2f> possibleNearestPoints(8);
        possibleNearestPoints[0] = cv::Point2f(0, 0);
        possibleNearestPoints[1] = cv::Point2f(imgDimensions.width - 1, 0);
        possibleNearestPoints[2] = cv::Point2f(0, imgDimensions.height - 1);
        possibleNearestPoints[3] = cv::Point2f(imgDimensions.width - 1, imgDimensions.height - 1);
        possibleNearestPoints[4] = cv::Point2f(epipole.x, 0);
        possibleNearestPoints[5] = cv::Point2f(epipole.x, imgDimensions.height - 1);
        possibleNearestPoints[6] = cv::Point2f(0, epipole.y);
        possibleNearestPoints[7] = cv::Point2f(imgDimensions.width - 1, epipole.y);
        
        for (vector <cv::Point2f>::iterator it = 
            possibleNearestPoints.begin(); it != possibleNearestPoints.end(); it++) {
            
            maxRho = max(maxRho,
                        sqrt((epipole.x - it->x) * (epipole.x - it->x) + 
                             (epipole.y - it->y) * (epipole.y - it->y)));
        }
        
        cout << "minTheta = " << minTheta << endl;
        cout << "maxTheta = " << maxTheta << endl;
        cout << "minRho = " << minRho << endl;
        cout << "maxRho = " << maxRho << endl;
    }
}

bool lineIntersection(const cv::Point2f & a1, const cv::Point2f & b1,
                           const cv::Point2f & a2, const cv::Point2f & b2,
                           cv::Point2f & intersection) {
    float s1_x, s1_y, s2_x, s2_y;
    s1_x = b1.x - a1.x;     s1_y = b1.y - a1.y;
    s2_x = b2.x - a2.x;     s2_y = b2.y - a2.y;
    
    float s, t;
    s = (-s1_y * (a1.x - a2.x) + s1_x * (a1.y - a2.y)) / (-s2_x * s1_y + s1_x * s2_y);
    t = ( s2_x * (a1.y - a2.y) - s2_y * (a1.x - a2.x)) / (-s2_x * s1_y + s1_x * s2_y);
    
    if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
        // Collision detected
        intersection.x = a1.x + (t * s1_x);
        intersection.y = a1.y + (t * s1_y);
        
        return true;
    }
    
    return false; // No collision
}

bool checkAB(const cv::Point2d &epipole, const double & theta, const cv::Size & imgDimensions, cv::Point2d & b) {
    double dy = -epipole.y / sin(theta);
    uint32_t x = epipole.x + dy * cos(theta);
    cout << "AB: ";
//     cout << "AB: " << cv::Point(x, 0) << endl;
//     cout << cos(theta) << endl;
    if ((x >= 0) && (x <= imgDimensions.width - 1)) {
//         cout << "ok" << endl;
        if (((cos(theta) >= 0) && (x >= epipole.x)) ||
            ((cos(theta) < 0) && (x < epipole.x))) {
//             cout << "ok" << endl;
            b = cv::Point2d(x, 0);
            return true;
        }
    }
    return false;
}

bool checkCD(const cv::Point2d &epipole, const double & theta, const cv::Size & imgDimensions, cv::Point2d & b) {
    double dy = ((imgDimensions.height - 1) - epipole.y) / sin(theta);
    uint32_t x = epipole.x + dy * cos(theta);
    cout << "CD: ";
//     cout << "CD: " << cv::Point2d(x, (imgDimensions.height - 1)) << endl;
//     cout << "dy " << dy << endl;
//     cout << "theta " << theta << endl;
//     cout << "x " << x << endl;
    if ((x >= 0) && (x <= imgDimensions.width - 1)) {
        if (((cos(theta) >= 0) && (x >= epipole.x)) ||
            ((cos(theta) < 0) && (x < epipole.x))) {
            
            b = cv::Point2d(x, (imgDimensions.height - 1));
            return true;
        }
    }
    return false;
}

bool checkAC(const cv::Point2d &epipole, const double & theta, const cv::Size & imgDimensions, cv::Point2d & b) {
    double dx = -epipole.x / cos(theta);
    uint32_t y = epipole.y + dx * sin(theta);
//     cout << "AC: ";
    cout << "AC: " << cv::Point2d(0, y) << endl;
//     cout << sin(theta) << endl;
    if ((y >= 0) && (y <= imgDimensions.height - 1)) {
//         cout << "ok" << endl;
        if (((sin(theta) >= 0) && (y >= epipole.y)) ||
            ((sin(theta) < 0) && (y < epipole.y))) {
            
//             cout << "ok" << endl;
            b = cv::Point2d(0, y);
            return true;
        }
    }
    return false;
}

bool checkBD(const cv::Point2d &epipole, const double & theta, const cv::Size & imgDimensions, cv::Point2d & b) {
    double dx = ((imgDimensions.width - 1) - epipole.x) / cos(theta);
    uint32_t y = epipole.y + dx * (double)sin(theta);
    cout << "BD: ";
//     cout << "BD: " << cv::Point2d(imgDimensions.width - 1, y) << endl;
    if ((y >= 0) && (y <= (imgDimensions.height - 1))) {
        if (((sin(theta) >= 0) && (y >= epipole.y)) ||
            ((sin(theta) < 0) && (y < epipole.y))) {
            
            b = cv::Point2d((imgDimensions.width - 1), y);
            return true;
        }
    }
    return false;
}

void getLineFromPoints(const cv::Point2d & p1, const cv::Point2d & p2, vector<cv::Vec3f> & line) {
    const double m = (p2.y - p1.y) / (p2.x - p1.x);
    const double n = p1.y - m * p1.x;
    
    line.resize(1);
    line[0][0] = m;
    line[0][1] = 1;
    line[0][2] = n;
}

void getLineFromAngle(/*const*/ cv::Point2d &epipole, /*const*/ double & theta,
                      const cv::Size & imgDimensions, cv::Point2d & b, vector<cv::Vec3f> & line) {
    
    b = cv::Point2d (-1, -1);
    // Get the point b
    // Watch out!!! The order must be always: AB, BD, CD, AC
    if (epipole.y < 0) { // Cases 1, 2 and 3
        if (epipole.x < 0) { // Case 1
            if (checkBD(epipole, theta, imgDimensions, b));
            else if(checkCD(epipole, theta, imgDimensions, b));
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 2
            if(checkBD(epipole, theta, imgDimensions, b));
            else if(checkCD(epipole, theta, imgDimensions, b));
            else if (checkAC(epipole, theta, imgDimensions, b));
        } else { // Case 3
            if(checkCD(epipole, theta, imgDimensions, b));
            else if (checkAC(epipole, theta, imgDimensions, b));
        }
    } else if (epipole.y <= imgDimensions.height - 1) { // Cases 4, 5 and 6
        if (epipole.x < 0) { // Case 4
            if (checkAB(epipole, theta, imgDimensions, b));
            else if(checkBD(epipole, theta, imgDimensions, b));
            else if(checkCD(epipole, theta, imgDimensions, b));
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 5
            if (checkAB(epipole, theta, imgDimensions, b));
            else if(checkBD(epipole, theta, imgDimensions, b));
            else if(checkCD(epipole, theta, imgDimensions, b));
            else if(checkAC(epipole, theta, imgDimensions, b));
        } else { // Case 6
            if (checkAB(epipole, theta, imgDimensions, b));
            else if(checkCD(epipole, theta, imgDimensions, b));
            else if(checkAC(epipole, theta, imgDimensions, b));
        }        
    } else { // Cases 7, 8 and 9
        if (epipole.x < 0) { // Case 7
            if (checkAB(epipole, theta, imgDimensions, b));
            else if(checkBD(epipole, theta, imgDimensions, b));
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 8
            if (checkAB(epipole, theta, imgDimensions, b));
            else if(checkBD(epipole, theta, imgDimensions, b));
            else if(checkAC(epipole, theta, imgDimensions, b));
        } else { // Case 9
            if (checkAB(epipole, theta, imgDimensions, b));
            else if(checkAC(epipole, theta, imgDimensions, b));
        }
    }
    
    getLineFromPoints(epipole, b, line);
}

double getThetaIncrement(const cv::Point2d & epipole, const cv::Size & imgDimensions, const cv::Point2d &b) {
    cv::Point2d a;
    if ((epipole.x > -1) && (epipole.x < imgDimensions.width))
        a = cv::Point2d(epipole.x, b.y);
    else
        a = cv::Point2d(b.x, epipole.y);
    
    //     cout << "a = " << a << ", ";
    //     cout << "b = " << b << ", ";
    //     cout << "c = " << epipole << " >>> ";
    
    double dist = sqrt((b.x - epipole.x) * (b.x - epipole.x) +
    (b.y - epipole.y) * (b.y - epipole.y)) / 
    sqrt((a.x - epipole.x) * (a.x - epipole.x) +
    (a.y - epipole.y) * (a.y - epipole.y));
    
    //     cout << "dist = " << sqrt((b.x - epipole.x) * (b.x - epipole.x) + (b.y - epipole.y) * (b.y - epipole.y)) << " / " << 
    //                          sqrt((a.x - epipole.x) * (a.x - epipole.x) + (a.y - epipole.y) * (a.y - epipole.y)) << " = " << 
    //                          dist << endl;
    dist = 1.0;
    
    cv::Point2d b1 = b;
    //     b1.x = b.x;
    //     b1.y = b.y;
    if (b.x == 0) b1.y -= dist;
    else if (b.x == imgDimensions.width - 1) b1.y += dist;
    else if (b.y == 0) b1.x += dist;
    else if (b.y == imgDimensions.height - 1) b1.x -= dist;
    
    //     exit (0);
    //     cv::Point2d b1(b.x + cos(theta + CV_PI / 2.0), b.y + sin(theta + CV_PI / 2.0));
    
    //     cout << b << " -- " << b1 << endl;
    //     cout << (b1.y - epipole.y) << " / " << (b1.x - epipole.x) << endl;
    //     cout << (b.y - epipole.y) << " / " << (b.x - epipole.x) << endl;
    //     cout << "b: " << (atan2((b.y - epipole.y), (b.x - epipole.x)) + 2.0 * CV_PI) * 180.0 / CV_PI << endl;
    //     cout << "b1: " << (atan2((b1.y - epipole.y), (b1.x - epipole.x)) + 2.0 * CV_PI) * 180.0 / CV_PI << endl;
    return atan2((b1.y - epipole.y), (b1.x - epipole.x)) + 2.0 * CV_PI;
}

// double getAwithCandB(const cv::Point2d & epipole, const cv::Point2d & b, const cv::Size & imgDimensions) {
//     cv::Point2i bInt(b.x, b.y);
//     if (bInt.x == 0) return cv::Point2d(0, epipole.y);
//     if (bInt.x == imgDimensions.width - 1) return cv::Point2d(epipole.x, b.y);
// 
//     if (bInt.y == 0) return cv::Point2d(epipole.x, b.y);
//     if (bInt.y == imgDimensions.height - 1) return cv::Point2d(epipole.x, b.y);
// }

void getNextThetaIncrement(/*const*/ cv::Point2d &epipole1, /*const*/ cv::Point2d &epipole2, 
                              /*const*/ double & theta1, /*const*/ double & theta2, 
                              /*const*/ double & minTheta1, /*const*/ double & maxTheta1, 
                              /*const*/ double & minTheta2, /*const*/ double & maxTheta2, 
                              /*const*/ double & maxRho1, /*const*/ double & maxRho2, 
                              const cv::Size & imgDimensions, const cv::Mat & F,
                              /*const*/ double & thetaInc1, /*const*/ double & thetaInc2) {
    
    cv::vector<cv::Vec3f> line1img1, line1img2, line2img1, line2img2;
    cv::Point2d b1img1, b1img2, b2img1, b2img2;
    getLineFromAngle(epipole1, theta1, imgDimensions, b1img1, line1img1);
    getLineFromAngle(epipole2, theta2, imgDimensions, b2img2, line2img2);
    
//     cout << "Showing for angle theta1 " << theta1 * 180 / CV_PI << ": b1img1 = " << b1img1 << endl;
//     cout << "Showing for angle theta2" << theta2 * 180 / CV_PI << ": b2img2 = " << b2img2 << endl;
    
    thetaInc1 = 0.001;
    thetaInc2 = 0.001;
}

void getNextThetaIncrement_v4(/*const*/ cv::Point2d &epipole1, /*const*/ cv::Point2d &epipole2, 
                             /*const*/ double & theta1, /*const*/ double & theta2, 
                             /*const*/ double & minTheta1, /*const*/ double & maxTheta1, 
                             /*const*/ double & minTheta2, /*const*/ double & maxTheta2, 
                             /*const*/ double & maxRho1, /*const*/ double & maxRho2, 
                                const cv::Size & imgDimensions, const cv::Mat & F,
                           /*const*/ double & thetaInc1, /*const*/ double & thetaInc2) {

    cv::vector<cv::Vec3f> line1img1, line1img2, line2img1, line2img2;
    cv::Point2d b1img1, b1img2, b2img1, b2img2;
    getLineFromAngle(epipole1, theta1, imgDimensions, b1img1, line1img1);
    getLineFromAngle(epipole2, theta2, imgDimensions, b2img2, line2img2);
    
    cout << "Showing for angle theta1 " << theta1 * 180 / CV_PI << ": b1img1 = " << b1img1 << endl;
    cout << "Showing for angle theta2" << theta2 * 180 / CV_PI << ": b2img2 = " << b2img2 << endl;
    
    vector<cv::Point2f> points(1);
    points[0] = b1img1;
    cv::computeCorrespondEpilines(points, 2, F, line1img2);
    points[0] = b2img2;
    cv::computeCorrespondEpilines(points, 1, F, line2img1);

    cout << line1img2[0] << endl;
    cout << line2img1[0] << endl;
    
    double theta1img2, theta2img1;
    getThetaFromEpilines(epipole2, imgDimensions, line1img2, theta1img2, minTheta2, maxTheta2);
    getThetaFromEpilines(epipole1, imgDimensions, line2img1, theta2img1, minTheta1, maxTheta1);
    
    cout << "theta1img2: " << theta1img2 * 180 / CV_PI << endl;
    cout << "theta2img1: " << theta2img1 * 180 / CV_PI << endl;
    
    getLineFromAngle(epipole1, theta2img1, imgDimensions, b2img1, line2img1);
    getLineFromAngle(epipole2, theta1img2, imgDimensions, b1img2, line1img2);
    
    cout << "Showing new line for angle theta1img2 " << theta1img2 * 180 / CV_PI << ": b1img2 = " << b1img2 << endl;
    cout << "Showing new line for angle theta2img1 " << theta2img1 * 180 / CV_PI << ": b2img1 = " << b2img1 << endl;
    
    exit(0);
    
    double increment1img1 = getThetaIncrement(epipole1, imgDimensions, b1img1);
    double increment2img1 = getThetaIncrement(epipole2, imgDimensions, b2img1);
    
    double increment1img2 = getThetaIncrement(epipole2, imgDimensions, b1img2);
    double increment2img2 = getThetaIncrement(epipole2, imgDimensions, b2img2);
    
    thetaInc1 = min(increment1img1, increment2img1);
    thetaInc2 = min(increment1img2, increment2img2);
}

double getNextThetaIncrement_v3(/*const*/ cv::Point2d &epipole, /*const*/ double & theta, /*const*/ double & maxRho,
                            const cv::Size & imgDimensions) {
    
//     epipole = cv::Point2d(imgDimensions.width / 2.0, imgDimensions.height / 2.0);
//     epipole = cv::Point2d(10, -10);
//     theta = 10 * CV_PI / 180 + 2 * CV_PI;
//     
//     cout << "epipole = " << epipole << endl;
//     cout << "theta = " << theta * 180 / CV_PI << endl;
    
    cv::Point2d b(-1, -1);
    // Get the point b
    // Watch out!!! The order must be always: AB, BD, CD, AC
    if (epipole.y < 0) { // Cases 1, 2 and 3
        if (epipole.x < 0) { // Case 1
            if (checkBD(epipole, theta, imgDimensions, b));
            else if(checkCD(epipole, theta, imgDimensions, b));
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 2
            if(checkBD(epipole, theta, imgDimensions, b));
            else if(checkCD(epipole, theta, imgDimensions, b));
            else if (checkAC(epipole, theta, imgDimensions, b));
        } else { // Case 3
            if(checkCD(epipole, theta, imgDimensions, b));
            else if (checkAC(epipole, theta, imgDimensions, b));
        }
    } else if (epipole.y <= imgDimensions.height - 1) { // Cases 4, 5 and 6
        if (epipole.x < 0) { // Case 4
            if (checkAB(epipole, theta, imgDimensions, b));
            else if(checkBD(epipole, theta, imgDimensions, b));
            else if(checkCD(epipole, theta, imgDimensions, b));
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 5
            if (checkAB(epipole, theta, imgDimensions, b));
            else if(checkBD(epipole, theta, imgDimensions, b));
            else if(checkCD(epipole, theta, imgDimensions, b));
            else if(checkAC(epipole, theta, imgDimensions, b));
        } else { // Case 6
            if (checkAB(epipole, theta, imgDimensions, b));
            else if(checkCD(epipole, theta, imgDimensions, b));
            else if(checkAC(epipole, theta, imgDimensions, b));
        }        
    } else { // Cases 7, 8 and 9
        if (epipole.x < 0) { // Case 7
            if (checkAB(epipole, theta, imgDimensions, b));
            else if(checkBD(epipole, theta, imgDimensions, b));
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 8
            if (checkAB(epipole, theta, imgDimensions, b));
            else if(checkBD(epipole, theta, imgDimensions, b));
            else if(checkAC(epipole, theta, imgDimensions, b));
        } else { // Case 9
            if (checkAB(epipole, theta, imgDimensions, b));
            else if(checkAC(epipole, theta, imgDimensions, b));
        }
    }
    
//     if (b.x == -1) {
//         cerr << "This point should never be reached. Something bad happened!!!" << endl;
//         cout << b << endl;
//         exit(0);
//     }
    cv::Point2d a;
    if ((epipole.x > -1) && (epipole.x < imgDimensions.width))
        a = cv::Point2d(epipole.x, b.y);
    else
        a = cv::Point2d(b.x, epipole.y);
    
//     cout << "a = " << a << ", ";
//     cout << "b = " << b << ", ";
//     cout << "c = " << epipole << " >>> ";
    
    double dist = sqrt((b.x - epipole.x) * (b.x - epipole.x) +
                        (b.y - epipole.y) * (b.y - epipole.y)) / 
                  sqrt((a.x - epipole.x) * (a.x - epipole.x) +
                      (a.y - epipole.y) * (a.y - epipole.y));
                  
//     cout << "dist = " << sqrt((b.x - epipole.x) * (b.x - epipole.x) + (b.y - epipole.y) * (b.y - epipole.y)) << " / " << 
//                          sqrt((a.x - epipole.x) * (a.x - epipole.x) + (a.y - epipole.y) * (a.y - epipole.y)) << " = " << 
//                          dist << endl;
    dist = 1.0;

    cv::Point2d b1 = b;
//     b1.x = b.x;
//     b1.y = b.y;
    if (b.x == 0) b1.y -= dist;
    else if (b.x == imgDimensions.width - 1) b1.y += dist;
    else if (b.y == 0) b1.x += dist;
    else if (b.y == imgDimensions.height - 1) b1.x -= dist;
    
//     exit (0);
//     cv::Point2d b1(b.x + cos(theta + CV_PI / 2.0), b.y + sin(theta + CV_PI / 2.0));
    
//     cout << b << " -- " << b1 << endl;
//     cout << (b1.y - epipole.y) << " / " << (b1.x - epipole.x) << endl;
//     cout << (b.y - epipole.y) << " / " << (b.x - epipole.x) << endl;
//     cout << "b: " << (atan2((b.y - epipole.y), (b.x - epipole.x)) + 2.0 * CV_PI) * 180.0 / CV_PI << endl;
//     cout << "b1: " << (atan2((b1.y - epipole.y), (b1.x - epipole.x)) + 2.0 * CV_PI) * 180.0 / CV_PI << endl;
    double nextTheta = atan2((b1.y - epipole.y), (b1.x - epipole.x)) + 2.0 * CV_PI;
//     cout << theta * 180.0 / CV_PI << " >>> " << nextTheta * 180.0 / CV_PI << endl;
    
    return nextTheta - theta;
    
//     exit(0);
}

void getNextTheta_v2(const cv::Point2d &epipole, const double & theta, const double & maxRho,
                  const cv::Size & imgDimensions, double & nextTheta) {
    
    cv::Point2f extremePoint(epipole.x + maxRho * cos(theta), epipole.y + maxRho * sin(theta));
//     cout << "extremePoint = " << extremePoint << endl;
    vector<cv::Point2f> cornerPoints(4);
    cornerPoints[0] = cv::Point2f(0, 0);
    cornerPoints[1] = cv::Point2f(imgDimensions.width - 1, 0);
    cornerPoints[2] = cv::Point2f(imgDimensions.width - 1, imgDimensions.height - 1);
    cornerPoints[3] = cv::Point2f(0, imgDimensions.height - 1);
    
    double maxDist = std::numeric_limits<double>::min();
    cv::Point2f oppositeIntersection;
    for (uint32_t i = 0; i < cornerPoints.size(); i++) {
        cv::Point2f intersection;
        if (lineIntersection(epipole, extremePoint, cornerPoints[i], cornerPoints[(i + 1) % cornerPoints.size()], intersection)) {
            double dist = sqrt((epipole.x - intersection.x) * (epipole.x - intersection.x) +
                        (epipole.y - intersection.y) * (epipole.y - intersection.y));
            if (dist > maxDist) {
                maxDist = dist;
                oppositeIntersection = intersection;
            }
        }
    }
    cv::Point2f newPoint = oppositeIntersection;
    newPoint.x += cos(CV_PI / 4 + theta);
    newPoint.y += sin(CV_PI / 4 + theta);
//     cout << "newPoint = " << newPoint << endl;
    nextTheta = pseudo_atan2(newPoint.y - epipole.y, newPoint.x - epipole.x);
    
//     exit(0);
}

void doTransformation(/*const*/ cv::Point2d &epipole1, /*const*/ cv::Point2d &epipole2, 
                      /*const*/ cv::Mat & imgInput1, /*const*/ cv::Mat & imgInput2,
                      cv::Mat & imgTransformed1, cv::Mat & imgTransformed2, 
                      /*const*/ double & minTheta1, /*const*/ double & minTheta2, 
                      /*const*/ double & maxTheta1, /*const*/ double & maxTheta2,
                      /*const*/ double & minRho1, /*const*/ double & minRho2, 
                      /*const*/ double & maxRho1, /*const*/ double & maxRho2,
                      const cv::Mat & F) {
                        
    imgTransformed1 = cv::Mat::zeros(imgInput1.rows, imgInput1.cols, CV_8UC1);
    imgTransformed2 = cv::Mat::zeros(imgInput2.rows, imgInput2.cols, CV_8UC1);
    
    cv::Mat tmpTransformed1 = cv::Mat::zeros(2 * (imgInput1.rows + imgInput1.cols),
                                             maxRho1 - minRho1,
                                             CV_8UC1);
    cv::Mat tmpTransformed2 = cv::Mat::zeros(2 * (imgInput2.rows + imgInput2.cols),
                                             maxRho2 - minRho2,
                                             CV_8UC1);
    uint32_t thetaIdx = 0;
//     double minTheta = max(minTheta1, minTheta2);
//     double maxTheta = min(maxTheta1, maxTheta2);
    
//     minTheta1 = minTheta2 = minTheta;
//     maxTheta1 = maxTheta2 = maxTheta;
    
    for (double theta1 = minTheta1, theta2 = minTheta2; theta1 < maxTheta1 /*&& theta2 < maxTheta2*/ && thetaIdx < tmpTransformed1.rows; thetaIdx++) {
        uint32_t rhoIdx1 = 0, rhoIdx2 = 0;
        for (double rho = minRho1; rho < maxRho1 && rhoIdx1 < tmpTransformed1.cols; rho += 1, rhoIdx1++) {
            // TODO: Do the transformation using remap
            cv::Point2i currPoint1(epipole1.x + cos(theta1) * rho, epipole1.y + sin(theta1) * rho);
            if ((currPoint1.x >= 0) && (currPoint1.y >= 0) &&
                (currPoint1.x < imgInput1.cols) && (currPoint1.y < imgInput1.rows)) {
                
                tmpTransformed1.at<uint8_t>(thetaIdx, rhoIdx1) = imgInput1.at<uint8_t>(currPoint1.y, currPoint1.x);
                imgTransformed1.at<uint8_t>(currPoint1.y, currPoint1.x) = 255;
            }
        }
//         theta2 += 0.5;
//         cout << theta2 * 180 / CV_PI << endl;
//         for (double rho = minRho2; rho < maxRho2 && rhoIdx2 < tmpTransformed2.cols; rho += 1, rhoIdx2++) {
//             cv::Point2i currPoint2(epipole2.x + cos(theta2) * rho, epipole2.y + sin(theta2) * rho);
// //             cout << currPoint2 << endl;
//             if ((currPoint2.x >= 0) && (currPoint2.y >= 0) &&
//                 (currPoint2.x < imgInput2.cols) && (currPoint2.y < imgInput2.rows)) {
//                 
//                 tmpTransformed2.at<uint8_t>(thetaIdx, rhoIdx2) = imgInput2.at<uint8_t>(currPoint2.y, currPoint2.x);
//                 imgTransformed2.at<uint8_t>(currPoint2.y, currPoint2.x) = 255;
//             }
//         }
//         exit(0);
        
        // Get the new theta
//         double incTheta1 = getNextThetaIncrement(epipole1, theta1, maxRho1, cv::Size(imgInput1.cols, imgInput1.rows));
//         double incTheta2 = std::numeric_limits< double>::max(); // getNextThetaIncrement(epipole2, theta2, maxRho2, cv::Size(imgInput1.cols, imgInput1.rows));
//         double incTheta = min(incTheta1, incTheta2);
//         theta1 += incTheta;
//         theta2 += 0;
        double thetaInc1, thetaInc2;
        getNextThetaIncrement(epipole1, epipole2, theta1, theta2, minTheta1, maxTheta1, minTheta2, maxTheta2, 
                                   maxRho1, maxRho2, cv::Size(imgInput1.cols, imgInput1.rows), F, thetaInc1, thetaInc2);
        theta1 += thetaInc1;
        theta2 += thetaInc2;
        
        cout << "Current Theta1 = " << theta1 * 180 / CV_PI << endl;
//         theta1 = newTheta;
        //TODO: Select the smaller increment
    }
    
    cv::imwrite("/tmp/transformed1.png", tmpTransformed1);
    
    
    cv::namedWindow("usedPixels1");
    cv::imshow("usedPixels1", imgTransformed1);
    cv::namedWindow("usedPixels2");
    cv::imshow("usedPixels2", imgTransformed2);
    //     
    cv::resize(tmpTransformed1, imgTransformed1, cv::Size(imgTransformed1.cols, imgTransformed1.rows));
    cv::resize(tmpTransformed2, imgTransformed2, cv::Size(imgTransformed2.cols, imgTransformed2.rows));
    //     cv::waitKey(0);
    //     exit(0);
}

void doTransformation_v1(/*const*/ cv::Point2d &epipole, /*const*/ cv::Mat & imgInput, cv::Mat & imgTransformed, 
                      /*const*/ double & minTheta, /*const*/ double & maxTheta,
                      /*const*/ double & minRho, /*const*/ double & maxRho) {
    
    cout << cv::Size(2 * (imgInput.rows + imgInput.cols), sqrt(imgInput.rows * imgInput.rows + imgInput.cols * imgInput.cols)) << endl;
    imgTransformed = cv::Mat::zeros(imgInput.rows, imgInput.cols, CV_8UC1);
    
    cv::Mat tmpTransformed = cv::Mat::zeros(2 * (imgInput.rows + imgInput.cols),
//                                             2 * (imgInput.rows + imgInput.cols),
//                                             (uint32_t)sqrt(imgInput.rows * imgInput.rows + imgInput.cols * imgInput.cols),
                                            maxRho - minRho,
                                            CV_8UC1);

    cout << "minThetaIn = " << minTheta * 180 / CV_PI << endl;
    
    
    cout << cv::Size(tmpTransformed.cols, tmpTransformed.rows) << endl;
    uint32_t thetaIdx = 0;
    for (double theta = minTheta; theta < maxTheta && thetaIdx < tmpTransformed.rows; thetaIdx++) {
        cout << "theta " << theta * 180 / CV_PI << endl;
        cv::waitKey(0);
        uint32_t rhoIdx = 0;
        for (double rho = minRho; rho < maxRho && rhoIdx < tmpTransformed.cols; rho += (tmpTransformed.cols / (maxRho - minRho)), rhoIdx++) {
//             cout << rhoIdx << endl;
            // TODO: Do the transformation using remap
            cv::Point2i currPoint(epipole.x + cos(theta) * rho, epipole.y + sin(theta) * rho);
            if ((currPoint.x >= 0) && (currPoint.y >= 0) &&
                (currPoint.x < imgInput.cols) && (currPoint.y < imgInput.rows)) {
                    
                if ((thetaIdx > 214) && (thetaIdx < 218)) {
                    cout << cv::Point2d(thetaIdx, rhoIdx) << " --> " << currPoint << " -- " << theta * 180 / CV_PI << endl;
                }

                tmpTransformed.at<uint8_t>(thetaIdx, rhoIdx) = imgInput.at<uint8_t>(currPoint.y, currPoint.x);
                imgTransformed.at<uint8_t>(currPoint.y, currPoint.x) = 255;
            }
        }

        // Get the new theta
        double newTheta = getNextThetaIncrement_v3(epipole, theta, maxRho, cv::Size(imgInput.cols, imgInput.rows));
        theta = newTheta;
    }
    
    cv::imwrite("/tmp/transformed.png", tmpTransformed);
    
    
    cv::namedWindow("usedPixels");
    cv::imshow("usedPixels", imgTransformed);
//     
    cv::resize(tmpTransformed, imgTransformed, cv::Size(imgTransformed.cols, imgTransformed.rows));
//     cv::waitKey(0);
//     exit(0);
}

int main(int argc, char * argv[]) {
  
    PolarCalibration calibrator;
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

        calibrator.compute(img1distorted, img2distorted, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2);
        
        cv::waitKey(0);
        
        cv::Mat img1(img1distorted.rows, img1distorted.cols, CV_8UC1);
        cv::Mat img2(img2distorted.rows, img2distorted.cols, CV_8UC1);
        
        cv::undistort(img1distorted, img1, cameraMatrix1, distCoeffs1);
        cv::undistort(img2distorted, img2, cameraMatrix2, distCoeffs2);
        
//         cv::imshow("img1", img1);
//         cv::imshow("img2", img2);
//         
//         cv::waitKey(0);
//         
//         exit(0);
        
        // We look for correspondences using SURF
        
        // vector of keypoints
        vector<cv::KeyPoint> keypoints1, keypoints2;
        
        cv::SurfFeatureDetector surf(50);
        surf.detect(img1, keypoints1);
        surf.detect(img2, keypoints2);
        
//         cv::Mat outImg1, outImg2;
//         cv::drawKeypoints(img1, keypoints1, outImg1, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DEFAULT);
//         cv::drawKeypoints(img2, keypoints2, outImg2, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DEFAULT);
//         cv::namedWindow("SURF detector img1");
//         cv::imshow("SURF detector img1", outImg1);
//         
//         cv::namedWindow("SURF detector img2");
//         cv::imshow("SURF detector img2", outImg2);
        
        // Descriptors are extracted
        cv::SurfDescriptorExtractor surfDesc;
        cv::Mat descriptors1, descriptors2;
        surfDesc.compute(img1, keypoints1, descriptors1);
        surfDesc.compute(img2, keypoints2, descriptors2);
        
        // Descriptors are matched
        cv::FlannBasedMatcher matcher;
        vector<cv::DMatch> matches;
//         vector < vector<cv::DMatch> > tmpMatches;
        
        matcher.match(descriptors1, descriptors2, matches);
//         matcher.radiusMatch(descriptors1, descriptors2, tmpMatches, 0.1f);
//         for (uint32_t i = 0; i < tmpMatches.size(); i++) {
//             matches.reserve(matches.size() + tmpMatches[i].size());
//             for (uint32_t j = 0; j < tmpMatches[i].size(); j++) {
//                 matches.push_back(tmpMatches[i][j]);
//             }
//         }
        
        nth_element(matches.begin(), matches.begin()+24, matches.end());
        matches.erase(matches.begin()+25, matches.end());
        
        cv::Mat imageMatches;
        cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, imageMatches, cv::Scalar(0,0,255));        
        cv::namedWindow("Matched");
        cv::imshow("Matched", imageMatches);
        
        // Fundamental matrix is found
//         vector<cv::Point2f> points1(matches.size());
//         vector<cv::Point2f> points2(matches.size());
        vector<cv::Point2f> points1, points2;
        
        points1.reserve(matches.size());
        points2.reserve(matches.size());
        
        for (int idx = 0; idx < matches.size(); idx++) {
            const cv::Point2f & p1 = keypoints1[matches[idx].queryIdx].pt;
            const cv::Point2f & p2 = keypoints2[matches[idx].trainIdx].pt;
            
            if (fabs(p1.x - p2.x < 10.0) && fabs(p1.y - p2.y < 10.0)) {
                points1.push_back(p1);
                points2.push_back(p2);
            }
//             points1[idx] = keypoints1[matches[idx].queryIdx].pt;
//             points2[idx] = keypoints2[matches[idx].trainIdx].pt;
        }
        
        cout << "sz = " << points1.size() << endl;
        
        cv::Mat F = cv::findFundamentalMat(points1, points2, CV_FM_LMEDS);
        cout << "F:\n" << F << endl;
        
        // We obtain the epipoles
        cv::SVD svd(F);
        
        cv::Mat e1 = svd.vt.row(2);
        cv::Mat e2 = svd.u.col(2);
        
        cout << "u:\n" << svd.u << endl;
        cout << "vt:\n" << svd.vt << endl;
        cout << "w:\n" << svd.w << endl;
//         cv::transpose(e2, e2);
        
        cout << "e1 = " << e1 << endl;
        cout << "e2 = " << e2 << endl;
        
//         cv::Mat Fe1, Fe2;
//         cv::multiply(F, e1.t(), Fe1);
//         cout << "Fe1\n" << F * e1.t() << endl;
//         cout << "e2TF\n" << e2.t() * F << endl;
//         cv::multiply(F, e2.t(), Fe2);
//         cout << "Fe2\n" << Fe2 << endl;
        
//         exit(0);
        
//         cout << F.mul(e1) << " --- " << F.mul(e2) << endl;

        
        cv::Point2d epipole1 = cv::Point2d(e1.at<double>(0, 0) / e1.at<double>(0, 2), e1.at<double>(0, 1) / e1.at<double>(0, 2));
        cv::Point2d epipole2 = cv::Point2d(e2.at<double>(0, 0) / e2.at<double>(2, 0), e2.at<double>(1, 0) / e2.at<double>(2, 0));
//         cv::Point2d epipole2 = cv::Point2d(e2.at<double>(0, 0) / e2.at<double>(0, 2), e2.at<double>(0, 1) / e2.at<double>(0, 2));
        
//         epipole1 = cv::Point2d(70, 50);
        
        cout << "epipole1: " << epipole1 << endl;
        cout << "epipole2: " << epipole2 << endl;

        // Determine common region
        // We look for epipolar lines
        vector<cv::Vec3f> epilines1, epilines2;
        vector<cv::Point2f> externalPoints1, externalPoints2;
        computeEpilinesBasedOnCase(epipole1, cv::Size(img1.cols, img1.rows), F, 2, externalPoints1, epilines2);
        computeEpilinesBasedOnCase(epipole2, cv::Size(img2.cols, img2.rows), F, 1, externalPoints2, epilines1);

        double minTheta1, maxTheta1, minRho1, maxRho1;
        double minTheta2, maxTheta2, minRho2, maxRho2;
        determineCommonRegion(epipole1, cv::Size(img1.cols, img1.rows), externalPoints1, epilines2, minTheta1, maxTheta1);
        determineCommonRegion(epipole2, cv::Size(img2.cols, img2.rows), externalPoints2, epilines1, minTheta2, maxTheta2);
        
        determineRhoRange(epipole1, cv::Size(img1.cols, img1.rows), externalPoints1, epilines2, minRho1, maxRho1);
        determineRhoRange(epipole2, cv::Size(img2.cols, img2.rows), externalPoints2, epilines1, minRho2, maxRho2);

//         minTheta1 = max(minTheta1, minTheta2);
//         maxTheta1 = min(maxTheta1, maxTheta2);
//         minRho1 = min(minRho1, minRho2);
//         maxRho1 = max(maxRho1, maxRho2);
        
//         minTheta1 = minTheta2 = 0;
//         maxTheta1 = maxTheta2 = 2.1 * CV_PI;
        
        cv::Mat imgTransformed1, imgTransformed2;
        doTransformation(epipole1, epipole2, img1, img2, imgTransformed1, imgTransformed2, 
                         minTheta1, minTheta2, maxTheta1, maxTheta2, minRho1, minRho2, maxRho1, maxRho2, F);
        
        cv::namedWindow("imgTransformed1");
        cv::imshow("imgTransformed1", imgTransformed1);
        cv::namedWindow("imgTransformed2");
        cv::imshow("imgTransformed2", imgTransformed2);
        
        //         
        cv::Mat epipolarOutput1, epipolarOutput2;
        img1.copyTo(epipolarOutput1);
        img2.copyTo(epipolarOutput2);
        for (cv::vector<cv::Vec3f>::const_iterator it = epilines1.begin(); it!=epilines1.end(); ++it)
        {
//             cv::line(epipolarOutput1,
            cv::line(epipolarOutput2,
                     cv::Point(0, -(*it)[2]/(*it)[1]), // y = (-ax - c) / b
                     cv::Point(epipolarOutput1.cols, (-(*it)[0] * epipolarOutput1.cols - (*it)[2]) / (*it)[1]), // y = (-ax - c) / b
                     cv::Scalar(255,0,0));
        }
        cv::circle(epipolarOutput1, epipole1, 5, cv::Scalar(255), -1);
// 
        for (cv::vector<cv::Vec3f>::const_iterator it = epilines2.begin(); it!=epilines2.end(); ++it)
        {
            cv::line(epipolarOutput1,
//             cv::line(epipolarOutput2,
                     cv::Point(0,-(*it)[2]/(*it)[1]),
                     cv::Point(epipolarOutput1.cols,-((*it)[2] + (*it)[0]*epipolarOutput1.cols)/(*it)[1]),
                     cv::Scalar(0,0,0));
        }
        cv::circle(epipolarOutput2, epipole2, 5, cv::Scalar(0), -1);
//         
        cv::namedWindow("Epipolar1");
        cv::imshow("Epipolar1", epipolarOutput1);
        cv::namedWindow("Epipolar2");
        cv::imshow("Epipolar2", epipolarOutput2);
        
        cv::waitKey(0);
//         
        
        
//         cv::Mat transform1, transform2;
//         if (! stereoRectifyUncalibrated(img1, img2, F, cv::Size(img1.cols, img1.rows), transform1, transform2)) {
//             cerr << "Rectification not found!!!" << endl;
//         }
        
//         polarTransform(img1, epipole1, externalPoints1, epilines1, transform1);
//         cout << "polar1" << endl;
//         polarTransform(img2, epipole2, externalPoints2, epilines2, transform2);
//         cout << "polar2" << endl;
        
//         cv::namedWindow("transform1");
//         cv::imshow("transform1", transform1);
//         cv::namedWindow("transform2");
//         cv::imshow("transform2", transform2);
        
//         cv::waitKey(0);
        
//         break;
   }
  
  return 0;
}


