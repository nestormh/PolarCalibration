/*
 *  Copyright 2013 Néstor Morales Hernández <nestor@isaatc.ull.es>
 * 
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <iostream>
#include <stdio.h>
#include <fstream>

#include "polarcalibration.h"

PolarCalibration::PolarCalibration() {

}

PolarCalibration::~PolarCalibration() {

}

void PolarCalibration::compute(const cv::Mat& img1distorted, const cv::Mat& img2distorted, 
                               const cv::Mat & cameraMatrix1, const cv::Mat & distCoeffs1, 
                               const cv::Mat & cameraMatrix2, const cv::Mat & distCoeffs2) {
    
    cv::Mat img1(img1distorted.rows, img1distorted.cols, CV_8UC1);
    cv::Mat img2(img2distorted.rows, img2distorted.cols, CV_8UC1);
    
    cv::undistort(img1distorted, img1, cameraMatrix1, distCoeffs1);
    cv::undistort(img2distorted, img2, cameraMatrix2, distCoeffs2);
    
    compute(img1, img2);
}

void PolarCalibration::compute(/*const*/ cv::Mat& img1, /*const*/ cv::Mat& img2) {
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

void PolarCalibration::computeEpilinesBasedOnCase(const cv::Point2d &epipole, const cv::Size imgDimensions,
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

bool PolarCalibration::getThetaAB(cv::Point2d & epipole, const cv::Vec3f &epiline, const cv::Size &imgDimensions, double & newTheta) {
    const double x = -epiline[2] / epiline[0];
    cout << "getThetaAB: " << cv::Point2d(x, 0) << endl;

    if ((x >= 0) && (x <= (imgDimensions.width - 1))) {
        newTheta = atan2(-epipole.y, x - epipole.x) + 2 * CV_PI;

        return true;
    }
    return false;
}

bool PolarCalibration::getThetaCD(cv::Point2d & epipole, const cv::Vec3f &epiline, const cv::Size &imgDimensions, double & newTheta) {
    const double x = -(epiline[1] * (imgDimensions.height - 1) + epiline[2]) / epiline[0];

    cout << "getThetaCD: " << cv::Point2d(x, imgDimensions.height - 1) << endl;

    if ((x >= 0) && (x <= (imgDimensions.width - 1))) {
        newTheta = atan2((imgDimensions.height - 1) - epipole.y, x - epipole.x) + 2 * CV_PI;

        return true;
    }
    return false;
}

bool PolarCalibration::getThetaBD(cv::Point2d & epipole, const cv::Vec3f &epiline, const cv::Size &imgDimensions, double & newTheta) {
    const double y = -(epiline[0] * (imgDimensions.width - 1) + epiline[2]) / epiline[1];

    cout << "getThetaBD: " << cv::Point2d(imgDimensions.width - 1, y) << endl;

    if ((y >= 0) && (y <= (imgDimensions.height - 1))) {
        newTheta = atan2(y - epipole.y, (imgDimensions.width - 1) - epipole.x) + 2 * CV_PI;

        return true;
    }
    return false;
}

bool PolarCalibration::getThetaAC(cv::Point2d & epipole, const cv::Vec3f &epiline, const cv::Size &imgDimensions, double & newTheta) {
    const double y = -epiline[2] / epiline[1];

    cout << "getThetaAC: " << cv::Point2d(0, y) << endl;

    if ((y >= 0) && (y <= (imgDimensions.height - 1))) {
        newTheta = atan2(y - epipole.y, -epipole.x) + 2 * CV_PI;

        return true;
    }
    return false;
}

void PolarCalibration::getThetaFromEpilines(/*const*/ cv::Point2d &epipole, const cv::Size imgDimensions,
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

void PolarCalibration::determineCommonRegion(/*const*/ cv::Point2d &epipole, const cv::Size imgDimensions,
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

void PolarCalibration::determineRhoRange(const cv::Point2d &epipole, const cv::Size imgDimensions,
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

bool PolarCalibration::checkAB(const cv::Point2d &epipole, const double & theta, const cv::Size & imgDimensions, cv::Point2d & b) {
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

bool PolarCalibration::checkCD(const cv::Point2d &epipole, const double & theta, const cv::Size & imgDimensions, cv::Point2d & b) {
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

bool PolarCalibration::checkAC(const cv::Point2d &epipole, const double & theta, const cv::Size & imgDimensions, cv::Point2d & b) {
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

bool PolarCalibration::checkBD(const cv::Point2d &epipole, const double & theta, const cv::Size & imgDimensions, cv::Point2d & b) {
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

void PolarCalibration::getLineFromPoints(const cv::Point2d & p1, const cv::Point2d & p2, vector<cv::Vec3f> & line) {
    const double m = (p2.y - p1.y) / (p2.x - p1.x);
    const double n = p1.y - m * p1.x;

    line.resize(1);
    line[0][0] = m;
    line[0][1] = 1;
    line[0][2] = n;
}

void PolarCalibration::getLineFromAngle(/*const*/ cv::Point2d &epipole, /*const*/ double & theta,
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

double PolarCalibration::getNextThetaIncrement(/*const*/ cv::Point2d &epipole, /*const*/ double & theta, /*const*/ double & maxRho,
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

void PolarCalibration::doTransformation(/*const*/ cv::Point2d &epipole1, /*const*/ cv::Point2d &epipole2,
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
        double incTheta1 = getNextThetaIncrement(epipole1, theta1, maxRho1, cv::Size(imgInput1.cols, imgInput1.rows));
        //         double incTheta = min(incTheta1, incTheta2);
        theta1 += incTheta1;
        //         theta2 += 0;
//         double thetaInc1, thetaInc2;
//         getNextThetaIncrement(epipole1, epipole2, theta1, theta2, minTheta1, maxTheta1, minTheta2, maxTheta2,
//                               maxRho1, maxRho2, cv::Size(imgInput1.cols, imgInput1.rows), F, thetaInc1, thetaInc2);
//         theta1 += thetaInc1;
//         theta2 += thetaInc2;

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

