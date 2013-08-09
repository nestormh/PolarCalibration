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
#include <boost/concept_check.hpp>

#include "polarcalibration.h"

PolarCalibration::PolarCalibration() {
    m_hessianThresh = 50;
    m_stepSize = STEP_SIZE;
}

PolarCalibration::~PolarCalibration() {

}

bool PolarCalibration::compute(const cv::Mat& img1distorted, const cv::Mat& img2distorted, 
                               const cv::Mat & cameraMatrix1, const cv::Mat & distCoeffs1, 
                               const cv::Mat & cameraMatrix2, const cv::Mat & distCoeffs2) {
    
//     AB: false
//     BD: true
//     DC: true
//     CA: true
//     [0.416418, 0.909173, -649.88]intersects? : true
//     AB: true
//     BD: true
//     DC: false
//     CA: true
//     [-0.768011, 0.640436, 387.121]intersects? : true
//     AB: false
//     BD: true
//     DC: false
//     CA: true
//     [-0.364761, -0.931101, 760.017]intersects? : true
//     AB: false
//     BD: true
//     DC: false
//     CA: true
//     [0.876247, -0.481862, -746.872]intersects? : true
    
//     cv::Vec3b line11(0.0246611, 0.999696, 10.5735);
//     cv::Vec3b line11(-0.364761, -0.931101, 760.017);
//     cout << cv::Point2d(img1distorted.cols - 1, 0) << endl;
//     cout << cv::Point2d(img1distorted.cols - 1, img1distorted.rows - 1) << endl;
//     cout << (lineIntersectsSegment(line11, cv::Point2d(img1distorted.cols - 1, 0), cv::Point2d(img1distorted.cols - 1, img1distorted.rows - 1))? "true" : "false") << endl;
//     
//     exit(0);
    
    cv::Mat img1(img1distorted.rows, img1distorted.cols, CV_8UC1);
    cv::Mat img2(img2distorted.rows, img2distorted.cols, CV_8UC1);
    
    cv::undistort(img1distorted, img1, cameraMatrix1, distCoeffs1);
    cv::undistort(img2distorted, img2, cameraMatrix2, distCoeffs2);
    
    return compute(img1, img2);
}

bool PolarCalibration::compute(/*const*/ cv::Mat& img1, /*const*/ cv::Mat& img2) {
//     epipole2: [203.469, 0.732156]
//     p1 = [56.7193, 479], p2 = [203.468, 0], line1 = [-477.451, -184.654, 115530], line2 = [0.999998, -0.0019771, -203.468]
    
//     getBorderIntersection(cv::Point2d(203.469, 0.732156), cv::Vec3f(0.999998, -0.0019771, -203.468), cv::Size(img1.cols, img1.rows));
//     
//     exit(0);
    
    
    cv::Mat F;
    cv::Point2d epipole1, epipole2, m;
    if (! findFundamentalMat(img1, img2, F, epipole1, epipole2, m))
        return false;
    
    cout << "epipole1: " << epipole1 << endl;
    cout << "epipole2: " << epipole2 << endl;

    // Determine common region
    // We look for epipolar lines
    vector<cv::Vec3f> epilines1, epilines2;
    vector<cv::Point2f> externalPoints1, externalPoints2;
    computeEpilinesBasedOnCase(epipole1, cv::Size(img1.cols, img1.rows), F, 2, m, externalPoints1, epilines2);
    computeEpilinesBasedOnCase(epipole2, cv::Size(img2.cols, img2.rows), F, 1, m, externalPoints2, epilines1);
    
//     double minTheta1, maxTheta1, minRho1, maxRho1;
//     double minTheta2, maxTheta2, minRho2, maxRho2;
    vector<cv::Vec3f> initialEpilines, finalEpilines;
    vector<cv::Point2f> epipoles(2);
    epipoles[0] = epipole1;
    epipoles[1] = epipole2;    
    determineCommonRegion(epipoles, cv::Size(img1.cols, img1.rows), F);
    
    doTransformation(img1, epipole1, epipole2, F);
    
//     exit(0);
    /*determineCommonRegion(epipole2, cv::Size(img2.cols, img2.rows), externalPoints2, epilines1, minTheta2, maxTheta2);
    
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
    cv::imshow("imgTransformed2", imgTransformed2);*/
    
    //         
    /*cv::Mat epipolarOutput1, epipolarOutput2;
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
    cv::imshow("Epipolar2", epipolarOutput2);*/
    
//     cv::waitKey(0);
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
    return true;
}

bool PolarCalibration::findFundamentalMat(const cv::Mat & img1, const cv::Mat & img2, cv::Mat & F, 
                                          cv::Point2d & epipole1, cv::Point2d & epipole2, cv::Point2d & m) {
    
    // We look for correspondences using SURF
    
    // vector of keypoints
    vector<cv::KeyPoint> keypoints1, keypoints2;
    
    cv::SurfFeatureDetector surf(m_hessianThresh);
    surf.detect(img1, keypoints1);
    surf.detect(img2, keypoints2);
    
    // Descriptors are extracted
    cv::SurfDescriptorExtractor surfDesc;
    cv::Mat descriptors1, descriptors2;
    surfDesc.compute(img1, keypoints1, descriptors1);
    surfDesc.compute(img2, keypoints2, descriptors2);
    
    // Descriptors are matched
    cv::FlannBasedMatcher matcher;
    vector<cv::DMatch> matches;
    
    matcher.match(descriptors1, descriptors2, matches);
    
    nth_element(matches.begin(), matches.begin()+24, matches.end());
    matches.erase(matches.begin()+25, matches.end());
    
//     cv::Mat imageMatches;
//     cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, imageMatches, cv::Scalar(0,0,255));        
//     cv::namedWindow("Matched");
//     cv::imshow("Matched", imageMatches);
    
    // Fundamental matrix is found
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
    }
    
    if (points1.size() < 8)
        return false;
    
    cout << "sz = " << points1.size() << endl;
    
    F = cv::findFundamentalMat(points1, points2, CV_FM_LMEDS);
    cout << "F:\n" << F << endl; 
    
    if (cv::countNonZero(F) == 0)
        return false;
    
    // We obtain the epipoles
    getEpipoles(F, epipole1, epipole2);
    
    checkF(F, epipole1, epipole2, points1[0], points2[0]);
        
    /// NOTE: Remove. Just for debugging (begin)
    cv::FileStorage file("/home/nestor/Dropbox/KULeuven/projects/PolarCalibration/testing/lastMat.xml", cv::FileStorage::READ);
    file["F"] >> F;
    file.release();
    cout << "F (from file)\n" << F << endl;
    getEpipoles(F, epipole1, epipole2);
    /// NOTE: Remove. Just for debugging (end)
    
    m = points1[0];
    
    return true;
}

void PolarCalibration::getEpipoles(const cv::Mat & F, cv::Point2d & epipole1, cv::Point2d & epipole2) {
    cv::SVD svd(F);
    
    cv::Mat e1 = svd.vt.row(2);
    cv::Mat e2 = svd.u.col(2);
    
    epipole1 = cv::Point2d(e1.at<double>(0, 0) / e1.at<double>(0, 2), e1.at<double>(0, 1) / e1.at<double>(0, 2));
    epipole2 = cv::Point2d(e2.at<double>(0, 0) / e2.at<double>(2, 0), e2.at<double>(1, 0) / e2.at<double>(2, 0));
}

void PolarCalibration::checkF(cv::Mat & F, cv::Point2d & epipole1, cv::Point2d & epipole2, const cv::Point2d & m, const cv::Point2d & m1) {
    cv::Vec3f line = getLineFromTwoPoints(epipole1, m);
    vector<cv::Point2f> points(1);
    points[0] = epipole1;
    vector<cv::Vec3f> lines;
    cv::computeCorrespondEpilines(points, 1, F, lines);
    cv::Vec3f line1 = lines[0];
    
    cv::Mat L(1, 3, CV_64FC1);
    L.at<double>(0, 0) = line[0];
    L.at<double>(0, 1) = line[1];
    L.at<double>(0, 2) = line[2];
    
    cv::Mat L1(1, 3, CV_64FC1);
    L1.at<double>(0, 0) = line1[0];
    L1.at<double>(0, 1) = line1[1];
    L1.at<double>(0, 2) = line1[2];
    
    cv::Mat M(3, 1, CV_64FC1);
    M.at<double>(0, 0) = m.x;
    M.at<double>(1, 0) = m.y;
    M.at<double>(2, 0) = 1.0;
    
    cv::Mat M1(3, 1, CV_64FC1);
    M1.at<double>(0, 0) = m1.x;
    M1.at<double>(1, 0) = m1.y;
    M1.at<double>(2, 0) = 1.0;
    
    cv::Mat fl = L * M;
    cv::Mat fl1 = L1 * M1;
    
    if (((fl.at<double>(0,0) < 0.0) && (fl1.at<double>(0,0) > 0.0)) ||
        ((fl.at<double>(0,0) > 0.0) && (fl1.at<double>(0,0) < 0.0))) {

        F = -F;
    }
}

void PolarCalibration::computeEpilinesBasedOnCase(const cv::Point2d &epipole, const cv::Size imgDimensions,
        const cv::Mat & F, const uint32_t & imgIdx, const cv::Point2d & m,
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

bool PolarCalibration::lineIntersectsSegment(const cv::Vec3d & line, const cv::Point2d & p1, const cv::Point2d & p2, cv::Point2d * intersection) {
    const cv::Vec3d segment = getLineFromTwoPoints(p1, p2);
    
    if (intersection != NULL)
        *intersection = cv::Point2d(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    // Lines are represented as ax + by + c = 0, so
    // y = -(ax+c)/b. If y1=y2, then we have to obtain x, which is
    // x = (b1 * c2 - b2 * c1) / (b2 * a1 - b1 * a2)
    if ((segment[1] * line[0] - line[1] * segment[0]) == 0)
        return false;
    double x = (line[1] * segment[2] - segment[1] * line[2]) / (segment[1] * line[0] - line[1] * segment[0]);
    double y = -(line[0] * x + line[2]) / line[1];
    
    cout << "Possible intersection " << cv::Point2d(x, y) << endl;
    
    if (((int32_t)round(x) >= (int32_t)min(p1.x, p2.x)) && ((int32_t)round(x) <= (int32_t)max(p1.x, p2.x))) {
        if (((int32_t)round(y) >= (int32_t)min(p1.y, p2.y)) && ((int32_t)round(y) <= (int32_t)max(p1.y, p2.y))) {
            if (intersection != NULL)
                *intersection = cv::Point2d(x, y);
            
            return true;
        }
    }
    
    return false;
}

bool PolarCalibration::lineIntersectsRect(const cv::Vec3d & line, const cv::Size & imgDimensions) {
    return lineIntersectsSegment(line, cv::Point2d(0, 0), cv::Point2d(imgDimensions.width - 1, 0)) ||
            lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, 0), cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1)) ||
            lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), cv::Point2d(0, imgDimensions.height - 1)) ||
            lineIntersectsSegment(line, cv::Point2d(0, imgDimensions.height - 1), cv::Point2d(0, 0));
}

/*bool PolarCalibration::lineIntersectsLine(const cv::Point2d & l1p1, const cv::Point2d & l1p2, 
                                          const cv::Point2d & l2p1, const cv::Point2d & l2p2)
{
    float q = (l1p1.y - l2p1.y) * (l2p2.x - l2p1.x) - (l1p1.x - l2p1.x) * (l2p2.y - l2p1.y);
    float d = (l1p2.x - l1p1.x) * (l2p2.y - l2p1.y) - (l1p2.y - l1p1.y) * (l2p2.x - l2p1.x);
    
    if( d == 0 )
    {
        return false;
    }
    
    float r = q / d;
    
    q = (l1p1.y - l2p1.y) * (l1p2.x - l1p1.x) - (l1p1.x - l2p1.x) * (l1p2.y - l1p1.y);
    float s = q / d;
    
    if( r < 0 || r > 1 || s < 0 || s > 1 )
    {
        return false;
    }
    
    return true;
}*/

bool PolarCalibration::isInsideImage(const cv::Point2d & point, const cv::Size & imgDimensions) {
    cv::Point2i p(point.x, point.y);
    if ((p.x >= 0) && (p.y >= 0) &&
        (p.x < imgDimensions.width) && (p.y < imgDimensions.height)) {
            return true;
    }
    return false;
}

// (py – qy)x + (qx – px)y + (pxqy – qxpy) = 0
cv::Vec3f PolarCalibration::getLineFromTwoPoints(const cv::Point2d & point1, const cv::Point2d & point2) {
    return cv::Vec3f(point1.y - point2.y, point2.x - point1.x, point1.x * point2.y - point2.x * point1.y);
}

bool PolarCalibration::isTheRightPoint(const cv::Point2d* lastPoint, const cv::Point2d& intersection)
{
    if (lastPoint != NULL) {
        double dist2 = (lastPoint->x - intersection.x) * (lastPoint->x - intersection.x) +
                        (lastPoint->y - intersection.y) * (lastPoint->y - intersection.y);
        
        if (dist2 < m_stepSize * m_stepSize)
            return true;
    }
    return false;
}

bool PolarCalibration::isTheRightPoint(const cv::Point2d & epipole, const cv::Point2d & intersection, const cv::Vec3d & line) {
    if ((line[0] > 0) && (epipole.y < intersection.y)) return false;
    if ((line[0] < 0) && (epipole.y > intersection.y)) return false;
    if ((line[1] > 0) && (epipole.x > intersection.x)) return false;
    if ((line[1] < 0) && (epipole.x < intersection.x)) return false;
//     cv::Vec3f v(-line[0], -line[1], 0);
//     cv::Vec3f v2(intersection.x - epipole.y, intersection.y - epipole.y, 0);
//     if (v.cross(v2)[2] > 0.0)
//         return false;
//     cout << "v " << v << endl;
//     cout << "v2 " << v2 << endl;
//     cout << "vCross " << v.cross(v2) << endl;
    
    return true;
}

cv::Point2d PolarCalibration::getBorderIntersection(const cv::Point2d & epipole, const cv::Vec3d & line, const cv::Size & imgDimensions, const cv::Point2d * lastPoint) {
    cv::Point2d intersection;
    // TODO: Look for the fartest point in the border
    // Then, create another function that looks for the nearest point
    // Hint: attending to the way in which lines are constructed, we can know which is the right cross with the border
    cout << "Looking for an intersection for " << line << ", using epipole " << epipole << endl;
//     cv::Vec3f v(-line[0], -line[1], 0);
    if (isInsideImage(epipole, imgDimensions)) {
        if (lineIntersectsSegment(line, cv::Point2d(0, 0), cv::Point2d(imgDimensions.width - 1, 0), &intersection)) {
            cout << "Testing AB " << intersection << endl;
            if (isTheRightPoint(epipole, intersection, line)) {
//             if (isTheRightPoint(lastPoint, intersection)) {
                cout << "VALID" << endl;
//                 cv::Vec3f v2(intersection.x - epipole.x, intersection.y - epipole.y, 0);
//                 cout << "v " << v << endl;
//                 cout << "v2 " << v2 << endl;
//                 cout << "vCross " << v.cross(v2) << endl;
//                 cout v.cross()
                return intersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, 0), cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), &intersection)) {
            cout << "Testing DB " << intersection << endl;
            if (isTheRightPoint(epipole, intersection, line)) {
//             if (isTheRightPoint(lastPoint, intersection)) {
                cout << "VALID" << endl;
                return intersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), cv::Point2d(0, imgDimensions.height - 1), &intersection)) {
            cout << "Testing CD " << intersection << endl;
//             cv::Vec3f v2(intersection.x - epipole.y, intersection.y - epipole.y, 0);
//             cout << "v " << v << endl;
//             cout << "v2 " << v2 << endl;
//             cout << "vCross " << v.cross(v2) << endl;
            if (isTheRightPoint(epipole, intersection, line)) {
//             if (isTheRightPoint(lastPoint, intersection)) {
                cout << "VALID" << endl;
                return intersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(0, imgDimensions.height - 1), cv::Point2d(0, 0), &intersection)) {
            cout << "Testing AC " << intersection << endl;
            if (isTheRightPoint(epipole, intersection, line)) {
//             if (isTheRightPoint(lastPoint, intersection)) {
                cout << "VALID" << endl;
                return intersection;
            }
        }
    } else {
        double maxDist = std::numeric_limits<double>::min();
        cv::Point2d tmpIntersection;
        cout << "AB" << endl;
        if (lineIntersectsSegment(line, cv::Point2d(0, 0), cv::Point2d(imgDimensions.width - 1, 0), &tmpIntersection)) {
            double dist2 = (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x) +
                            (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x);
            
            if (dist2 > maxDist) {
                maxDist = dist2;
                intersection = tmpIntersection;
            }
        }
        cout << "DB" << endl;
        if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, 0), cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), &tmpIntersection)) {
            double dist2 = (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x) +
            (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x);
            
            if (dist2 > maxDist) {
                maxDist = dist2;
                intersection = tmpIntersection;
            }
        }
        cout << "DC" << endl;
        if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), cv::Point2d(0, imgDimensions.height - 1), &tmpIntersection)) {
            double dist2 = (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x) +
            (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x);
            
            if (dist2 > maxDist) {
                maxDist = dist2;
                intersection = tmpIntersection;
            }
        }
        cout << "AC" << endl;
        if (lineIntersectsSegment(line, cv::Point2d(0, imgDimensions.height - 1), cv::Point2d(0, 0), &tmpIntersection)) {
            double dist2 = (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x) +
            (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x);
            
            if (dist2 > maxDist) {
                maxDist = dist2;
                intersection = tmpIntersection;
            }
        }
        return intersection;
    }
}

void PolarCalibration::getExternalPoints(const cv::Point2d &epipole, const cv::Size imgDimensions,
                                         vector<cv::Point2f> &externalPoints) {
    
    if (epipole.y < 0) { // Cases 1, 2 and 3
        if (epipole.x < 0) { // Case 1
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(imgDimensions.width - 1, 0);
            externalPoints[1] = cv::Point2f(0, imgDimensions.height - 1);
        } else if (epipole.x <= imgDimensions.width - 1) { // Case 2
            externalPoints.resize(2);
            externalPoints[0] = cv::Point2f(0, 0);
            externalPoints[1] = cv::Point2f(imgDimensions.width - 1, 0);
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
}

bool PolarCalibration::sign(const double & val) {
    return val >= 0.0;
}

void PolarCalibration::computeEpilines(const vector<cv::Point2f> & points, const uint32_t &whichImage, 
                                       const cv::Mat & F, const vector <cv::Vec3f> & oldlines, vector <cv::Vec3f> & newLines) {
 
    cv::computeCorrespondEpilines(points, whichImage, F, newLines);
    
    for (uint32_t i = 0; i < oldlines.size(); i++) {
        if ((sign(oldlines[i][0]) != sign(newLines[i][0])) && 
            (sign(oldlines[i][1]) != sign(newLines[i][1]))) {
            newLines[i] *= -1;
        }
    }
}

/**
 * This function is more easily understandable after reading section 3.4 of 
 * ftp://cmp.felk.cvut.cz/pub/cmp/articles/matousek/Sandr-TR-2009-04.pdf
 * */
void PolarCalibration::determineCommonRegion(/*const*/ vector<cv::Point2f> &epipoles, 
                                             const cv::Size imgDimensions, const cv::Mat & F) {

    cout << "************** determineCommonRegion **************" << endl;
    vector<cv::Point2f> externalPoints1, externalPoints2;
    getExternalPoints(epipoles[0], imgDimensions, externalPoints1);
    getExternalPoints(epipoles[1], imgDimensions, externalPoints2);

    if (!isInsideImage(epipoles[0], imgDimensions) && !isInsideImage(epipoles[1], imgDimensions)) {
        // CASE 1: Both outside
        cout << "CASE 1: Both outside" << endl;
        const cv::Vec3f line11 = getLineFromTwoPoints(epipoles[0], externalPoints1[0]);
        const cv::Vec3f line12 = getLineFromTwoPoints(epipoles[0], externalPoints1[1]);
        
        const cv::Vec3f line23 = getLineFromTwoPoints(epipoles[1], externalPoints2[0]);
        const cv::Vec3f line24 = getLineFromTwoPoints(epipoles[1], externalPoints2[1]);
        
//         vector <cv::Vec3f> tmpLines;
//         cv::computeCorrespondEpilines(externalPoints2, 2, F, tmpLines);
        vector <cv::Vec3f> inputLines(2), outputLines;
        inputLines[0] = line23;
        inputLines[1] = line24;
        computeEpilines(externalPoints2, 2, F, inputLines, outputLines);
        const cv::Vec3f line13 = outputLines[0];
        const cv::Vec3f line14 = outputLines[1];
        
        
//         cv::computeCorrespondEpilines(externalPoints1, 1, F, outputLines);
        inputLines[0] = line11;
        inputLines[1] = line12;
        computeEpilines(externalPoints1, 1, F, inputLines, outputLines);
        const cv::Vec3f line21 = outputLines[0];
        const cv::Vec3f line22 = outputLines[1];
        
//         cv::Vec3f m_line1B, m_line1E, m_line2B, m_line2E;
        m_line1B = lineIntersectsRect(line13, imgDimensions)? line13 : line11;
        m_line1E = lineIntersectsRect(line14, imgDimensions)? line14 : line12;
        m_line2B = lineIntersectsRect(line21, imgDimensions)? line21 : line23;
        m_line2E = lineIntersectsRect(line22, imgDimensions)? line22 : line24;
        
        cout << "m_line1B: " << m_line1B << endl;
        cout << "m_line1E: " << m_line1E << endl;
        cout << "m_line2B: " << m_line2B << endl;
        cout << "m_line2E: " << m_line2E << endl;
        
        m_b1 = getBorderIntersection(epipoles[0], m_line1B, imgDimensions);
        m_b2 = getBorderIntersection(epipoles[1], m_line2B, imgDimensions);
        
        showCommonRegion(epipoles[0], line11, line12, line13, line14, m_line1B, m_line1E, m_b1, imgDimensions, 
                         externalPoints1, std::string("leftCommonRegion"));
        showCommonRegion(epipoles[1], line23, line24, line21, line22, m_line2B, m_line2E, m_b2, imgDimensions, 
                         externalPoints2, std::string("rightCommonRegion"));
        
    } else if (isInsideImage(epipoles[0], imgDimensions) && isInsideImage(epipoles[1], imgDimensions)) {
        // CASE 2: Both inside
        cout << "CASE 2: Both inside" << endl;
        m_line1B = getLineFromTwoPoints(epipoles[0], externalPoints1[0]);
        m_line1E = m_line1B;
        
//         vector <cv::Vec3f> tmpLines;
//         cv::computeCorrespondEpilines(externalPoints1, 1, F, tmpLines);
        vector <cv::Vec3f> inputLines(1), outputLines;
        inputLines[0] = m_line1B;
        computeEpilines(externalPoints1, 1, F, inputLines, outputLines);
        
        m_line2B = outputLines[0];
        m_line2E = outputLines[0];
        
        m_b1 = getBorderIntersection(epipoles[0], m_line1B, imgDimensions);
        m_b2 = getBorderIntersection(epipoles[1], m_line2B, imgDimensions);
        
        showCommonRegion(epipoles[0], m_line1B, m_line1E, m_line2B, m_line2E, m_line1B, m_line1E, m_b1, imgDimensions, 
                         externalPoints1, std::string("leftCommonRegion"));
//         showCommonRegion(epipoles[1], m_line2B, m_line2E, m_line1B, m_line1E, m_line2B, m_line2E, m_b2, imgDimensions, 
//                          externalPoints2, std::string("rightCommonRegion"));
    } else {
        // CASE 3: One inside and one outside
        cout << "CASE 3: One inside and one outside" << endl;
        if (isInsideImage(epipoles[0], imgDimensions)) {
            // CASE 3.1: Only the first epipole is inside
            cout << "CASE 3.1: Only the first epipole is inside" << endl;
            
            const cv::Vec3f line23 = getLineFromTwoPoints(epipoles[1], externalPoints2[0]);
            const cv::Vec3f line24 = getLineFromTwoPoints(epipoles[1], externalPoints2[1]);
            
//             vector <cv::Vec3f> tmpLines;
//             cv::computeCorrespondEpilines(externalPoints2, 2, F, tmpLines);
            vector <cv::Vec3f> inputLines(1), outputLines;
            inputLines[0] = line23;
            computeEpilines(externalPoints2, 2, F, inputLines, outputLines);
            const cv::Vec3f & line13 = outputLines[0];
            const cv::Vec3f & line14 = outputLines[1];
            
            m_line1B = line13;
            m_line1E = line14;
            m_line2B = line23;
            m_line2E = line24;
            
            m_b1 = getBorderIntersection(epipoles[0], m_line1B, imgDimensions);
            m_b2 = getBorderIntersection(epipoles[1], m_line2B, imgDimensions);
            
            showCommonRegion(epipoles[0], line13, line14, line13, line14, m_line1B, m_line1E, m_b1, imgDimensions, 
                             externalPoints1, std::string("leftCommonRegion"));
//             showCommonRegion(epipoles[1], line23, line24, line23, line24, m_line2B, m_line2E, m_b2, imgDimensions, 
//                              externalPoints2, std::string("rightCommonRegion"));
//             cv::waitKey(0);
//             exit(0);
        } else {
            // CASE 3.2: Only the second epipole is inside
            cout << "CASE 3.2: Only the second epipole is inside" << endl;
            const cv::Vec3f line11 = getLineFromTwoPoints(epipoles[0], externalPoints1[0]);
            const cv::Vec3f line12 = getLineFromTwoPoints(epipoles[0], externalPoints1[1]);
            
//             vector <cv::Vec3f> tmpLines;
//             cv::computeCorrespondEpilines(externalPoints1, 1, F, tmpLines);
            vector <cv::Vec3f> inputLines(1), outputLines;
            inputLines[0] = line11;
            computeEpilines(externalPoints1, 1, F, inputLines, outputLines);
            const cv::Vec3f & line21 = outputLines[0];
            const cv::Vec3f & line22 = outputLines[1];
            
            m_line1B = line11;
            m_line1E = line12;
            m_line2B = line21;
            m_line2E = line22;
            
            m_b1 = getBorderIntersection(epipoles[0], m_line1B, imgDimensions);
            m_b2 = getBorderIntersection(epipoles[1], m_line2B, imgDimensions);
            
            showCommonRegion(epipoles[0], line11, line12, line11, line12, m_line1B, m_line1E, m_b1, imgDimensions, 
                             externalPoints1, std::string("leftCommonRegion"));
//             showCommonRegion(epipoles[1], line21, line22, line21, line22, m_line2B, m_line2E, m_b2, imgDimensions, 
//                              externalPoints2, std::string("rightCommonRegion"));
        }
    }
    
//     cv::moveWindow("rightCommonRegion", imgDimensions.width + 10, 0);
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

// return nextTheta - theta;
return 0.001;

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

        cout << "Current Theta1 = " << theta1 * 180 / CV_PI << " < " << maxTheta1 * 180 / CV_PI << endl;
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

cv::Point2d PolarCalibration::image2World(const cv::Point2d & point, const cv::Size & imgDimensions) {
//     cv::Rect realWorldLimits(-imgDimensions.width / 2.0, -imgDimensions.height / 2.0, imgDimensions.width * 1.5, imgDimensions.width * 1.5);
    return cv::Point(point.x * 0.5 + imgDimensions.width / 4.0, /*imgDimensions.height -*/ (point.y * 0.5  + imgDimensions.height / 4.0));
}

cv::Point2d PolarCalibration::getPointFromLineAndX(const double & x, const cv::Vec3f line) {
    return cv::Point2d(x, -(line[0] * x + line[2]) / line[1]);
}

cv::Point2d PolarCalibration::getPointFromLineAndY(const double & y, const cv::Vec3f line) {
    return cv::Point2d(-(line[1] * y + line[2]) / line[0]);
}

void PolarCalibration::getNewPointAndLineSingleImage(const cv::Point2d epipole1, const cv::Point2d epipole2, const cv::Size & imgDimensions, 
                                   const cv::Mat & F, const uint32_t & whichImage, const cv::Point2d & pOld,
                                   /*const*/ cv::Vec3f & prevLine, cv::Point2d & pNew1, cv::Vec3f & newLine1, 
                                   cv::Point2d & pNew2, cv::Vec3f & newLine2) {
    
    cv::Vec3f vBegin(m_line1B[0], m_line1B[1], m_line1B[2]);
    cv::Vec3f vEnd(m_line1E[0], m_line1E[1], m_line1E[2]);
    cv::Vec3f vCross = vEnd.cross(vBegin);
    cv::Vec3f v = vCross.cross(vBegin);
    
//     cout << "vBegin " << vBegin << endl;
//     cout << "vEnd " << vEnd << endl;
    cout << "vCross " << vCross << endl;
//     cout << "pOld = " << pOld << endl;
    
    if (isInsideImage(epipole1, imgDimensions)) {
        cout << "It is inside the image" << endl;
        vBegin = cv::Vec3f(m_line2B[0], m_line2B[1], m_line2B[2]);
        vEnd = cv::Vec3f(m_line2E[0], m_line2E[1], m_line2E[2]);
        vCross = vEnd.cross(vBegin);
        v = vCross.cross(vBegin);
        
        if (isInsideImage(epipole1, imgDimensions)) {
            v = cv::Vec3f(-prevLine[0], -prevLine[1], 0);
        }
//         cout << "vBegin " << vBegin << endl;
//         cout << "vEnd " << vEnd << endl;
//         cout << "vCross " << vCross << endl;
//         exit(0);
    }
    
    prevLine = getLineFromTwoPoints(epipole1, pOld);
    
//     cv::Vec2f v(-prevLine[0], -prevLine[1]);
//     cv::Vec3f v = vCross.cross(m_line1B);
    cout << "v " << v << endl;
    if (vCross[2] < 0.0) {
        v = cv::Vec3f(prevLine[0], prevLine[1], 0);
    }
    double v_norm = sqrt(v[0] * v[0] + v[1] * v[1]);
    v /= v_norm;
    
    
    pNew1 = cv::Point2d(pOld.x + v[0] * m_stepSize, pOld.y + v[1] * m_stepSize);
    
    newLine1 = getLineFromTwoPoints(epipole1, pNew1);
    pNew1 = getBorderIntersection(epipole1, newLine1, imgDimensions/*, &pOld*/);

    vector<cv::Point2f> points(1);
    points[0] = pNew1;
    vector<cv::Vec3f> inLines(1);
    inLines[0] = prevLine;
    vector<cv::Vec3f> outLines(1);
    computeEpilines(points, whichImage, F, inLines, outLines);
    cout << "prevLine " << prevLine << ", v " << v << ", pNew1 " << pNew1 << ", newLine1 " << newLine1 << ", pNew2 " << pNew2 << ", newLine2 " << newLine2 << endl;

    newLine2 = outLines[0];
    cv::Point2d tmpPoint = getBorderIntersection(epipole2, newLine2, imgDimensions/*, &pNew2*/);
    pNew2 = tmpPoint;
}

bool PolarCalibration::isEndReached(const cv::Vec3f & currLine, const cv::Vec3f & endLine) {
    cv::Vec2f vCurrent(-currLine[0], -currLine[1]);
    cv::Vec2f vStop(-endLine[0], -endLine[1]);
    
    
    cout << "vCurrent " << vCurrent << endl;
    cout << "vStop " << vStop << endl;
    cout << "dot " << vCurrent.dot(vStop) << endl;
    cout << "dot " << vStop.dot(vCurrent) << endl;
    
    return false;
}

void PolarCalibration::getNewEpiline(const cv::Point2d epipole1, const cv::Point2d epipole2, const cv::Size & imgDimensions, 
                                     const cv::Mat & F, const cv::Point2d pOld1, const cv::Point2d pOld2, 
                                     /*const*/ cv::Vec3f prevLine1, /*const*/ cv::Vec3f prevLine2, 
                                     cv::Point2d & pNew1, cv::Point2d & pNew2, cv::Vec3f & newLine1, cv::Vec3f & newLine2) {
    
    getNewPointAndLineSingleImage(epipole1, epipole2, imgDimensions, F, 1, pOld1, prevLine1, pNew1, newLine1, pNew2, newLine2);
    
//     double distImg2 = sqrt((pOld2.x - pNew2.x) * (pOld2.x - pNew2.x) + (pOld2.y - pNew2.y) * (pOld2.y - pNew2.y));
//     if (distImg2 > STEP_SIZE) {
//         cout << distImg2 << endl;
//         cout << "It's bigger!!!" << endl;
//         
//         getNewPointAndLineSingleImage(epipole2, epipole1, imgDimensions, F, 2, pOld2, prevLine2, pNew2, newLine2, pNew1, newLine1);
//      
//     }
    
    showNewEpiline(epipole1, m_line1B, m_line1E, newLine1, pOld1, pNew1, imgDimensions, std::string("newEpiline1"));
    showNewEpiline(epipole2, m_line2B, m_line2E, newLine2, pOld2, pNew2, imgDimensions, std::string("newEpiline2"));
    cv::moveWindow("newEpiline2", imgDimensions.width +10, 0);
}

void PolarCalibration::doTransformation(const cv::Mat& img, const cv::Point2d epipole1, const cv::Point2d epipole2, const cv::Mat & F) {
   
    cv::Point2d p1 = m_b1, p2 = m_b2;
    cv::Vec3f line1 = m_line1B, line2 = m_line2B;

    bool lastCrossProduct = (m_line1E.cross(line1))[2] >= 0.0;
    uint32_t crossesLeft = 1;
    if (isInsideImage(epipole1, cv::Size(img.cols, img.rows)) &&
        isInsideImage(epipole2, cv::Size(img.cols, img.rows)))
        crossesLeft++;
    cout << "crossesLeft " << crossesLeft << endl;

    while (true) {
        cv::Point2d oldP1 = p1, oldP2 = p2;
        cv::Vec3f oldLine1 = line1, oldLine2 = line2;
        cout << "oldP1 = " << p1 << ", oldP2 = " << p2 << ", oldLine1 = " << line1 << ", oldLine2 = " << line2 << endl;
        getNewEpiline(epipole1, epipole2, cv::Size(img.cols, img.rows), F, p1, p2, line1, line2, p1, p2, line1, line2);
        cout << "p1 = " << p1 << ", p2 = " << p2 << ", line1 = " << line1 << ", line2 = " << line2 << endl;
        
        bool currentCrossProduct = (m_line1E.cross(line1))[2] >= 0.0;

        cout << "lastCrossProduct " << lastCrossProduct << endl;
        cout << "currentCrossProduct " << currentCrossProduct << endl;
        cout << "crossesLeft " << crossesLeft << endl;
        if (lastCrossProduct != currentCrossProduct) {
            if (crossesLeft == 0) {
                cout << "**********************************************************" << endl;
                cout << "********************* End ********************************" << endl;
                cout << "**********************************************************" << endl;
            
                break;
            } else {
                crossesLeft--;
            }
        }
        lastCrossProduct = currentCrossProduct;
        
        int keycode = cv::waitKey(0);
        
        cout << "keycode " << keycode << endl;
        if (keycode == 113) {
            exit(0);
        }
        if (keycode == 115) {
            cv::FileStorage file("/home/nestor/Dropbox/KULeuven/projects/PolarCalibration/testing/lastMat.xml", cv::FileStorage::WRITE);
            file << "F" << F;
            file.release();
        }
        if (keycode == 49) {
            cout << "stepSize = 1" << endl;
            m_stepSize = 1;
        }
        if (keycode == 50) {
            cout << "stepSize = 10" << endl;
            m_stepSize = 10;
        }
        if (keycode == 51) {
            cout << "stepSize = 50" << endl;
            m_stepSize = 50;
        }
        if (keycode == 51) {
            cout << "stepSize = 100" << endl;
            m_stepSize = 100;
        }
        if (keycode == 110) {
            break;
        }
        if (keycode == 114) {
            p1 = m_b1;
            p2 = m_b2;
            line1 = m_line1B;
            line2 = m_line2B;
        }
    }
}

void PolarCalibration::showNewEpiline(const cv::Point2d epipole, const cv::Vec3f & lineB, const cv::Vec3f & lineE, 
                                      const cv::Vec3f & newLine, const cv::Point2d & pOld, const cv::Point2d & pNew, 
                                      const cv::Size & imgDimensions, std::string windowName) {
    
    cv::Mat img = cv::Mat::zeros(imgDimensions.height, imgDimensions.width, CV_8UC3);
    
    // Image limits are drawn
    cv::line(img, image2World(cv::Point2d(0, 0), imgDimensions), 
             image2World(cv::Point2d(imgDimensions.width - 1, 0), imgDimensions), cv::Scalar::all(255));
    cv::line(img, image2World(cv::Point2d(0, 0), imgDimensions), 
             image2World(cv::Point2d(0, imgDimensions.height - 1), imgDimensions), cv::Scalar::all(255));
    cv::line(img, image2World(cv::Point2d(imgDimensions.width - 1, 0), imgDimensions), 
             image2World(cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), imgDimensions), cv::Scalar::all(255));
    cv::line(img, image2World(cv::Point2d(0, imgDimensions.height - 1), imgDimensions), 
             image2World(cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), imgDimensions), cv::Scalar::all(255));
    
    cv::line(img, image2World(epipole, imgDimensions), 
             image2World(pNew, imgDimensions), cv::Scalar(0, 255, 255));
    cv::line(img, image2World(pOld, imgDimensions), 
             image2World(pNew, imgDimensions), cv::Scalar(0, 255, 255));
    
    cv::circle(img, image2World(epipole, imgDimensions), 4, cv::Scalar(0, 255, 0), -1);
        
    cv::line(img, image2World(getPointFromLineAndX(-2 * imgDimensions.width, lineB), imgDimensions), 
             image2World(getPointFromLineAndX(2 * imgDimensions.width, lineB), imgDimensions), cv::Scalar(255, 0, 255), 1);
    cv::line(img, image2World(getPointFromLineAndX(-2 * imgDimensions.width, lineE), imgDimensions),
             image2World(getPointFromLineAndX(2 * imgDimensions.width, lineE), imgDimensions), cv::Scalar(0, 255, 255), 1);
    
    cv::line(img, image2World(getPointFromLineAndX(-2 * imgDimensions.width, newLine), imgDimensions), 
             image2World(getPointFromLineAndX(2 * imgDimensions.width, newLine), imgDimensions), cv::Scalar(0, 255, 0), 2);
    
    cv::circle(img, image2World(pOld, imgDimensions), 3, cv::Scalar(255, 0, 0), -1);
    cv::circle(img, image2World(pNew, imgDimensions), 3, cv::Scalar(0, 0, 255), -1);
    
    cv::namedWindow(windowName.c_str());
    cv::imshow(windowName.c_str(), img);
}

void PolarCalibration::showCommonRegion(const cv::Point2d epipole, const cv::Vec3f & line11, const cv::Vec3f & line12,
                                        const cv::Vec3f & line13, const cv::Vec3f & line14, 
                                        const cv::Vec3f & lineB, const cv::Vec3f & lineE, 
                                        const cv::Point2d & b, const cv::Size & imgDimensions, 
                                        const vector<cv::Point2f> & externalPoints, std::string windowName) {
                                            
    cv::Mat img = cv::Mat::zeros(imgDimensions.height, imgDimensions.width, CV_8UC3);
    
    // Image limits are drawn
    cv::line(img, image2World(cv::Point2d(0, 0), imgDimensions), 
             image2World(cv::Point2d(imgDimensions.width - 1, 0), imgDimensions), cv::Scalar::all(255));
    cv::line(img, image2World(cv::Point2d(0, 0), imgDimensions), 
             image2World(cv::Point2d(0, imgDimensions.height - 1), imgDimensions), cv::Scalar::all(255));
    cv::line(img, image2World(cv::Point2d(imgDimensions.width - 1, 0), imgDimensions), 
             image2World(cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), imgDimensions), cv::Scalar::all(255));
    cv::line(img, image2World(cv::Point2d(0, imgDimensions.height - 1), imgDimensions), 
             image2World(cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), imgDimensions), cv::Scalar::all(255));
    
    cv::circle(img, image2World(epipole, imgDimensions), 4, cv::Scalar(0, 255, 0), -1);
    cv::circle(img, image2World(externalPoints[0], imgDimensions), 3, cv::Scalar(255, 255, 0), -1);
    cv::circle(img, image2World(externalPoints[1], imgDimensions), 3, cv::Scalar(255, 255, 0), -1);
    
    cv::line(img, image2World(getPointFromLineAndX(epipole.x, line11), imgDimensions), 
                image2World(getPointFromLineAndX(0, line11), imgDimensions), cv::Scalar(255, 0, 0));
   
    cv::line(img, image2World(getPointFromLineAndX(-2 * imgDimensions.width, lineB), imgDimensions), 
             image2World(getPointFromLineAndX(2 * imgDimensions.width, lineB), imgDimensions), cv::Scalar(255, 0, 255), 3);
    cv::line(img, image2World(getPointFromLineAndX(-2 * imgDimensions.width, lineE), imgDimensions),
             image2World(getPointFromLineAndX(2 * imgDimensions.width, lineE), imgDimensions), cv::Scalar(0, 255, 255), 3);
    
    cv::line(img, image2World(getPointFromLineAndX(-2 * imgDimensions.width, line11), imgDimensions), 
             image2World(getPointFromLineAndX(2 * imgDimensions.width, line11), imgDimensions), cv::Scalar(255, 0, 0));
    cv::line(img, image2World(getPointFromLineAndX(-2 * imgDimensions.width, line12), imgDimensions),
             image2World(getPointFromLineAndX(2 * imgDimensions.width, line12), imgDimensions), cv::Scalar(255, 0, 0));
    
    cv::line(img, image2World(getPointFromLineAndX(-2 * imgDimensions.width, line13), imgDimensions),
             image2World(getPointFromLineAndX(2 * imgDimensions.width, line13), imgDimensions), cv::Scalar(0, 0, 255));
    cv::line(img, image2World(getPointFromLineAndX(-2 * imgDimensions.width, line14), imgDimensions),
             image2World(getPointFromLineAndX(2 * imgDimensions.width, line14), imgDimensions), cv::Scalar(0, 0, 255));

    cv::circle(img, image2World(b, imgDimensions), 10, cv::Scalar(128, 255, 128), 1);
    
    cv::namedWindow(windowName.c_str());
    cv::imshow(windowName.c_str(), img);
    
//     cv::waitKey(0);
}
