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
#include <time.h>
#include <omp.h>

#include "polarcalibration.h"


PolarCalibration::PolarCalibration() {
    m_hessianThresh = 50;
    m_stepSize = STEP_SIZE;
    m_showCommonRegion = false;
    m_showIterations = false;
}

PolarCalibration::~PolarCalibration() {

}

bool PolarCalibration::compute(const cv::Mat& img1distorted, const cv::Mat& img2distorted, 
                               const cv::Mat & cameraMatrix1, const cv::Mat & distCoeffs1, 
                               const cv::Mat & cameraMatrix2, const cv::Mat & distCoeffs2, 
                               const uint32_t method) {
    
    cv::Mat img1(img1distorted.rows, img1distorted.cols, CV_8UC1);
    cv::Mat img2(img2distorted.rows, img2distorted.cols, CV_8UC1);
    
    cv::undistort(img1distorted, img1, cameraMatrix1, distCoeffs1);
    cv::undistort(img2distorted, img2, cameraMatrix2, distCoeffs2);
    
    return compute(img1, img2);
}


bool PolarCalibration::compute(const cv::Mat& img1, const cv::Mat& img2, cv::Mat F, 
                               vector< cv::Point2f > points1, vector< cv::Point2f > points2, const uint32_t method)
{
    clock_t begin = clock();
    
//     cv::Mat F;
    cv::Point2d epipole1, epipole2;
    if (! findFundamentalMat(img1, img2, F, points1, points2, epipole1, epipole2, method))
        return false;

    clock_t end = clock();
    double elapsedFMat = double(end - begin) / CLOCKS_PER_SEC;
    
//     cout << "epipole1: " << epipole1 << endl;
//     cout << "epipole2: " << epipole2 << endl;

    begin = clock();
    // Determine common region
    vector<cv::Vec3f> initialEpilines, finalEpilines;
    vector<cv::Point2f> epipoles(2);
    epipoles[0] = epipole1;
    epipoles[1] = epipole2;    
    determineCommonRegion(epipoles, cv::Size(img1.cols, img1.rows), F);
    
    doTransformation(img1, img2, epipole1, epipole2, F);
    
    end = clock();
    double elapsedPolar = double(end - begin) / CLOCKS_PER_SEC;
    
    cout << "Time F: " << elapsedFMat << endl;
    cout << "Time transformation: " << elapsedPolar << endl;
    
    return true;
}


inline bool PolarCalibration::findFundamentalMat(const cv::Mat& img1, const cv::Mat& img2, cv::Mat & F, 
                                                 vector<cv::Point2f> points1, vector<cv::Point2f> points2, 
                                                 cv::Point2d& epipole1, cv::Point2d& epipole2, const uint32_t method)
{
//     vector<cv::Point2f> points1, points2;
    
    if (F.empty()) {
        switch(method) {
            case FMAT_METHOD_OFLOW:
                findPairsOFlow(img1, img2, points1, points2);
                break;
            case FMAT_METHOD_SURF:
                findPairsSURF(img1, img2, points1, points2);
                break;
        }
        
        if (points1.size() < 8)
            return false;
        
    //     cout << "sz = " << points1.size() << endl;
        
        F = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
    //     cout << "F:\n" << F << endl; 
        
        if (cv::countNonZero(F) == 0)
            return false;
    }
    
    // We obtain the epipoles
    getEpipoles(F, epipole1, epipole2);
    
    checkF(F, epipole1, epipole2, points1[0], points2[0]);
    
    /// NOTE: Remove. Just for debugging (begin)
    {
        cout << "***********************" << endl;
        cv::FileStorage file("/home/nestor/Dropbox/KULeuven/projects/PolarCalibration/testing/lastMat_0597.xml", cv::FileStorage::READ);
//         cv::FileStorage file("/home/nestor/Dropbox/KULeuven/projects/PolarCalibration/testing/results/mats/lastMat_0925.xml", cv::FileStorage::READ);
//         cv::FileStorage file("/tmp/results/mats/lastMat_0039.xml", cv::FileStorage::READ);
        
        file["F"] >> F;
        file.release();
        cout << "F (from file)\n" << F << endl;
        getEpipoles(F, epipole1, epipole2);
        cout << "epipole1 " << epipole1 << endl;
        cout << "epipole2 " << epipole2 << endl;
    }
    /// NOTE: Remove. Just for debugging (end)
    
//     cv::FileStorage file("/home/nestor/Dropbox/KULeuven/projects/PolarCalibration/testing/lastMat_O.xml", cv::FileStorage::WRITE);
//     file << "F" << F;
//     file.release();
    
        if (SIGN(epipole1.x) != SIGN(epipole2.x) &&
            SIGN(epipole1.y) != SIGN(epipole2.y)) {
            
            epipole2 *= -1;
        }
            
        
    return true;
}

inline void PolarCalibration::findPairsOFlow(const cv::Mat & img1, const cv::Mat & img2, 
                                             vector<cv::Point2f> & outPoints1, vector<cv::Point2f> & outPoints2) {
                                                
    // We look for correspondences using Optical flow
    // vector of keypoints
    vector<cv::KeyPoint> keypoints1;
    cv::FastFeatureDetector fastDetector(50);
    fastDetector.detect(img1, keypoints1);
    
    if (keypoints1.size() == 0)
        return;
    
    vector<cv::Point2f> points1(keypoints1.size()), points2, points1B;
    {
        uint32_t idx = 0;
        for (vector<cv::KeyPoint>::iterator it = keypoints1.begin(); it != keypoints1.end(); it++, idx++) {
            points1[idx] = it->pt;
        }
    }    
    // Optical flow
    vector<uint8_t> status, statusB;
    vector<float_t> error, errorB;
    
    cv::calcOpticalFlowPyrLK(img1, img2, points1, points2, status, error, cv::Size(3, 3), 3);
    cv::calcOpticalFlowPyrLK(img2, img1, points2, points1B, statusB, errorB, cv::Size(3, 3), 3);
    
    vector<cv::Point2f> pointsA(points1.size()), pointsB(points2.size());
    {
        uint32_t idx = 0;
        for (uint32_t i = 0; i < points1.size(); i++) {
            if ((status[i] == 1) && (statusB[i] == 1)) {
                if (cv::norm(points1[i] - points1B[i]) < 1.0) {
                    pointsA[idx] = points1[i];
                    pointsB[idx] = points2[i];
                }
            }
            idx++;
        }
        pointsA.resize(idx);
        pointsB.resize(idx);
    }
    
    outPoints1 = pointsA;
    outPoints2 = pointsB;

}

inline void PolarCalibration::findPairsSURF(const cv::Mat & img1, const cv::Mat & img2,
                                                     vector<cv::Point2f> & outPoints1, vector<cv::Point2f> & outPoints2) {
    
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
    
    points1.resize(matches.size());
    points2.resize(matches.size());
    
    {
        uint32_t idx2 = 0;
        for (int idx = 0; idx < matches.size(); idx++) {
            const cv::Point2f & p1 = keypoints1[matches[idx].queryIdx].pt;
            const cv::Point2f & p2 = keypoints2[matches[idx].trainIdx].pt;
            
            if (fabs(p1.x - p2.x < 10.0) && fabs(p1.y - p2.y < 10.0)) {
                points1[idx2] = p1;
                points2[idx2] = p2;
                idx2++;
            }
        }
        points1.resize(idx2);
        points2.resize(idx2);
    }
    
    outPoints1 = points1;
    outPoints2 = points2;
}

inline void PolarCalibration::getEpipoles(const cv::Mat & F, cv::Point2d & epipole1, cv::Point2d & epipole2) {
    cv::SVD svd(F);

    cv::Mat e1 = svd.vt.row(2);
    cv::Mat e2 = svd.u.col(2);
    
    epipole1 = cv::Point2d(e1.at<double>(0, 0) / e1.at<double>(0, 2), e1.at<double>(0, 1) / e1.at<double>(0, 2));
    epipole2 = cv::Point2d(e2.at<double>(0, 0) / e2.at<double>(2, 0), e2.at<double>(1, 0) / e2.at<double>(2, 0));

//     cv::Mat eigenVal, eigenVect;
//     if (! cv::eigen(F.t() * F, eigenVal,  eigenVect)) {
//         cout << "Failure" << endl;
//         exit(0);
//     }
//     
//     epipole1 = cv::Point2d(eigenVect.at<double>(2, 0) / eigenVect.at<double>(2, 2), eigenVect.at<double>(2, 1) / eigenVect.at<double>(2, 2));
//     
//     if (! cv::eigen(F * F.t(), eigenVal,  eigenVect)) {
//         cout << "Failure" << endl;
//         exit(0);
//     }
//     
//     epipole2 = cv::Point2d(eigenVect.at<double>(2, 0) / eigenVect.at<double>(2, 2), eigenVect.at<double>(2, 1) / eigenVect.at<double>(2, 2));
}

inline void PolarCalibration::checkF(cv::Mat & F, cv::Point2d & epipole1, cv::Point2d & epipole2, const cv::Point2d & m, const cv::Point2d & m1) {
    cv::Vec3f line = GET_LINE_FROM_POINTS(epipole1, m);
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
    
//     if (((fl.at<double>(0,0) < 0.0) && (fl1.at<double>(0,0) > 0.0)) ||
//         ((fl.at<double>(0,0) > 0.0) && (fl1.at<double>(0,0) < 0.0))) {

    if (SIGN(fl.at<double>(0,0)) != SIGN(fl1.at<double>(0,0))) {
        cout << "Test failed" << endl;
        
        F = -F;
    
        getEpipoles(F, epipole1, epipole2);
    }
}

inline bool PolarCalibration::lineIntersectsSegment(const cv::Vec3d & line, const cv::Point2d & p1, const cv::Point2d & p2, cv::Point2d * intersection) {
    const cv::Vec3d segment = GET_LINE_FROM_POINTS(p1, p2);
    
    if (intersection != NULL)
        *intersection = cv::Point2d(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    
    // Lines are represented as ax + by + c = 0, so
    // y = -(ax+c)/b. If y1=y2, then we have to obtain x, which is
    // x = (b1 * c2 - b2 * c1) / (b2 * a1 - b1 * a2)
    if ((segment[1] * line[0] - line[1] * segment[0]) == 0)
        return false;
    double x = (line[1] * segment[2] - segment[1] * line[2]) / (segment[1] * line[0] - line[1] * segment[0]);
    double y = -(line[0] * x + line[2]) / line[1];
    
    cout << "possible intersection at " << cv::Point2d(x, y) << " -- " << m_b1 << endl;
    if (((int32_t)round(x) >= (int32_t)min(p1.x, p2.x)) && ((int32_t)round(x) <= (int32_t)max(p1.x, p2.x))) {
        if (((int32_t)round(y) >= (int32_t)min(p1.y, p2.y)) && ((int32_t)round(y) <= (int32_t)max(p1.y, p2.y))) {
            if (intersection != NULL)
                *intersection = cv::Point2d(x, y);
            
            return true;
        }
    }
    
    return false;
}

inline bool PolarCalibration::lineIntersectsRect(const cv::Vec3d & line, const cv::Size & imgDimensions, cv::Point2d * intersection) {
    return lineIntersectsSegment(line, cv::Point2d(0, 0), cv::Point2d(imgDimensions.width - 1, 0), intersection) ||
            lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, 0), cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), intersection) ||
            lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), cv::Point2d(0, imgDimensions.height - 1), intersection) ||
            lineIntersectsSegment(line, cv::Point2d(0, imgDimensions.height - 1), cv::Point2d(0, 0), intersection);
}

inline bool PolarCalibration::isTheRightPoint(const cv::Point2d & epipole, const cv::Point2d & intersection, const cv::Vec3d & line,
                                       const cv::Point2d * lastPoint)
{
    if (lastPoint != NULL) {
        cv::Vec3f v1(lastPoint->x - epipole.x, lastPoint->y - epipole.y, 0.0);
        v1 /= cv::norm(v1);
        cv::Vec3f v2(intersection.x - epipole.x, intersection.y - epipole.y, 0.0);
        v2 /= cv::norm(v2);
        
        if (fabs(acos(v1.dot(v2))) > CV_PI / 2.0)
            return false;
        else
            return true;
        cout << "rightPoint" << endl;
    } else {
        cout << "***********NULL************" << endl;
        if ((line[0] > 0) && (epipole.y < intersection.y)) return false;
        if ((line[0] < 0) && (epipole.y > intersection.y)) return false;
        if ((line[1] > 0) && (epipole.x > intersection.x)) return false;
        if ((line[1] < 0) && (epipole.x < intersection.x)) return false;
        
        return true;
    }
    return false;
}

inline void PolarCalibration::getBorderIntersections(const cv::Point2d & epipole, const cv::Vec3d & line, const cv::Size & imgDimensions, 
                                                     vector<cv::Point2d> & intersections) {
    
    cv::Point2d intersection(-1, -1);
    intersections.reserve(2);
    
    if (lineIntersectsSegment(line, cv::Point2d(0, 0), cv::Point2d(imgDimensions.width - 1, 0), &intersection)) {
        intersections.push_back(intersection);
    }
    if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, 0), cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), &intersection)) {
        intersections.push_back(intersection);
    }
    if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), cv::Point2d(0, imgDimensions.height - 1), &intersection)) {
        intersections.push_back(intersection);
    }
    if (lineIntersectsSegment(line, cv::Point2d(0, imgDimensions.height - 1), cv::Point2d(0, 0), &intersection)) {
        intersections.push_back(intersection);
    }
}

inline cv::Point2d PolarCalibration::getBorderIntersection(const cv::Point2d & epipole, const cv::Vec3d & line, const cv::Size & imgDimensions, 
                                                            const cv::Point2d * lastPoint) {
    
    cv::Point2d intersection(-1, -1);

    if (IS_INSIDE_IMAGE(epipole, imgDimensions)) {
        cout << "INSIDE" << endl;
        if (lineIntersectsSegment(line, cv::Point2d(0, 0), cv::Point2d(imgDimensions.width - 1, 0), &intersection)) {
            if (isTheRightPoint(epipole, intersection, line, lastPoint)) {
                cout << "Right point" << endl;
                
                return intersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, 0), cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), &intersection)) {
            if (isTheRightPoint(epipole, intersection, line, lastPoint)) {
                cout << "Right point" << endl;

                return intersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), cv::Point2d(0, imgDimensions.height - 1), &intersection)) {
            if (isTheRightPoint(epipole, intersection, line, lastPoint)) {
                cout << "Right point" << endl;
                
                return intersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(0, imgDimensions.height - 1), cv::Point2d(0, 0), &intersection)) {
            if (isTheRightPoint(epipole, intersection, line, lastPoint)) {
                cout << "Right point" << endl;
                
                return intersection;
            }
        }
    } else {
        cout << "OUTSIDE" << endl;
        double maxDist = std::numeric_limits<double>::min();
        cv::Point2d tmpIntersection(-1, -1);
        if (lineIntersectsSegment(line, cv::Point2d(0, 0), cv::Point2d(imgDimensions.width - 1, 0), &tmpIntersection)) {
            cout << "intersection at AB" << endl;
            const double dist2 = (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x) +
                                 (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x);
            
            if (dist2 > maxDist) {
                cout << "WINNER" << endl;
                maxDist = dist2;
                intersection = tmpIntersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, 0), cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), &tmpIntersection)) {
            cout << "intersection at BD" << endl;
            const double dist2 = (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x) +
            (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x);
            
            if (dist2 > maxDist) {
                cout << "WINNER" << endl;
                maxDist = dist2;
                intersection = tmpIntersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), cv::Point2d(0, imgDimensions.height - 1), &tmpIntersection)) {
            cout << "intersection at CD" << endl;
            const double dist2 = (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x) +
            (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x);
            
            if (dist2 > maxDist) {
                cout << "WINNER" << endl;
                maxDist = dist2;
                intersection = tmpIntersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(0, imgDimensions.height - 1), cv::Point2d(0, 0), &tmpIntersection)) {
            cout << "intersection at AC" << endl;
            const double dist2 = (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x) +
            (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x);
            
            if (dist2 > maxDist) {
                cout << "WINNER" << endl;
                maxDist = dist2;
                intersection = tmpIntersection;
            }
        }
        return intersection;
    }
}

inline void PolarCalibration::getExternalPoints(const cv::Point2d &epipole, const cv::Size imgDimensions,
                                                vector<cv::Point2f> &externalPoints) {
    
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
    
    cout << "externalPoints[0] " << externalPoints[0] << endl;
    cout << "externalPoints[1] " << externalPoints[1] << endl;
}

inline void PolarCalibration::computeEpilines(const vector<cv::Point2f> & points, const uint32_t &whichImage, 
                                            const cv::Mat & F, const vector <cv::Vec3f> & oldlines, vector <cv::Vec3f> & newLines) {
 
    cv::computeCorrespondEpilines(points, whichImage, F, newLines);
    
    for (uint32_t i = 0; i < oldlines.size(); i++) {
        if ((SIGN(oldlines[i][0]) != SIGN(newLines[i][0])) && 
            (SIGN(oldlines[i][1]) != SIGN(newLines[i][1]))) {
            newLines[i] *= -1;
        }
    }
}

inline cv::Point2d PolarCalibration::getBorderIntersectionFromOutside(const cv::Point2d & epipole, const cv::Vec3d & line, const cv::Size & imgDimensions)
{
    vector<cv::Point2d> intersections;
    getBorderIntersections(epipole, line, imgDimensions, intersections);
    double maxDist = std::numeric_limits<double>::min();
    cv::Point2d returnPoint;
    for (uint32_t i = 0; i < intersections.size(); i++) {
        const double dist = cv::norm(epipole - intersections[i]);
        if (dist > maxDist) {
            maxDist = dist;
            returnPoint = intersections[i];
        }
    }
    
    return returnPoint;
}

/**
 * This function is more easily understandable after reading section 3.4 of 
 * ftp://cmp.felk.cvut.cz/pub/cmp/articles/matousek/Sandr-TR-2009-04.pdf
 * */
inline void PolarCalibration::determineCommonRegion(const vector<cv::Point2f> &epipoles, 
                                             const cv::Size imgDimensions, const cv::Mat & F) {

    vector<cv::Point2f> externalPoints1, externalPoints2;
    getExternalPoints(epipoles[0], imgDimensions, externalPoints1);
    getExternalPoints(epipoles[1], imgDimensions, externalPoints2);
    
    determineRhoRange(epipoles[0], imgDimensions, externalPoints1, m_minRho1, m_maxRho1);
    determineRhoRange(epipoles[1], imgDimensions, externalPoints2, m_minRho2, m_maxRho2);

    if (!IS_INSIDE_IMAGE(epipoles[0], imgDimensions) && !IS_INSIDE_IMAGE(epipoles[1], imgDimensions)) {
        // CASE 1: Both outside
        const cv::Vec3f line11 = GET_LINE_FROM_POINTS(epipoles[0], externalPoints1[0]);
        const cv::Vec3f line12 = GET_LINE_FROM_POINTS(epipoles[0], externalPoints1[1]);
        
        const cv::Vec3f line23 = GET_LINE_FROM_POINTS(epipoles[1], externalPoints2[0]);
        const cv::Vec3f line24 = GET_LINE_FROM_POINTS(epipoles[1], externalPoints2[1]);
        
        vector <cv::Vec3f> inputLines(2), outputLines;
        inputLines[0] = line23;
        inputLines[1] = line24;
        computeEpilines(externalPoints2, 2, F, inputLines, outputLines);
        const cv::Vec3f line13 = outputLines[0];
        const cv::Vec3f line14 = outputLines[1];
        
        inputLines[0] = line11;
        inputLines[1] = line12;
        computeEpilines(externalPoints1, 1, F, inputLines, outputLines);
        const cv::Vec3f line21 = outputLines[0];
        const cv::Vec3f line22 = outputLines[1];
        
        // Beginning and ending lines
//         if (lineIntersectsRect(line13, imgDimensions)) {
//             cout << "Begin1 " << endl;
//             m_line1B = line13;
//             m_line2B = line23;
//         } else {
//             cout << "Begin2 " << endl;
//             m_line1B = line11;
//             m_line2B = line21;
//         }
//         
//         if (lineIntersectsRect(line14, imgDimensions)) {
//             cout << "End 1 " << endl;
//             m_line1E = line14;
//             m_line2E = line24;
//         } else {
//             cout << "End 2" << endl;
//             m_line1E = line12;
//             m_line2E = line22;
//         }
        
        m_line1B = lineIntersectsRect(line13, imgDimensions)? line13 : line11;
        m_line1E = lineIntersectsRect(line14, imgDimensions)? line14 : line12;
        m_line2B = lineIntersectsRect(line21, imgDimensions)? line21 : line23;
        m_line2E = lineIntersectsRect(line22, imgDimensions)? line22 : line24;
                
        // Beginning and ending lines intersection with the borders
        {
            vector<cv::Point2d> intersections;
            getBorderIntersections(epipoles[0], m_line1B, imgDimensions, intersections);
            double maxDist = std::numeric_limits<double>::min();
            for (uint32_t i = 0; i < intersections.size(); i++) {
                const cv::Point2f intersect = intersections[i];
                const double dist = cv::norm(epipoles[0] - intersect);
                if (dist > maxDist) {
                    maxDist = dist;
                    m_b1 = intersections[i];
                }
            }
        }
        {
            vector<cv::Point2d> intersections;
            getBorderIntersections(epipoles[1], m_line2B, imgDimensions, intersections);
            double maxDist = std::numeric_limits<double>::min();
            for (uint32_t i = 0; i < intersections.size(); i++) {
                const cv::Point2f intersect = intersections[i];
                const double dist = cv::norm(epipoles[1] - intersect);
                if (dist > maxDist) {
                    maxDist = dist;
                    m_b2 = intersections[i];
                }
            }
        }
        {
            vector<cv::Point2d> intersections;
            getBorderIntersections(epipoles[0], m_line1E, imgDimensions, intersections);
            double maxDist = std::numeric_limits<double>::min();
            for (uint32_t i = 0; i < intersections.size(); i++) {
                const cv::Point2f intersect = intersections[i];
                const double dist = cv::norm(epipoles[0] - intersect);
                if (dist > maxDist) {
                    maxDist = dist;
                    m_e1 = intersections[i];
                }
            }
        }
        {
            vector<cv::Point2d> intersections;
            getBorderIntersections(epipoles[1], m_line2E, imgDimensions, intersections);
            double maxDist = std::numeric_limits<double>::min();
            for (uint32_t i = 0; i < intersections.size(); i++) {
                const cv::Point2f intersect = intersections[i];
                const double dist = cv::norm(epipoles[1] - intersect);
                if (dist > maxDist) {
                    maxDist = dist;
                    m_e1 = intersections[i];
                }
            }
        }
        
//         m_b1 = getBorderIntersectionFromOutside(epipoles[0], m_line1B, imgDimensions);
//         m_b2 = getBorderIntersectionFromOutside(epipoles[1], m_line2B, imgDimensions);
//         m_e1 = getBorderIntersectionFromOutside(epipoles[0], m_line1E, imgDimensions);
//         m_e2 = getBorderIntersectionFromOutside(epipoles[1], m_line2E, imgDimensions);
        
        if (m_showCommonRegion) {
            showCommonRegion(epipoles[0], line11, line12, line13, line14, m_line1B, m_line1E, m_b1, m_e1, imgDimensions, 
                            externalPoints1, std::string("leftCommonRegion"));
            showCommonRegion(epipoles[1], line23, line24, line21, line22, m_line2B, m_line2E, m_b2, m_e2, imgDimensions, 
                            externalPoints2, std::string("rightCommonRegion"));
        }
        
    } else if (IS_INSIDE_IMAGE(epipoles[0], imgDimensions) && IS_INSIDE_IMAGE(epipoles[1], imgDimensions)) {
        // CASE 2: Both inside
        m_line1B = GET_LINE_FROM_POINTS(epipoles[0], externalPoints1[0]);
        m_line1E = m_line1B;
        
        vector <cv::Vec3f> inputLines(1), outputLines;
        inputLines[0] = m_line1B;
        computeEpilines(externalPoints1, 1, F, inputLines, outputLines);
        
        m_line2B = outputLines[0];
        m_line2E = outputLines[0];
        
        m_b1 = getBorderIntersection(epipoles[0], m_line1B, imgDimensions);
//         m_b2 = getBorderIntersection(epipoles[1], m_line2B, imgDimensions);
        m_e1 = getBorderIntersection(epipoles[0], m_line1E, imgDimensions);
//         m_e2 = getBorderIntersection(epipoles[1], m_line2E, imgDimensions);
        
        vector<cv::Point2d> intersections;
        getBorderIntersections(epipoles[1], m_line2B, imgDimensions, intersections);
        
        cv::Vec3d v0(intersections[0].x - epipoles[1].x, intersections[0].y - epipoles[1].y, 1.0);
        cv::Vec3d v1(intersections[1].x - epipoles[1].x, intersections[1].y - epipoles[1].y, 1.0);
        
        cv::Vec3d v1B(m_b1.x - epipoles[0].x, m_b1.y - epipoles[0].y, 1.0);
        
        v0 /= cv::norm(v0);
        v1 /= cv::norm(v1);
        v1B /= cv::norm(v1B);
        
        if (acos(v0.dot(v1B)) < acos(v1.dot(v1B))) {
            m_b2 = intersections[0];
        } else {
            m_b2 = intersections[1];
        }
        m_e2 = m_b2;
                
        if (m_showCommonRegion) {
            showCommonRegion(epipoles[0], m_line1B, m_line1E, m_line2B, m_line2E, m_line1B, m_line1E, m_b1, m_e1, imgDimensions, 
                            externalPoints1, std::string("leftCommonRegion"));
            showCommonRegion(epipoles[1], m_line2B, m_line2E, m_line1B, m_line1E, m_line2B, m_line2E, m_b2, m_e2, imgDimensions, 
                            externalPoints2, std::string("rightCommonRegion"));
        }
    } else {
        // CASE 3: One inside and one outside
        if (IS_INSIDE_IMAGE(epipoles[0], imgDimensions)) {
            // CASE 3.1: Only the first epipole is inside
            
            const cv::Vec3f line23 = GET_LINE_FROM_POINTS(epipoles[1], externalPoints2[0]);
            const cv::Vec3f line24 = GET_LINE_FROM_POINTS(epipoles[1], externalPoints2[1]);
            
            vector <cv::Vec3f> inputLines(2), outputLines;
            inputLines[0] = line23;
            inputLines[1] = line24;
            computeEpilines(externalPoints2, 2, F, inputLines, outputLines);
            const cv::Vec3f & line13 = outputLines[0];
            const cv::Vec3f & line14 = outputLines[1];
            
            m_line1B = line13;
            m_line1E = line14;
            m_line2B = line23;
            m_line2E = line24;
           
            m_b2 = getBorderIntersection(epipoles[1], m_line2B, imgDimensions);
            m_e2 = getBorderIntersection(epipoles[1], m_line2E, imgDimensions);
            
            vector<cv::Point2d> intersectionsB, intersectionsE;
            getBorderIntersections(epipoles[0], m_line1B, imgDimensions, intersectionsB);
            getBorderIntersections(epipoles[0], m_line1E, imgDimensions, intersectionsE);
            
            cv::Vec3d v0(intersectionsB[0].x - epipoles[0].x, intersectionsB[0].y - epipoles[0].y, 1.0);
            cv::Vec3d v1(intersectionsB[1].x - epipoles[0].x, intersectionsB[1].y - epipoles[0].y, 1.0);
            
            cv::Vec3d v2B(m_b2.x - epipoles[1].x, m_b2.y - epipoles[1].y, 1.0);
            cv::Vec3d v2E(m_e2.x - epipoles[1].x, m_e2.y - epipoles[1].y, 1.0);
            
            v0 /= cv::norm(v0);
            v1 /= cv::norm(v1);
            v2B /= cv::norm(v2B);
            v2E /= cv::norm(v2E);
            
            cout << "acos(v0.dot(v2B)) " << acos(v0.dot(v2B)) << endl;
            cout << "acos(v1.dot(v2B)) " << acos(v1.dot(v2B)) << endl;
            cout << "intersectionsB[0] " << intersectionsB[0] << endl;
            cout << "intersectionsB[1] " << intersectionsB[1] << endl;
            cout << "v0 " << v0 << endl;
            cout << "v1 " << v1 << endl;
            cout << "v2B " << v2B << endl;
            
//             if (SIGN(v0.cross(v1)[2]) != SIGN(v2B.cross(v2E)[2])) {
            if (acos(v0.dot(v2B)) < acos(v1.dot(v2B))) {
                m_b1 = intersectionsB[0];
            } else {
                m_b1 = intersectionsB[1];
            }
            
            v0 = cv::Vec3d(intersectionsE[0].x - epipoles[0].x, intersectionsE[0].y - epipoles[0].y, 1.0);
            v1 = cv::Vec3d(intersectionsE[1].x - epipoles[0].x, intersectionsE[1].y - epipoles[0].y, 1.0);
            
//             if (SIGN(v0.cross(v1)[2]) != SIGN(v2B.cross(v2E)[2])) {
            if (acos(v0.dot(v2E)) < acos(v1.dot(v2E))) {
                m_e1 = intersectionsE[1];
            } else {
                m_e1 = intersectionsE[0];
            }
            
//             cout << "m_line1B " << m_line1B << endl;
//             cout << "m_line1E " << m_line1E << endl;
//             cout << "m_line2B " << m_line2B << endl;
//             cout << "m_line2E " << m_line2E << endl;
            
//             m_b1 = getBorderIntersection(epipoles[0], m_line1B, imgDimensions);
//             m_b2 = getBorderIntersection(epipoles[1], m_line2B, imgDimensions);
//             m_e1 = getBorderIntersection(epipoles[0], m_line1E, imgDimensions);
//             m_e2 = getBorderIntersection(epipoles[1], m_line2E, imgDimensions);
            
//             cout << "m_b1 " << m_b1 << endl;
//             cout << "m_e1 " << m_e1 << endl;
//             
//             exit(0);
            
            if (m_showCommonRegion) {
                showCommonRegion(epipoles[0], line13, line14, line13, line14, m_line1B, m_line1E, m_b1, m_e1, imgDimensions, 
                                externalPoints1, std::string("leftCommonRegion"));
                showCommonRegion(epipoles[1], line23, line24, line23, line24, m_line2B, m_line2E, m_b2, m_e2, imgDimensions, 
                                externalPoints2, std::string("rightCommonRegion"));
            }
        } else {
            // CASE 3.2: Only the second epipole is inside
            const cv::Vec3f line11 = GET_LINE_FROM_POINTS(epipoles[0], externalPoints1[0]);
            const cv::Vec3f line12 = GET_LINE_FROM_POINTS(epipoles[0], externalPoints1[1]);
            
            vector <cv::Vec3f> inputLines(2), outputLines;
            inputLines[0] = line11;
            inputLines[1] = line12;
            computeEpilines(externalPoints1, 1, F, inputLines, outputLines);
            const cv::Vec3f & line21 = outputLines[0];
            const cv::Vec3f & line22 = outputLines[1];
            
            m_line1B = line11;
            m_line1E = line12;
            m_line2B = line21;
            m_line2E = line22;
            
            m_b1 = getBorderIntersection(epipoles[0], m_line1B, imgDimensions);
            m_e1 = getBorderIntersection(epipoles[0], m_line1E, imgDimensions);
            
            /*vector<cv::Point2d> intersections;
            getBorderIntersections(epipoles[1], m_line2B, imgDimensions, intersections);
            
            cv::Vec3d v0(intersections[0].x - epipoles[1].x, intersections[0].y - epipoles[1].y, 1.0);
            cv::Vec3d v1(intersections[1].x - epipoles[1].x, intersections[1].y - epipoles[1].y, 1.0);
            
            cv::Vec3d v1B(m_b1.x - epipoles[0].x, m_b1.y - epipoles[0].y, 1.0);
            cv::Vec3d v1E(m_e1.x - epipoles[0].x, m_e1.y - epipoles[0].y, 1.0);
            
            if (SIGN(v0.cross(v1)[2]) != SIGN(v1B.cross(v1E)[2])) {
                m_b2 = intersections[0];
                m_e2 = intersections[1];
            } else {
                m_b2 = intersections[1];
                m_e2 = intersections[0];
            }*/
            vector<cv::Point2d> intersectionsB, intersectionsE;
            getBorderIntersections(epipoles[1], m_line2B, imgDimensions, intersectionsB);
            getBorderIntersections(epipoles[1], m_line2E, imgDimensions, intersectionsE);
            
            cv::Vec3d v0(intersectionsB[0].x - epipoles[1].x, intersectionsB[0].y - epipoles[1].y, 1.0);
            cv::Vec3d v1(intersectionsB[1].x - epipoles[1].x, intersectionsB[1].y - epipoles[1].y, 1.0);
            
            cv::Vec3d v1B(m_b1.x - epipoles[0].x, m_b1.y - epipoles[0].y, 1.0);
            cv::Vec3d v1E(m_e1.x - epipoles[0].x, m_e1.y - epipoles[0].y, 1.0);
            
            v0 /= cv::norm(v0);
            v1 /= cv::norm(v1);
            v1B /= cv::norm(v1B);
            v1E /= cv::norm(v1E);
            
//             if (SIGN(v0.cross(v1)[2]) != SIGN(v1B.cross(v1E)[2])) {
            if (acos(v0.dot(v1B)) < acos(v1.dot(v1B))) {
                m_b2 = intersectionsB[0];
            } else {
                m_b2 = intersectionsB[1];
            }
            
            v0 = cv::Vec3d(intersectionsE[0].x - epipoles[1].x, intersectionsE[0].y - epipoles[1].y, 1.0);
            v1 = cv::Vec3d(intersectionsE[1].x - epipoles[1].x, intersectionsE[1].y - epipoles[1].y, 1.0);
            
//             if (SIGN(v0.cross(v1)[2]) != SIGN(v1B.cross(v1E)[2])) {
            if (acos(v0.dot(v1E)) < acos(v1.dot(v1E))) {
                m_e2 = intersectionsE[1];
            } else {
                m_e2 = intersectionsE[0];
            }
            
//             m_b1 = getBorderIntersection(epipoles[0], m_line1B, imgDimensions);
//             m_b2 = getBorderIntersection(epipoles[1], m_line2B, imgDimensions);
//             m_e1 = getBorderIntersection(epipoles[0], m_line1E, imgDimensions);
//             m_e2 = getBorderIntersection(epipoles[1], m_line2E, imgDimensions);

            if (m_showCommonRegion) {
                showCommonRegion(epipoles[0], line11, line12, line11, line12, m_line1B, m_line1E, m_b1, m_e1, imgDimensions, 
                                externalPoints1, std::string("leftCommonRegion"));
                showCommonRegion(epipoles[1], line21, line22, line21, line22, m_line2B, m_line2E, m_b2, m_e2, imgDimensions, 
                                externalPoints2, std::string("rightCommonRegion"));
            }
        }
    }
    
    if (m_showCommonRegion) {
        cv::moveWindow("rightCommonRegion", imgDimensions.width + 10, 0);
    }
}

inline void PolarCalibration::determineRhoRange(const cv::Point2d &epipole, const cv::Size imgDimensions,
                       const vector<cv::Point2f> &externalPoints, double & minRho, double & maxRho) {
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
}

inline void PolarCalibration::getNewPointAndLineSingleImage(const cv::Point2d epipole1, const cv::Point2d epipole2, const cv::Size & imgDimensions, 
                                    const cv::Mat & F, const uint32_t & whichImage, const cv::Point2d & pOld1, const cv::Point2d & pOld2,
                                   cv::Vec3f & prevLine, cv::Point2d & pNew1, cv::Vec3f & newLine1, 
                                   cv::Point2d & pNew2, cv::Vec3f & newLine2) {

    
    // We obtain vector v
    cv::Vec2f v;

    cv::Vec3f vBegin(m_b1.x - epipole1.x, m_b1.y - epipole1.y, 0.0);
    cv::Vec3f vCurr(pOld1.x - epipole1.x, pOld1.y - epipole1.y, 0.0);
    cv::Vec3f vEnd(m_e1.x - epipole1.x, m_e1.y - epipole1.y, 0.0);
    
    vBegin /= cv::norm(vBegin);
    vCurr /= cv::norm(vCurr);
    vEnd /= cv::norm(vEnd);
    
    cout << "epipole1 " << epipole1 << endl;
    cout << "epipole2 " << epipole2 << endl;
    cout << "[nextEpiline] vCurr " << vCurr << endl;
    
    if (IS_INSIDE_IMAGE(epipole1, imgDimensions)) {
        if (IS_INSIDE_IMAGE(epipole2, imgDimensions)) {
            cout << "CASE1" << endl;
            v = cv::Vec2f(vCurr[1], -vCurr[0]);
            cout << "[nextEpiline] v " << v << endl;
        } else {
            cout << "CASE3" << endl;
            vBegin = cv::Vec3f(m_b2.x - epipole2.x, m_b2.y - epipole2.y, 0.0);
            vCurr = cv::Vec3f(pOld2.x - epipole2.x, pOld2.y - epipole2.y, 0.0);
            vEnd = cv::Vec3f(m_e2.x - epipole2.x, m_e2.y - epipole2.y, 0.0);
            
            vBegin /= cv::norm(vBegin);
            vCurr /= cv::norm(vCurr);
            vEnd /= cv::norm(vEnd);
            
            const cv::Vec3f vCross = vBegin.cross(vEnd);
            
            v = cv::Vec2f(vCurr[1], -vCurr[0]);
            cout << "[newEpiline] //////////////////////////" << endl;
            cout << "[newEpiline] v " << v << endl;
            if (vCross[2] > 0.0) {
                v = -v;
            }
            cout << "[newEpiline] vBegin " << vBegin << endl;
            cout << "[newEpiline] vEnd " << vEnd << endl;
            cout << "[newEpiline] v " << v << endl;
            cout << "[newEpiline] vCross " << vCross << endl;
//                 exit(0);
        }
    } else {
        cout << "CASE2" << endl;
        const cv::Vec3f vCross = vBegin.cross(vEnd);
    
        v = cv::Vec2f(vCurr[1], -vCurr[0]);
        cout << "vPrev " << v << endl;
        if (vCross[2] > 0.0) {
            v = -v;
        }
        cout << "vCross " << vCross << endl;
    }
    cout << "vBegin " << vBegin << endl;
    cout << "vEnd " << vEnd << endl;
    cout << "v " << v << endl;

    pNew1 = cv::Point2d(pOld1.x + v[0] * m_stepSize, pOld1.y + v[1] * m_stepSize);
//     double distPOld = cv::norm(epipole1 - pOld1);
//     cv::Point2d startingPoint(epipole1.x + vCurr[0] * distPOld, epipole1.y + vCurr[1] * distPOld);
//     pNew1 = cv::Point2d(startingPoint.x + v[0] * m_stepSize, startingPoint.y + v[1] * m_stepSize);
    cout << "epipole1 " << epipole1 << endl;
    cout << "epipole2 " << epipole2 << endl;
    cout << "pOld1 " << pOld1 << endl;
    cout << "m_e1 " << m_e1 << endl;
    cout << "prevLine " << prevLine << endl;
    cout << "[nextEpiline] Testing " << pNew1 << " from " << pOld1 << endl;
    
    
    newLine1 = GET_LINE_FROM_POINTS(epipole1, pNew1);
    cout << "[nextEpiline] newLine1 " << newLine1 << endl;
    if (! IS_INSIDE_IMAGE(epipole1, imgDimensions)) {
        pNew1 = getBorderIntersection(epipole1, newLine1, imgDimensions, &pOld1);
    } else {
        vector <cv::Point2d> intersections;
        getBorderIntersections(epipole1, newLine1, imgDimensions, intersections);
        pNew1 = cv::Point2d(-1, -1);
        
//         cv::Vec3d v0(intersections[0].x - epipole1.x, intersections[0].y - epipole1.y, 1.0);
//         cv::Vec3d v1(intersections[1].x - epipole1.x, intersections[1].y - epipole1.y, 1.0);
//         cv::Vec3d v2(pOld1.x - epipole1.x, pOld1.y - epipole1.y, 1.0);
//         
//         v0 /= cv::norm(v0);
//         v1 /= cv::norm(v1);
//         v2 /= cv::norm(v2);
//         
//         cout << "[nextIt] int0 " << intersections[0] << endl;
//         cout << "[nextIt] int1 " << intersections[1] << endl;
//         cout << "[nextIt] acos0 " << acos(v0.dot(v2)) << endl;
//         cout << "[nextIt] acos1 " << acos(v1.dot(v2)) << endl;
//         
//         if (acos(v0.dot(v2)) > acos(v1.dot(v2))) {
//             pNew1 = intersections[1];
//         } else if (acos(v0.dot(v2)) > acos(v1.dot(v2))) {
//             
//         }
        
//         double dist1 = cv::norm(pOld1 - intersections[0]);
//         double dist2 = cv::norm(pOld1 - intersections[1]);
        
//         cv::Vec3d v0(intersections[0].x - epipole1.x, intersections[0].y - epipole1.y, 1.0);
        cv::Vec3f vOld(pOld1.x - epipole1.x, pOld1.y - epipole1.y, 0.0);
        cv::Vec3f vDir(v[0], v[1], 0.0);
        
//         v0 /= cv::norm(v0);
        vOld /= cv::norm(vOld);
        vDir /= cv::norm(vDir);
        
        cout << "nextEpiline vOld " << vOld << endl;
        cout << "nextEpiline vDir " << vDir << endl;
        cv::Vec3f vCross1 = vOld.cross(cv::Vec3f(vDir));
        for (uint32_t i = 0; i < intersections.size(); i++) {
            cv::Vec3f vNew(intersections[i].x - epipole1.x, intersections[i].y - epipole1.y, 0.0);
            vNew /= cv::norm(vNew);
            cv::Vec3f vCross2 = vOld.cross(vNew);
            cout << "nextEpiline vNew " << vNew << endl;
            
            cout << "nextEpiline vCross1 " << vCross1 << endl;
            cout << "nextEpiline vCross2 " << vCross2 << endl;
            cout << "nextEpiline intersections[1] " << intersections[1] << endl;
            cout << "nextEpiline intersections[0] " << intersections[0] << endl;
            
            if (SIGN(vOld.cross(vDir)[2]) == SIGN(vOld.cross(vNew)[2])) {
                pNew1 = intersections[i];
                break;
            }
        }
        
//         double minDist = std::numeric_limits<double>::max();
//         for (uint32_t i = 0; i < intersections.size(); i++) {
//             const double dist = (pOld1.x - intersections[i].x) * (pOld1.x - intersections[i].x) +
//                                 (pOld1.y - intersections[i].y) * (pOld1.y - intersections[i].y);
//             if (minDist > dist) {
//                 minDist = dist;
//                 pNew1 = intersections[i];
//             }
//         }
    }
    
    cout << "new intersection: " << pNew1 << endl;

    vector<cv::Point2f> points(1);
    points[0] = pNew1;
    vector<cv::Vec3f> inLines(1);
    inLines[0] = newLine1;
    vector<cv::Vec3f> outLines(1);
    computeEpilines(points, whichImage, F, inLines, outLines);
    newLine2 = outLines[0];
    
    if (! IS_INSIDE_IMAGE(epipole2, imgDimensions)) {
        cv::Point2d tmpPoint = getBorderIntersection(epipole2, newLine2, imgDimensions, &pOld2);
        pNew2 = tmpPoint;
    } else {
        vector <cv::Point2d> intersections;
        getBorderIntersections(epipole2, newLine2, imgDimensions, intersections);
        
        pNew2 = intersections[0];
//         cv::Vec3d v0(intersections[0].x - epipole2.x, intersections[0].y - epipole2.y, 1.0);
//         cv::Vec3d v1(intersections[1].x - epipole2.x, intersections[1].y - epipole2.y, 1.0);
//         cv::Vec3d v2(pOld2.x - epipole2.x, pOld2.y - epipole2.y, 1.0);
//         
//         v0 /= cv::norm(v0);
//         v1 /= cv::norm(v1);
//         v2 /= cv::norm(v2);
//         
//         if (fabs(acos(v0.dot(v2))) > fabs(acos(v1.dot(v2)))) {
//             pNew2 = intersections[1];
//         }
        
        double minDist = std::numeric_limits<double>::max();
        for (uint32_t i = 0; i < intersections.size(); i++) {
            const double dist = (pOld2.x - intersections[i].x) * (pOld2.x - intersections[i].x) +
                                (pOld2.y - intersections[i].y) * (pOld2.y - intersections[i].y);
            if (minDist > dist) {
                minDist = dist;
                pNew2 = intersections[i];
            }
        }
    }

}

inline void PolarCalibration::getNewEpiline(const cv::Point2d epipole1, const cv::Point2d epipole2, const cv::Size & imgDimensions, 
                                     const cv::Mat & F, const cv::Point2d pOld1, const cv::Point2d pOld2, 
                                     cv::Vec3f prevLine1, cv::Vec3f prevLine2, 
                                     cv::Point2d & pNew1, cv::Point2d & pNew2, cv::Vec3f & newLine1, cv::Vec3f & newLine2) {
    
    getNewPointAndLineSingleImage(epipole1, epipole2, imgDimensions, F, 1, pOld1, pOld2, prevLine1, pNew1, newLine1, pNew2, newLine2);
    
    // If the distance is too big in image 2, we do it in the opposite sense
//     double distImg2 = (pOld2.x - pNew2.x) * (pOld2.x - pNew2.x) + (pOld2.y - pNew2.y) * (pOld2.y - pNew2.y);
//     if (distImg2 > m_stepSize * m_stepSize)
//         getNewPointAndLineSingleImage(epipole2, epipole1, imgDimensions, F, 2, pOld2, pOld1, prevLine2, pNew2, newLine2, pNew1, newLine1);
    
    if (m_showIterations) {
        showNewEpiline(epipole1, m_line1B, m_line1E, newLine1, pOld1, pNew1, imgDimensions, std::string("newEpiline1"));
        showNewEpiline(epipole2, m_line2B, m_line2E, newLine2, pOld2, pNew2, imgDimensions, std::string("newEpiline2"));
        cv::moveWindow("newEpiline2", imgDimensions.width +10, 0);
    }
}

inline void PolarCalibration::transformLine(const cv::Point2d& epipole, const cv::Point2d& p2, const cv::Mat& inputImage, 
                                            const uint32_t & thetaIdx, const double &minRho, const double & maxRho, cv::Mat& mapX, cv::Mat& mapY)
{
    cv::Vec2f v(p2.x - epipole.x, p2.y - epipole.y);
    double maxDist = cv::norm(v);
    v /= maxDist;
    
    {
        uint32_t rhoIdx = 0;
        for (double rho = minRho; rho <= min(maxDist, maxRho); rho += 1.0, rhoIdx++) {
            cv::Point2d target(v[0] * rho + epipole.x, v[1] * rho + epipole.y);
            if ((target.x >= 0) && (target.x < inputImage.cols) &&
                (target.y >= 0) && (target.y < inputImage.rows)) {

                mapX.at<float>(thetaIdx, rhoIdx) = target.x;
                mapY.at<float>(thetaIdx, rhoIdx) = target.y;
            }
        }
    }
}

void PolarCalibration::doTransformation(const cv::Mat& img1, const cv::Mat& img2, const cv::Point2d epipole1, const cv::Point2d epipole2, const cv::Mat & F) {
   
    const double rhoRange1 = m_maxRho1 - m_minRho1 + 1;
    const double rhoRange2 = m_maxRho2 - m_minRho2 + 1;
    
    const double rhoRange = max(rhoRange1, rhoRange2);
    
    m_mapX1 = cv::Mat::zeros(2 * (img1.rows + img1.cols), rhoRange, CV_32FC1);
    m_mapY1 = cv::Mat::zeros(2 * (img1.rows + img1.cols), rhoRange, CV_32FC1);
    m_mapX2 = cv::Mat::zeros(2 * (img1.rows + img1.cols), rhoRange, CV_32FC1);
    m_mapY2 = cv::Mat::zeros(2 * (img1.rows + img1.cols), rhoRange, CV_32FC1);
    
    //TODO: Remove
//     m_b1 = cv::Point2d(360, 0);
    
    {
        cv::Point2d p1 = m_b1, p2 = m_b2;
        cv::Vec3f line1 = m_line1B, line2 = m_line2B;

        int32_t crossesLeft = 0;
        if (IS_INSIDE_IMAGE(epipole1, img1.size()) && IS_INSIDE_IMAGE(epipole2, img2.size()))
            crossesLeft++;
        
        uint32_t thetaIdx = 0;
        double lastAngle = std::numeric_limits< double >::min();
        double lastAngleIncrement = 0.0;
        bool hadNegativeIncr = false;
        cv::Vec3d lastVector(0, 0, 0);
        cv::Vec3d lastDirection(0, 0, 0);
        double lastIncrement = 0.0;
        
        bool lastReached = false;
        double lastCrossProd = 0;
//         bool lastDirection = false;
        while (true) {
            transformLine(epipole1, p1, img1, thetaIdx, m_minRho1, m_maxRho1, m_mapX1, m_mapY1);
            transformLine(epipole2, p2, img2, thetaIdx, m_minRho2, m_maxRho2, m_mapX2, m_mapY2);
            
            // TODO: Visualize while doing the transformation. I'm not planning to do that unless I
            // need it due to a bug
//             if (m_showIterations) {
//                 cv::Size newSize;
//                 if (m_mapX1.cols > m_mapX1.rows) {
//                     newSize = cv::Size(600, 600 * (thetaIdx + 100) / (m_maxRho1 - m_minRho1));
//                 } else {
//                     newSize = cv::Size(600 * (m_maxRho1 - m_minRho1) / (thetaIdx + 100), 600);
//                 }
// 
//                 cout << "thetaIdx " << thetaIdx << endl;
//                 cout << "rho " << (m_maxRho1 - m_minRho1) << endl;
//                 cout << "newSize " << newSize << endl;
//                 
// //                 exit(0);
//                 
//                 cv::Mat rectified1, rectified2;
//                 cv::Mat scaled1, scaled2;
//                 getRectifiedImages(img1, img2, rectified1, rectified2);
//                 cv::resize(rectified1, scaled1, newSize);
//                 cv::resize(rectified2, scaled2, newSize);
//                 const double proportion = (2 * (img1.rows + img1.cols)) / (maxRho - minRho + 1);
//                 cv::imshow("outputImage1", scaled1);
//                 cv::imshow("outputImage2", scaled2);
//             }

            cv::Vec3f v0(p1.x - epipole1.x, p1.y - epipole1.y, 1.0);
            v0 /= cv::norm(v0);
            cv::Point2d oldP1 = p1;
            
            getNewEpiline(epipole1, epipole2, cv::Size(img1.cols, img1.rows), F, p1, p2, line1, line2, p1, p2, line1, line2);

            // Check if we reached the end
            cv::Vec3f v1(p1.x - epipole1.x, p1.y - epipole1.y, 0.0);
            v1 /= cv::norm(v1);
            cv::Vec3f v2(m_e1.x - epipole1.x, m_e1.y - epipole1.y, 0.0);
            v2 /= cv::norm(v2);
            cv::Vec3f v3(oldP1.x - epipole1.x, oldP1.y - epipole1.y, 0.0);
            v3 /= cv::norm(v3);
            
            cout << "v1 " << v1 << ", v2 " << v2 << ", v3 " << v3 << endl;
            cout << "p1 " << p1 << ", m_e1 " << m_e1 << endl;
//             double angle = acos(v1.dot(v2));
//             double angleIncrement = angle - lastAngle;
            
//             cv::Vec3f currVector = v1.cross(v2);
            
            bool reached = REACHED(v1, v2); 
            double crossProd = v1.cross(v2)[2];
            double angle = fabs(acos(v0.dot(v1)));
            double angleIncrement = angle-lastAngle;
//             cout << "REACHED " << v1.cross(v2) << " == " << REACHED(v1, v2) << endl;
            cout << "crossProd " << crossProd << ", lastCrossProd " << lastCrossProd << ", angle " << angle * 180 / CV_PI << endl;
//             cout << "angleIncrement " << angleIncrement << ", lastAngleIncrement " << lastAngleIncrement << endl;
//             cout << "oldP1 " << oldP1 << ", p1 " << p1 << ", cond " << (IS_A_CORNER(oldP1, img1.size()) && IS_A_CORNER(p1, img1.size()) && (cv::norm(p1 - oldP1) > 1.1)) << endl;
            if (thetaIdx != 0) {
                if (crossProd < 5.0) {
                    cout << "Reaching " << endl;
                    if ((SIGN(lastCrossProd) != SIGN(crossProd))  || (p1 == cv::Point2d(-1, -1)) || (fabs(acos(v1.dot(-v3))) < 0.01))
//                         (IS_A_CORNER(oldP1, img1.size()) && IS_A_CORNER(p1, img1.size()) && (cv::norm(p1 - oldP1) > 1.1)))
                        crossesLeft--;
                    cout << "crossesLeft " << crossesLeft << endl;
                    if ((crossesLeft < 0)) {
                        cout << "FINISHED!!!!" << endl;
                        break;
                    }
                }
//                 if (lastReached != reached)
//                 if (lastReached && !reached)
//                     crossesLeft--;
//                 double angleIncrement = currVector[2] - lastVector[2];
//                 if (thetaIdx > 1) {
// //                     cout << "lastVector " << lastVector[2] << ", increment " << angleIncrement << ", lastIncrement " << lastIncrement << endl;
//                     cout << "[DEBUG] lastVector " << lastVector[2] << ", currVector[2] " << currVector[2] << endl;
// //                     if ((lastVector[2] < 0.01) && (SIGN(angleIncrement) != SIGN(lastIncrement)))
//                     if (SIGN(lastVector[2]) != SIGN(currVector[2]))
//                         crossesLeft--;
//                     cout << "[DEBUG] crossesLeft " << crossesLeft << endl;
//                     }
// //                     cout << crossesLeft << endl;
//                 }
//                 lastIncrement = angleIncrement;
            }
            lastReached = reached;
            lastCrossProd = crossProd;
            lastAngle = angle;
            lastAngleIncrement = angleIncrement;
//             lastVector = currVector;
        
//             double angleIncrement = angle - lastAngle;
            
            
//             if (SIGN(lastAngleIncrement) != SIGN(angleIncrement)) {
//                 cout << "crossesLeft " << crossesLeft << endl;
//                 if (crossesLeft == 0) {
//                     break;
//                 } else {
//                     crossesLeft--;
//                 }
//             }
            
//             if (angleIncrement < 0) 
//                 hadNegativeIncr = true;
            
//             cout << "angle " << angle << endl;
//             cout << "angleIncrement " << angleIncrement << endl;
            
//             cv::Point2d intersection;
//             getBorderIntersection(epipole1, line1, img1.size(), &intersection);
            /*cv::Vec3d currVector(m_e1.x - p1.x, m_e1.y - p1.y, 0.0);
            currVector /= cv::norm(currVector);
            
            cout << "[ENDING] lastDirection " << lastDirection << endl;
            cout << "[ENDING] lastVector " << lastVector << endl;
            cout << "[ENDING] currVector " << currVector << endl;
            
            if (cv::norm(lastVector) != 0.0) {
                const cv::Vec3d & currDirection = currVector.cross(lastVector);
                cout << "[ENDING] currDirection " << currDirection << endl;
                if (cv::norm(lastDirection) != 0.0) {
                    if (SIGN(currDirection[2]) != SIGN(lastDirection[2])) {
//                         if (cv::norm(p1 - m_e1) < 5.0) {
                            cout << "FINISHED!!!" << endl;
                            break;
//                         }
                    }
                } else {
                    cout << "[ENDING] No last direction!!!" << endl;
                }
                lastDirection = currDirection;
            } else {
                cout << "[ENDING] No last vector!!!" << endl;
            }
            lastVector = currVector;
            cout << "[ENDING] lastDirection2 " << lastDirection << endl;
            cout << "[ENDING] lastVector2 " << lastVector << endl;
            cout << "[ENDING] ********************* " << endl;*/
            
            
//             if ((cv::norm(p1 - m_e1) < 5.0) || (cv::norm(p2 - m_e2) < 5.0)) {
//                 cout << "FINISHED!!!" << endl;
//                 break;
//             }
                
            
            
            /*if (SIGN(lastAngleIncrement) != SIGN(angleIncrement)) {
                cout << "crossesLeft " << crossesLeft << endl;
                if (crossesLeft == 0) {
//                     break;
                } else {
                    crossesLeft--;
                }
            }*/
//             if ((angleIncrement >= 0.0) && hadNegativeIncr) {
//                 cout << "FINISHED!!!" << endl;
//                 break;
//             }
            thetaIdx++;
//             lastAngle = angle;
//             lastAngleIncrement = angleIncrement;

            if (m_showIterations) {
                int keycode = cv::waitKey(0);
                
                // q: exit
                if (keycode == 113) {
                    exit(0);
                }
                // 1: stepSize = 1
                if (keycode == 49) {
                    m_stepSize = 1;
                }
                // 2: stepSize = 50
                if (keycode == 50) {
                    m_stepSize = 10;
                }
                // 3: stepSize = 50
                if (keycode == 51) {
                    m_stepSize = 50;
                }
                // 4: stepSize = 100
                if (keycode == 51) {
                    m_stepSize = 100;
                }
                // n: next image
                if (keycode == 110) {
                    break;
                }
                // r: reset current image
                if (keycode == 114) {
                    p1 = m_b1;
                    p2 = m_b2;
                    line1 = m_line1B;
                    line2 = m_line2B;
                }
            }
        }
        m_mapX1 = m_mapX1(cv::Range(0, thetaIdx), cv::Range::all());
        m_mapY1 = m_mapY1(cv::Range(0, thetaIdx), cv::Range::all());
        m_mapX2 = m_mapX2(cv::Range(0, thetaIdx), cv::Range::all());
        m_mapY2 = m_mapY2(cv::Range(0, thetaIdx), cv::Range::all());
    }
    
//     getRectifiedImages(img1, img2, m_rectified1, m_rectified2);
}

void PolarCalibration::getRectifiedImages(const cv::Mat& img1, const cv::Mat& img2, 
                                          cv::Mat& rectified1, cv::Mat& rectified2, int interpolation) {
    
    cv::remap(img1, rectified1, m_mapX1, m_mapY1, interpolation, cv::BORDER_TRANSPARENT);
    cv::remap(img2, rectified2, m_mapX2, m_mapY2, interpolation, cv::BORDER_TRANSPARENT);
}


