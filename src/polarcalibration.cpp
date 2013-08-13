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
                               const cv::Mat & cameraMatrix2, const cv::Mat & distCoeffs2) {
    
    cv::Mat img1(img1distorted.rows, img1distorted.cols, CV_8UC1);
    cv::Mat img2(img2distorted.rows, img2distorted.cols, CV_8UC1);
    
    cv::undistort(img1distorted, img1, cameraMatrix1, distCoeffs1);
    cv::undistort(img2distorted, img2, cameraMatrix2, distCoeffs2);
    
    return compute(img1, img2);
}

bool PolarCalibration::compute(const cv::Mat& img1, const cv::Mat& img2) {
    
    clock_t begin = clock();
    
    cv::Mat F;
    cv::Point2d epipole1, epipole2;
    if (! findFundamentalMat(img1, img2, F, epipole1, epipole2, FMAT_METHOD_OFLOW))
        return false;
    
    clock_t end = clock();
    double elapsedFMat = double(end - begin) / CLOCKS_PER_SEC;
    
    cout << "epipole1: " << epipole1 << endl;
    cout << "epipole2: " << epipole2 << endl;

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

inline bool PolarCalibration::findFundamentalMat(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& F, 
                                                 cv::Point2d& epipole1, cv::Point2d& epipole2, const uint32_t method)
{
    vector<cv::Point2f> points1, points2;
    
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
    
    cout << "sz = " << points1.size() << endl;
    
    F = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
    cout << "F:\n" << F << endl; 
    
    if (cv::countNonZero(F) == 0)
        return false;
    
    // We obtain the epipoles
    getEpipoles(F, epipole1, epipole2);
    
    checkF(F, epipole1, epipole2, points1[0], points2[0]);
    
    /// NOTE: Remove. Just for debugging (begin)
    //     cv::FileStorage file("/home/nestor/Dropbox/KULeuven/projects/PolarCalibration/testing/lastMat_B.xml", cv::FileStorage::READ);
    //     file["F"] >> F;
    //     file.release();
    //     cout << "F (from file)\n" << F << endl;
    //     getEpipoles(F, epipole1, epipole2);
    /// NOTE: Remove. Just for debugging (end)
    
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
    
    vector<cv::Point2f> points1(keypoints1.size()), points2;
    {
        uint32_t idx = 0;
        for (vector<cv::KeyPoint>::iterator it = keypoints1.begin(); it != keypoints1.end(); it++, idx++) {
            points1[idx] = it->pt;
        }
    }    
    // Optical flow
    vector<cv::Mat> pyramid2;
    vector<uint8_t> status;
    vector<float_t> error;
    
    cv::calcOpticalFlowPyrLK(img1, img2, points1, points2, status, error, cv::Size(3, 3), 3);
    
    vector<cv::Point2f> pointsA(points1.size()), pointsB(points2.size());
    {
        uint32_t idx = 0;
        for (uint32_t i = 0; i < points1.size(); i++) {
            if (status[i] == 1) {
                pointsA[idx] = points1[i];
                pointsB[idx] = points2[i];
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
    
    if (((fl.at<double>(0,0) < 0.0) && (fl1.at<double>(0,0) > 0.0)) ||
        ((fl.at<double>(0,0) > 0.0) && (fl1.at<double>(0,0) < 0.0))) {

        F = -F;
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
    
    if (((int32_t)round(x) >= (int32_t)min(p1.x, p2.x)) && ((int32_t)round(x) <= (int32_t)max(p1.x, p2.x))) {
        if (((int32_t)round(y) >= (int32_t)min(p1.y, p2.y)) && ((int32_t)round(y) <= (int32_t)max(p1.y, p2.y))) {
            if (intersection != NULL)
                *intersection = cv::Point2d(x, y);
            
            return true;
        }
    }
    
    return false;
}

inline bool PolarCalibration::lineIntersectsRect(const cv::Vec3d & line, const cv::Size & imgDimensions) {
    return lineIntersectsSegment(line, cv::Point2d(0, 0), cv::Point2d(imgDimensions.width - 1, 0)) ||
            lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, 0), cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1)) ||
            lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), cv::Point2d(0, imgDimensions.height - 1)) ||
            lineIntersectsSegment(line, cv::Point2d(0, imgDimensions.height - 1), cv::Point2d(0, 0));
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
    } else {
        if ((line[0] > 0) && (epipole.y < intersection.y)) return false;
        if ((line[0] < 0) && (epipole.y > intersection.y)) return false;
        if ((line[1] > 0) && (epipole.x > intersection.x)) return false;
        if ((line[1] < 0) && (epipole.x < intersection.x)) return false;
        
        return true;
    }
    return false;
}

inline cv::Point2d PolarCalibration::getBorderIntersection(const cv::Point2d & epipole, const cv::Vec3d & line, const cv::Size & imgDimensions, 
                                                            const cv::Point2d * lastPoint) {
    
    cv::Point2d intersection;

    if (IS_INSIDE_IMAGE(epipole, imgDimensions)) {
        if (lineIntersectsSegment(line, cv::Point2d(0, 0), cv::Point2d(imgDimensions.width - 1, 0), &intersection)) {
            if (isTheRightPoint(epipole, intersection, line, lastPoint)) {

                return intersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, 0), cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), &intersection)) {
            if (isTheRightPoint(epipole, intersection, line, lastPoint)) {

                return intersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), cv::Point2d(0, imgDimensions.height - 1), &intersection)) {
            if (isTheRightPoint(epipole, intersection, line, lastPoint)) {

                return intersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(0, imgDimensions.height - 1), cv::Point2d(0, 0), &intersection)) {
            if (isTheRightPoint(epipole, intersection, line, lastPoint)) {

                return intersection;
            }
        }
    } else {
        double maxDist = std::numeric_limits<double>::min();
        cv::Point2d tmpIntersection;
        if (lineIntersectsSegment(line, cv::Point2d(0, 0), cv::Point2d(imgDimensions.width - 1, 0), &tmpIntersection)) {
            const double dist2 = (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x) +
                                 (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x);
            
            if (dist2 > maxDist) {
                maxDist = dist2;
                intersection = tmpIntersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, 0), cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), &tmpIntersection)) {
            const double dist2 = (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x) +
            (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x);
            
            if (dist2 > maxDist) {
                maxDist = dist2;
                intersection = tmpIntersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(imgDimensions.width - 1, imgDimensions.height - 1), cv::Point2d(0, imgDimensions.height - 1), &tmpIntersection)) {
            const double dist2 = (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x) +
            (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x);
            
            if (dist2 > maxDist) {
                maxDist = dist2;
                intersection = tmpIntersection;
            }
        }
        if (lineIntersectsSegment(line, cv::Point2d(0, imgDimensions.height - 1), cv::Point2d(0, 0), &tmpIntersection)) {
            const double dist2 = (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x) +
            (tmpIntersection.x - epipole.x) * (tmpIntersection.x - epipole.x);
            
            if (dist2 > maxDist) {
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
        m_line1B = lineIntersectsRect(line13, imgDimensions)? line13 : line11;
        m_line1E = lineIntersectsRect(line14, imgDimensions)? line14 : line12;
        m_line2B = lineIntersectsRect(line21, imgDimensions)? line21 : line23;
        m_line2E = lineIntersectsRect(line22, imgDimensions)? line22 : line24;
                
        // Beginning and ending lines intersection with the borders
        m_b1 = getBorderIntersection(epipoles[0], m_line1B, imgDimensions);
        m_b2 = getBorderIntersection(epipoles[1], m_line2B, imgDimensions);
        m_e1 = getBorderIntersection(epipoles[0], m_line1E, imgDimensions);
        m_e2 = getBorderIntersection(epipoles[1], m_line2E, imgDimensions);
        
        if (m_showCommonRegion) {
            showCommonRegion(epipoles[0], line11, line12, line13, line14, m_line1B, m_line1E, m_b1, imgDimensions, 
                            externalPoints1, std::string("leftCommonRegion"));
            showCommonRegion(epipoles[1], line23, line24, line21, line22, m_line2B, m_line2E, m_b2, imgDimensions, 
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
        m_b2 = getBorderIntersection(epipoles[1], m_line2B, imgDimensions);
        m_e1 = getBorderIntersection(epipoles[0], m_line1E, imgDimensions);
        m_e2 = getBorderIntersection(epipoles[1], m_line2E, imgDimensions);
                
        if (m_showCommonRegion) {
            showCommonRegion(epipoles[0], m_line1B, m_line1E, m_line2B, m_line2E, m_line1B, m_line1E, m_b1, imgDimensions, 
                            externalPoints1, std::string("leftCommonRegion"));
            showCommonRegion(epipoles[1], m_line2B, m_line2E, m_line1B, m_line1E, m_line2B, m_line2E, m_b2, imgDimensions, 
                            externalPoints2, std::string("rightCommonRegion"));
        }
    } else {
        // CASE 3: One inside and one outside
        if (IS_INSIDE_IMAGE(epipoles[0], imgDimensions)) {
            // CASE 3.1: Only the first epipole is inside
            
            const cv::Vec3f line23 = GET_LINE_FROM_POINTS(epipoles[1], externalPoints2[0]);
            const cv::Vec3f line24 = GET_LINE_FROM_POINTS(epipoles[1], externalPoints2[1]);
            
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
            m_e1 = getBorderIntersection(epipoles[0], m_line1E, imgDimensions);
            m_e2 = getBorderIntersection(epipoles[1], m_line2E, imgDimensions);
                        
            if (m_showCommonRegion) {
                showCommonRegion(epipoles[0], line13, line14, line13, line14, m_line1B, m_line1E, m_b1, imgDimensions, 
                                externalPoints1, std::string("leftCommonRegion"));
                showCommonRegion(epipoles[1], line23, line24, line23, line24, m_line2B, m_line2E, m_b2, imgDimensions, 
                                externalPoints2, std::string("rightCommonRegion"));
            }
        } else {
            // CASE 3.2: Only the second epipole is inside
            const cv::Vec3f line11 = GET_LINE_FROM_POINTS(epipoles[0], externalPoints1[0]);
            const cv::Vec3f line12 = GET_LINE_FROM_POINTS(epipoles[0], externalPoints1[1]);
            
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
            m_e1 = getBorderIntersection(epipoles[0], m_line1E, imgDimensions);
            m_e2 = getBorderIntersection(epipoles[1], m_line2E, imgDimensions);

            if (m_showCommonRegion) {
                showCommonRegion(epipoles[0], line11, line12, line11, line12, m_line1B, m_line1E, m_b1, imgDimensions, 
                                externalPoints1, std::string("leftCommonRegion"));
                showCommonRegion(epipoles[1], line21, line22, line21, line22, m_line2B, m_line2E, m_b2, imgDimensions, 
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
    const cv::Vec3f vBegin(m_line1B[0], m_line1B[1], m_line1B[2]);
    const cv::Vec3f vEnd(m_line1E[0], m_line1E[1], m_line1E[2]);
    const cv::Vec3f vCross = vEnd.cross(vBegin);
    
    prevLine = GET_LINE_FROM_POINTS(epipole1, pOld1);
    
    cv::Vec2f v(-prevLine[0], -prevLine[1]);
    v /= cv::norm(v);
    if (vCross[2] < 0.0) {
        v = -v;
    }

    pNew1 = cv::Point2d(pOld1.x + v[0] * m_stepSize, pOld1.y + v[1] * m_stepSize);
    
    newLine1 = GET_LINE_FROM_POINTS(epipole1, pNew1);
    pNew1 = getBorderIntersection(epipole1, newLine1, imgDimensions, &pOld1);

    vector<cv::Point2f> points(1);
    points[0] = pNew1;
    vector<cv::Vec3f> inLines(1);
    inLines[0] = newLine1;
    vector<cv::Vec3f> outLines(1);
    computeEpilines(points, whichImage, F, inLines, outLines);

    newLine2 = outLines[0];
    cv::Point2d tmpPoint = getBorderIntersection(epipole2, newLine2, imgDimensions, &pOld2);
    pNew2 = tmpPoint;
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
        for (double rho = minRho; rho <= maxDist; rho += 1.0, rhoIdx++) {
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
   
    double minRho = min(m_minRho1, m_minRho2);
    double maxRho = max(m_maxRho1, m_maxRho2);

    m_mapX1 = cv::Mat::zeros(2 * (img1.rows + img1.cols), maxRho - minRho + 1, CV_32FC1);
    m_mapY1 = cv::Mat::zeros(2 * (img1.rows + img1.cols), maxRho - minRho + 1, CV_32FC1);
    m_mapX2 = cv::Mat::zeros(2 * (img1.rows + img1.cols), maxRho - minRho + 1, CV_32FC1);
    m_mapY2 = cv::Mat::zeros(2 * (img1.rows + img1.cols), maxRho - minRho + 1, CV_32FC1);
    
    {
        cv::Point2d p1 = m_b1, p2 = m_b2;
        cv::Vec3f line1 = m_line1B, line2 = m_line2B;

        uint32_t crossesLeft = 1;
        
        uint32_t thetaIdx = 0;
        double lastAngle = 0.0;
        double lastAngleIncrement = 0.0;
        while (true) {
            transformLine(epipole1, p1, img1, thetaIdx, m_minRho1, m_maxRho1, m_mapX1, m_mapY1);
            transformLine(epipole2, p2, img2, thetaIdx, m_minRho2, m_maxRho2, m_mapX2, m_mapY2);
            
            // TODO: Visualize while doing the transformation. I'm not planning to do that unless I
            // need it due to a bug
//             if (m_showIterations) {
//                 const double proportion = (2 * (img1.rows + img1.cols)) / (maxRho - minRho + 1);
//                 cv::resize(outputImg1, showImg1, cv::Size((uint32_t)(600.0 / proportion), 600));
//                 cv::imshow("outputImage1", showImg1);
//                 cv::resize(outputImg2, showImg2, cv::Size((uint32_t)(600.0 / proportion), 600));
//                 cv::imshow("outputImage2", showImg2);
//             }
            
            getNewEpiline(epipole1, epipole2, cv::Size(img1.cols, img1.rows), F, p1, p2, line1, line2, p1, p2, line1, line2);

            // Check if we reached the end
            cv::Vec3f v1(p1.x - epipole1.x, p1.y - epipole1.y, 0.0);
            v1 /= cv::norm(v1);
            cv::Vec3f v2(m_e1.x - epipole1.x, m_e1.y - epipole1.y, 0.0);
            v2 /= cv::norm(v2);

            double angle = acos(v1.dot(v2));
            double angleIncrement = angle - lastAngle;
            
            if (SIGN(lastAngleIncrement) != SIGN(angleIncrement)) {
                if (crossesLeft == 0) {
                    break;
                } else {
                    crossesLeft--;
                }
            }
            thetaIdx++;
            lastAngle = angle;
            lastAngleIncrement = angleIncrement;

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
    
    getRectifiedImages(img1, img2, m_rectified1, m_rectified2);
}

void PolarCalibration::getRectifiedImages(const cv::Mat& img1, const cv::Mat& img2, 
                                          cv::Mat& rectified1, cv::Mat& rectified2, int interpolation) {
    
    cv::remap(img1, rectified1, m_mapX1, m_mapY1, interpolation);
    cv::remap(img2, rectified2, m_mapX2, m_mapY2, interpolation);
}


