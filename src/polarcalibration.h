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


#ifndef POLARCALIBRATION_H
#define POLARCALIBRATION_H

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <vector>

using namespace std;

class PolarCalibration
{
public:
    PolarCalibration();
    ~PolarCalibration();
    void compute(const cv::Mat& img1distorted, const cv::Mat& img2distorted,
                 const cv::Mat & cameraMatrix1, const cv::Mat & distCoeffs1,
                 const cv::Mat & cameraMatrix2, const cv::Mat & distCoeffs2);
    void compute(/*const*/ cv::Mat & img1, /*const*/ cv::Mat & img2);
private:
    void computeEpilinesBasedOnCase(const cv::Point2d &epipole, const cv::Size imgDimensions,
                                    const cv::Mat & F, const uint32_t & imgIdx,
                                    vector<cv::Point2f> &externalPoints, vector<cv::Vec3f> &epilines);

    bool getThetaAB(cv::Point2d & epipole, const cv::Vec3f &epiline, const cv::Size &imgDimensions, double & newTheta);
    bool getThetaCD(cv::Point2d & epipole, const cv::Vec3f &epiline, const cv::Size &imgDimensions, double & newTheta);
    bool getThetaBD(cv::Point2d & epipole, const cv::Vec3f &epiline, const cv::Size &imgDimensions, double & newTheta);
    bool getThetaAC(cv::Point2d & epipole, const cv::Vec3f &epiline, const cv::Size &imgDimensions, double & newTheta);
    void getThetaFromEpilines(/*const*/ cv::Point2d &epipole, const cv::Size imgDimensions,
                                        const vector<cv::Vec3f> &epilines, double & newTheta, double & minTheta, double & maxTheta);
    void determineCommonRegion(/*const*/ cv::Point2d &epipole, const cv::Size imgDimensions,
                                         const vector<cv::Point2f> &externalPoints, const vector<cv::Vec3f> &epilines,
                                         double & minTheta, double & maxTheta);
    void determineRhoRange(const cv::Point2d &epipole, const cv::Size imgDimensions,
                           const vector<cv::Point2f> &externalPoints, const vector<cv::Vec3f> &epilines,
                           double & minRho, double & maxRho);
    bool checkAB(const cv::Point2d &epipole, const double & theta, const cv::Size & imgDimensions, cv::Point2d & b);
    bool checkCD(const cv::Point2d &epipole, const double & theta, const cv::Size & imgDimensions, cv::Point2d & b);
    bool checkAC(const cv::Point2d &epipole, const double & theta, const cv::Size & imgDimensions, cv::Point2d & b);
    bool checkBD(const cv::Point2d &epipole, const double & theta, const cv::Size & imgDimensions, cv::Point2d & b);
    void getLineFromPoints(const cv::Point2d & p1, const cv::Point2d & p2, vector<cv::Vec3f> & line);
    void getLineFromAngle(/*const*/ cv::Point2d &epipole, /*const*/ double & theta,
                                    const cv::Size & imgDimensions, cv::Point2d & b, vector<cv::Vec3f> & line);
    double getNextThetaIncrement(/*const*/ cv::Point2d &epipole, /*const*/ double & theta, /*const*/ double & maxRho,
                                    const cv::Size & imgDimensions);
    void doTransformation(/*const*/ cv::Point2d &epipole1, /*const*/ cv::Point2d &epipole2,
                                    /*const*/ cv::Mat & imgInput1, /*const*/ cv::Mat & imgInput2,
                                    cv::Mat & imgTransformed1, cv::Mat & imgTransformed2,
                                    /*const*/ double & minTheta1, /*const*/ double & minTheta2,
                                    /*const*/ double & maxTheta1, /*const*/ double & maxTheta2,
                                    /*const*/ double & minRho1, /*const*/ double & minRho2,
                                    /*const*/ double & maxRho1, /*const*/ double & maxRho2,
                                    const cv::Mat & F);
};

#endif // POLARCALIBRATION_H
