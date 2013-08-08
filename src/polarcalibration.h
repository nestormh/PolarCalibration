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
    bool compute(const cv::Mat& img1distorted, const cv::Mat& img2distorted,
                 const cv::Mat & cameraMatrix1, const cv::Mat & distCoeffs1,
                 const cv::Mat & cameraMatrix2, const cv::Mat & distCoeffs2);
    bool compute(/*const*/ cv::Mat & img1, /*const*/ cv::Mat & img2);
    
    void setHessianThresh(const uint32_t & hessianThresh) { m_hessianThresh = hessianThresh; }
private:
    void computeEpilinesBasedOnCase(const cv::Point2d &epipole, const cv::Size imgDimensions,
                                    const cv::Mat & F, const uint32_t & imgIdx, const cv::Point2d & m,
                                    vector<cv::Point2f> &externalPoints, vector<cv::Vec3f> &epilines);

    bool getThetaAB(cv::Point2d & epipole, const cv::Vec3f &epiline, const cv::Size &imgDimensions, double & newTheta);
    bool getThetaCD(cv::Point2d & epipole, const cv::Vec3f &epiline, const cv::Size &imgDimensions, double & newTheta);
    bool getThetaBD(cv::Point2d & epipole, const cv::Vec3f &epiline, const cv::Size &imgDimensions, double & newTheta);
    bool getThetaAC(cv::Point2d & epipole, const cv::Vec3f &epiline, const cv::Size &imgDimensions, double & newTheta);
    void getThetaFromEpilines(/*const*/ cv::Point2d &epipole, const cv::Size imgDimensions,
                                        const vector<cv::Vec3f> &epilines, double & newTheta, double & minTheta, double & maxTheta);
    void determineCommonRegion(/*const*/ vector<cv::Point2f> &epipoles, 
                               const cv::Size imgDimensions, const cv::Mat & F);
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
    bool findFundamentalMat(const cv::Mat & img1, const cv::Mat & img2, cv::Mat & F,
                            cv::Point2d & epipole1, cv::Point2d & epipole2, cv::Point2d & m);
    void getEpipoles(const cv::Mat & F, cv::Point2d & epipole1, cv::Point2d & epipole2);
    void checkF(cv::Mat & F, cv::Point2d & epipole1, cv::Point2d & epipole2, const cv::Point2d & m, const cv::Point2d & m1);
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
    bool isInsideImage(const cv::Point2d & point, const cv::Size & imgDimensions);
    cv::Vec3f getLineFromTwoPoints(const cv::Point2d & point1, const cv::Point2d & point2);
    void getExternalPoints(const cv::Point2d &epipole, const cv::Size imgDimensions,
                           vector<cv::Point2f> &externalPoints);
    bool lineIntersectsSegment(const cv::Vec3d & line, const cv::Point2d & p1, const cv::Point2d & p2, cv::Point2d * intersection = NULL);
    bool lineIntersectsRect(const cv::Vec3d & line, const cv::Size & imgDimensions);
    bool isTheRightPoint(const cv::Point2d & epipole, const cv::Point2d & intersection, const cv::Vec3d & line);
    cv::Point2d getBorderIntersection(const cv::Point2d & epipole, const cv::Vec3d & line, const cv::Size & imgDimensions);        
    cv::Point2d image2World(const cv::Point2d & point, const cv::Size & imgDimensions);
    cv::Point2d getPointFromLineAndX(const double & x, const cv::Vec3f line);
    cv::Point2d getPointFromLineAndY(const double & y, const cv::Vec3f line);
    void computeEpilines(const vector<cv::Point2f> & points, const uint32_t &whichImage, 
                        const cv::Mat & F, const vector <cv::Vec3f> & oldlines, vector <cv::Vec3f> & newLines);
    void showCommonRegion(const cv::Point2d epipole, const cv::Vec3f & line11, const cv::Vec3f & line12,
                          const cv::Vec3f & line13, const cv::Vec3f & line14, 
                          const cv::Vec3f & lineB, const cv::Vec3f & lineE, 
                          const cv::Point2d & b, const cv::Size & imgDimensions, 
                          const vector<cv::Point2f> & externalPoints, std::string windowName);
    
    uint32_t m_hessianThresh;
    
    cv::Vec3f m_line1B, m_line1E, m_line2B, m_line2E;
};

#endif // POLARCALIBRATION_H
