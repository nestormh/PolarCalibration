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

#define STEP_SIZE 1.0

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

    void determineCommonRegion(/*const*/ vector<cv::Point2f> &epipoles, 
                               const cv::Size imgDimensions, const cv::Mat & F);
    void determineRhoRange(const cv::Point2d &epipole, const cv::Size imgDimensions,
                           const vector<cv::Point2f> &externalPoints, const vector<cv::Vec3f> &epilines,
                           double & minRho, double & maxRho);
    bool findFundamentalMat(const cv::Mat & img1, const cv::Mat & img2, cv::Mat & F,
                            cv::Point2d & epipole1, cv::Point2d & epipole2, cv::Point2d & m);
    void getEpipoles(const cv::Mat & F, cv::Point2d & epipole1, cv::Point2d & epipole2);
    void checkF(cv::Mat & F, cv::Point2d & epipole1, cv::Point2d & epipole2, const cv::Point2d & m, const cv::Point2d & m1);
    bool isInsideImage(const cv::Point2d & point, const cv::Size & imgDimensions);
    cv::Vec3f getLineFromTwoPoints(const cv::Point2d & point1, const cv::Point2d & point2);
    void getExternalPoints(const cv::Point2d &epipole, const cv::Size imgDimensions,
                           vector<cv::Point2f> &externalPoints);
    bool lineIntersectsSegment(const cv::Vec3d & line, const cv::Point2d & p1, const cv::Point2d & p2, cv::Point2d * intersection = NULL);
    bool lineIntersectsRect(const cv::Vec3d & line, const cv::Size & imgDimensions);
    bool isTheRightPoint(const cv::Point2d & epipole, const cv::Point2d & intersection, const cv::Vec3d & line,
                         const cv::Point2d * lastPoint);
    cv::Point2d getBorderIntersection(const cv::Point2d & epipole, const cv::Vec3d & line, const cv::Size & imgDimensions, 
                                      const cv::Point2d * lastPoint = NULL);        
    void computeEpilines(const vector<cv::Point2f> & points, const uint32_t &whichImage, 
                        const cv::Mat & F, const vector <cv::Vec3f> & oldlines, vector <cv::Vec3f> & newLines);
    bool sign(const double & val);
    void getNewPointAndLineSingleImage(const cv::Point2d epipole1, const cv::Point2d epipole2, const cv::Size & imgDimensions, 
                                       const cv::Mat & F, const uint32_t & whichImage, const cv::Point2d & pOld1, const cv::Point2d & pOld2,
                                        /*const*/ cv::Vec3f & prevLine, cv::Point2d & pNew1, cv::Vec3f & newLine1, 
                                        cv::Point2d & pNew2, cv::Vec3f & newLine2);
    void getNewEpiline(const cv::Point2d epipole1, const cv::Point2d epipole2, const cv::Size & imgDimensions, 
                       const cv::Mat & F, const cv::Point2d pOld1, const cv::Point2d pOld2, 
                       /*const*/ cv::Vec3f prevLine1, /*const*/ cv::Vec3f prevLine2, 
                       cv::Point2d & pNew1, cv::Point2d & pNew2, cv::Vec3f & newLine1, cv::Vec3f & newLine2);
    void transformLine(const cv::Point2d& epipole, const cv::Point2d& p2, const cv::Mat& inputImage, 
                       const uint32_t & thetaIdx, const double &minRho, const double & maxRho, cv::Mat& outputImage);
    void doTransformation(const cv::Mat& img1, const cv::Mat& img2, const cv::Point2d epipole1, const cv::Point2d epipole2, const cv::Mat & F);
    
    // Visualization functions
    cv::Point2d image2World(const cv::Point2d & point, const cv::Size & imgDimensions);
    cv::Point2d getPointFromLineAndX(const double & x, const cv::Vec3f line);
    void showCommonRegion(const cv::Point2d epipole, const cv::Vec3f & line11, const cv::Vec3f & line12,
                          const cv::Vec3f & line13, const cv::Vec3f & line14, 
                          const cv::Vec3f & lineB, const cv::Vec3f & lineE, 
                          const cv::Point2d & b, const cv::Size & imgDimensions, 
                          const vector<cv::Point2f> & externalPoints, std::string windowName);
    void showNewEpiline(const cv::Point2d epipole, const cv::Vec3f & lineB, const cv::Vec3f & lineE, 
                        const cv::Vec3f & newLine, const cv::Point2d & pOld, const cv::Point2d & pNew, 
                        const cv::Size & imgDimensions, std::string windowName);
    
    uint32_t m_hessianThresh;
    
    cv::Vec3f m_line1B, m_line1E, m_line2B, m_line2E;
    cv::Point2d m_b1, m_b2, m_e1, m_e2;
    double m_stepSize;
    
    double m_minRho1, m_maxRho1, m_minRho2, m_maxRho2;
    
    bool m_showCommonRegion, m_showIterations;
};

#endif // POLARCALIBRATION_H
