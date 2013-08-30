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

#define SIGN(val) (bool)(val >= 0.0)

#define REACHED(v1, v2) (bool)(v1.cross(v2)[2] < 0.5)

#define IS_INSIDE_IMAGE(point, imgDimensions) \
    ((point.x >= 0) && (point.y >= 0) && \
    (point.x < (imgDimensions.width - 1.0)) && (point.y <= (imgDimensions.height - 1.0)))
    
// (py – qy)x + (qx – px)y + (pxqy – qxpy) = 0
#define GET_LINE_FROM_POINTS(point1, point2) \
    cv::Vec3f(point1.y - point2.y, point2.x - point1.x, point1.x * point2.y - point2.x * point1.y)
    
#define IS_A_CORNER(point, imgDimensions) \
    (bool)(((uint32_t)point.x == 0) || ((uint32_t)point.x == imgDimensions.width - 1)) && \
          (((uint32_t)point.y == 0) || ((uint32_t)point.y == imgDimensions.height - 1))

class PolarCalibration
{
public:
    static const uint32_t FMAT_METHOD_OFLOW=0;
    static const uint32_t FMAT_METHOD_SURF=1;
    
    PolarCalibration();
    ~PolarCalibration();
    
    bool compute(const cv::Mat& img1distorted, const cv::Mat& img2distorted,
                 const cv::Mat & cameraMatrix1, const cv::Mat & distCoeffs1,
                 const cv::Mat & cameraMatrix2, const cv::Mat & distCoeffs2, 
                 const uint32_t method = FMAT_METHOD_OFLOW);
    bool compute(const cv::Mat & img1, const cv::Mat & img2, cv::Mat F = cv::Mat(),
                 vector<cv::Point2f> points1 = vector<cv::Point2f>(), 
                 vector<cv::Point2f> points2 = vector<cv::Point2f>(), 
                 const uint32_t method = FMAT_METHOD_OFLOW);
    
    void setHessianThresh(const uint32_t & hessianThresh) { m_hessianThresh = hessianThresh; }
    
    void toggleShowCommonRegion(const bool & showCommonRegion) { m_showCommonRegion = showCommonRegion; }
    void toggleShowIterations(const bool & showIterations) { m_showIterations = showIterations; }
    
    void getRectifiedImages(const cv::Mat & img1, const cv::Mat & img2, 
                            cv::Mat & rectified1, cv::Mat & rectified2, int interpolation = cv::INTER_CUBIC);
//     cv::Mat getRectifiedImage1() { return m_rectified1; }
//     cv::Mat getRectifiedImage2() { return m_rectified2; }
private:
    void determineCommonRegion(const vector<cv::Point2f> &epipoles, 
                               const cv::Size imgDimensions, const cv::Mat & F);
    void determineRhoRange(const cv::Point2d &epipole, const cv::Size imgDimensions,
                           const vector<cv::Point2f> &externalPoints, double & minRho, double & maxRho);
    bool findFundamentalMat(const cv::Mat & img1, const cv::Mat & img2, cv::Mat & F,
                            vector<cv::Point2f> points1, vector<cv::Point2f> points2, 
                            cv::Point2d & epipole1, cv::Point2d & epipole2, const uint32_t method = FMAT_METHOD_OFLOW);
    void findPairsSURF(const cv::Mat & img1, const cv::Mat & img2,
                        vector<cv::Point2f> & outPoints1, vector<cv::Point2f> & outPoints2);
    void findPairsOFlow(const cv::Mat & img1, const cv::Mat & img2,
                       vector<cv::Point2f> & outPoints1, vector<cv::Point2f> & outPoints2);
    void getEpipoles(const cv::Mat & F, cv::Point2d & epipole1, cv::Point2d & epipole2);
    void checkF(cv::Mat & F, cv::Point2d & epipole1, cv::Point2d & epipole2, const cv::Point2d & m, const cv::Point2d & m1);
    void getExternalPoints(const cv::Point2d &epipole, const cv::Size imgDimensions,
                           vector<cv::Point2f> &externalPoints);
    bool lineIntersectsSegment(const cv::Vec3d & line, const cv::Point2d & p1, const cv::Point2d & p2, cv::Point2d * intersection = NULL);
    bool lineIntersectsRect(const cv::Vec3d & line, const cv::Size & imgDimensions, cv::Point2d * intersection = NULL);
    bool isTheRightPoint(const cv::Point2d & epipole, const cv::Point2d & intersection, const cv::Vec3d & line,
                         const cv::Point2d * lastPoint);
    cv::Point2d getBorderIntersection(const cv::Point2d & epipole, const cv::Vec3d & line, const cv::Size & imgDimensions, 
                                      const cv::Point2d * lastPoint = NULL);        
    void getBorderIntersections(const cv::Point2d & epipole, const cv::Vec3d & line, const cv::Size & imgDimensions, 
                                vector<cv::Point2d> & intersections);
    cv::Point2d getNearestIntersection(const cv::Point2d & oldEpipole, const cv::Point2d & newEpipole, const cv::Vec3d & line, 
                                       const cv::Point2d & oldPoint, const cv::Size & imgDimensions);
    cv::Point2d getBorderIntersectionFromOutside(const cv::Point2d & epipole, const cv::Vec3d & line, const cv::Size & imgDimensions);
    void computeEpilines(const vector<cv::Point2f> & points, const uint32_t &whichImage, 
                        const cv::Mat & F, const vector <cv::Vec3f> & oldlines, vector <cv::Vec3f> & newLines);
    void getNewPointAndLineSingleImage(const cv::Point2d epipole1, const cv::Point2d epipole2, const cv::Size & imgDimensions, 
                                       const cv::Mat & F, const uint32_t & whichImage, const cv::Point2d & pOld1, const cv::Point2d & pOld2,
                                        cv::Vec3f & prevLine, cv::Point2d & pNew1, cv::Vec3f & newLine1, 
                                        cv::Point2d & pNew2, cv::Vec3f & newLine2);
    void getNewEpiline(const cv::Point2d epipole1, const cv::Point2d epipole2, const cv::Size & imgDimensions, 
                       const cv::Mat & F, const cv::Point2d pOld1, const cv::Point2d pOld2, 
                       cv::Vec3f prevLine1, cv::Vec3f prevLine2, 
                       cv::Point2d & pNew1, cv::Point2d & pNew2, cv::Vec3f & newLine1, cv::Vec3f & newLine2);
    void transformLine(const cv::Point2d& epipole, const cv::Point2d& p2, const cv::Mat& inputImage, 
                       const uint32_t & thetaIdx, const double &minRho, const double & maxRho, cv::Mat& mapX, cv::Mat& mapY);
    void doTransformation(const cv::Mat& img1, const cv::Mat& img2, const cv::Point2d epipole1, const cv::Point2d epipole2, const cv::Mat & F);
    
    // Visualization functions
    cv::Point2d image2World(const cv::Point2d & point, const cv::Size & imgDimensions);
    cv::Point2d getPointFromLineAndX(const double & x, const cv::Vec3f line);
    void showCommonRegion(const cv::Point2d epipole, const cv::Vec3f & line11, const cv::Vec3f & line12,
                          const cv::Vec3f & line13, const cv::Vec3f & line14, 
                          const cv::Vec3f & lineB, const cv::Vec3f & lineE, 
                          const cv::Point2d & b, const cv::Point2d & e, const cv::Size & imgDimensions, 
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
    
    cv::Mat m_mapX1, m_mapY1, m_mapX2, m_mapY2;
    cv::Mat m_rectified1, m_rectified2;
};

#endif // POLARCALIBRATION_H
