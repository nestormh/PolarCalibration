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

cv::Point2d PolarCalibration::image2World(const cv::Point2d & point, const cv::Size & imgDimensions) {
    return cv::Point(point.x * 0.5 + imgDimensions.width / 4.0, /*imgDimensions.height -*/ (point.y * 0.5  + imgDimensions.height / 4.0));
}

cv::Point2d PolarCalibration::getPointFromLineAndX(const double & x, const cv::Vec3f line) {
    return cv::Point2d(x, -(line[0] * x + line[2]) / line[1]);
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
                                        const cv::Point2d & b, const cv::Point2d & e, const cv::Size & imgDimensions,
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
    cv::circle(img, image2World(e, imgDimensions), 10, cv::Scalar(128, 255, 128), -1);

    cv::namedWindow(windowName.c_str());
    cv::imshow(windowName.c_str(), img);

    //     cv::waitKey(0);
}
