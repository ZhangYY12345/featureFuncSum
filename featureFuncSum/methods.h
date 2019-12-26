#pragma once
#include <opencv2/opencv.hpp>

void getOriginLBPFeature(cv::Mat src, cv::Mat& dst);
void getLBP_Circle(cv::Mat src, cv::Mat& dst, int radius, int neighborPts = 8);
void getLBP_RoatationInvariant(cv::Mat src, cv::Mat& dst);
int  hopCount(uchar i);
void getLBP_UniformPattern(cv::Mat src, cv::Mat& dst);
void getLBP_MultiscaleBlock(cv::Mat src, cv::Mat& dst, int scale);
void getLBP_StatisticallyEffectiveMB(cv::Mat src, cv::Mat& dst, int scale);

void getLBPH(cv::Mat src, cv::Mat& dst, int patternNum, int gridX = 8, int gridY = 8, bool normed = true);
void getLocalRegionLBPH(cv::Mat src, cv::Mat&dst, int minValue, int maxValue, bool normed);



bool biggerSort(std::vector<cv::Point> v1, std::vector<cv::Point> v2);
void contourMatch(cv::Mat src1, cv::Mat src2, cv::Mat& dst);
void drawHist(cv::Mat& image, double& maxValueBin, double& minValueBin, int& maxLocation, int& minLocation, cv::Mat& imgHist);

void featureDetectSURF_FLANN(cv::Mat srcObj, cv::Mat srcScene, cv::Mat& transformMatrix);
void featureDetectSIFT_FLANN(cv::Mat srcObj, cv::Mat srcScene, cv::Mat& transformMatrix);
void featureDetectORB_BF(cv::Mat srcObj, cv::Mat srcScene, cv::Mat& transformMatrix);
void featureDetectFAST_BF(cv::Mat srcObj, cv::Mat srcScene, cv::Mat& transformMatrix);

void colorIndexGrey_ExR(cv::Mat src, cv::Mat& dst);
void colorIndexGrey_singleChannel(cv::Mat src, cv::Mat& dst);
void featureDetectORB_BF_ColorIndex(cv::Mat srcObj, cv::Mat srcScene, cv::Mat& transformMatrix);

void getCascadeTransRes(cv::Mat src, cv::Mat& dst, cv::Mat& transformMatrix, std::vector<cv::Point2f> cornersObj);
void getTransRes(cv::Mat src, cv::Mat& dst, cv::Mat& transformMatrix, std::vector<cv::Point2f> cornersObj);
void getCascadeTransRes_Narrow(cv::Mat src, cv::Mat& dst, cv::Mat& transformMatrix, std::vector<cv::Point2f> cornersObj, float scaleValue);
void getTransRes_Narrow(cv::Mat src, cv::Mat& dst, cv::Mat& transformMatrix, std::vector<cv::Point2f> cornersObj, float scaleValue);

void pointTransform(cv::Point2f& pts1, float value);