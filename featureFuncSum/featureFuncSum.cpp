// featureFuncSum.cpp: 定义控制台应用程序的入口点。
//
#include "methods.h"

using namespace cv;
using namespace std;

int main()
{
	Mat obj_full_mat = imread("D:/studying/PCBCheck/PCBCheckWithQt/CoatingCheck-color_index_based approaches/testingData/2.jpg");
	Mat obj = obj_full_mat(Range(140, 3553), Range(800, 4997));
	Mat srcScene = imread("D:/studying/PCBCheck/PCBCheckWithQt/CoatingCheck-color_index_based approaches/testingData/4-2.jpg");

	//Mat imgLBPFeature;
	//getOriginLBPFeature(obj_full_mat, imgLBPFeature);
	//imwrite("lbp.jpg", imgLBPFeature);

	Mat transformMatrix;

	featureDetectSIFT_FLANN(obj, srcScene, transformMatrix);
	//featureDetectSURF_FLANN(obj, srcScene, transformMatrix);
	//featureDetectORB_BF(obj, srcScene, transformMatrix);
	//featureDetectFAST_BF(obj, srcScene, transformMatrix);
	//featureDetectORB_BF_ColorIndex(obj, srcScene, transformMatrix);

	if (!transformMatrix.empty())
	{
		std::vector<Point2f> obj_corners(4);
		//obj_corners[0] = cvPoint(800.0, 140.0);
		//obj_corners[1] = cvPoint(4997.0, 140.0);
		//obj_corners[2] = cvPoint(4997.0, 3553.0);
		//obj_corners[3] = cvPoint(800.0, 3553.0);
		obj_corners[0] = cvPoint(0.0, 0.0);
		obj_corners[1] = cvPoint(4196.0, 0.0);
		obj_corners[2] = cvPoint(4196.0, 3412.0);
		obj_corners[3] = cvPoint(0.0, 3412.0);

		Rect rect = boundingRect(obj_corners);
		Mat originImg = obj(rect);
		imwrite("originalImage.jpg", originImg);

		Mat sceneROI;
		getCascadeTransRes(srcScene, sceneROI, transformMatrix, obj_corners);

		Mat sceneROI2;
		getTransRes(srcScene, sceneROI2, transformMatrix, obj_corners);

		//Mat sceneROI;
		//getCascadeTransRes_Narrow(srcScene, sceneROI, transformMatrix, obj_corners, 0.25);

		//Mat sceneROI2;
		//getTransRes_Narrow(srcScene, sceneROI2, transformMatrix, obj_corners, 0.25);
	}
	else
	{
		cout << "cannot find the carrier" << endl;
	}
    return 0;
}

