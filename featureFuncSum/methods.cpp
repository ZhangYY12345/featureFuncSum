#include "methods.h"
#include "nonfree.hpp"


using namespace cv;

/**
 * \brief  output the LBP feature image of the input src image
 * \param src :the input image should be BGR-color image
 * \param dst :output feature image
 */
void getOriginLBPFeature(Mat src, Mat& dst)
{
	if (src.channels() != 1)
	{
		cvtColor(src, src, CV_BGR2GRAY);
	}
	Mat res(src.rows - 2, src.cols - 2, src.type());
	res.setTo(0);
	dst = res.clone();

	for (int i = 1; i<src.rows - 1; i++)
	{
		for (int j = 1; j<src.cols - 1; j++)
		{
			uchar center = src.at<uchar>(i, j);
			unsigned char lbpCode = 0;
			lbpCode |= (src.at<uchar>(i - 1, j - 1) >= center) << 7;
			lbpCode |= (src.at<uchar>(i - 1, j) >= center) << 6;
			lbpCode |= (src.at<uchar>(i - 1, j + 1) >= center) << 5;
			lbpCode |= (src.at<uchar>(i, j + 1) >= center) << 4;
			lbpCode |= (src.at<uchar>(i + 1, j + 1) >= center) << 3;
			lbpCode |= (src.at<uchar>(i + 1, j) >= center) << 2;
			lbpCode |= (src.at<uchar>(i + 1, j - 1) >= center) << 1;
			lbpCode |= (src.at<uchar>(i, j - 1) >= center) << 0;
			dst.at<uchar>(i - 1, j - 1) = lbpCode;
		}
	}
}

/**
 * \brief  circular LBP (has some propblems on efficiency)
 * \param src 
 * \param dst 
 * \param radius 
 * \param neigborPts 
 */
void getLBP_Circle(cv::Mat src, cv::Mat& dst, int radius, int neighborPts) 
{
	if (src.channels() != 1)
	{
		cvtColor(src, src, CV_BGR2GRAY);
	}

	Mat res(src.rows - radius * 2, src.cols - radius * 2, src.type());
	res.setTo(0);

	for (int n = 0; n < neighborPts; n++)
	{
		float x = static_cast<float>(radius * cos(2.0 * CV_PI * n / static_cast<float>(neighborPts)));
		float y = static_cast<float>(-radius * sin(2.0 * CV_PI * n / static_cast<float>(neighborPts)));

		int fx = static_cast<int>(floor(x));  //floor()向下取整
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x)); //ceil()向上取整
		int cy = static_cast<int>(ceil(y));

		float tx = x - fx; //将坐标映射到0-1之间
		float ty = y - fy;

		float w1 = (1 - tx) * (1 - ty);//计算插值权重
		float w2 = tx * (1 - ty);
		float w3 = (1 - tx) * ty;
		float w4 = tx * ty;

		for(int i = radius; i < src.rows - radius; i++)
		{
			for(int j = radius; j < src.cols - radius; j++)
			{
				float t = static_cast<float>(w1 * src.at<uchar>(i + fy, j + fx) + w2 * src.at<uchar>(i + fy, j + cx)
											+ w3 * src.at<uchar>(i + cy, j + fx) + w4 * src.at<uchar>(i + cy, j + cx));
				res.at<uchar>(i - radius, j - radius) += ((t > src.at<uchar>(i, j))
					|| (abs(t - src.at<uchar>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
	dst = res.clone();
}

void getLBP_RoatationInvariant(cv::Mat src, cv::Mat& dst)
{
	if (src.channels() != 1)
	{
		cvtColor(src, src, CV_BGR2GRAY);
	}

	uchar RITable[256];
	int temp, val;
	Mat res(src.rows - 2, src.cols - 2, src.type());
	res.setTo(0);

	for(int i = 0; i < 256; i++)
	{
		val = i;
		for(int j = 0; j < 7; j++)
		{
			temp = i >> 1;
			if (val > temp)
				val = temp;
		}
		RITable[i] = val;
	}

	for (int i = 1; i < src.rows - 1; i++)
	{
		for(int j = 1; j < src.cols - 1; j++)
		{
			uchar center = src.at<uchar>(i, j);
			uchar code = 0;
			code |= (src.at<uchar>(i - 1, j - 1) >= center) << 7;
			code |= (src.at<uchar>(i - 1, j) >= center) << 6;
			code |= (src.at<uchar>(i - 1, j + 1) >= center) << 5;
			code |= (src.at<uchar>(i, j + 1) >= center) << 4;
			code |= (src.at<uchar>(i + 1, j + 1) >= center) << 3;
			code |= (src.at<uchar>(i + 1, j) >= center) << 2;
			code |= (src.at<uchar>(i + 1, j - 1) >= center) << 1;
			code |= (src.at<uchar>(i, j - 1) >= center) << 0;
			res.at<uchar>(i - 1, j - 1) = RITable[code];
		}
	}
	dst = res.clone();
}

int hopCount(uchar i)
{
	uchar a[8] = { 0 };
	int cnt = 0;
	int k = 7;
	while(k)
	{
		a[k] = i & 1;
		i = i >> 1;
		k--;
	}

	for(int j = 0; j < 7; j++)
	{
		if (a[j] != a[j + 1])
			cnt++;
	}
	if(a[0] != a[7])
	{
		cnt++;
	}
	return cnt;
}

void getLBP_UniformPattern(cv::Mat src, cv::Mat& dst)
{
	uchar UPTable[256];
	memset(UPTable, 0, 256 * sizeof(uchar));
	uchar temp = 1;
	for(int i = 0; i < 256; i++)
	{
		if(hopCount(i) <= 2)
		{
			UPTable[i] = temp;
			temp++;
		}
	}

	if (src.channels() != 1)
	{
		cvtColor(src, src, CV_BGR2GRAY);
	}

	Mat res(src.rows - 2, src.cols - 2, src.type());
	res.setTo(0);

	for(int i = 1; i < src.rows - 1; i ++)
	{
		for(int j = 1; j < src.cols - 1; j++)
		{
			uchar center = src.at<uchar>(i, j);
			uchar code = 0;
			code |= (src.at<uchar>(i - 1, j - 1) >= center) << 7;
			code |= (src.at<uchar>(i - 1, j) >= center) << 6;
			code |= (src.at<uchar>(i - 1, j + 1) >= center) << 5;
			code |= (src.at<uchar>(i, j + 1) >= center) << 4;
			code |= (src.at<uchar>(i + 1, j + 1) >= center) << 3;
			code |= (src.at<uchar>(i + 1, j) >= center) << 2;
			code |= (src.at<uchar>(i + 1, j - 1) >= center) << 1;
			code |= (src.at<uchar>(i, j - 1) >= center) << 0;
			res.at<uchar>(i - 1, j - 1) = UPTable[code];
		}
	}
	dst = res.clone();
}

void getLBP_MultiscaleBlock(cv::Mat src, cv::Mat& dst, int scale)
{
	int cellSize = scale / 3; 
	int offset = cellSize / 2;

	if (src.channels() != 1)
	{
		cvtColor(src, src, CV_BGR2GRAY);
	}

	Mat cellImg(src.rows - 2 * offset, src.cols - 2 * offset, src.type()); 
	for(int i = offset; i < src.rows - offset; i++)
	{
		for(int j = offset; j < src.cols - offset; j++)
		{
			int temp = 0;
			for(int m = -offset; m < offset + 1; m++)
			{
				for(int n = -offset; n < offset + 1; n++)
				{
					temp += src.at<uchar>(i + n, j + m);
				}
			}
			temp /= (cellSize * cellSize);
			cellImg.at<uchar>(i - cellSize / 2, j - cellSize / 2) = uchar(temp);
		}
	}
	getOriginLBPFeature(cellImg, dst);
}

void getLBP_StatisticallyEffectiveMB(cv::Mat src, cv::Mat& dst, int scale)
{
	Mat imgMBLBP;
	getLBP_MultiscaleBlock(src, imgMBLBP, scale);

	Mat histImg;
	int histSize = 256;
	float range[] = { float(0), float(255) };
	const float* ranges = { range };
	calcHist(&imgMBLBP, 1, 0, Mat(), histImg, 1, &histSize, &ranges, true, false);
	histImg.reshape(1, 1);
	std::vector<float> histVector(histImg.rows * histImg.cols);
	uchar table[256];
	memset(table, 64, 256);
	if(histImg.isContinuous())
	{
		histVector.assign((float*)histImg.datastart, (float*)histImg.dataend); //将直方图histImg变为vector向量histVector
		std::vector<float> histVectorCopy(histVector);
		sort(histVector.begin(), histVector.end(), std::greater<float>()); //对LBP特征值的数量进行排序，降序排序
		for(int i = 0; i < 63; i++)
		{
			for(int j = 0; j < histVectorCopy.size(); j++)
			{
				if(histVectorCopy[j] == histVector[i])
				{
					table[j] = i;
				}
			}
		}
	}

	dst = imgMBLBP;
	for(int i = 0; i < dst.rows; i++)
	{
		for(int j = 0; j < dst.cols; j++)
		{
			dst.at<uchar>(i, j) = table[dst.at<uchar>(i, j)];
		}
	}
}

/**
 * \brief 
 * \param src 
 * \param dst 
 * \param patternNum :LBP值的模式种类
 * \param gridX 
 * \param gridY ：表示将图像分割成 gridX * gridY 块
 * \param normed 
 */
void getLBPH(cv::Mat src, cv::Mat& dst, int patternNum, int gridX, int gridY, bool normed)
{
	int width = src.cols / gridX;
	int height = src.rows / gridY;

	dst = Mat::zeros(gridX * gridY, patternNum, CV_32FC1);
	if(src.empty())
	{
		dst.reshape(1, 1);
		return;
	}

	int dstRowIndex = 0;
	for(int i = 0; i < gridX; i++)
	{
		for(int j = 0; j < gridY; j++)
		{
			Mat srcCell = Mat(src, Range(i*height, (i + 1)*height), Range(j*width, (j + 1)*width));
			Mat histCell;
			getLocalRegionLBPH(srcCell, histCell, 0, (patternNum - 1), true);
			Mat rowRes = dst.row(dstRowIndex);
			histCell.reshape(1, 1).convertTo(rowRes, CV_32FC1);
			dstRowIndex++;
		}
	}
	dst.reshape(1, 1);
}

void getLocalRegionLBPH(cv::Mat src, cv::Mat& dst, int minValue, int maxValue, bool normed)
{
	int histSize = maxValue - minValue + 1;
	float range[] = { static_cast<float>(minValue), static_cast<float>(maxValue + 1) };
	const float* ranges = { range };
	calcHist(&src, 1, 0, Mat(), dst, 1, &histSize, &ranges, true, false);

	if(normed)
	{
		dst /= (int)src.total();
	}

	dst.reshape(1, 1);
}

bool biggerSort(std::vector<Point> v1, std::vector<Point> v2)
{
	return contourArea(v1) > contourArea(v2);
}

void contourMatch(cv::Mat src1, cv::Mat src2, cv::Mat& dst)
{
	//Mat edge1, edge2;
	//Canny(src1, edge1, 3, 9);
	//Canny(src2, edge2, 3, 9);
	//imwrite("edge1.jpg", edge1);
	//imwrite("edge2.jpg", edge2);
	
	//Mat closerRect = getStructuringElement(MORPH_RECT, Size(3, 3));
	//morphologyEx(edge1, edge1, MORPH_OPEN, closerRect);

	//std::vector<std::vector<Point>> contours1, contours2;
	//findContours(edge1, contours1, CV_RETR_TREE, CHAIN_APPROX_NONE);
	//findContours(edge2, contours2, CV_RETR_TREE, CHAIN_APPROX_NONE);
	//sort(contours1.be gin(), contours1.end(), biggerSort);
	//sort(contours2.begin(), contours2.end(), biggerSort);

	//Mat contourImg = Mat::zeros(src1.size(), CV_8UC1);
	//drawContours(contourImg, contours1, 0, 255, CV_FILLED);
	//imwrite("edge1Bin.jpg", contourImg);
}

void drawHist(cv::Mat& image, double& maxValueBin, double& minValueBin, int& maxLocation, int& minLocation,
	cv::Mat& imgHist)
{
	int nimages = 1;
	int channel = 0;
	Mat outputHist;  //列向量width:1 height:256
	int dims = 1;
	int histSize = 256;
	float hranges[] = { 0, 255 };
	const float *ranges[] = { hranges };
	bool uni = false;
	bool accum = false;

	cv::calcHist(&image, nimages, &channel, Mat(), outputHist, dims, &histSize, ranges);

	Point maxLocal;
	Point minLocal;

	cv::minMaxLoc(outputHist, &minValueBin, &maxValueBin, &minLocal, &maxLocal);
	maxLocation = maxLocal.y;
	minLocation = minLocal.y;

	int scale = 1;
	double rate = (histSize / maxValueBin) * 0.9;
	Mat histPic(histSize * scale, histSize, CV_8U, Scalar(255));

	for (int i = 0; i < histSize; i++)
	{
		float value = outputHist.at<float>(i);
		line(histPic, Point(i*scale, histSize), Point(i*scale, histSize - value * rate), Scalar(0));
	}
	imgHist = histPic.clone();

}

void featureDetectSURF_FLANN(cv::Mat srcObj, cv::Mat srcScene, cv::Mat& transformMatrix)
{
	Mat srcObjGrey, srcSceneGrey;
	cvtColor(srcObj, srcObjGrey, CV_BGR2GRAY);
	cvtColor(srcScene, srcSceneGrey, CV_BGR2GRAY);

	resize(srcObjGrey, srcObjGrey , Size(srcObjGrey.cols * 0.25, srcObjGrey.rows * 0.25));
	resize(srcSceneGrey, srcSceneGrey, Size(srcSceneGrey.cols * 0.25, srcSceneGrey.rows * 0.25));

	//检测和匹配SURF关键点
	int minHessian = 300;
	Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(minHessian);
	Ptr<DescriptorExtractor> descriptor = xfeatures2d::SURF::create();
	Ptr<DescriptorMatcher> matcher1 = DescriptorMatcher::create("FlannBased");

	std::vector<KeyPoint> keyPoint1, keyPoint2;
	Mat descriptors1, descriptors2;
	std::vector<DMatch> matches;

	detector->detect(srcObjGrey, keyPoint1);
	detector->detect(srcSceneGrey, keyPoint2);

	//[4]提取特征描述子
	descriptor->compute(srcObjGrey, keyPoint1, descriptors1);
	descriptor->compute(srcSceneGrey, keyPoint2, descriptors2);

	//[5]匹配图像中的特征点描述子
	matcher1->match(descriptors1, descriptors2, matches);

	//Mat img_keyPoint1, img_keyPoint2;
	//drawKeypoints(srcObjGrey, keyPoint1, img_keyPoint1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//imwrite("img1_keypoints_SURF.jpg", img_keyPoint1);
	//drawKeypoints(srcSceneGrey, keyPoint2, img_keyPoint2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//imwrite("img2_keypoints_SURF.jpg", img_keyPoint2);

	double max_dist = 0;
	double min_dist = 100;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	std::vector<DMatch> goodMatches;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		if (matches[i].distance <= max(min_dist, 10.0))
		{
			goodMatches.push_back(matches[i]);
		}
	}
	Mat imgMatches;
	drawMatches(srcObjGrey, keyPoint1, srcSceneGrey, keyPoint2, goodMatches, imgMatches);
	imwrite("matches_SURF.jpg", imgMatches);

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (unsigned int i = 0; i < goodMatches.size(); i++)
	{
		obj.push_back(keyPoint1[goodMatches[i].queryIdx].pt);
		scene.push_back(keyPoint2[goodMatches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, CV_RANSAC);
  	transformMatrix = H;
}

void featureDetectSIFT_FLANN(cv::Mat srcObj, cv::Mat srcScene, cv::Mat& transformMatrix)
{
	Mat srcObjGrey, srcSceneGrey;
	cvtColor(srcObj, srcObjGrey, CV_BGR2GRAY);
	cvtColor(srcScene, srcSceneGrey, CV_BGR2GRAY);

	resize(srcObjGrey, srcObjGrey, Size(srcObjGrey.cols * 0.25, srcObjGrey.rows * 0.25));
	resize(srcSceneGrey, srcSceneGrey, Size(srcSceneGrey.cols * 0.25, srcSceneGrey.rows * 0.25));

	//检测和匹配SIFT关键点
	int minHessian = 400;
	Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create(minHessian);
	Ptr<DescriptorExtractor> descriptor = xfeatures2d::SIFT::create();
	Ptr<DescriptorMatcher> matcher1 = DescriptorMatcher::create("FlannBased");

	std::vector<KeyPoint> keyPoint1, keyPoint2;
	Mat descriptors1, descriptors2;
	std::vector<DMatch> matches;

	detector->detect(srcObjGrey, keyPoint1);
	detector->detect(srcSceneGrey, keyPoint2);

	//[4]提取特征描述子
	descriptor->compute(srcObjGrey, keyPoint1, descriptors1);
	descriptor->compute(srcSceneGrey, keyPoint2, descriptors2);

	//[5]匹配图像中的特征点描述子
	matcher1->match(descriptors1, descriptors2, matches);

	double max_dist = 0;
	double min_dist = 100;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	std::vector<DMatch> goodMatches;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		if (matches[i].distance <= max(3 * min_dist, 40.0))
		{
			goodMatches.push_back(matches[i]);
		}
	}
	Mat imgMatches;
	drawMatches(srcObjGrey, keyPoint1, srcSceneGrey, keyPoint2, goodMatches, imgMatches);
	imwrite("matches_SIFT.jpg", imgMatches);

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (unsigned int i = 0; i < goodMatches.size(); i++)
	{
		obj.push_back(keyPoint1[goodMatches[i].queryIdx].pt);
		scene.push_back(keyPoint2[goodMatches[i].trainIdx].pt);
	}

	transformMatrix = findHomography(obj, scene, CV_RANSAC);
}

void featureDetectORB_BF(cv::Mat srcObj, cv::Mat srcScene, cv::Mat& transformMatrix)
{
	Mat srcObjGrey, srcSceneGrey;
	cvtColor(srcObj, srcObjGrey, CV_BGR2GRAY);
	cvtColor(srcScene, srcSceneGrey, CV_BGR2GRAY);

	//【2】检测和匹配SIFT关键点
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	Ptr<DescriptorMatcher> matcher1 = DescriptorMatcher::create("BruteForce-Hamming");

	std::vector<KeyPoint> keyPoint1, keyPoint2;
	Mat descriptors1, descriptors2;
	std::vector<DMatch> matches;

	detector->detect(srcObjGrey, keyPoint1);
	detector->detect(srcSceneGrey, keyPoint2);

	//[4]提取特征描述子
	descriptor->compute(srcObjGrey, keyPoint1, descriptors1);
	descriptor->compute(srcSceneGrey, keyPoint2, descriptors2);

	//[5]匹配图像中的特征点描述子
	matcher1->match(descriptors1, descriptors2, matches);

	double max_dist = 0;
	double min_dist = 100;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	std::vector<DMatch> goodMatches;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 30.0))
		{
			goodMatches.push_back(matches[i]);
		}
	}
	Mat imgMatches;
	drawMatches(srcObjGrey, keyPoint1, srcSceneGrey, keyPoint2, goodMatches, imgMatches);
	imwrite("matches_SIFT.jpg", imgMatches);

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (unsigned int i = 0; i < goodMatches.size(); i++)
	{
		obj.push_back(keyPoint1[goodMatches[i].queryIdx].pt);
		scene.push_back(keyPoint2[goodMatches[i].trainIdx].pt);
	}

	transformMatrix = findHomography(obj, scene, CV_RANSAC);
}

void featureDetectFAST_BF(cv::Mat srcObj, cv::Mat srcScene, cv::Mat& transformMatrix)
{
	Mat srcObjGrey, srcSceneGrey;
	cvtColor(srcObj, srcObjGrey, CV_BGR2GRAY);
	cvtColor(srcScene, srcSceneGrey, CV_BGR2GRAY);

	std::vector<Mat> objChannelBGR;
	split(srcObj, objChannelBGR);
	std::vector<Mat> sceneChannelBGR;
	split(srcScene, sceneChannelBGR);


	//FAST关键点,ORB特征描述，FLANN匹配
	int minHessian = 100;
	Ptr<FeatureDetector> detector = FastFeatureDetector::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	Ptr<DescriptorMatcher> matcher1 = DescriptorMatcher::create("BruteForce-Hamming");

	std::vector<KeyPoint> keyPoint1, keyPoint2;
	Mat descriptors1, descriptors2;
	std::vector<DMatch> matches;

	detector->detect(objChannelBGR[2], keyPoint1);
	detector->detect(sceneChannelBGR[2], keyPoint2);

	//[4]提取特征描述子
	descriptor->compute(srcObjGrey, keyPoint1, descriptors1);
	descriptor->compute(srcSceneGrey, keyPoint2, descriptors2);

	//[5]匹配图像中的特征点描述子
	matcher1->match(descriptors1, descriptors2, matches);

	double max_dist = 0;
	double min_dist = 100;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	std::vector<DMatch> goodMatches;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		if (matches[i].distance <= 2 * min_dist)
		{
			goodMatches.push_back(matches[i]);
		}
	}
	Mat imgMatches;
	drawMatches(srcObjGrey, keyPoint1, srcSceneGrey, keyPoint2, goodMatches, imgMatches);
	imwrite("matches_SIFT.jpg", imgMatches);

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (unsigned int i = 0; i < goodMatches.size(); i++)
	{
		obj.push_back(keyPoint1[goodMatches[i].queryIdx].pt);
		scene.push_back(keyPoint2[goodMatches[i].trainIdx].pt);
	}

	transformMatrix = findHomography(obj, scene, CV_RANSAC);
}

void colorIndexGrey_ExR(cv::Mat src, cv::Mat& dst)
{
	std::vector<Mat> channelsBGR;
	split(src, channelsBGR);

	Mat b = channelsBGR[0].clone();
	b.convertTo(b, CV_32FC1);
	Mat g = channelsBGR[1].clone();
	g.convertTo(g, CV_32FC1);
	Mat r = channelsBGR[2].clone();
	r.convertTo(r, CV_32FC1);

	Mat color1p3r_b = 1.3 * r - b;
	Mat colorBB = color1p3r_b;
	imwrite("ExR_color_1.3R-B.jpg", colorBB);
	dst = colorBB.clone();
	dst.convertTo(dst, CV_8UC1);
}

void colorIndexGrey_singleChannel(cv::Mat src, cv::Mat& dst)
{
	std::vector<Mat> channelsBGR;
	split(src, channelsBGR);

	dst = channelsBGR[0].clone(); //b
	//dst = channelsBGR[1].clone(); //g
	//dst = channelsBGR[2].clone(); //r
}

void featureDetectORB_BF_ColorIndex(cv::Mat srcObj, cv::Mat srcScene, cv::Mat& transformMatrix)
{
	Mat srcObjGrey, srcSceneGrey;

	//colorIndexGrey_ExR(srcObj, srcObjGrey);
	//colorIndexGrey_ExR(srcScene, srcSceneGrey);

	//colorIndexGrey_singleChannel(srcObj, srcObjGrey);
	//colorIndexGrey_singleChannel(srcScene, srcSceneGrey);

	//getOriginLBPFeature(srcObj, srcObjGrey);
	//getOriginLBPFeature(srcScene, srcSceneGrey);

	//getLBP_Circle(srcObj, srcObjGrey, 5);
	//getLBP_Circle(srcScene, srcSceneGrey, 5);

	//getLBP_UniformPattern(srcObj, srcObjGrey);
	//getLBP_UniformPattern(srcScene, srcSceneGrey);

	//getLBP_MultiscaleBlock(srcObj, srcObjGrey, 3);
	//getLBP_MultiscaleBlock(srcScene, srcSceneGrey, 3);

	getLBP_StatisticallyEffectiveMB(srcObj, srcObjGrey, 3); 
	Mat lbpH;
	getLBPH(srcObjGrey, lbpH, 64);
	getLBP_StatisticallyEffectiveMB(srcScene, srcSceneGrey, 3);

	//【2】检测和匹配SIFT关键点
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	Ptr<DescriptorMatcher> matcher1 = DescriptorMatcher::create("BruteForce-Hamming");

	std::vector<KeyPoint> keyPoint1, keyPoint2;
	Mat descriptors1, descriptors2;
	std::vector<DMatch> matches;

	detector->detect(srcObjGrey, keyPoint1);
	detector->detect(srcSceneGrey, keyPoint2);

	//[4]提取特征描述子
	descriptor->compute(srcObjGrey, keyPoint1, descriptors1);
	descriptor->compute(srcSceneGrey, keyPoint2, descriptors2);

	//[5]匹配图像中的特征点描述子
  	matcher1->match(descriptors1, descriptors2, matches);

	double max_dist = 0;
	double min_dist = 100;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	std::vector<DMatch> goodMatches;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 30.0))
		{
			goodMatches.push_back(matches[i]);
		}
	}
	Mat imgMatches;
	drawMatches(srcObjGrey, keyPoint1, srcSceneGrey, keyPoint2, goodMatches, imgMatches);
	imwrite("matches_SIFT.jpg", imgMatches);

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (unsigned int i = 0; i < goodMatches.size(); i++)
	{
		obj.push_back(keyPoint1[goodMatches[i].queryIdx].pt);
		scene.push_back(keyPoint2[goodMatches[i].trainIdx].pt);
	}

	transformMatrix = findHomography(obj, scene, CV_RANSAC);

}

void getCascadeTransRes(cv::Mat src, cv::Mat& dst, cv::Mat& transformMatrix, std::vector<cv::Point2f> cornersObj)
{
	if (cornersObj.size() == 4)
	{
		Rect rect = boundingRect(cornersObj);

		std::vector<Point2f> scene_corners(4);
		perspectiveTransform(cornersObj, scene_corners, transformMatrix);

		Rect sceneRect = boundingRect(scene_corners);
		if (sceneRect.width > rect.width * 0.75 && sceneRect.height > rect.height * 0.75)
		{
			Mat mask = Mat::zeros(src.size(), CV_8UC1);
			std::vector<std::vector<Point>> contoursMask;
			std::vector<Point> obj_corners2(4);
			obj_corners2[0] = scene_corners[0];
			obj_corners2[1] = scene_corners[1];
			obj_corners2[2] = scene_corners[2];
			obj_corners2[3] = scene_corners[3];
			contoursMask.push_back(obj_corners2);

			drawContours(mask, contoursMask, -1, 1, CV_FILLED);
			Mat targetImg;
			src.copyTo(targetImg, mask);
			targetImg = targetImg(boundingRect(obj_corners2)).clone();
			imwrite("irregular_quadrilateral.jpg", targetImg);

			std::vector<Point2f> dstSize(4);
			dstSize[0] = cvPoint(0, 0);
			dstSize[1] = cvPoint(rect.width - 1, 0);
			dstSize[2] = cvPoint(rect.width - 1, rect.height - 1);
			dstSize[3] = cvPoint(0, rect.height - 1);

			Rect areaRect = boundingRect(obj_corners2);
			scene_corners[0] = cvPoint(obj_corners2[0].x - areaRect.x, obj_corners2[0].y - areaRect.y);
			scene_corners[1] = cvPoint(obj_corners2[1].x - areaRect.x, obj_corners2[1].y - areaRect.y);
			scene_corners[2] = cvPoint(obj_corners2[2].x - areaRect.x, obj_corners2[2].y - areaRect.y);
			scene_corners[3] = cvPoint(obj_corners2[3].x - areaRect.x, obj_corners2[3].y - areaRect.y);

			Mat trans = getPerspectiveTransform(scene_corners, dstSize);

			warpPerspective(targetImg, dst, trans, Size(rect.width, rect.height));
			imwrite("cascadeTrans_carrierRegion.jpg", dst);
		}
		else
		{
			std::cout << "Cannot find the carrier" << std::endl;
		}
	}
}

void getTransRes(cv::Mat src, cv::Mat& dst, cv::Mat& transformMatrix, std::vector<cv::Point2f> cornersObj)
{
	if (cornersObj.size() == 4)
	{
		Rect rect = boundingRect(cornersObj);

		std::vector<Point2f> scene_corners(4);
		perspectiveTransform(cornersObj, scene_corners, transformMatrix);

		Rect sceneRect = boundingRect(scene_corners);
		if (sceneRect.width > rect.width * 0.75 && sceneRect.height > rect.height * 0.75)
		{
			int centerX = (scene_corners[0].x + scene_corners[1].x + scene_corners[2].x + scene_corners[3].x) / 4;
			int centerY = (scene_corners[0].y + scene_corners[1].y + scene_corners[2].y + scene_corners[3].y) / 4;
			int pointTPX = centerX - rect.width / 2;
			int pointTPY = centerY - rect.height / 2;

			dst = src(Rect(pointTPX, pointTPY, rect.width, rect.height));

			//dst = src(Rect(scene_corners[0].x, scene_corners[0].y, rect.width, rect.height));
			imwrite("trans_carrierRegion.jpg", dst);
		}
		else
		{
			std::cout << "Cannot find the carrier" << std::endl;
		}
	}
}

void getCascadeTransRes_Narrow(cv::Mat src, cv::Mat& dst, cv::Mat& transformMatrix, std::vector<cv::Point2f> cornersObj, float scaleValue)
{
	if (cornersObj.size() == 4)
	{
		Rect rect = boundingRect(cornersObj);

		pointTransform(cornersObj[0], scaleValue);
		pointTransform(cornersObj[1], scaleValue);
		pointTransform(cornersObj[2], scaleValue);
		pointTransform(cornersObj[3], scaleValue);

		std::vector<Point2f> scene_corners(4);

		perspectiveTransform(cornersObj, scene_corners, transformMatrix);
		float invScale = 1 / scaleValue;
		pointTransform(scene_corners[0], invScale);
		pointTransform(scene_corners[1], invScale);
		pointTransform(scene_corners[2], invScale);
		pointTransform(scene_corners[3], invScale);

		Rect sceneRect = boundingRect(scene_corners);
		if (sceneRect.width > rect.width * 0.75 && sceneRect.height > rect.height * 0.75)
		{
			Mat mask = Mat::zeros(src.size(), CV_8UC1);
			std::vector<std::vector<Point>> contoursMask;
			std::vector<Point> obj_corners2(4);
			obj_corners2[0] = scene_corners[0];
			obj_corners2[1] = scene_corners[1];
			obj_corners2[2] = scene_corners[2];
			obj_corners2[3] = scene_corners[3];
			contoursMask.push_back(obj_corners2);

			drawContours(mask, contoursMask, -1, 1, CV_FILLED);
			Mat targetImg;
			src.copyTo(targetImg, mask);
			targetImg = targetImg(boundingRect(obj_corners2)).clone();
			imwrite("irregular_quadrilateral.jpg", targetImg);

			std::vector<Point2f> dstSize(4);
			dstSize[0] = cvPoint(0, 0);
			dstSize[1] = cvPoint(rect.width - 1, 0);
			dstSize[2] = cvPoint(rect.width - 1, rect.height - 1);
			dstSize[3] = cvPoint(0, rect.height - 1);

			Rect areaRect = boundingRect(obj_corners2);
			scene_corners[0] = cvPoint(obj_corners2[0].x - areaRect.x, obj_corners2[0].y - areaRect.y);
			scene_corners[1] = cvPoint(obj_corners2[1].x - areaRect.x, obj_corners2[1].y - areaRect.y);
			scene_corners[2] = cvPoint(obj_corners2[2].x - areaRect.x, obj_corners2[2].y - areaRect.y);
			scene_corners[3] = cvPoint(obj_corners2[3].x - areaRect.x, obj_corners2[3].y - areaRect.y);

			Mat trans = getPerspectiveTransform(scene_corners, dstSize);

			warpPerspective(targetImg, dst, trans, Size(rect.width, rect.height));
			imwrite("cascadeTrans_narrow_carrierRegion.jpg", dst);
		}
		else
		{
			std::cout << "Cannot find the carrier" << std::endl;
		}
	}
}

void getTransRes_Narrow(cv::Mat src, cv::Mat& dst, cv::Mat& transformMatrix, std::vector<cv::Point2f> cornersObj, float scaleValue)
{
	if (cornersObj.size() == 4)
	{
		Rect rect = boundingRect(cornersObj);

		pointTransform(cornersObj[0], scaleValue);
		pointTransform(cornersObj[1], scaleValue);
		pointTransform(cornersObj[2], scaleValue);
		pointTransform(cornersObj[3], scaleValue);

		std::vector<Point2f> scene_corners(4);

		perspectiveTransform(cornersObj, scene_corners, transformMatrix);
		float invScale = 1 / scaleValue;
		pointTransform(scene_corners[0], invScale);
		pointTransform(scene_corners[1], invScale);
		pointTransform(scene_corners[2], invScale);
		pointTransform(scene_corners[3], invScale);

		Rect sceneRect = boundingRect(scene_corners);
		if (sceneRect.width > rect.width * 0.75 && sceneRect.height > rect.height * 0.75)
		{
			int centerX = (scene_corners[0].x + scene_corners[1].x + scene_corners[2].x + scene_corners[3].x) / 4;
			int centerY = (scene_corners[0].y + scene_corners[1].y + scene_corners[2].y + scene_corners[3].y) / 4;

			int pointTPX = centerX - rect.width / 2;
			int pointTPY = centerY - rect.height / 2;

			dst = src(Rect(pointTPX, pointTPY, rect.width, rect.height));

			//dst = src(Rect(scene_corners[0].x, scene_corners[0].y, rect.width, rect.height));
			imwrite("trans_narrow_carrierRegion.jpg", dst);
		}
		else
		{
			std::cout << "Cannot find the carrier" << std::endl;
		}
	}
}

void pointTransform(cv::Point2f& pts1, float value)
{
	pts1.x = pts1.x * value;
	pts1.y = pts1.y * value;
}
