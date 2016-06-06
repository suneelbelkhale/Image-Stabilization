/*
* ImageStabilize.cpp
*
*  Created on: Apr 21, 2016
*      Author: suneelbelkhale1
*/

#include "ImageStabilize.h"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2\cudaimgproc.hpp>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <Shlwapi.h>
//#include <thread>         // std::this_thread::sleep_for
//#include <pthread.h>
//#include <chrono>
#include <time.h>
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"

//#include <termios.h>
#include <windows.h>
#pragma comment(lib, "vfw32.lib")
#pragma comment( lib, "comctl32.lib" )
#pragma comment(lib, "Shlwapi.lib")

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace cv::cuda;


#define FPS 20.0;
#define RADIUS 30
#define DT_SMOOTH_RADIUS 45
#define HESSIAN 270.0

static Size gaus_size(3,3);
static int sig_x = 0;



ImageStabilize::ImageStabilize() {
	// TODO Auto-generated constructor stub

}

ImageStabilize::~ImageStabilize() {
	// TODO Auto-generated destructor stub
}

//for allowing easier surf detection bc im lazy
struct SURFDetector
{
	Ptr<Feature2D> surf;

	SURFDetector(double hessian = 800.0)
	{
		surf = SURF::create(hessian);
	}
	template<class T>
	void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
	{
		surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
	}
};


struct rectE {
	int x;
	int y;
	int height;
	int width;
};

///FOR IMAGE TRAJECTORY BASED STABILIZATION
struct Traj {

	double dx;
	double dy;
	double da;

	void set(double x, double y, double a){
		dx = x;
		dy = y;
		da = a;
	}
};


//CALCULATES AN AVERAGE TRAJECTORY BASED ON THE PREVIOUS ELEMENTS
struct AverageWindow {
	vector <Traj> dt;

	vector <Traj> loc;

	//based on inputted dx, dy, and dz
	double x,y,a;

	void push(Traj change){
		//updating the current stored change in x vals
		dt.push_back(change);
		if (dt.size() > 80){
			dt.erase(dt.begin()); //get rid of first element
		}
		dt.shrink_to_fit();

		//updating the current image "location" based on change
		Traj newX;
		if (loc.size() > 0){
			newX.set(loc[loc.size()-1].dx + change.dx, loc[loc.size()-1].dy + change.dy, loc[loc.size()-1].da + change.da);
		} else{
			// if this is the first element
			newX.set(0 + change.dx, 0 + change.dy, 0 + change.da);
		}

		//push that new x value
		loc.push_back(newX);
		if (loc.size() > 80){
			loc.erase(loc.begin()); //get rid of first element
		}
		loc.shrink_to_fit();
	}

	// void full_sum(vector<Traj> els, Traj* j){

	// 	double sumX = 0, sumY = 0, sumA = 0;

	// 	for (int i = 0 ; i < els.size(); i++){
	// 		sumX += els[i].dx;
	// 		sumY += els[i].dy;
	// 		sumA += els[i].da;
	// 	}

	// 	j->set(sumX, sumY, sumA);
	// }


	Traj averageWindow(vector <Traj> els, int frames = 5){

		Traj currentTraj; //to be returned

//		cout << "MOTION" << endl;
		int count = 0;
		double sumX = 0, sumY = 0, sumA = 0;

		if (els.size() == 0){
			cout << "1" << endl;

			//returns empty
			currentTraj.set(0,0,0);
			return currentTraj;
		}

		else {


//			cout << "2: size: " << t.size() << endl;

			//prevents array bound errors
			if (els.size() < frames) {
				frames = els.size();
			}

			for (int i = 0; i < frames; i++){
				//accum

//				cout << "-- in loop : " << i << ", until:  " << t.size() - frames << endl;

				sumA += els[els.size() - 1 - i].da; //Angle
				sumY += els[els.size() - 1 - i].dy; //Y
				sumX += els[els.size() - 1 - i].dx; //X

				count++;
			}

			//accumulates the most recent ones
			currentTraj.set(sumX / count, sumY / count, sumA / count);
			return currentTraj;
		}

	}

};

//template<class KPMatcher>
//struct SURFMatcher
//{
//    KPMatcher matcher;
//    template<class T>
//    void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
//    {
//        matcher.match(in1, in2, matches);
//    }
//};

static Size fr_s;
static AverageWindow accumTraj;
Mat global_prevShown;
Mat global_curr;

static double prev_diff_x = 0;
static double prev_diff_y = 0;
static double prev_diff_a = 0;

vector<KeyPoint> keypoints_prev;
Mat descriptors_prev;
//GpuMat keypoints_prev_GPU;
//GpuMat descriptors_prev_GPU;
//static SURF_CUDA surf(1000);
static SURFDetector surf(600);





/**********************************************IMAGE STABILIZATION ALGORITHMS********************************************/




/************************ << BETTER TRAJECTORY ALGORITHM - OPTICAL FLOW>> *************************/

Mat trajAlgorithmOptFlow(Mat prev, Mat curr) {
	Rect roi(Point_<float>(curr.cols * 0.1f, curr.rows * 0.1f), Point_<float>(curr.cols * 0.9f, curr.rows * 0.9f));

	try{
		BFMatcher matcher;
		vector<KeyPoint> keypoints_2;
		Mat descriptors2;
		vector< vector <DMatch> > matches;

		surf(curr, Mat(), keypoints_2, descriptors2);

		matcher.knnMatch(descriptors_prev, descriptors2, matches, 2);

		//stores filtered keypoints, these two sets are identical, just in different form
		vector<KeyPoint> prevKeypts, currKeypts;
		vector<Point2f> prevPoints, currPoints;
		vector< DMatch > new_matches;

		//CHECK IF WE GOT NO GOOD KEYPOINTS FROM SURF, which is highly unlikely
		if (keypoints_prev.size() == 0 || keypoints_2.size() == 0){
			cout << "EMPTY RAW PTS YO" << endl;
			return curr(roi);
		};
		

		//filtering #####3: based on distance between matches
		int count = 0;
		double accumHyp = 0;
		for (size_t i = 0; i < matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			KeyPoint pr(keypoints_prev[matches[i][0].queryIdx].pt, keypoints_prev[0].size);
			KeyPoint cr(keypoints_2[matches[i][0].trainIdx].pt, keypoints_2[0].size);
			//check hypotenuse length
			double hypL = sqrt((pr.pt.x - cr.pt.x)*(pr.pt.x - cr.pt.x) + (pr.pt.y - cr.pt.y)*(pr.pt.y - cr.pt.y));
			if (hypL < 8){
				//Keypoint array
				prevKeypts.push_back(pr);
				currKeypts.push_back(cr);
				//point array
				prevPoints.push_back(pr.pt);
				currPoints.push_back(cr.pt);
				new_matches.push_back(matches[i][0]);
			}
			else{
				accumHyp += hypL;
				count++;
			}
		}

		//check if we still have stuff after filtering
		if (prevKeypts.size() == 0 || currKeypts.size() == 0){
			cout << "EMPTY FILTERED PTS YO" << endl;
			return curr(roi);
		};

		//PREV STUFF, muy importante, si si si si si
		keypoints_prev = keypoints_2;
		descriptors_prev.release();
		descriptors2.copyTo(descriptors_prev);


		
		//kept as a redundancy measure
		Mat T = estimateRigidTransform(prevPoints, currPoints, false);

		//-------------------------OPTICAL FLOW PART-------------------------//
		vector <Point2f> flowPoints;
		vector<uchar> featuresFound;
		Mat err;
		calcOpticalFlowPyrLK(prev,curr, prevPoints, flowPoints, featuresFound, err);

		int ct = 0; //in many ways a redundant variable but like YOLO, AMIRITE???
		double x_accum = 0, y_accum = 0;
		for (int i = 0; i < prevPoints.size(); i++){
			if (featuresFound[i]){
				x_accum += flowPoints[i].x - prevPoints[i].x;
				y_accum += flowPoints[i].y - prevPoints[i].y;
				ct++;
			}
		}
		//cout << endl << "Average X: " << x_accum/ct << ", Average Y: " << y_accum/ct << endl << endl;


		//-------------------------SMOOTHING TRAJECTORY PART-------------------------//
		double dx, dy, da;
		if (ct > 3 && !T.empty()){ //ct being greater than 3 = more than 3 matching feature sets found, arbitrary but means we have a valid average

			dx = x_accum/ct;
			dy = y_accum/ct;
			da = 0;


			//update the lists
			//adds to the trajectory averaging window
			Traj currTr;
			currTr.set(dx,dy,da);
			accumTraj.push(currTr); //updates everything

			//calculate the smoothed trajectory
			Traj smoothed_trajectory = accumTraj.averageWindow(accumTraj.loc, DT_SMOOTH_RADIUS);

			double diff_x = prev_diff_x;
			double diff_y = prev_diff_y;
			double diff_a = prev_diff_a;

			if (accumTraj.loc.size() > 0){ //this will always be true, but just in case

				Traj actualTraj = accumTraj.loc[accumTraj.loc.size() - 1];

				//WRITES
				// raw_traj_file << actualTraj.dx << ", " << actualTraj.dy << endl;
				// est_traj_file << smoothed_trajectory.dx << ", " << smoothed_trajectory.dy << endl;

				diff_x = smoothed_trajectory.dx - actualTraj.dx;
				diff_y = smoothed_trajectory.dy - actualTraj.dy;
				diff_a = smoothed_trajectory.da - actualTraj.da;
			}


			Mat TRANSFORM(2,3,CV_64F);
			TRANSFORM.at<double>(0,0) = 1;//cos(diff_a);
			TRANSFORM.at<double>(0,1) = 0;//-sin(diff_a);
			TRANSFORM.at<double>(1,0) = 0;//sin(diff_a);
			TRANSFORM.at<double>(1,1) = 1;//cos(diff_a);

			TRANSFORM.at<double>(0,2) = diff_x;
			TRANSFORM.at<double>(1,2) = diff_y;

			Mat curr2;
			warpAffine(curr, curr2, TRANSFORM, curr.size());

//			Rect roi2(Point_<float>(roi.x + diff_x / 1.0, roi.y + diff_y / 1.0), Point_<float>(roi.x + roi.width + diff_x / 1.0, roi.y + roi.height + diff_y / 1.0));
//			rectangle(curr, roi2, Scalar(0, 255, 0), 1);


			//x,y,z
			prev_diff_x = diff_x;
			prev_diff_y = diff_y;
			prev_diff_a = diff_a;

			return curr2(roi);


		}

		//IF THERE WAS A MISTRANSFORM
		else {
			cout << ">>>>>>>>>>>MISSED TRANSFORM!!!!!!1";
			return curr(roi);
		}


	}
	catch (cv::Exception& e) {
		cout << "\nERROR SOMEWHERE HERE: " << e.msg << endl << endl;
		return curr(roi);
	}
}







/************************ << TRAJECTORY ALGORITHM - ESTIMATE RIGID TRANSFORM>> *************************/


Mat trajAlgorithm(Mat prev, Mat curr) {
	Rect roi(Point_<float>(curr.cols * 0.1f, curr.rows * 0.1f), Point_<float>(curr.cols * 0.9f, curr.rows * 0.9f));

	try{
		BFMatcher matcher;
		vector<KeyPoint> keypoints_2;
		Mat descriptors2;
		vector< vector <DMatch> > matches;

		surf(curr, Mat(), keypoints_2, descriptors2);

		matcher.knnMatch(descriptors_prev, descriptors2, matches, 2);

		vector<Point2f> prevScene;
		vector<Point2f> currScene;
		vector< DMatch > new_matches;

		int count = 0;
		double accumHyp = 0;

		//filtering #####3
		for (size_t i = 0; i < matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			Point2f pr = keypoints_prev[matches[i][0].queryIdx].pt;
			Point2f cr = keypoints_2[matches[i][0].trainIdx].pt;
			//check hypotenuse length
			double hypL = sqrt((pr.x - cr.x)*(pr.x - cr.x) + (pr.y - cr.y)*(pr.y - cr.y));
			if (hypL < 35){
				prevScene.push_back(pr);
				currScene.push_back(cr);
				new_matches.push_back(matches[i][0]);
			}
			else{
				accumHyp += hypL;
				count++;
			}
		}

		//PREV STUFF
		keypoints_prev = keypoints_2;
		descriptors_prev.release();
		descriptors2.copyTo(descriptors_prev);


		if (prevScene.size() == 0 || currScene.size() == 0){
			return curr(roi);
		};


		Mat T = estimateRigidTransform(prevScene, currScene, false);

		if (!T.empty()){

			double dx = T.at<double>(0, 2);
			double dy = T.at<double>(1, 2);
			double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));


			//update the lists
			//adds to the trajectory averaging window
			Traj currTr;
			currTr.set(dx,dy,da);
			accumTraj.push(currTr); //updates everything

			//calculate the smoothed trajectory
			Traj smoothed_trajectory = accumTraj.averageWindow(accumTraj.loc, RADIUS);

			double diff_x = prev_diff_x;
			double diff_y = prev_diff_y;
			double diff_a = prev_diff_a;

			if (accumTraj.loc.size() > 0){ //this will always be true, but jsut in case

				Traj actualTraj = accumTraj.loc[accumTraj.loc.size() - 1];

				//WRITES
				// raw_traj_file << actualTraj.dx << ", " << actualTraj.dy << endl;
				// est_traj_file << smoothed_trajectory.dx << ", " << smoothed_trajectory.dy << endl;

				diff_x = smoothed_trajectory.dx - actualTraj.dx;
				diff_y = smoothed_trajectory.dy - actualTraj.dy;
				diff_a = smoothed_trajectory.da - actualTraj.da;
			}


			Mat TRANSFORM(2,3,CV_64F);
			TRANSFORM.at<double>(0,0) = 1;//cos(diff_a);
			TRANSFORM.at<double>(0,1) = 0;//-sin(diff_a);
			TRANSFORM.at<double>(1,0) = 0;//sin(diff_a);
			TRANSFORM.at<double>(1,1) = 1;//cos(diff_a);

			TRANSFORM.at<double>(0,2) = diff_x;
			TRANSFORM.at<double>(1,2) = diff_y;

			Mat curr2;
			warpAffine(curr, curr2, TRANSFORM, curr.size());

//			Rect roi2(Point_<float>(roi.x + diff_x / 1.0, roi.y + diff_y / 1.0), Point_<float>(roi.x + roi.width + diff_x / 1.0, roi.y + roi.height + diff_y / 1.0));
//			rectangle(curr, roi2, Scalar(0, 255, 0), 1);


			//x,y,z
			prev_diff_x = diff_x;
			prev_diff_y = diff_y;
			prev_diff_a = diff_a;

			return curr2(roi);


		}

		//IF THERE WAS A MISTRANSFORM
		else {
			cout << ">>>>>>>>>>>MISSED TRANSFORM!!!!!!1";
			return curr(roi);
		}


	}
	catch (cv::Exception& e) {
		cout << "\nERROR SOMEWHERE HERE: " << e.msg << endl << endl;
		return curr(roi);
	}
}








/************************ << ALGORITHM 1 : BASED ON SOLE IMAGE >> *************************/


Mat alg1(Mat curr) {


	try{

		long int first = GetTickCount(), original = GetTickCount();

		Rect roi(Point_<float>(curr.cols * 0.1f, curr.rows * 0.1f), Point_<float>(curr.cols * 0.9f, curr.rows * 0.9f));

		//cout << "BEGIN SURF || ";
		first = GetTickCount();


		//GpuMat g_curr(curr);


		// detecting keypoints & computing descriptors 
		//GpuMat keypoints2GPU;
		//GpuMat descriptors2GPU;
		//surf(g_curr, GpuMat(), keypoints2GPU, descriptors2GPU);


		// matching descriptors 

		//Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
		//vector<DMatch> matches;
		//matcher->match(descriptors_prev_GPU, descriptors2GPU, matches);

		// downloading results 
		//vector<KeyPoint> keypoints2;
		//vector<float> descriptors2;
		//surf.downloadKeypoints(keypoints2GPU, keypoints2);
		//surf.downloadDescriptors(descriptors2GPU, descriptors2);


		BFMatcher matcher;
		vector<KeyPoint> keypoints_2;
		Mat descriptors2;
		vector< vector <DMatch> > matches;

		surf(curr, Mat(), keypoints_2, descriptors2);

		//cout << "SURF: " << GetTickCount() - first << " || ";
		first = GetTickCount();

		// matcher.match(descriptors_prev, descriptors2, matches);
		matcher.knnMatch(descriptors_prev, descriptors2, matches, 2);

		//cout << "MATCHER: " << GetTickCount() - first << " || ";
		//first = GetTickCount();

		//			Mat img_matches;
		//			vector< DMatch > good_matches;

		//filtering #####1
		//			for (int i = 0; i < (int)matches.size(); i++){
		//	//			if (matches[i][0].distance < 0.6f * matches[i][1].distance){
		//					good_matches.push_back(matches[i][0]);
		//	//			}
		//			}

		//			cout << "GM size: " << matches.size() << endl;


		//filtering #####2
		//		matches.erase(remove_if(matches.begin(),matches.end(),bad_dist),matches.end());

		//		sort(matches.begin(), matches.end());
		//		double minDist = matches.front().distance;
		//		double maxDist = matches.back().distance;

		//		const int ptsPairs = std::min(800, (int)(matches.size() * 0.15f));
		//		for( int i = 0; i < ptsPairs; i++ )
		//		{
		//			good_matches.push_back( matches[i] );
		//		}



		vector<Point2f> prevScene;
		vector<Point2f> currScene;
		vector< DMatch > new_matches;
		//			cout << "Choosing best matches" << endl;

		int count = 0;
		double accumHyp = 0;

		//filtering #####3
		for (size_t i = 0; i < matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			Point2f pr = keypoints_prev[matches[i][0].queryIdx].pt;
			Point2f cr = keypoints_2[matches[i][0].trainIdx].pt;
			//check hypotenuse length
			double hypL = sqrt((pr.x - cr.x)*(pr.x - cr.x) + (pr.y - cr.y)*(pr.y - cr.y));
			if (hypL < 35){
				prevScene.push_back(pr);
				currScene.push_back(cr);
				new_matches.push_back(matches[i][0]);
			}
			else{
				accumHyp += hypL;
				count++;
			}
		}

		//cout << "FILTERING: " << GetTickCount() - first << " || ";
		//first = GetTickCount();

		//			if (count != 0) { cout << endl << "DIFF/TOTAL::: " << count << " / " << good_matches.size() << ", average Missed Hyp::: " << accumHyp/(double)count << endl; }
		//			else { cout << endl << "DIFF::: None" << endl; }


		//			cout << endl;



		//DRAW
		//			drawMatches( global_prevShown, keypoints_prev, curr, keypoints_2,
		//									 new_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		//									 vector<char>(), DrawMatchesFlags::DEFAULT  );



		//			cout << "PrevScene: "  << prevScene << endl;
		//			cout << "CurrScene: "  << currScene << endl;


		//			Mat H = findHomography( prevScene, currScene, RANSAC );

		if (prevScene.size() == 0 || currScene.size() == 0){
			//							cout << "Empty Scenes" << endl;
			//cout << "NA keypoints >>>>>> TOTAL ELAPSED TIME ALG 1: " << GetTickCount() - original << endl;
			return curr(roi);
		};


		Mat rigidTransform = estimateRigidTransform(prevScene, currScene, false);

		if ((rigidTransform.rows == 0 && rigidTransform.cols == 0) || rigidTransform.empty()){
			//				cout << "Empty RIGID" << endl;

			//cout << "Empty Rigid >>>>>> TOTAL ELAPSED TIME ALG 1: " << GetTickCount() - original << endl;
			return curr(roi);
		};

		//			cout << "RIGID::: " << rigidTransform << endl;
		double dx = rigidTransform.at<double>(0, 2);
		double dy = rigidTransform.at<double>(1, 2);
		double da = atan2(rigidTransform.at<double>(1, 0), rigidTransform.at<double>(0, 0));
		//			cout << "RIGID (x,y,a): " << dx << ", " << dy << ", " << da << endl << endl;


		//cout << "RIGID TRANSFORM: " << GetTickCount() - first << " || ";
		//first = GetTickCount();

		Rect roi2(Point_<float>(roi.x + dx / 1.0, roi.y + dy / 1.0), Point_<float>(roi.x + roi.width + dx / 1.0, roi.y + roi.height + dy / 1.0));

		rectE roiC;
		roiC.x = roi2.x;
		roiC.y = roi2.y;
		roiC.width = roi2.width;
		roiC.height = roi2.height;

		Mat temp;
		curr.convertTo(temp, CV_32FC1);

		//too far right
		if (roi2.x + roi2.width >= temp.cols) {
			roiC.x = temp.cols - 1 - roi2.width;
			//Rect roiC(Point_<float>(temp.cols - 1 - roi2.width, roi2.y), Point_<float>(temp.cols-1, roi2.y + roi2.height));
			//cout << ">> TOO FAR RIGHT -- ROI 2 : " << roi2 << " ROI C : " << roiC << endl;
		}
		//too far left
		else if (roi2.x < 0){
			roiC.x = 0;
			//Rect roiC(Point_<float>(0, roi2.y), Point_<float>(roi2.width, roi2.y + roi2.height));
			//cout << ">> TOO FAR LEFT -- ROI 2 : " << roi2 << " ROI C : " << roiC << endl;
		}
		//if too far down
		if (roi2.y + roi2.height >= temp.rows) {
			roiC.y = temp.rows - 1 - roi2.height;
			//Rect roiC(Point_<float>(roi2.x, temp.rows - 1 - roi2.height), Point_<float>(roi2.x + roi2.width, temp.rows - 1));
			//cout << ">> TOO FAR DOWN -- ROI 2 : " << roi2  << " ROI C : " << roiC << endl;
		}
		//too far up
		else if (roi2.y < 0){
			roiC.y = 0;
			//Rect roiC(Point_<float>(roi2.x, 0), Point_<float>(roi2.x + roi2.width, roi2.height));
			//cout << ">> TOO FAR UP -- ROI 2 : " << roi2 << " ROI C : " << roiC << endl;
		}


		//cout << "RECT STUFF: " << GetTickCount() - first << " || ";
		//first = GetTickCount();

		Rect correctedROI(roiC.x, roiC.y, roiC.width, roiC.height);

		Mat subby = curr(correctedROI).clone();

		rectangle(curr, correctedROI, Scalar(0, 255, 0), 1);


		//cout << "ENDING STUFF: " << GetTickCount() - first << " || ";
		//first = GetTickCount();

		//		cout << "Type: " << H.type() << ", D {" << H.cols << ", " << H.rows << "}" << endl;

		//-- Get the corners from the image_1 ( the object to be "detected" )
		//			vector<Point2f> obj_corners(4);
		//			obj_corners[0] = Point2f((float)roi.tl().x, (float)roi.tl().y) ;
		//			obj_corners[1] = Point2f((float)roi.tl().x + roi.width, (float)roi.tl().y);
		//			obj_corners[2] = Point2f((float)roi.br().x, (float)roi.br().y) ;
		//			obj_corners[3] = Point2f((float)roi.br().x - roi.width, (float)roi.br().y);
		//			vector<Point2f> scene_corners(4);

		//			cout << obj_corners << endl;

		//			perspectiveTransform( obj_corners, scene_corners, H);


		//			line( curr, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 2 );
		//			line( curr, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 2 );
		//			line( curr, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 2 );
		//			line( curr, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 2 );


		//cout << ">>>>>> TOTAL ELAPSED TIME ALG 1: " << GetTickCount() - original << endl;

		//return img_matches;
		return subby;
		//return curr;

	}
	catch (cv::Exception& e) {
		cout << "\nERROR SOMEWHERE HERE: " << e.msg << endl << endl;
		return curr;
	}
}




/************************ << ALGORITHM 2 >> *************************/


Mat alg2(Mat prev, Mat curr){
	cv::cvtColor(curr, curr, COLOR_BGR2GRAY);
	//		inRange(curr, Scalar(0), Scalar(100), curr);
	curr.convertTo(curr, CV_8UC1);
	Rect roi(Point_<float>(prev.cols * 0.2f, prev.rows * 0.2f), Point_<float>(prev.cols * 0.8f, prev.rows * 0.8f));
	vector<KeyPoint> keypoints;

	curr.convertTo(curr, CV_8UC1);

	FAST(curr, keypoints, 9);
	Ptr<GeneralizedHoughBallard> ght_detector = cv::createGeneralizedHoughBallard();
	vector<Vec4i> out;
	ght_detector->detect(curr, out);
	//		cout << "Positions: " << out << endl;
	return curr;
}


bool bad_dist(const DMatch &m) {
	return m.distance > 150;
}








bool onOrOff;
bool isRecording;
bool prevToggleRecord = false;
bool prevScreenShotButton = false;
bool prevToggleOISButton = false;
bool prevCaptureButton = false;
bool progExit = false;

static VideoWriter outputVideo;
static VideoCapture vcap;

//static string outputDir = "/Users/suneelbelkhale1/Documents/code/opencv/OIS/outputs/";
static string outputDir = "C:\\Users\\vixel\\Desktop\\OxidationImages\\";
static string outputFile;
//static string recStat = "OFF";

void onChangeRecord(){

	if (isRecording){
		time_t rawtime = time(0);
		tm *t = localtime(&rawtime);
		std::ostringstream oss;
		oss << outputDir << "output_vid__" << t->tm_hour << "_" << t->tm_min << "_" << t->tm_sec << ".avi";
		outputFile = oss.str();
		//vcap.get(CV_CAP_PROP_FOURCC), vcap.get(CV_CAP_PROP_FPS)

		int fps = vcap.get(CV_CAP_PROP_FPS);
		int fourcc = CV_FOURCC('M', 'J', 'P', 'G');
		if (fps < 1){
			//this means you are reading from a live stream so it doesnt know the frame rate
			fps = FPS; //fastest it could be, so YOLO
		}

		new (&outputVideo) VideoWriter(outputFile, fourcc , fps, fr_s);
		cout << "SIZE: " << fr_s << endl;
		cout << "Outputting video to: " << outputFile << ", FCC: " << vcap.get(CV_CAP_PROP_FOURCC) << endl;
	}
	else{
		cout << "Releasing video" << endl;
		outputVideo.release();
	}
}

void onChangeOIS(){
	cout << "FIRST TIME -- cloning this image as basis" << endl;
	Mat newcurr = global_curr.clone();
	surf(newcurr, Mat(), keypoints_prev, descriptors_prev);
	global_prevShown.release();
	newcurr.copyTo(global_prevShown);

}

void capture(){
	time_t rawtime = time(0);
	tm *t = localtime(&rawtime);
	std::ostringstream oss;
	oss << outputDir << "screenshot__" << t->tm_hour << "_" << t->tm_min << "_" << t->tm_sec << ".bmp";
	outputFile = oss.str();

	cout << "Capturing to: " << outputFile;

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);
	bool result;

	if (global_curr.empty()){ cout << "Empty prevShown" << endl; }
	try {
		result = imwrite(outputFile, global_curr, compression_params);
	}
	catch (exception &e){

	}

	cout << " ------ Success? " << result << endl;
}
//void *updateKeyboardState(void *threadID){
DWORD WINAPI updateKeyboardState(LPVOID lpParam){
	onOrOff = false;
	prevToggleOISButton = false;
	isRecording = false;

	while (!progExit){
		cout << "On (1) or Off (0)? " << onOrOff << ", reading... " << endl;
		char input = getchar();
		//		cout << "got: " << input << endl;

		if (input == 's' && !prevToggleOISButton){
			onOrOff = !onOrOff;
			//if setting on, redo some stuff
			// if (onOrOff){
			// 	onChangeOIS();
			// }
		}

		if (input == 'c' && !prevCaptureButton){
			capture();
		}

		if (input == 'r' && !prevToggleRecord){
			isRecording = !isRecording;
			onChangeRecord();
		}
		//to avoid enters and duplicates
		//		this_thread::sleep_for(std::chrono::milliseconds(50));

		prevToggleOISButton = (input == 's');
		prevToggleRecord = (input == 'r');
		prevCaptureButton = (input == 'c');
	}


	cout << "STOPPED reading" << endl;

	return 0;
}



int main(int argc, char** argv){

	cout << "STARTING ----- \n" << endl;

	int vidSource = 0; //default


	//if (argc == 2){
	//	string s = argv[1];
	//	vidSource = stoi(s);
	//}

	if (argc >= 2){
		for (int i = 1; i < argc; i++){
			string param = argv[i];

			if (param == "-i" && i + 1 < argc){
				string s = argv[i + 1];
				vidSource = stoi(s);
				cout << "--->video input set to : " << vidSource << endl;
			}
			if (param == "-o" && i + 1 < argc){
				// for output video dir
				char* original = argv[i + 1];
				bool isDir = PathFileExists(original);
				//if already a valid path, do nothing, else, try to convert it
				if (!isDir){

					cout << "--->INVALID PATH ENTERED FOR -o/: " << original << endl;
					return -1;
				}
				else{
					cout << "--->Path entered is a valid path" << endl;
					//int len = strlen(original);
					//char temp[1000];
					//int t = 0;
					//for (int j = 0; j < len; j++){
					//	if (original[j] == "" ){
					//		temp[t] = original[j];
					//	}
					//}
					outputDir = original;
				}
			}
		}
	}

	cout << "Starting video from input# : " << vidSource << endl;

	onOrOff = false; //decides when to start recording;
	progExit = false;

	namedWindow("Stabilized", CV_WINDOW_AUTOSIZE);

	//	namedWindow("Unadjusted", CV_WINDOW_AUTOSIZE);

	//	vcap.open("/Users/suneelbelkhale1/Documents/code/opencv/OIS/sample.avi");
	vcap.open(vidSource);//"C:\\Users\\Vidya\\Documents\\testOpencv\\sample.avi");

	//	VideoCapture vcap(0);
	assert(vcap.isOpened());


	if (!vcap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}

	//sketch af lol
	//    this_thread::sleep_for(std::chrono::milliseconds(100));
	Sleep(100);

	Mat prev;
	vcap.read(prev);

	//preliminary to avoid errors shhhh
	surf(prev, Mat(), keypoints_prev, descriptors_prev);

	try {
		GpuMat gm;
		gm.upload(prev);
		cuda::cvtColor(gm, gm, COLOR_BGR2GRAY);
		cout << "	Y   -> CUDA --IS-- AVAILABLE ON THIS COMPUTER" << endl << endl;
	}
	catch (exception& e){
		cout << "	N   -> CUDA --NOT-- AVAILABLE ON THIS COMPUTER: Check Graphics Card" << endl << endl;
	}
	catch (Exception& e){
		cout << "	N   -> CUDA --NOT-- AVAILABLE ON THIS COMPUTER: Check Graphics Card" << endl << endl;
	}

	fr_s = Size(prev.cols, prev.rows);


	//	cout << prev << endl;

	float aspect_ratio = prev.rows / (double)prev.cols; //y:x
	resize(prev, prev, Size(700, 700 * aspect_ratio));

	//	cout << "Aspect Ratio: " << prev.cols << ", " << prev.rows << " --- R: " << aspect_ratio <<endl;

	HANDLE thread;
	DWORD dwThreadId, dwThrdParam = 1;

	thread = CreateThread(NULL, // default security attributes
		0, // use default stack size
		updateKeyboardState, // thread function
		&dwThrdParam, // argument to thread function
		0, // use default creation flags
		&dwThreadId); // returns the thread identifier)
	//	pthread_t thread;


	//	if (!pthread_create(&thread, NULL,
	//            updateKeyboardState, (void *)1)){
	////		cout << endl << "Success! Created Thread" << endl;
	//	}
	//	else{
	////		cout << endl << "FAILED to create Thread" << endl;
	//		exit(-1);
	//	}

	//	imshow("Stabilized", prev);

	Mat curr;

	bool prevStateOnOrOff = false;

	//	pthread_t recThread;
	//	pthread_create(&recThread, NULL, updateRecordingStatus, (void *)2); //run the rec loop checker




	//////*******************LOOP****************//////

	while (vcap.read(curr)){
		try{
			long int first = GetTickCount();

			resize(curr, curr, Size(700, 700 * aspect_ratio));
			//			cout << "Aspect Ratio: " << curr.cols << ", " << curr.rows << " --- R: " << aspect_ratio <<endl;

			curr.copyTo(global_curr);
			String stat, recStat;

			//OIS
			Mat s;
			if (onOrOff){
				// if (!prevStateOnOrOff){ //for the first time
				// 	cout << "FIRST TIME -- cloning this image as basis" << endl;
				// 	Mat newcurr = curr.clone();
				// 	surf(newcurr, Mat(), keypoints_prev, descriptors_prev);
				// 	//GpuMat old(curr);
				// 	//creating keypoints as well
				// 	//surf(old, GpuMat(), keypoints_prev_GPU, descriptors_prev_GPU);
				// 	//surf.downloadKeypoints(keypoints_prev_GPU, keypoints_prev);
				// 	//surf.downloadDescriptors(descriptors_prev_GPU, descriptors_prev);

				// }
				// s = alg1(curr);
				s = trajAlgorithmOptFlow(prev, curr);
				stat = "ON";
			}
			else{
				s = curr.clone();
				stat = "OFF";
			}

			global_curr.copyTo(prev);


			resize(s, s, curr.size());


			


			//		Mat s = alg2(prev, curr);

			Rect roi(Point_<float>(prev.cols * 0.1f, prev.rows * 0.1f), Point_<float>(prev.cols * 0.9f, prev.rows * 0.9f));
			//		Mat subby = s(roi).clone();

			putText(s, "OIS: " + stat, Point2f(15, 10), FONT_HERSHEY_COMPLEX, 0.3, Scalar(255, 0, 0));
			putText(s, "Recording: " + recStat, Point2f(s.cols - 90, 10), FONT_HERSHEY_COMPLEX, 0.3, Scalar(0, 0, 255));


			fr_s = Size(s.cols, s.rows);

			imshow("Stabilized", s);

			//cout << ">>>>>>>>>>>FULL THANG: " << GetTickCount() - first << endl << endl;

			

			//while (GetTickCount() - first < 1000 / 1.0){
			//	cout << "...waiting for frame" << endl;
			//}

			//RECORDING
			if (isRecording && outputVideo.isOpened()){
				recStat = "REC";
				outputVideo.write(s);

			}
			else{
				recStat = "OFF";
				outputVideo.release();
			}


			if (waitKey(1) >= 0) break;
			//curr.copyTo(prev);


		}
		catch (cv::Exception& e){
			cout << "WOOPS " << endl;
		}

		prevStateOnOrOff = onOrOff;
	}

	// Wait for a keystroke in the window

	progExit = true;

	return 0;

}
