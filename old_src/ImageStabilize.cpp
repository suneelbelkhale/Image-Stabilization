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

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace cv::cuda;


ImageStabilize::ImageStabilize() {
	// TODO Auto-generated constructor stub

}

ImageStabilize::~ImageStabilize() {
	// TODO Auto-generated destructor stub
}


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


vector<KeyPoint> keypoints_prev;
Mat descriptors_prev;
//GpuMat keypoints_prev_GPU;
//GpuMat descriptors_prev_GPU;
//static SURF_CUDA surf(1000);
SURFDetector surf(1000);

Mat alg1 (Mat curr) {


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
			vector<DMatch> matches;

			surf(curr, Mat(), keypoints_2, descriptors2);

			//cout << "SURF: " << GetTickCount() - first << " || ";
			first = GetTickCount();

			matcher.match(descriptors_prev, descriptors2, matches);
			//matcher.knnMatch(descriptors_prev, descriptors2, matches, 2);

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
			for( size_t i = 0; i < matches.size(); i++ )
			{
			//-- Get the keypoints from the good matches
				Point2f pr =  keypoints_prev[ matches[i].queryIdx].pt;
				Point2f cr = keypoints_2[ matches[i].trainIdx].pt;
				//check hypotenuse length
				double hypL = sqrt((pr.x - cr.x)*(pr.x - cr.x) + (pr.y - cr.y)*(pr.y - cr.y));
				if (hypL < 35){
					prevScene.push_back( pr );
					currScene.push_back( cr );
					new_matches.push_back(matches[i]);
				}
				else{
					accumHyp+=hypL;
					count++;
				}
			}

			//cout << "FILTERING: " << GetTickCount() - first << " || ";
			//first = GetTickCount();

//			if (count != 0) { cout << endl << "DIFF/TOTAL::: " << count << " / " << good_matches.size() << ", average Missed Hyp::: " << accumHyp/(double)count << endl; }
//			else { cout << endl << "DIFF::: None" << endl; }


//			cout << endl;



			//DRAW
//			drawMatches( subPrev, keypoints_1, curr, keypoints_2,
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
			double da = atan2(rigidTransform.at<double>(1,0), rigidTransform.at<double>(0,0));
//			cout << "RIGID (x,y,a): " << dx << ", " << dy << ", " << da << endl << endl;


			//cout << "RIGID TRANSFORM: " << GetTickCount() - first << " || ";
			//first = GetTickCount();

			Rect roi2(Point_<float>(roi.x + dx/1.0, roi.y + dy/1.0), Point_<float>(roi.x+ roi.width + dx/1.0, roi.y + roi.height + dy/1.0));

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

			//rectangle(curr, correctedROI, Scalar(0, 255, 0), 1);


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

//			return img_matches;
			return subby;
//			return curr;

		}
		catch (cv::Exception& e) {
			cout << "\nERROR SOMEWHERE HERE: " << e.msg << endl << endl;
			return curr;
		}
}

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
		ght_detector->detect(curr,out);
//		cout << "Positions: " << out << endl;
		return curr;
}


bool bad_dist(const DMatch &m) {
    return m.distance > 150;
}







Size fr_s;
Mat prevShown;


bool onOrOff;
bool isRecording;
bool prevToggleRecord = false;
bool prevScreenShotButton = false;
bool prevToggleOISButton = false;
bool prevCaptureButton = false;
bool progExit = false;

VideoWriter outputVideo;
VideoCapture vcap;

//static string outputDir = "/Users/suneelbelkhale1/Documents/code/opencv/OIS/outputs/";
static string outputDir = "C:\\Users\\vixel\\Desktop\\OxidationImages\\";
string outputFile;
string recStat = "OFF";

void onChangeRecord(){

	if (isRecording){
			time_t rawtime = time(0);
			tm *t = localtime(&rawtime);
			std::ostringstream oss;
			oss << outputDir << "output_vid__" << t->tm_hour << "_" << t->tm_min << "_" << t->tm_sec << ".avi";
			outputFile = oss.str();
			outputVideo.open(outputFile,vcap.get(CV_CAP_PROP_FOURCC), vcap.get(CV_CAP_PROP_FPS), fr_s);
			cout << "Outputting video to: " << outputFile << ", FCC: " << vcap.get(CV_CAP_PROP_FOURCC) <<  endl;
	}
	else{
		cout << "Releasing video" << endl;
		outputVideo.release();
	}
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

	if (prevShown.empty()){ cout << "Empty prevShown" <<endl; }
	try {
	        result = imwrite(outputFile, prevShown, compression_params);
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

	if (argc == 2){
		string s = argv[1];
		vidSource = stoi(s);
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

	while(vcap.read(curr)){
		try{
			long int first = GetTickCount();
			resize(curr, curr, Size(700, 700 * aspect_ratio));
//			cout << "Aspect Ratio: " << curr.cols << ", " << curr.rows << " --- R: " << aspect_ratio <<endl;

			String stat;

	//		imshow("Stabilized", subby);
			Mat s;
			if (onOrOff){
				if (!prevStateOnOrOff){ //for the first time
					cout << "FIRST TIME -- cloning this image as basis" << endl;
					Mat newcurr = curr.clone();
					surf(newcurr, Mat(), keypoints_prev, descriptors_prev);
					//GpuMat old(curr);
					//creating keypoints as well
					//surf(old, GpuMat(), keypoints_prev_GPU, descriptors_prev_GPU);
					//surf.downloadKeypoints(keypoints_prev_GPU, keypoints_prev);
					//surf.downloadDescriptors(descriptors_prev_GPU, descriptors_prev);
					
				}
				s = alg1(curr);
				stat = "ON";
			}
			else{
				s = curr.clone();
				stat = "OFF";
			}


			resize(s, s, curr.size());

			if (isRecording && outputVideo.isOpened()){
				recStat = "REC";
				outputVideo.write(s);

			}
			else{
				recStat = "OFF";
			}


	//		Mat s = alg2(prev, curr);

			Rect roi(Point_<float>(prev.cols * 0.1f, prev.rows * 0.1f), Point_<float>(prev.cols * 0.9f, prev.rows * 0.9f));
	//		Mat subby = s(roi).clone();

			putText(s,"OIS: " + stat, Point2f(15,10), FONT_HERSHEY_COMPLEX, 0.3, Scalar(255,0,0));
			putText(s,"Recording: " + recStat, Point2f(s.cols - 90,10), FONT_HERSHEY_COMPLEX, 0.3, Scalar(0,0,255));


			fr_s = Size(s.cols, s.rows);

			imshow("Stabilized", s);
			s.copyTo(prevShown);

			//cout << ">>>>>>>>>>>FULL THANG: " << GetTickCount() - first << endl << endl;
			

			if(waitKey(1) >= 0) break;
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
