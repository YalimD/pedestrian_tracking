#include <opencv2/core.hpp>
#include <opencv2/closecv.hpp>

//#include <opencv2/bgsegm.hpp>
//#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/ximgproc.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <string>
#include <random>
#include <cmath>
#include <fstream>

#include "debug.h"


using namespace std;
using namespace cv;

/***************************************************************************************************/
//Global variables used in code
constexpr int START_FRAME = 0;
constexpr int FRAME_SCALE = 1; //Resize frames (!!NOT WORKING RIGHT NOW (KEEP IT AT 1)!!)
constexpr int NUM_FRAMES = 10000; //Number of frames that will be processed

constexpr int TRACKER_MAX_DISTANCE  = 50; // Used in score function
constexpr int TRACKER_ACTIVATE_COUNT = 15;
constexpr int TRACKER_DEATH_COUNT   = 300;
constexpr int TRACKER_INACTIVATE_COUNT = 30;

constexpr double TRACKER_SCORE_MIX_RATIO = 0.7;

constexpr double MULTI_TRACKER_ASSOCIATION_THRESHOLD = 0.3;

constexpr int HISTOGRAM_SIZE = 16;

constexpr double DETECTION_SCALE = 3.5;

constexpr const char* OUTPUT_FILE_PATH = "out.txt";

constexpr bool DRAW_TRACKERS = true;

constexpr const char* SAVE_FORMAT = "jpg"; // jpg 

constexpr bool USE_STABILIZATION = true;

constexpr bool VIDEO_OUTPUT = true; //If true save as video otherwise save as image sequence

constexpr bool SHOW_OUTPUT = true;
/**************************************************************************************************/

template <typename T>
T clamp(T value,T min = 0 , T max = 1)
{
    return ( value < min ) ? ( min ) : ( ( value > max ) ? ( max ) : ( value ) );
}

double sqr(double d)
{
    return d*d;
}

float euclideanDist(const cv::Point2f& a,const cv::Point2f& b)
{
    cv::Point2f diff = a - b;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

//Returns a unique integer when called
int next_id()
{
    static int id = 0;
    id++;
    return id;
}

//Tracks one kind of detection
//Uses Kalman Filter (Constant velocity model)
class Tracker
{
public:
    enum State{START,INACTIVE,ACTIVE,DEAD};
    
    KalmanFilter filter; //Used to filter movement (Uses constant velocity model)
    int time_passed_since_correction; //How many times object is undetected (consecutive)
    int correction_count;	//How many times object detected

    int frame_width,frame_height; //Width and height of the screen / detection region
    double scale_x,scale_y; //Scale of the tracked region
    
    int id; //Unique id for the tracker
    
    //Is tracker detected in the last frame
    bool detected = true;
    
    State state;
    Mat color_data; //Feature data used to discriminate the tracker from other trackers
    
    Tracker(const Rect &initial_bounding_box,int frame_width,int frame_height,const Mat& feature)
    {
        id = next_id();
        time_passed_since_correction = 0;
        correction_count = 0;
        state = START;

        this->frame_width = frame_width;
        this->frame_height = frame_height;
        scale_x = initial_bounding_box.width;
        scale_y = initial_bounding_box.height;
        color_data = feature.clone();
        initializeKalmanFilter(initial_bounding_box);
        
        predict();

        correct(initial_bounding_box,feature);
    }

    //Predict next position of the object using Kalman Filter
    Point predict()
    {
        detected = false;
        time_passed_since_correction++;
        
        if (time_passed_since_correction >= TRACKER_INACTIVATE_COUNT && state == ACTIVE)
        {
            state = INACTIVE;
        }
        
        if (time_passed_since_correction >= TRACKER_DEATH_COUNT / 50 && state == START   )
        {
            state = DEAD;
        }
        if (time_passed_since_correction >= TRACKER_DEATH_COUNT)
        {
            state = DEAD;
        }
        
        Mat prediction = filter.predict();
        return Point(prediction.at<float>(0),prediction.at<float>(1));
    }

    //Performs correction
    //Check Kalman Filter correction
    //Correction means object is detected
    void correct(const Rect& bounding_box,const Mat& feature)
    {
        detected = true;
        correction_count++;
        time_passed_since_correction = 0;
        
        scale_x = bounding_box.width * 0.5 + scale_x * 0.5;
        scale_y = bounding_box.height * 0.5 + scale_y * 0.5;
        Point2f position = toPoint(bounding_box);
        
        if (correction_count >= TRACKER_ACTIVATE_COUNT && (state == INACTIVE || state == START ) )
        {
            state = ACTIVE;
            //On reactivation : reset kalman filter
            //
            initializeKalmanFilter(bounding_box);
            predict();
        }


        Mat_<float> measurement(2,1);
        measurement(0) = position.x;
        measurement(1) = position.y;

        filter.correct(measurement);
        
        if (state != INACTIVE)
           addWeighted(color_data,0.6 , feature , 0.4,0 ,color_data);
    }

    //Returns cost
    //Cost should be between 0 and 1
    //0 -> means detected object probably is the tracked object
    //1 -> means it's not
    double cost(const Rect& bounding_box,const Mat& feature)
    {
        Point2f position = toPoint(bounding_box);
        Point2f prediction(filter.statePre.at<float>(0),filter.statePre.at<float>(1));

        double positional_cost = euclideanDist(prediction,position);
        positional_cost = positional_cost > TRACKER_MAX_DISTANCE ? 1 : (positional_cost / TRACKER_MAX_DISTANCE);
       
        double similarity_cost = norm(feature, this->color_data , NORM_L2);

        if (state == INACTIVE)
            return similarity_cost * 0.7 + positional_cost * 0.3;

        
        return  (positional_cost * TRACKER_SCORE_MIX_RATIO) + (similarity_cost * (1.0 - TRACKER_SCORE_MIX_RATIO)) ;
    }

    //Get Rekt :P
    //Returns estimated bounding box
    Rect getRect()
    {
        Point2f last_position(filter.statePost.at<float>(0),filter.statePost.at<float>(1));
        return Rect(last_position.x - scale_x/2,last_position.y - scale_y/2,scale_x,scale_y);
    }

    //Returns the estimated velocity of the target
    Point2f getVelocity()
    {
        return Point2f(filter.statePost.at<float>(2),filter.statePost.at<float>(3));
    }

private:
    //initializeKalmanFilter
    void initializeKalmanFilter(Rect2f initial_bounding_box)
    {
        //Create Kalman Filter object
        filter = KalmanFilter(4,2,0);
        filter.transitionMatrix = (Mat_<float>(4, 4) << 1.0,0.0,1.0,0.0,   0.0,1.0,0.0,1.0,  0.0,0.0,1.0,0.0,  0.0,0.0,0.0,1.0);


        // Init Kalman Filter
        Point2f initial_position = toPoint(initial_bounding_box);
        filter.statePre.at<float>(0) = initial_position.x;
        filter.statePre.at<float>(1) = initial_position.y;
        filter.statePre.at<float>(2) = 0;
        filter.statePre.at<float>(3) = 0;

        filter.statePost.at<float>(0) = initial_position.x;
        filter.statePost.at<float>(1) = initial_position.y;
        filter.statePost.at<float>(2) = 0;
        filter.statePost.at<float>(3) = 0;
        setIdentity(filter.measurementMatrix);
        setIdentity(filter.processNoiseCov, Scalar::all(1e-4));
        setIdentity(filter.measurementNoiseCov, Scalar::all(1e-1));
        setIdentity(filter.errorCovPost, Scalar::all(.1));
    }
    
    
    //Convert detected rectangle into point (center of the rectangle)
    Point2f toPoint(const Rect &rect) 
    {
        return Point2f((rect.x + (double)rect.width/2), (rect.y + (double)rect.height/2));
    }
};

//Tracks and manages multiple Tracker objects
class MultiTracker
{
public:
    vector<Tracker> trackers;
    int frame_width;
    int frame_height;

    MultiTracker(int width,int height)
    {
        frame_width = width;
        frame_height = height;
    }

    
    //Predicts the future state of trackers
    //Must be called before update in each frame
    void predict(vector<Rect> &predicted_trackers)
    {
        bool detected;
        //For each trackers
        //Perform prediction
        for (Tracker &t : trackers)
        {
            detected = true;
            t.predict();
            if ((t.state == Tracker::ACTIVE || t.state == Tracker::INACTIVE) && detected)
                predicted_trackers.push_back(t.getRect());
        }
    }
    
    //Updates the model using detections
    void update(const vector<Rect> &detections,const vector<Mat> &features)
    {
        int tracker_size = trackers.size();
        int detection_size = detections.size();
        bool* is_tracker_updated = NULL;
        bool* is_detection_used = NULL;
        int trackers_updated = 0;
        int detections_used = 0;

        is_detection_used = new bool[detection_size];

        for (int i = 0 ; i < detection_size; i++)
            is_detection_used[i] = false;

        //Associate && update trackers using detections
        //Greedy association:
        // - Find the tracker-detection pair with min cost
        // - Assign detection to tracker
        // - Remove tracker/detection from tracker/detection list
        // - Repeat this until there is no tracker or detection left or min score is greater than a threshold
        if (!trackers.empty() && !detections.empty())
        {
            is_tracker_updated = new bool[tracker_size];
            for (int i = 0 ; i < tracker_size; i++)
                is_tracker_updated[i] = false;

            while (trackers_updated < tracker_size && detections_used < detection_size)
            {
                double min_cost = std::numeric_limits<double>::max();
                
                int best_detection_index = -1;
                int best_tracker_index = -1;

                for (int tracker_index = 0; tracker_index < tracker_size ; tracker_index++)
                {
                    if (is_tracker_updated[tracker_index] == false)
                    {
                        for (int detection_index = 0 ; detection_index < detection_size ; detection_index++)
                        {
                            if (is_detection_used[detection_index] == false)
                            {
                                double cost = trackers[tracker_index].cost(detections[detection_index],features[detection_index]);
                                
                                if (cost < min_cost)
                                {
                                    best_detection_index = detection_index;
                                    best_tracker_index = tracker_index;
                                    min_cost = cost;
                                }
                            }
                        }
                    }
                } // End for tracker_index

                if ( best_detection_index == -1 || best_tracker_index == -1)
                    break;

                //Association threshold
                if (min_cost > MULTI_TRACKER_ASSOCIATION_THRESHOLD)
                    break;

                //Update tracker
                trackers[best_tracker_index].correct(detections[best_detection_index],features[best_detection_index]);
                is_detection_used[best_detection_index] = true;
                is_tracker_updated[best_tracker_index] = true;
                trackers_updated++;
                detections_used++;
            }
        }

        //If there are extra detections not associated. Create new trackers
        if (detection_size > 0 && detections_used < detection_size)
        {
            for (int i = 0 ; i < detection_size ; i++)
            {
                Rect d = detections[i];
                if (is_detection_used[i] == false)
                {
                    trackers.push_back(Tracker(detections[i],frame_width,frame_height,features[i]));
                }
            }
        }
        
        //Remove trackers
        trackers.erase(remove_if(begin(trackers), end(trackers), [](Tracker &t) {
            return t.state == Tracker::DEAD;
        }), end(trackers));

        if (is_tracker_updated != NULL)
            delete[] is_tracker_updated;

        if (is_detection_used != NULL)
            delete[] is_detection_used;
    }
    
    //Draws position of the detected objects.
    void draw(Mat &img,bool drawInactive = false)
    {
        Rect window(0,0,img.cols,img.rows);
        for (int i = 0 ; i < trackers.size() ; i++)
        {
            Rect p = trackers[i].getRect();
            p = p & window;
            
            if (p.x >= 0 && p.x+p.width <= img.cols &&
                p.y >= 0 && p.y + p.height <= img.rows)
            {
                if (trackers[i].state == Tracker::ACTIVE)
                {
                    Point center = Point(p.x+p.width/2,p.y+p.height/2);
                    Point2f velocity = trackers[i].getVelocity();

                    line(img,center,center + Point(velocity.x,velocity.y), Scalar(255,0,0),7);
                    rectangle(img, p, Scalar(255,0,0),5);
                }
                else
                {
                    if (drawInactive)
                    {
                        
                        Scalar color;
                        if (trackers[i].state == Tracker::INACTIVE)
                        {
                            color = Scalar(0,0,255);
                        }
                        else
                        {
                            color = Scalar(0,255,0);
                        }
                        Point center = Point(p.x+p.width/2,p.y+p.height/2);
                        Point2f velocity = trackers[i].getVelocity();

                        line(img,center,center + Point(velocity.x,velocity.y), Scalar(0,0,255),5);
                        rectangle(img, p, color,2);
                    }
                }
            }
        }
    }

    void printTrackers(ofstream& file,const Mat &img)
    {
        for (int i = 0 ; i < trackers.size() ; i++)
        {
            Rect p = trackers[i].getRect();
            if (p.x > 0 && p.x+p.width < img.cols &&
                p.y > 0 && p.y + p.height < img.rows &&
                trackers[i].state == Tracker::ACTIVE)
            {
                
                Point center = Point(p.x+p.width/2,p.y+p.height/2);
                Point2f velocity = trackers[i].getVelocity();

                Point2f scale = Vec2f(trackers[i].scale_x,trackers[i].scale_y);

                int tracker_id = trackers[i].id;
                file << tracker_id << "," << center.x << "," << center.y << "," << velocity.x << "," << velocity.y << "," << scale.x << "," << scale.y << ",";
            }
        }
    }
};

//Extracts histogram from image
void extractFeatures(const Mat& img,Mat& features)
{
    Mat lab;
    cvtColor(img,lab,COLOR_BGR2Lab);

    int dims = 1;
    const int sizes[] = {HISTOGRAM_SIZE};
    const int channels[] = { 0 , 1 , 2};
    float range[] = {0,255};
    const float *ranges[] = {range};

    Mat hist[3];
    calcHist(&lab, 1, channels, Mat(), hist[0], dims, sizes, ranges);
    calcHist(&lab, 1, channels + 1, Mat(), hist[1], dims, sizes, ranges);
    calcHist(&lab, 1, channels + 2, Mat(), hist[2], dims, sizes, ranges);
    
    vconcat(hist[0], hist[1], features);
    vconcat(features, hist[2], features);
    
    transpose(features,features);
    
    normalize(features, features, 1, 0, NORM_L2);
}

/******************************************************************************************************************/

/**
 * Finds the contours on given image
 * 
 * 
 */
void findContours(Mat &image,Mat &mask,vector< vector<Point> > &contours )
{
    //Perform some filtering to remove noises
    Mat kernel = getStructuringElement(MORPH_ELLIPSE,Size(11,11));
    morphologyEx(mask, mask,MORPH_CLOSE, kernel);
    morphologyEx(mask, mask,MORPH_CLOSE, kernel);

    //Find initial contour
    vector<Vec4i> hierarchy;
    findContours( mask, contours, hierarchy, CV_RETR_TREE , CHAIN_APPROX_SIMPLE, Point(0, 0) );

    //If contour area is small ignore it
    contours.erase(remove_if(begin(contours), end(contours), [](vector<Point> &i) {
        return contourArea(i) < 100;
    }), end(contours));

    //Perform contour operation again to refine contours
    mask.setTo(0);
    for( size_t i = 0; i < contours.size(); i++ )
    {
        drawContours( mask, contours, (int)i, Scalar(255,255,255), -1, 8, hierarchy, 0, Point() );
    }
    
    morphologyEx(mask, mask,MORPH_OPEN, kernel);
    morphologyEx(mask, mask,MORPH_OPEN, kernel);
    contours.clear();
    findContours( mask, contours, hierarchy, CV_RETR_EXTERNAL , CHAIN_APPROX_SIMPLE, Point(0, 0) );
}

// VideoStabilizer class
// Which uses the given reference frame for stabilization of the rest
class VideoStabilizer
{
public:
    VideoStabilizer(const Mat &ref)
    {
        reference_frame_BGR = ref.clone();
	//The reason for BGR: OpenCV stores them in the order of Blue, Green and Red
	//in order to work correctly, the input color type must be grayscale
        cvtColor(reference_frame_BGR,reference_frame,CV_BGR2GRAY);
    }

    //Stabilizes given image wrt given reference image
    Mat stabilize(const Mat &src,Mat &dst)
    {
	
        massert(src.type() == CV_8U || src.type() == CV_8UC1 , "SRC must be a grayscale image");

        //Initialize new points to tracked points
        if (tracked_points.size() < MIN_TRACKED_POINTS)
        {
            initializePoints();
        }

        //Find corresponding points in src frame
        vector<Point2f> points;
        lkTrack(reference_frame,src,points);

        massert(points.size() > 0 , "No points tracked");

        //Find the correspondence between reference frame and current frame
        //Correspondence is a 3x3 Matrix
        Mat homo = findHomography(points,tracked_points,RANSAC,10,noArray(),2000,0.99999);

        //Remove outliers from trackedPoints
        vector<Point2f> warped_points;
        perspectiveTransform(points,warped_points,homo);

        for (int i = tracked_points.size() - 1 ; i >= 0 ; --i)
        {
            if (euclideanDist(warped_points[i] , tracked_points[i]) >= 1)
            {
                tracked_points.erase(tracked_points.begin() + i);
            }
        }

        warp(src,dst,homo);
        return homo;
    }

    /*
     * Warps the given image using homography given
     * 
     */
    void warp(const Mat &src,Mat &dst,const Mat &homo)
    {
        Mat temp;
        if (src.type() == CV_8U || src.type() == CV_8UC1)
        {
            temp = reference_frame.clone();
        }
        else
        {
            temp = reference_frame_BGR.clone();
        }

        Size s {reference_frame.size[1],reference_frame.size[0]};

        warpPerspective(src,temp,homo,s,INTER_LINEAR,BORDER_TRANSPARENT);
        
        dst = temp;
    }
private:
    /**
     * Finds corner points on reference_frame
     * 
     * Corner points are used to find correspondence between reference image and given image
     */
    void initializePoints()
    {
        //Termination criteria for the iterative programs is defined as TermCriteria (as the name suggests)
        TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
        vector<Point2f> new_points;

        //A method from OpenCV
        //Quality level: Best corner quality * quality level = quality threshold for other corners
        goodFeaturesToTrack ( reference_frame,
                              new_points,
                              MAX_TRACKED_POINTS,//Max Corners
                              0.001, //Quality level
                              20// Min distance between found points
                            );
        //Refines the corner locations
        cornerSubPix(reference_frame, new_points, Size(10,10), Size(-1,-1), termcrit);

        tracked_points = new_points;
    }

    /*
     * Finds the location of "corner" points on given image
     * 
     * Lucas - Kanade tracking method is used to find the point correspondences
     * The algorithm tracks interest points between two frames.
     * 
     * Back validation is used to filter the noises
     */
    
    //Size of the pyramid as the search window
    Size WIN_SIZE = Size(11,11);
    void lkTrack(const Mat &src,const Mat &dst,vector<Point2f> &dst_corners)
    {
        vector<uchar> status1;
        vector<float> err1;

        //This method seems to be the bone of the algortihm,
        calcOpticalFlowPyrLK( src,
                              dst,
                              tracked_points,
                              dst_corners,
                              status1, //If the flow of the corresponding features is found between the images
                              err1, //The flow ?
                              WIN_SIZE, //winSize
                              3  //max level
                            );

        //Back validation
        //Now calculate the error between frames in reverse order
        vector<uchar> status2;
        vector<float> err2;
        vector<Point2f> src_corners;
        calcOpticalFlowPyrLK( dst,
                              src,
                              dst_corners,
                              src_corners,
                              status2,
                              err2,
                              WIN_SIZE, //winSize
                              3  //max level
                            );

        //For each tracked point
        for (int i = tracked_points.size() - 1 ; i > 0  ; i--)
        {
            //If the feature is not found on the other image or the point location
            //changes more than the threshold upon backtracking them to previous frame
            if (status1[i] != 1 || status2[i] != 1  ||
                    euclideanDist(tracked_points[i],src_corners[i]) >= 0.5
               )
            {
                //Remove the feature
                dst_corners.erase(dst_corners.begin() + i);
                tracked_points.erase(tracked_points.begin() + i);
            }
        }
    }

    Mat reference_frame;
    Mat reference_frame_BGR;

    vector<Point2f> tracked_points;

    static constexpr int MIN_TRACKED_POINTS = 300;
    static constexpr int MAX_TRACKED_POINTS = 500;
};

//Performs hog detection on given frame
//foreground_bounding_boxes is used to filter the background noises
void detect(Mat &frame,vector<Rect> &foreground_bounding_boxes , HOGDescriptor &hog,vector<Rect> &detections)
{
    Rect frameWindow(0,0,frame.cols,frame.rows);
    //For each foreground bounding boxes:
    // - Perform detection 
    // - Filter detections if one detection is inside the other
    for (auto &bb : foreground_bounding_boxes)
    {
        bb = Rect(bb.tl() - Point(10,10),bb.br() + Point(10,10)) & frameWindow;
        Rect window(0,0,bb.width * DETECTION_SCALE,bb.height * DETECTION_SCALE);
        Mat resized_frame;

        resize(frame(bb).clone(),resized_frame,Size(window.width,window.height));

        std::vector<cv::Rect> rect , rect2;

        Size win_size = hog.winSize;
        if (window.width > win_size.width && window.height > win_size.height)
        {
			//TODO: To be changed to tensor
            hog.detectMultiScale(resized_frame,rect,0,Size(),Size(),1.1,1,false); //Find pedestrians in image

            //if (bb.width > win_size.width && bb.height > win_size.height)
              //  hog.detectMultiScale(frame(bb).clone(),rect2,0,Size(),Size(),1.1,1,false); //Find pedestrians in image
        }
        
        //Detection filtering
        vector<Rect> filtered_rect;
        for (int i = 0 ; i < rect.size() ; i++)
        {
            rect[i] = rect[i] & window;
            
            rect[i] = Rect( bb.x + ( rect[i].x / DETECTION_SCALE ) ,
                            bb.y + ( rect[i].y / DETECTION_SCALE ) ,
                            rect[i].width / DETECTION_SCALE,
                            rect[i].height / DETECTION_SCALE);
        }
        
        for (int i = 0 ; i < rect2.size() ; i++)
        {
            rect.push_back(rect2[i]);
        }

        for (int i = 0; i < rect.size() ; i++)
        {
            bool contained = false;
            for (int j = 0; j < rect.size() ; j++)
            {
                if (rect[j].contains(rect[i].tl()) && rect[j].contains(rect[i].br()))
                {
                    contained = true;
                }
            }
            if (!contained)
                filtered_rect.push_back(rect[i]);
        }
        for (int i = 0 ; i < filtered_rect.size() ; i++)
        {
            detections.push_back(filtered_rect[i]);
        }

    }  
    
    vector<Rect> detections2;
    for (int i = 0; i < detections.size() ; i++)
    {
        bool contained = false;
        for (int j = 0; j < detections.size() ; j++)
        {
            if (detections[j].contains(detections[i].tl()) && detections[j].contains(detections[i].br()))
            {
                contained = true;
            }
						if (detections[j].area() > detections[i].area() && (detections[j] & detections[i]).area() / detections[i].area() > 0.8)
						{
								contained = true;
						}
        }
        if (!contained)
            detections2.push_back(detections[i]);
    }
    detections = detections2;
}


void createVideoWriter(VideoCapture& cap , VideoWriter &writer , string path)
{
    Size frame_size 
    {
        static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT))
    };
    
    writer = VideoWriter(path,cap.get(CAP_PROP_FOURCC), cap.get(CAP_PROP_FPS), frame_size);
}

class OutputStream
{
public:
    OutputStream(string path)
    {
        this->path = path;
    }
    
    virtual bool write(const Mat& data) = 0;
protected:
    string path;
};

class VideoOutputStream : public OutputStream
{
public:
    VideoOutputStream(string path , VideoCapture& cap) : OutputStream(path)
    {
        createVideoWriter(cap , writer , path);
    }
    
    bool write(const Mat& img) override
    {
        writer.write(img);
        return true;
    }
private:
    VideoWriter writer;
};

class ImageSequenceOutputStream : public OutputStream
{
public:
    ImageSequenceOutputStream(string path , int frame_number = 0 , string file_ext = "bmp") : OutputStream(path)
    {
        this->frame_number = frame_number;
        this->file_ext = file_ext;
    }
    
    bool write(const Mat& img) override
    {
        bool success = imwrite(path + to_string(frame_number) + "." + file_ext , img);
        frame_number++;
        return success;
    }
private:
    int frame_number;
    string file_ext;
};

int main(int argc,char* argv[])
{
    //Argument parsing / validation
    if(!(argc == 2 ||  argc == 3))
    {
        cout << "./Tracker <filename> [<video_save_path>]" << endl;
        return EXIT_FAILURE;
    }
	
	// Depending on the save option, define the path for the output video to be saved
    bool save = (argc == 3);
    string video_path = argv[1];
    
    string video_save_path;
    if (save)
        video_save_path = argv[2];
    
    //Display properties
    if (SHOW_OUTPUT)
    {
        namedWindow("Display",CV_WINDOW_NORMAL);
        resizeWindow("Display", 640,480);

        namedWindow("BG" , CV_WINDOW_NORMAL);
        resizeWindow("BG" , 640 , 480);
    }
    Mat current_frame,gray_frame,bg_mask;

    //Get the first frame (for reference)
    //VideoCapture is an OpenCV class, which does what is says
    VideoCapture capture(video_path);

    if(!capture.isOpened())
        return -1;

    Mat reference_frame;
    capture >> reference_frame;

    //Stabilize the frame by finding the homography betweeen it and the reference frame. Then warp the former using the
    // 3x3 homography matrix
    VideoStabilizer stabilizer(reference_frame);
    //Uses OpenCV background subtraction, which relies on mixture of gaussians. Their weights determine their duration,
    //higher means pixel color stays longer therefore a background pixel.
    Ptr<BackgroundSubtractor> background_subtractor = createBackgroundSubtractorMOG2(200,32,false);
    
    //Use couple frames to fit an initial background model
    for (int i = 0 ; i < 5 ; i++)
    {
        capture >> current_frame;

        if (current_frame.empty())
            return EXIT_FAILURE;

        cvtColor(current_frame,gray_frame , CV_BGR2GRAY);

        if (USE_STABILIZATION)
        {
            Mat homo = stabilizer.stabilize(gray_frame,gray_frame);
            stabilizer.warp(current_frame,current_frame,homo);
        }
        //medianBlur(current_frame,current_frame,3);
        background_subtractor->apply(current_frame,bg_mask,0.9);
    }
    capture.release();

    capture = VideoCapture(video_path);
    
    //Initialize hog detector and multi tracker
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    MultiTracker multi_tracker(capture.get(CV_CAP_PROP_FRAME_WIDTH),capture.get(CV_CAP_PROP_FRAME_HEIGHT));
    
    //File output
    ofstream file;
  	file.open (OUTPUT_FILE_PATH);
	file << argv[1] << " " << capture.get(CV_CAP_PROP_FRAME_WIDTH) << " "
         << capture.get(CV_CAP_PROP_FRAME_HEIGHT) << " " 
         << FRAME_SCALE << " " << capture.get(CAP_PROP_FPS)
         << endl;
    
    capture.set(CAP_PROP_POS_FRAMES , START_FRAME);
    
    unique_ptr<OutputStream> videoOutput;
    
    if (save)
    {
        if (VIDEO_OUTPUT)
        {
            videoOutput = unique_ptr<VideoOutputStream>(new VideoOutputStream(video_save_path , capture));
        }
        else
        {
            videoOutput = unique_ptr<ImageSequenceOutputStream>(new ImageSequenceOutputStream(video_save_path , START_FRAME , SAVE_FORMAT));
        }
    }
    
    for (int frame = START_FRAME ; frame < ( START_FRAME + NUM_FRAMES ) ; frame++ )
    {
        capture >> current_frame;

        if (current_frame.empty())
            break;

        cvtColor(current_frame,gray_frame , CV_BGR2GRAY);
        
        //Stabilization
        if (USE_STABILIZATION)
        {
            Mat homo = stabilizer.stabilize(gray_frame,gray_frame);
            stabilizer.warp(current_frame,current_frame,homo);
        }
        
        //medianBlur(current_frame,current_frame,3);
        //Find foreground contours
        background_subtractor->apply(current_frame,bg_mask);
        
        vector<Rect> predicted_trackers;
        multi_tracker.predict(predicted_trackers);
        for (Rect r : predicted_trackers)
        {
            rectangle(bg_mask,r,Scalar(255),1);
        }
        
        vector< vector<Point> > contours;
        findContours(current_frame,bg_mask,contours);

        vector<Rect> contour_bounding_boxes;
        for (auto &con : contours)
        {
            contour_bounding_boxes.push_back(boundingRect(con));
        }

				vector<Rect> fbb;
        {
            vector<Rect>::iterator it = contour_bounding_boxes.begin();
            while (it != contour_bounding_boxes.end())
            {
                Rect bb = *(it);
                it = remove_if( it,contour_bounding_boxes.end() ,
                    [&bb](auto r)
                    {
                        Rect rr = bb;
                        rr.x -= 5;
                        rr.y -= 5;
                        rr.width += 10;
                        rr.height += 10;
                        if ((r & rr).area() > 0)
                        {
                            bb = r | bb;
                            return false;
                        }
                        return true;
                    });
                
                if (it != contour_bounding_boxes.end())
                    fbb.push_back(bb);
            }
        }
        contour_bounding_boxes = fbb;

        //Pedestrian detection
        vector<Rect> detections;
        detect(current_frame,contour_bounding_boxes, hog,detections);
        
        //Feature extraction
        vector<Mat> features(detections.size());
        for (int i = 0 ; i < detections.size() ; i++)
        {
            extractFeatures(current_frame(detections[i]),features[i]);
        }
        
        //Track pedestrians
        multi_tracker.update(detections,features);

        //Print trackers to the file
        file << (capture.get(CV_CAP_PROP_POS_FRAMES)-1) << ",";
		multi_tracker.printTrackers(file,current_frame);
		file << endl;

        if (save)
        {   
            bool success = videoOutput->write(current_frame);
            massert(success , "Could not able to save output frame.");
        }
        
        //Draw data
        Mat draw_layer = current_frame.clone(); //Draw layer (used to draw things on screen)
        
        if (SHOW_OUTPUT)
        {
            if (DRAW_TRACKERS)
                multi_tracker.draw(draw_layer,true);
            
            addWeighted(current_frame,0.3,draw_layer,0.7,0,draw_layer);
            
            imshow("BG",bg_mask);
            imshow("Display", draw_layer);
        
            int key = waitKey(1);   
            if (key == 27)
            {
                break;
            }
            cout << frame << endl;
        }
        

    }
    
    file.close();

    destroyAllWindows();
    return 0;
}
