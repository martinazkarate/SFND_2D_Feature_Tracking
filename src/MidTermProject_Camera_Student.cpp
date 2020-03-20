/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <cstdlib>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

struct perfStats {
  string detectorType;
  string descriptorType;
  string matcherType;
  string selectorType;
  int   numKeyPointsPerframe[10];
  int   numKeyPointsPerROI[10];
  int   numMatchedKeyPoints[10];
  double detectorTime[10];
  double descriptorTime[10];
  double matcherTime[10];
};

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    if (argc != 5)
    {
        cout << "Please provide as CLI arguments the detector, descriptor, matcher and selector types to be used. Exiting now. " << endl;
        return EXIT_FAILURE;
    }
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    // struct to hold performances for evalutation 
    perfStats performances;
    
    std::string filename = "../data.csv";
    std::ofstream output_stream(filename, std::ios::binary | std::ios::app);

    if (!output_stream.is_open()) {
        std::cerr << "failed to open file: " << filename << std::endl;
        return EXIT_FAILURE;
    }
  
    // write CSV header row
    output_stream << "Detector Type" << ","
                << "Descriptor Type" << ","
                << "Frame#" << ","
                << "#KeyPointsPerFrame" << ","
                << "#KeyPointsPerROI" << ","
                << "DetectorTime(ms)" << ","
                << "DescriptorTime(ms)" << ","
                << "Matcher Type" << ","
                << "Selector Type" << ","
                << "#MatchedPoints" << "," 
                << "MatchingTime(ms))" << std::endl;

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);
        if (dataBuffer.size() > dataBufferSize)
        {
            dataBuffer.erase(dataBuffer.begin());
        }

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = argv[1]; // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        performances.detectorType = detectorType;

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0)
        {
            performances.detectorTime[imgIndex] = detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if(detectorType.compare("HARRIS") == 0)
        {
            performances.detectorTime[imgIndex] = detKeypointsHarris(keypoints, imgGray, false);
        }
        else if (detectorType.compare("FAST") == 0 || detectorType.compare("BRISK") == 0 || detectorType.compare("ORB") == 0 || detectorType.compare("AKAZE") == 0 || detectorType.compare("SIFT") == 0)
        {
            performances.detectorTime[imgIndex] = detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
        else
        {
            cout << "The detector type is not implemented. Exiting now." << endl;
            return EXIT_FAILURE;
        }

        performances.numKeyPointsPerframe[imgIndex] = keypoints.size();
        
        
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            vector<cv::KeyPoint> keypoints_vehicle; // create empty feature list for keypoints within the rectangle
            for (auto it = keypoints.begin(); it !=keypoints.end(); it++)
            {
                if (vehicleRect.contains((*it).pt))
                    keypoints_vehicle.push_back(*it);
            }
            keypoints = keypoints_vehicle;
        }

        performances.numKeyPointsPerROI[imgIndex] = keypoints.size();

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT " << detectorType << " KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        string descriptorType = argv[2]; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
        performances.descriptorType = descriptorType;
        performances.descriptorTime[imgIndex] = descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        
        if (performances.descriptorTime[imgIndex] < 0)
        {
            cout << "Descriptor extraction failed." << endl;
            return EXIT_FAILURE;
        }          
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT " << descriptorType << " DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = argv[3];        // MAT_BF, MAT_FLANN
            performances.matcherType = matcherType;
            //string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = argv[4];       // SEL_NN, SEL_KNN
            performances.selectorType = selectorType;            

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            performances.matcherTime[imgIndex] = matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            performances.numMatchedKeyPoints[imgIndex] = matches.size();

            // Evaluation of quality of matches. 
            // 1.- Statistics of match distance metric
            vector<float> distances(matches.size());
            for (auto match = matches.begin(); match != matches.end(); match++)
            {
                distances.push_back((*match).distance);
            }
            // Mean of distances
            float sum = std::accumulate(distances.begin(), distances.end(), 0.0);
            float mean = sum / distances.size();

            // Standard Dev of distances
            vector<float> diff(distances.size());
            transform(distances.begin(), distances.end(), diff.begin(), [mean](float x) { return x - mean; });
            float sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
            float stdev = sqrt(sq_sum / distances.size());

            // 2.- Statistics of keypoints euclidean distance in image
            vector<float> distances2(matches.size());
            for (auto match = matches.begin(); match != matches.end(); match++)
            {
                cv::KeyPoint queryKpt = (dataBuffer.end() - 2)->keypoints[(*match).queryIdx];
                cv::KeyPoint trainKpt = (dataBuffer.end() - 1)->keypoints[(*match).trainIdx];
                distances2.push_back(norm(queryKpt.pt-trainKpt.pt));
            }
            // Mean of distances
            float sum2 = std::accumulate(distances2.begin(), distances2.end(), 0.0);
            float mean2 = sum2 / distances2.size();

            // Standard Dev of distances
            vector<float> diff2(distances2.size());
            transform(distances2.begin(), distances2.end(), diff2.begin(), [mean2](float x) { return x - mean2; });
            float sq_sum2 = inner_product(diff2.begin(), diff2.end(), diff2.begin(), 0.0);
            float stdev2 = sqrt(sq_sum2 / distances2.size());
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS WITH MATCHER " << matcherType << " AND SELECTOR " << selectorType << " done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis && imgIndex==imgEndIndex)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << endl;
                cout << "Press key to continue to next image" << endl;
                
                //Store matchImg on disk
                cv::Point2f top_left(50, 50);
                string overlay_text = "Detector: " + detectorType + ", Descriptor: " + descriptorType;
                cv::Scalar font_color(0, 0, 255);
                cv::putText(matchImg, overlay_text, top_left, cv::FONT_HERSHEY_COMPLEX, 1, font_color);
                cv::imwrite("../"+detectorType+"-"+descriptorType+"_output_image.png", img);
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }
        else
        {
            performances.matcherType = argv[3];
            performances.selectorType = argv[4];
            performances.matcherTime[imgIndex] = 0.0;
            performances.numMatchedKeyPoints[imgIndex] = 0;
            cout << endl;
        }
        

    } // eof loop over all images
    for (int i = 0; i < 10; i++) 
    {
        output_stream << performances.detectorType
                << "," << performances.descriptorType
                << "," << i
                << "," << performances.numKeyPointsPerframe[i]
                << "," << performances.numKeyPointsPerROI[i]
                << "," << std::fixed << std::setprecision(3) << performances.detectorTime[i]
                << "," << std::fixed << std::setprecision(3) << performances.descriptorTime[i]
                << "," << performances.matcherType
                << "," << performances.selectorType
                << "," << performances.numMatchedKeyPoints[i]
                << "," << std::fixed << std::setprecision(3) << performances.matcherTime[i] << std::endl;
    }
    output_stream << std::endl;
    output_stream.close();

    return 0;
}
