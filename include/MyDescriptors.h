/*
Intensity Order based Local Features
https://github.com/foelin/IntensityOrderFeature

Reference:
[1] Zhenhua Wang, Bin Fan and Fuchao Wu, “Local Intensity Order Pattern for Feature Description”,
IEEE International Conference on Computer Vision (ICCV) , Nov. 2011
[2] Zhenhua Wang, Bin Fan, Gang Wang and Fuchao Wu, “Exploring Local and Overall Ordinal Information for Robust Feature Description”,
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), Dec. 2016.

Contact: zhwang.me@gmail.com

This is a free software.
You can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
You should have received a copy of the GNU General Public License
along with this software.  If not, see <http://www.gnu.org/licenses/>.
*/



#ifndef _MY_DESCRIPTORS_H
#define _MY_DESCRIPTORS_H

#include "Common.h"
#include "opencv2/opencv.hpp"
#include <fstream>


class MyDescriptors
{
public:
	MyDescriptors(Params& params);
	virtual ~MyDescriptors(void);

	virtual int descriptorSize() const;
	virtual int descriptorType() const;


    void compute(const cv::Mat& image, CV_OUT CV_IN_OUT std::vector<AffineKeyPoint>& keypoints, CV_OUT cv::Mat& descriptors) const;

    void computePatchImage(const cv::Mat& image, int patch_per_row, int patch_per_col, int patch_length, int max_patch_num,  cv::Mat& descriptors) const;

protected:

    virtual void computeImpl(const cv::Mat& image, std::vector<AffineKeyPoint>& keypoints, cv::Mat& descriptors) const;


    void createLIOP		(const cv::Mat& outPatch, const cv::Mat& flagPatch, int inPatchSz, float* des) const;;
    void createOIOP		(const cv::Mat& outPatch, const cv::Mat& flagPatch, int inPatchSz, float* des) const;
    void createMIOP		(const cv::Mat& outPatch, const cv::Mat& flagPatch, int inPatchSz, float* des) const;
    void createMIOP_FAST(const cv::Mat& outPatch, const cv::Mat& flagPatch, int inPatchSz, float* des) const;

    void removeOutBound(const cv::Mat& image, const std::vector<AffineKeyPoint>& keypoints,  std::vector<AffineKeyPoint>& keypointsInBounds) const;
    bool readPCA(const std::string& file);


	
private:
	int m_dim;
	int m_bytes;
	int m_dataType;
	Params& m_params;
	bool m_computeDomiOri;

    cv::Mat m_PCABasis;	//each row vector is a orthon basis
    cv::Mat m_PCAMean;	//a row vector
	const float* m_fenceRatio;


    void (MyDescriptors::*m_ptrCreateFeatFunc)(const cv::Mat& outPatch, const cv::Mat& flagPatch, int inPatchSz, float* des) const;
};



#endif
