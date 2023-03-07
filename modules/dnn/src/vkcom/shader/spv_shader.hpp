// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SPV_SHADER_HPP
#define OPENCV_DNN_SPV_SHADER_HPP


namespace cv { namespace dnn { namespace vkcom {

extern const unsigned int gemm_v10_spv[1528];
extern const unsigned int max_pool_spv[1449];
extern const unsigned int lrn_spv[1845];
extern const unsigned int relu_spv[502];
extern const unsigned int prior_box_spv[1480];
extern const unsigned int gemm_v11_spv[2884];
extern const unsigned int conv48_nobias_spv[7182];
extern const unsigned int gemm_v6_spv[1532];
extern const unsigned int permute_spv[765];
extern const unsigned int conv48_spv[7458];
extern const unsigned int gemm_spv[753];
extern const unsigned int conv_spv[1904];
extern const unsigned int softmax_spv[1496];
extern const unsigned int gemm_v7_spv[3114];
extern const unsigned int conv_4x4_spv[3000];
extern const unsigned int gemm_v4_spv[2580];
extern const unsigned int conv_in_repack_spv[2227];
extern const unsigned int gemm_v8_spv[2177];
extern const unsigned int moo_test2_spv[494];
extern const unsigned int gemm_v9_spv[1552];
extern const unsigned int moo_test_spv[1066];
extern const unsigned int gemm_v5_spv[349];
extern const unsigned int dw_conv_spv[1762];
extern const unsigned int concat_spv[541];
extern const unsigned int avg_pool_spv[1538];
extern const unsigned int gemm_v2_spv[1478];
extern const unsigned int gemm_v3_spv[2197];

extern std::map<std::string, std::pair<const unsigned int *, size_t> > SPVMaps;

void initSPVMaps();

}}} // namespace cv::dnn::vkcom

#endif /* OPENCV_DNN_SPV_SHADER_HPP */
