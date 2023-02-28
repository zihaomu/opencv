// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "spv_shader.hpp"

namespace cv { namespace dnn { namespace vkcom {

std::map<std::string, std::pair<const unsigned int *, size_t> > SPVMaps;

void initSPVMaps()
{
    SPVMaps.insert(std::make_pair("dw_conv_spv", std::make_pair(dw_conv_spv, 1762)));
    SPVMaps.insert(std::make_pair("conv48_spv", std::make_pair(conv48_spv, 7458)));
    SPVMaps.insert(std::make_pair("avg_pool_spv", std::make_pair(avg_pool_spv, 1538)));
    SPVMaps.insert(std::make_pair("prior_box_spv", std::make_pair(prior_box_spv, 1480)));
    SPVMaps.insert(std::make_pair("moo_test_spv", std::make_pair(moo_test_spv, 1066)));
    SPVMaps.insert(std::make_pair("softmax_spv", std::make_pair(softmax_spv, 1496)));
    SPVMaps.insert(std::make_pair("lrn_spv", std::make_pair(lrn_spv, 1845)));
    SPVMaps.insert(std::make_pair("conv_in_repack_spv", std::make_pair(conv_in_repack_spv, 2227)));
    SPVMaps.insert(std::make_pair("max_pool_spv", std::make_pair(max_pool_spv, 1449)));
    SPVMaps.insert(std::make_pair("conv48_nobias_spv", std::make_pair(conv48_nobias_spv, 7182)));
    SPVMaps.insert(std::make_pair("conv_spv", std::make_pair(conv_spv, 1904)));
    SPVMaps.insert(std::make_pair("relu_spv", std::make_pair(relu_spv, 502)));
    SPVMaps.insert(std::make_pair("concat_spv", std::make_pair(concat_spv, 541)));
    SPVMaps.insert(std::make_pair("permute_spv", std::make_pair(permute_spv, 765)));
    SPVMaps.insert(std::make_pair("moo_test2_spv", std::make_pair(moo_test2_spv, 494)));
    SPVMaps.insert(std::make_pair("conv_4x4_spv", std::make_pair(conv_4x4_spv, 3000)));
}

}}} // namespace cv::dnn::vkcom
