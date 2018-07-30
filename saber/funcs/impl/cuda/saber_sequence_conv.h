/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SEQUENCE_CONV_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SEQUENCE_CONV_H

#include "saber/funcs/impl/impl_sequence_conv.h"
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/cuda/base/sass_funcs.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
class SaberSequenceConv<NV, OpDtype> :
    public ImplBase <
    NV, OpDtype, SequenceConvParam<NV> > {
public:

    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;
    typedef Tensor<NV> OpTensor;

    SaberSequenceConv() = default;

    ~SaberSequenceConv() {}

    virtual SaberStatus init(const std::vector<OpTensor*>& inputs,
                             std::vector<OpTensor*>& outputs,
                             SequenceConvParam<NV>& param,
                             Context<NV>& ctx) {
        this->_ctx = &ctx;
        CHECK_EQ(param.padding_trainable, false) << "not support padding_trainable==true";
        CHECK_EQ(param.context_stride, 1) << "not support context_stride!=1";

        if (param.padding_tensor != nullptr) {
            CHECK_EQ(1, 0) << "not support padding_tensor";
        }

        CHECK_NOTNULL(param.filter_tensor);
        _hidden_size = param.filter_tensor->height() / param.context_length;
        _feature_size = param.filter_tensor->width();
        _up_pad = std::max(0, -param.context_start);
        _down_pad = std::max(0, param.context_start + param.context_length - 1);
        _hidden_kernel_size = _hidden_size * param.context_length;
        _word_start=std::max(0, param.context_start);
        return create(inputs, outputs, param, ctx);
    };

    virtual SaberStatus create(const std::vector<OpTensor*>& inputs,
                               std::vector<OpTensor*>& outputs,
                               SequenceConvParam<NV>& param,
                               Context<NV>& ctx) {

        int word_num=inputs[0]->num();
        _gemm_im2col = saber_find_fast_sass_gemm(false, false, word_num, _feature_size, _hidden_kernel_size);
        return SaberSuccess;
    };

    virtual SaberStatus dispatch(const std::vector<OpTensor*>& inputs,
                                 std::vector<OpTensor*>& outputs,
                                 SequenceConvParam<NV>& param);
private:
    OpTensor _temp_im2col_tensor;
    int _hidden_size;
    int _feature_size;
    int _hidden_kernel_size;
    int _up_pad;
    int _down_pad;
    int _word_start;

    std::function<void(const int, const int, const int,
                       const float, const float*, const float,
                       const float*, float*, cudaStream_t)> _gemm_im2col;
};

}
}

#endif