#include "core/tensor.h"
#include "env.h"

#include "bmlib_runtime.h"
#include "bmdnn_api.h"
#include "bmlib_utils.h"

#ifdef USE_BM
const char* bmdnn_get_errorstring(bm_status_t error) {
    switch (error) {
        case BM_SUCCESS:
            return "BM API call correct";
        case BM_ERR_FAILURE:
            return "BM API fail to return";
        case BM_ERR_TIMEOUT:
            return "BM API time out";
        case BM_ERR_PARAM:
            return "BM API invalid parameter";
        case BM_ERR_NOMEM:
            return "BM API insufficient memory";
        case BM_ERR_DATA:
            return "BM API invalid data";
        case BM_ERR_BUSY:
            return "BM device is busy";
        case BM_NOT_SUPPORTED:
            return "BM unsupported operate";
    }
    return "Unknown bmdnn status";
}
#endif

namespace anakin{

namespace saber{

#ifdef USE_BM

typedef TargetWrapper<BM, __device_target> BM_API;

// Init handle only once in the lifetime
static bm_handle_t handle;
static bm_status_t init_handle{bmdnn_init(&handle)};

void BM_API::get_device_count(int &count) {
    BMDNN_CHECK(bm_dev_getcount(&count));
}

void BM_API::set_device(int id){
    //(bm_handle_t &handle, bool bmkernel_used, int id){
    //BMDNN_CHECK(bm_dev_request(&handle, 0, id));
}

//TODO: Do we have this functionality?
int BM_API::get_device_id(){
    return 0;
}
        
void BM_API::mem_alloc(void** ptr, size_t n){
    handle = get_bm_handle();
    /* bm_device_mem_t *mem = reinterpret_cast<struct bm_mem_desc *>(*ptr); */
    bm_device_mem_t *mem = new bm_device_mem_t();
    BMDNN_CHECK(bm_malloc_device_byte(handle, mem, n));
    *ptr = mem;
}
        
void BM_API::mem_free(void* ptr){
    if(ptr != nullptr){
        handle = get_bm_handle();
        bm_free_device(handle, *(struct bm_mem_desc*)(ptr));
        delete ptr;
    }
}
        
void BM_API::mem_set(void* ptr, int value, size_t n){
    //(bm_handle_t handle, const int value, bm_device_mem_t mem){
    BMDNN_CHECK(bm_memset_device(handle, value, bm_mem_from_system(ptr)));
    //bm_device_mem_t* pmem = (struct bm_mem_desc *)(ptr);
    //BMDNN_CHECK(bm_memset_device(handle, value, *pmem));
}

void BM_API::sync_memcpy(void* dst, int dst_id, const void* src, int src_id, \
    size_t count, __DtoD) {
    handle = get_bm_handle(); 
    //BMDNN_CHECK(bm_memcpy_d2d(handle, bm_mem_from_device(dst), dst_id, bm_mem_from_device(src), src_id, count));
    BMDNN_CHECK(bm_memcpy_d2d(handle, *(bm_device_mem_t *)(dst), dst_id, *(bm_device_mem_t *)(src), src_id, count));
    LOG(INFO) << "BM sync_memcpy: device to device, finished";
};

void BM_API::sync_memcpy(void* dst, int dst_id, const void* src, int src_id, \
    size_t count, __HtoD) {
    handle = get_bm_handle(); 
    BMDNN_CHECK(bm_memcpy_s2d(handle, *(bm_device_mem_t *)(dst), bm_mem_from_system(src)));

    #ifdef DEBUG
    for(int i=0; i<10; i++)
	    LOG(INFO) << "HtoD src: " << *((float *)(src)+i);
    #endif
    
    LOG(INFO) << "BM sync_memcpy: host to device, finished";
};

void BM_API::sync_memcpy(void* dst, int dst_id, const void* src, int src_id, \
    size_t count, __DtoH) {
    handle = get_bm_handle(); 
    BMDNN_CHECK(bm_memcpy_d2s(handle, bm_mem_from_system(dst), *(bm_device_mem_t *)(src)));

    #ifdef DEBUG
    for(int i=0; i<10; i++)
        LOG(INFO) << "DtoH dst: " << *((float *)(dst)+i);
    #endif

    LOG(INFO) << "BM sync_memcpy: device to host, finished";
};

void BM_API::sync_memcpy_p2p(void* dst, int dst_dev, const void* src, \
    int src_dev, size_t count) { 

    LOG(INFO) << "BM sync_memcpy_p2p: temporarily no used";
};


//! target wrapper
template struct TargetWrapper<BM, __device_target>;

//! BM Buffer
template class Buffer<BM>;

//! BM Tensor
INSTANTIATE_TENSOR(BM, AK_BM, NCHW);

#ifdef USE_BM
/**
 * \brief Constructor with allocated data ptr and entire memory shape. only for BM
*/
template <>
Tensor<BM,AK_BM, NCHW>::Tensor(Dtype_p data_ptr, TargetType_t target, int id, Shape shape) {
    CHECK_EQ(shape.dims(), TensorAPI::layout_dims::value) << \
        "shape dims is not matched to layout type";
    _shape = shape;
    _valid_shape = shape;
    _offset = Shape::zero(shape.dims());

    std::shared_ptr<Buffer<TargetType_t>> buf_from_date = \
        std::make_shared<Buffer<TargetType_t>>(&bm_mem_from_system(const_cast<Dtype_p>(data_ptr)), shape.count() * _type_len, id);

    BufferMemShare(_buf, buf_from_date);

    _is_subbuf = false;
}
#endif

#ifdef USE_BM

#ifndef BM_TENSOR_COPY
#define BM_TENSOR_COPY

template<>
template<> inline
SaberStatus Tensor<BM, AK_BM, NCHW>::copy_from<X86, AK_FLOAT, NCHW>(const Tensor<X86, AK_FLOAT, NCHW>& tensor) {
            LOG(INFO) << "BM copy_from X86";
            CHECK_EQ(valid_size(), tensor.valid_size()) << "sizes of two valid shapes must be the same";

    auto* device_data_ptr = mutable_data();
    BMDNN_CHECK(bm_memcpy_s2d(get_bm_handle(), *device_data_ptr, bm_mem_from_system(const_cast<float *>(tensor.data()))));
    return SaberSuccess;
}

template<>
template<> inline
SaberStatus Tensor<X86, AK_FLOAT, NCHW>::copy_from<BM, AK_BM, NCHW>(const Tensor<BM, AK_BM, NCHW>& tensor) {
            LOG(INFO) << "X86 copy_from BM";
            CHECK_EQ(valid_size(), tensor.valid_size()) << "sizes of two valid shapes must be the same";

    auto* device_data_ptr = const_cast<bm_device_mem_t *>(tensor.data());
    BMDNN_CHECK(bm_memcpy_d2s(get_bm_handle(), bm_mem_from_system(mutable_data()), *device_data_ptr));
    return SaberSuccess;
}

/*


    template<>
    template<> inline
    SaberStatus Tensor<BM, AK_BM, NCHW>::copy_from<X86, AK_FLOAT, NCHW>(const Tensor<X86, AK_FLOAT, NCHW>& tensor) {
        LOG(INFO) << "BM copy_from X86";
        CHECK_EQ(valid_size(), tensor.valid_size()) << "sizes of two valid shapes must be the same";

        auto* device_data_ptr = mutable_data();
        BMDNN_CHECK(bm_memcpy_s2d(get_bm_handle(), *device_data_ptr, bm_mem_from_system(const_cast<float *>(tensor.data()))));
        //BMDNN_CHECK(bm_memcpy_s2d(get_bm_handle(), *(bm_device_mem_t *)(mutable_data()), bm_mem_from_system(tensor.data())));
        return SaberSuccess;
    }

    template<>
    template<> inline
    SaberStatus Tensor<X86, AK_FLOAT, NCHW>::copy_from<BM, AK_BM, NCHW>(const Tensor<BM, AK_BM, NCHW>& tensor) {
        LOG(INFO) << "X86 copy_from BM";
        CHECK_EQ(valid_size(), tensor.valid_size()) << "sizes of two valid shapes must be the same";

        auto* device_data_ptr = const_cast<bm_device_mem_t *>(tensor.data());
        BMDNN_CHECK(bm_memcpy_d2s(get_bm_handle(), bm_mem_from_system(mutable_data()), *device_data_ptr));
        //BMDNN_CHECK(bm_memcpy_d2s(get_bm_handle(), bm_mem_from_system(mutable_data()), *(bm_device_mem_t *)(tensor.data())));
        return SaberSuccess;
    }

    template<>
    template<> inline
    SaberStatus Tensor<BM, AK_BM, NCHW>::copy_from<BM, AK_BM, NCHW>(const Tensor<BM, AK_BM, NCHW>& tensor) {
        LOG(INFO) << "BM copy_from BM";
        CHECK_EQ(valid_size(), tensor.valid_size()) << "sizes of two valid shapes must be the same";

        auto* device_data_ptr = const_cast<bm_device_mem_t *>(tensor.data());
        //BMDNN_CHECK(bm_memcpy_d2s(get_bm_handle(), bm_mem_from_system(mutable_data()), *device_data_ptr));
        //BMDNN_CHECK(bm_memcpy_d2s(get_bm_handle(), bm_mem_from_system(mutable_data()), *(bm_device_mem_t *)(tensor.data())));
        return SaberSuccess;
    }
*/

#endif

#endif

template struct Env<BM>;

#endif //USE_BM

} //namespace saber

} //namespace anakin
