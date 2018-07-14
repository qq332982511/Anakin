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

typedef  TargetWrapper<BM, __device_target>BM_API;


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
        
void BM_API::mem_alloc(TPtr* ptr, size_t n){
    handle = get_bm_handle();
    /* bm_device_mem_t *mem = reinterpret_cast<struct bm_mem_desc *>(*ptr); */
//    bm_device_mem_t *mem = new bm_device_mem_t();
    bm_device_mem_t mem;
    BMDNN_CHECK(bm_malloc_device_byte(handle, &mem, n));
    *ptr = TPtr(mem);
}
        
void BM_API::mem_free(TPtr ptr){
    if((ptr != BM_MEM_NULL)){
        handle = get_bm_handle();
        bm_free_device(handle, ptr);
//        delete ptr;
    }
}
        
void BM_API::mem_set(TPtr ptr, int value, size_t n){
    //(bm_handle_t handle, const int value, bm_device_mem_t mem){
    BMDNN_CHECK(bm_memset_device(handle, value, ptr));
    //bm_device_mem_t* pmem = (struct bm_mem_desc *)(ptr);
    //BMDNN_CHECK(bm_memset_device(handle, value, *pmem));
}

void BM_API::sync_memcpy(TPtr dst, int dst_id, const TPtr src, int src_id, \
    size_t count, __DtoD) {
    handle = get_bm_handle(); 
    //BMDNN_CHECK(bm_memcpy_d2d(handle, bm_mem_from_device(dst), dst_id, bm_mem_from_device(src), src_id, count));
    BMDNN_CHECK(bm_memcpy_d2d(handle, dst, dst_id, src, src_id, count));
    LOG(INFO) << "BM sync_memcpy: device to device, finished";
};

void BM_API::sync_memcpy(TPtr dst, int dst_id, const void* src, int src_id, \
    size_t count, __HtoD) {
    handle = get_bm_handle(); 
    BMDNN_CHECK(bm_memcpy_s2d(handle, dst, bm_mem_from_system(src)));

    #ifdef DEBUG
    for(int i=0; i<10; i++)
	    LOG(INFO) << "HtoD src: " << *((float *)(src)+i);
    #endif
    
    LOG(INFO) << "BM sync_memcpy: host to device, finished";
};

void BM_API::sync_memcpy(void* dst, int dst_id, const TPtr src, int src_id, \
    size_t count, __DtoH) {
    handle = get_bm_handle(); 
    BMDNN_CHECK(bm_memcpy_d2s(handle, bm_mem_from_system(dst), src));

    #ifdef DEBUG
    for(int i=0; i<10; i++)
        LOG(INFO) << "DtoH dst: " << *((float *)(dst)+i);
    #endif

    LOG(INFO) << "BM sync_memcpy: device to host, finished";
};

void BM_API::sync_memcpy_p2p(TPtr dst, int dst_dev, const TPtr src, \
    int src_dev, size_t count) { 

    LOG(INFO) << "BM sync_memcpy_p2p: temporarily no used";
};




//! BM Buffer
template class Buffer<BM>;

//! BM Tensor



/**
 * \brief Constructor with allocated data ptr and entire memory shape. only for BM
*/
template <>
template <typename TargetType_t>
Tensor<BM,AK_FLOAT, NCHW>::Tensor(Dtype*  data_ptr, TargetType_t target, int id, Shape shape) {
    CHECK_EQ(shape.dims(), TensorAPI::layout_dims::value) << \
        "shape dims is not matched to layout type";
    _shape = shape;
    _valid_shape = shape;
    _offset = Shape::zero(shape.dims());

    std::shared_ptr<Buffer<TargetType_t>> buf_from_date = \
        std::make_shared<Buffer<TargetType_t>>(&bm_mem_from_system(const_cast<void*>(data_ptr)), shape.count() * _type_len, id);

    BufferMemShare(_buf, buf_from_date);

    _is_subbuf = false;
}
INSTANTIATE_TENSOR(BM, AK_FLOAT, NCHW);




template struct Env<BM>;

#endif //USE_BM

} //namespace saber

} //namespace anakin
