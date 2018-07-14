/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef ANAKIN_SABER_CORE_DATA_TRAITS_H
#define ANAKIN_SABER_CORE_DATA_TRAITS_H

#include "common.h"
//#include "iostream"
#ifdef USE_BM
#include "bmlib_runtime.h"
#include "bmdnn_api.h"
#include "bmlib_utils.h"
#endif

namespace anakin{

namespace saber{

template <typename Ttype, DataType datatype>
struct DataTrait{
    typedef __invalid_type Dtype;
    typedef __invalid_type PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_HALF> {
    typedef short Dtype;
    typedef short* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_FLOAT> {
    typedef float Dtype;
    typedef float* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_DOUBLE> {
    typedef double Dtype;
    typedef double* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_INT8> {
    typedef char Dtype;
    typedef char* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_INT16> {
    typedef short Dtype;
    typedef short* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_INT32> {
    typedef int Dtype;
    typedef int* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_INT64> {
    typedef long Dtype;
    typedef long* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_UINT8> {
    typedef unsigned char Dtype;
    typedef unsigned char* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_UINT16> {
    typedef unsigned short Dtype;
    typedef unsigned short* PtrDtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_UINT32> {
    typedef unsigned int Dtype;
    typedef unsigned int* PtrDtype;
};

template <typename Ttype>
struct PtrTrait {
    typedef void* PtrType;
};


#ifdef USE_BM


struct BM_mem_addr:bm_mem_desc{

    BM_mem_addr(){};

    BM_mem_addr(void *k){
        if(k== nullptr){
            *this=BM_MEM_NULL;
        }else{
            CHECK(false)<<"not suport construct not null ptr";
        }
    }



//    BM_mem_addr& operator=(bm_mem_desc& right) {
//        *this = right;
//        return *this;
//    }
//
    inline bool compare_char_array(const unsigned char* a,const unsigned char* b,int size)const{
        for(int i=0;i<size;++i){
            if(a[i]!=b[i]){
                return false;
            }
        }
        return true;
    }

    bool operator==(const bm_mem_desc& right) {
        return compare_char_array(desc , right.desc,sizeof(desc));
    }
    bool operator!=(const bm_mem_desc& right) {
        return !compare_char_array(desc , right.desc,sizeof(desc));
    }

//    bool operator==(const BM_mem_addr& right) {
//        return *this == right;
//    }
//
    bool operator==(const void* right) {
        if(right== nullptr){
            return *this == BM_MEM_NULL;
        }else{
            CHECK(false)<<"not suport compare not null BM_mem_addr with nullptr";
            return false;
        }
    }


    bool operator!=(const void* right) {
        return !(*this == right);
    }

    BM_mem_addr(struct bm_mem_desc init_desc):bm_mem_desc(init_desc){

        ;
    }

    BM_mem_addr& operator+(int offset){
        if(offset!=0) {
            unsigned long long target_addr=bm_mem_get_device_addr(*this);
            bm_mem_set_device_addr(*this, target_addr + offset*sizeof(float));
        }
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& out, const BM_mem_addr& s){
        out << " [print BM_mem_addr] ";
        return out;
    }

};

template <>
struct PtrTrait<BM> {
    typedef  BM_mem_addr PtrType;
    int WHAT_HAPPEND;
};

template <>
struct DataTrait<BM,AK_FLOAT> {
    typedef  float Dtype;
    typedef  BM_mem_addr PtrDtype;
};

#endif

#ifdef USE_OPENCL
struct ClMem{
    ClMem(){
        dmem = nullptr;
        offset = 0;
    }

    ClMem(cl_mem* mem_in, int offset_in = 0) {
        dmem = mem_in;
        offset = offset_in;
    }

    ClMem(ClMem& right) {
        dmem = right.dmem;
        offset = right.offset;
    }

    ClMem& operator=(ClMem& right) {
        this->dmem = right.dmem;
        this->offset = right.offset;
        return *this;
    }

    ClMem& operator+(int offset_in) {
        this->offset += offset_in;
        return *this;
    }

    int offset{0};
    cl_mem* dmem{nullptr};
};

template <>
struct DataTrait<AMD, AK_FLOAT> {
    typedef ClMem Dtype;
    typedef float dtype;
};

template <>
struct DataTrait<AMD, AK_DOUBLE> {
    typedef ClMem Dtype;
    typedef double dtype;
};

template <>
struct DataTrait<AMD, AK_INT8> {
    typedef ClMem Dtype;
    typedef char dtype;
};

#endif

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_CORE_DATA_TRAITS_H
