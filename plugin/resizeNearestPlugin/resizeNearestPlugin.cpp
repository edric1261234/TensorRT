/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "resizeNearestPlugin.h"
#include "plugin.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include<fstream>
#include<iostream>

#define DEBUG 1

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::ResizeNearest;
using nvinfer1::plugin::ResizeNearestPluginCreator;

namespace
{
const char* RESIZE_PLUGIN_VERSION{"1"};
const char* RESIZE_PLUGIN_NAME{"ResizeNearest_TRT"};
} // namespace

PluginFieldCollection ResizeNearestPluginCreator::mFC{};
std::vector<PluginField> ResizeNearestPluginCreator::mPluginAttributes;

ResizeNearestPluginCreator::ResizeNearestPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("width", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("height", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("nchw", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("index", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ResizeNearestPluginCreator::getPluginName() const
{
    return RESIZE_PLUGIN_NAME;
};

const char* ResizeNearestPluginCreator::getPluginVersion() const
{
    return RESIZE_PLUGIN_VERSION;
};

const PluginFieldCollection* ResizeNearestPluginCreator::getFieldNames()
{
    return &mFC;
};

IPluginV2Ext* ResizeNearestPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "width"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            mWidth = *(static_cast<const int32_t*>(fields[i].data));
        } else  if (!strcmp(attrName, "height"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            mHeight = *(static_cast<const int32_t*>(fields[i].data));
        }
    }
    return new ResizeNearest(mWidth, mHeight);
};

IPluginV2Ext* ResizeNearestPluginCreator::deserializePlugin(const char* name, const void* data, size_t length)
{
    return new ResizeNearest(data, length);
};

ResizeNearest::ResizeNearest(int width, int height)
    : mWidth(width)
    , mHeight(height)
{
    assert(width > 0);
    assert(height > 0);
};

int ResizeNearest::getNbOutputs() const
{
    return 1;
};

Dims ResizeNearest::getOutputDimensions(int index, const Dims* inputDims, int nbInputs)
{
    assert(nbInputs == 1);
    nvinfer1::Dims const& input = inputDims[0];
    assert(index == 0);
    nvinfer1::Dims output;
    output.nbDims = input.nbDims;
    output.d[0] = input.d[0];
    output.d[1] = mHeight;
    output.d[2] = mWidth;
//    std::ofstream OutFile("/home/tal/test.txt", std::ios::app);
//    OutFile <<"input dims " << std::to_string(input.d[0]) << " " << std::to_string(input.d[1]) << " " << std::to_string(input.d[2]) << std::endl;
//    OutFile << mHeight << " " << mWidth << std::endl;
//    OutFile <<"output dims " << std::to_string(output.d[0]) << " " << std::to_string(output.d[1]) << " " << std::to_string(output.d[2]) << std::endl;
//    OutFile.close();
    return output;
};

//Dims ResizeNearest::getOutputDimensions(int index, const Dims* inputDims, int nbInputs)
//{
//    assert(nbInputs == 1);
//    nvinfer1::Dims const& input = inputDims[0];
//    assert(index == 0);
//    nvinfer1::Dims output;
//    output.nbDims = input.nbDims;
//    for (int d = 0; d < input.nbDims; ++d)
//    {
//        if (d == input.nbDims - 2)
//        {
//            output.d[d] = mWidth;
//        }
//        else if (d == input.nbDims - 1)
//        {
//            output.d[d] = mHeight;
//        }
//        else
//        {
//            output.d[d] = input.d[d];
//        }
//    }
//    std::ofstream OutFile("/home/tal/Test.txt");
//    OutFile <<"input dims " << std::to_string(input.d[0]) << " " << std::to_string(input.d[1]) << " " << std::to_string(input.d[2]) << std::endl;
//    OutFile << mHeight << " " << mWidth << std::endl;
//    OutFile <<"output dims " << std::to_string(output.d[0]) << " " << std::to_string(output.d[1]) << " " << std::to_string(output.d[2]) << std::endl;
//    OutFile.close();
//    return output;
//};


int ResizeNearest::initialize()
{
    return 0;
};

void ResizeNearest::terminate(){

};

void ResizeNearest::destroy(){

};

size_t ResizeNearest::getWorkspaceSize(int) const
{
    return 0;
}

size_t ResizeNearest::getSerializationSize() const
{
    // height, width, dimensions: 3 * 2
    return sizeof(int) * 2 + sizeof(int) * 3 * 2;
};

void ResizeNearest::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mHeight);
    write(d, mWidth);
    write(d, mInputDims.d[0]);
    write(d, mInputDims.d[1]);
    write(d, mInputDims.d[2]);
    write(d, mOutputDims.d[0]);
    write(d, mOutputDims.d[1]);
    write(d, mOutputDims.d[2]);

    ASSERT(d == a + getSerializationSize());
};

ResizeNearest::ResizeNearest(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mHeight = read<int>(d);
    mWidth = read<int>(d);
    mInputDims = Dims3();
    mInputDims.d[0] = read<int>(d);
    mInputDims.d[1] = read<int>(d);
    mInputDims.d[2] = read<int>(d);
    mOutputDims = Dims3();
    mOutputDims.d[0] = read<int>(d);
    mOutputDims.d[1] = read<int>(d);
    mOutputDims.d[2] = read<int>(d);

    ASSERT(d == a + length);
};

const char* ResizeNearest::getPluginType() const
{
    return "ResizeNearest_TRT";
};

const char* ResizeNearest::getPluginVersion() const
{
    return "1";
};

IPluginV2Ext* ResizeNearest::clone() const
{
    return new ResizeNearest(*this);
};

void ResizeNearest::setPluginNamespace(const char* libNamespace)
{
    mNameSpace = libNamespace;
};

const char* ResizeNearest::getPluginNamespace() const
{
    return mNameSpace.c_str();
}

bool ResizeNearest::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
};


int ResizeNearest::enqueue(
    int batch_size, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    int nchan = mOutputDims.d[0];

    int2 osize = {mOutputDims.d[2], mOutputDims.d[1]};
    int istride = mInputDims.d[2];
    int ostride = mOutputDims.d[2];
    float scale = 1.0f * ostride / istride;
    int ibatchstride =  mInputDims.d[1] * istride;
    int obatchstride = mOutputDims.d[1] * ostride;
    dim3 block(32, 16);
    dim3 grid((osize.x - 1) / block.x + 1, (osize.y - 1) / block.y + 1, std::min(batch_size * nchan, 65535));

    resizeNearest(grid, block, stream, batch_size * nchan, scale, osize, static_cast<float const*>(inputs[0]), istride,
                  ibatchstride, static_cast<float*>(outputs[0]), ostride, obatchstride);

    return cudaGetLastError() != cudaSuccess;
};

// Return the DataType of the plugin output at the requested index
DataType ResizeNearest::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only 1 input and 1 output from the plugin layer
    ASSERT(index == 0);

    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool ResizeNearest::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool ResizeNearest::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Configure the layer with input and output data types.
void ResizeNearest::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    assert(nbInputs == 1);
    mInputDims = inputDims[0];

    assert(nbOutputs == 1);
    mOutputDims = outputDims[0];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void ResizeNearest::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void ResizeNearest::detachFromContext() {}
