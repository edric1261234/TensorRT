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
#include "onesPlugin.h"
#include "plugin.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include<fstream>
#include<iostream>

#define DEBUG 1

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::Ones;
using nvinfer1::plugin::OnesPluginCreator;

namespace
{
const char* RESIZE_PLUGIN_VERSION{"1"};
const char* RESIZE_PLUGIN_NAME{"Ones_TRT"};
} // namespace

PluginFieldCollection OnesPluginCreator::mFC{};
std::vector<PluginField> OnesPluginCreator::mPluginAttributes;

OnesPluginCreator::OnesPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("width", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("height", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("interpolation", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* OnesPluginCreator::getPluginName() const
{
    return RESIZE_PLUGIN_NAME;
};

const char* OnesPluginCreator::getPluginVersion() const
{
    return RESIZE_PLUGIN_VERSION;
};

const PluginFieldCollection* OnesPluginCreator::getFieldNames()
{
    return &mFC;
};

IPluginV2Ext* OnesPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
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
        } else  if (!strcmp(attrName, "interpolation"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            mInterpolation = *(static_cast<const int32_t*>(fields[i].data));
        }
    }
    return new Ones(mWidth, mHeight, mInterpolation);
};

IPluginV2Ext* OnesPluginCreator::deserializePlugin(const char* name, const void* data, size_t length)
{
    return new Ones(data, length);
};

Ones::Ones(int width, int height, int interpolation)
    : mWidth(width)
    , mHeight(height)
    , mInterpolation(interpolation)
{
    std::cout << "===========start1" << std::endl;
    assert(width > 0);
    assert(height > 0);
    assert(interpolation >= 0);
};

int Ones::getNbOutputs() const
{
    std::cout << "===========start2" << std::endl;
    return 1;
};

Dims Ones::getOutputDimensions(int index, const Dims* inputDims, int nbInputs)
{
    std::cout << "===========start3 " << " " << nbInputs << std::endl;
//    assert(nbInputs == 1);
//    nvinfer1::Dims const& input = inputDims[0];
//    assert(index == 0);
    nvinfer1::Dims output;
    output.nbDims = 3;
    output.d[0] = 3;
    output.d[1] = 1080;
    output.d[2] = 1920;
    return output;
};

int Ones::initialize()
{
    std::cout << "===========start init" << std::endl;
    return 0;
};

void Ones::terminate(){

};

void Ones::destroy(){

};

size_t Ones::getWorkspaceSize(int) const
{
    return 0;
}

size_t Ones::getSerializationSize() const
{
    // height, width, interpolation, dimensions: 3 * 2
    return sizeof(int) * 3 + sizeof(int) * 3 * 2;
};

void Ones::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mHeight);
    write(d, mWidth);
    write(d, mInterpolation);
    write(d, mInputDims.d[0]);
    write(d, mInputDims.d[1]);
    write(d, mInputDims.d[2]);
    write(d, mOutputDims.d[0]);
    write(d, mOutputDims.d[1]);
    write(d, mOutputDims.d[2]);

    ASSERT(d == a + getSerializationSize());
};

Ones::Ones(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mHeight = read<int>(d);
    mWidth = read<int>(d);
    mInterpolation = read<int>(d);
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

const char* Ones::getPluginType() const
{
    std::cout << "===========start getplugin type" << std::endl;
    return "Ones_TRT";
};

const char* Ones::getPluginVersion() const
{
    std::cout << "===========start getplugin version" << std::endl;
    return "1";
};

IPluginV2Ext* Ones::clone() const
{
    return new Ones(*this);
};

void Ones::setPluginNamespace(const char* libNamespace)
{
    mNameSpace = libNamespace;
};

const char* Ones::getPluginNamespace() const
{
    return mNameSpace.c_str();
}

bool Ones::supportsFormat(DataType type, PluginFormat format) const
{
    std::cout << "===========start4" << std::endl;
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
};


int Ones::enqueue(
    int batch_size, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    std::cout << "===========start5" << std::endl;
    return cudaGetLastError() != cudaSuccess;
};

// Return the DataType of the plugin output at the requested index
DataType Ones::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    std::cout << "===========start6" << std::endl;
    // Only 1 input and 1 output from the plugin layer
    ASSERT(index == 0);

    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool Ones::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool Ones::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Configure the layer with input and output data types.
void Ones::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    std::cout << "===========start7" << std::endl;
    assert(nbInputs == 1);
    mInputDims = inputDims[0];

    assert(nbOutputs == 1);
    mOutputDims = outputDims[0];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void Ones::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void Ones::detachFromContext() {}
