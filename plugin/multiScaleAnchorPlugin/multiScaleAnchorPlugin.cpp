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
#include "multiScaleAnchorPlugin.h"
#include <cstring>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

using namespace nvinfer1;
using nvinfer1::plugin::MultiScaleAnchorGenerator;
using nvinfer1::plugin::MultiScaleAnchorPluginCreator;

namespace
{
const char* MULTI_SCALE_ANCHOR_PLUGIN_VERSION{"1"};
const char* MULTI_SCALE_ANCHOR_PLUGIN_NAME{"MultiScaleAnchor_TRT"};
} // namespace
PluginFieldCollection MultiScaleAnchorPluginCreator::mFC{};
std::vector<PluginField> MultiScaleAnchorPluginCreator::mPluginAttributes;

MultiScaleAnchorGenerator::MultiScaleAnchorGenerator(const MultiScaleAnchorParameters* paramIn, int mNumLayers)
    : mNumLayers(mNumLayers)
{
    CUASSERT(cudaMallocHost((void**) &mNumPriors, mNumLayers * sizeof(int)));
    CUASSERT(cudaMallocHost((void**) &mDeviceWidths, mNumLayers * paramIn[0].scalesPerOctave * sizeof(Weights)));
    CUASSERT(cudaMallocHost((void**) &mDeviceHeights, mNumLayers * paramIn[0].scalesPerOctave * sizeof(Weights)));

    mParam.resize(mNumLayers);
    for (int id = 0; id < mNumLayers; id++)
    {
        mParam[id] = paramIn[id];
        //std::cout << "mParam[id] H " << mParam[id].H << std::endl;
        ASSERT(mParam[id].numAspectRatios >= 0 && mParam[id].aspectRatios != nullptr);

        mParam[id].aspectRatios = (float*) malloc(sizeof(float) * mParam[id].numAspectRatios);

        for (int i = 0; i < paramIn[id].numAspectRatios; ++i)
        {
            mParam[id].aspectRatios[i] = paramIn[id].aspectRatios[i];
        }

        std::vector<float> aspect_ratios;
        std::vector<float> scales;

        for (int i = 0; i < mParam[id].numAspectRatios; i++)
        {
            aspect_ratios.push_back(mParam[id].aspectRatios[i]);
        }
        // scales
        for (int j = 0; j < mParam[id].scalesPerOctave; j++) {
            scales.push_back(
                mParam[id].anchorScale * (1.0 / mParam[id].H) *
                pow(2, 1.0 * j / paramIn[id].scalesPerOctave));
        }

        mNumPriors[id] = mParam[id].numAspectRatios * paramIn[id].scalesPerOctave;

        std::vector<float> tmpWidths;
        std::vector<float> tmpHeights;
        // Calculate the width and height of the prior boxes
        for (int i = 0; i < mParam[id].numAspectRatios; i++)
        {
            float sqrt_AR = sqrt(aspect_ratios[i]);
            for (int j = 0; j < scales.size(); j++) {
                tmpWidths.push_back(scales[j] * sqrt_AR);
                tmpHeights.push_back(scales[j] / sqrt_AR);
            }
        }

        mDeviceWidths[id] = copyToDevice(&tmpWidths[0], tmpWidths.size());
        mDeviceHeights[id] = copyToDevice(&tmpHeights[0], tmpHeights.size());
    }
}

MultiScaleAnchorGenerator::MultiScaleAnchorGenerator(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mNumLayers = read<int>(d);
    int scalesPerOctave = read<int>(d);
    CUASSERT(cudaMallocHost((void**) &mNumPriors, mNumLayers * sizeof(int)));
    CUASSERT(cudaMallocHost((void**) &mDeviceWidths, mNumLayers  * sizeof(Weights)));
    CUASSERT(cudaMallocHost((void**) &mDeviceHeights, mNumLayers  * sizeof(Weights)));
    mParam.resize(mNumLayers);
    for (int id = 0; id < mNumLayers; id++)
    {
        // we have to deserialize GridAnchorParameters by hand
        mParam[id].minLevel = read<int>(d);
        mParam[id].maxLevel = read<int>(d);
        mParam[id].numAspectRatios = read<int>(d);
        mParam[id].aspectRatios = (float*) malloc(sizeof(float) * mParam[id].numAspectRatios);
        for (int i = 0; i < mParam[id].numAspectRatios; ++i)
        {
            mParam[id].aspectRatios[i] = read<float>(d);
        }
        mParam[id].H = read<int>(d);
        mParam[id].W = read<int>(d);
        mNumPriors[id] = read<int>(d);
        mParam[id].anchorScale = read<float>(d);
        mParam[id].scalesPerOctave = scalesPerOctave;

        mDeviceWidths[id] = deserializeToDevice(d, mNumPriors[id]);
        mDeviceHeights[id] = deserializeToDevice(d, mNumPriors[id]);
    }
    ASSERT(d == a + length);
}

MultiScaleAnchorGenerator::~MultiScaleAnchorGenerator()
{
    for (int id = 0; id < mNumLayers; id++)
    {
        CUERRORMSG(cudaFree(const_cast<void*>(mDeviceWidths[id].values)));
        CUERRORMSG(cudaFree(const_cast<void*>(mDeviceHeights[id].values)));
        free(mParam[id].aspectRatios);
    }
    CUERRORMSG(cudaFreeHost(mNumPriors));
    CUERRORMSG(cudaFreeHost(mDeviceWidths));
    CUERRORMSG(cudaFreeHost(mDeviceHeights));
}

int MultiScaleAnchorGenerator::getNbOutputs() const
{
    return mNumLayers;
}

Dims MultiScaleAnchorGenerator::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Particularity of the PriorBox layer: no batchSize dimension needed
    // 2 channels. First channel stores the mean of each prior coordinate.
    // Second channel stores the variance of each prior coordinate.
    return DimsCHW(2, mParam[index].H * mParam[index].W * mNumPriors[index] * 4, 1);
}

int MultiScaleAnchorGenerator::initialize()
{
    return STATUS_SUCCESS;
}

void MultiScaleAnchorGenerator::terminate() {}

size_t MultiScaleAnchorGenerator::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int MultiScaleAnchorGenerator::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    // Generate prior boxes for each layer
    for (int id = 0; id < mNumLayers; id++)
    {
        void* outputData = outputs[id];
        pluginStatus_t status = multiScaleGridInference(
            stream, mParam[id], mNumPriors[id], mDeviceWidths[id].values, mDeviceHeights[id].values, outputData);
        ASSERT(status == STATUS_SUCCESS);
        // Weights w = copyToHost(outputData, 64);
        // const float* ptr = static_cast<const float*>(w.values);
        // for (int i = 0; i < 64 / 4; i++) {
        //     std::cout << i 
        //     << ":" << *(ptr + 4 * i ) * 640
        //     << " " << *(ptr + 4 * i + 1) * 640
        //     << " " << *(ptr + 4 * i + 2) * 640
        //     << " " << *(ptr + 4 * i + 3) * 640
        //     << std::endl;
        // }
        // std::cout << "=================" << std::endl;
    }
    return STATUS_SUCCESS;
}

size_t MultiScaleAnchorGenerator::getSerializationSize() const
{
    size_t sum = 2 * sizeof(int); // mNumLayers, scalesPerOctave
    for (int i = 0; i < mNumLayers; i++)
    {
        sum += 3 * sizeof(int); // minLevel, maxLevel, numAspectRatios
        sum += mParam[i].numAspectRatios * sizeof(float); //  aspectRatios
        sum += 3 * sizeof(int); // H, W, mNumPriors
        sum += sizeof(float); // anchorScale
        sum += mDeviceWidths[i].count * sizeof(float);
        sum += mDeviceHeights[i].count * sizeof(float);
    }
    return sum;
}

void MultiScaleAnchorGenerator::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mNumLayers);
    std::cout << mNumLayers << std::endl;
    write(d, mParam[0].scalesPerOctave);
    for (int id = 0; id < mNumLayers; id++)
    {
        // we have to serialize GridAnchorParameters by hand
        write(d, mParam[id].minLevel);
        write(d, mParam[id].maxLevel);
        write(d, mParam[id].numAspectRatios);
        for (int i = 0; i < mParam[id].numAspectRatios; ++i)
        {
            write(d, mParam[id].aspectRatios[i]);
        }
        write(d, mParam[id].H);
        write(d, mParam[id].W);
        write(d, mNumPriors[id]);
        write(d, mParam[id].anchorScale);

        serializeFromDevice(d, mDeviceWidths[id]);
        serializeFromDevice(d, mDeviceHeights[id]);
    }
    ASSERT(d == a + getSerializationSize());
}

Weights MultiScaleAnchorGenerator::copyToDevice(const void* hostData, size_t count)
{
    void* deviceData;
    CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
    CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
    return Weights{DataType::kFLOAT, deviceData, int64_t(count)};
}

Weights MultiScaleAnchorGenerator::copyToHost(const void* deviceData, size_t count)
{
    void* hostData;
    hostData = malloc(count * sizeof(float));
    CUASSERT(cudaMemcpy(hostData, deviceData, count * sizeof(float), cudaMemcpyDeviceToHost));
    return Weights{DataType::kFLOAT, hostData, int64_t(count)};
}

void MultiScaleAnchorGenerator::serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const
{
    cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost);
    hostBuffer += deviceWeights.count * sizeof(float);
}

Weights MultiScaleAnchorGenerator::deserializeToDevice(const char*& hostBuffer, size_t count)
{
    Weights w = copyToDevice(hostBuffer, count);
    hostBuffer += count * sizeof(float);
    return w;
}
bool MultiScaleAnchorGenerator::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

const char* MultiScaleAnchorGenerator::getPluginType() const
{
    return MULTI_SCALE_ANCHOR_PLUGIN_NAME;
}

const char* MultiScaleAnchorGenerator::getPluginVersion() const
{
    return MULTI_SCALE_ANCHOR_PLUGIN_VERSION;
}

// Set plugin namespace
void MultiScaleAnchorGenerator::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* MultiScaleAnchorGenerator::getPluginNamespace() const
{
    return mPluginNamespace;
}

#include <iostream>
// Return the DataType of the plugin output at the requested index
DataType MultiScaleAnchorGenerator::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(index < mNumLayers);
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool MultiScaleAnchorGenerator::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool MultiScaleAnchorGenerator::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Configure the layer with input and output data types.
void MultiScaleAnchorGenerator::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    ASSERT(nbOutputs == mNumLayers);
    ASSERT(outputDims[0].nbDims == 3);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void MultiScaleAnchorGenerator::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void MultiScaleAnchorGenerator::detachFromContext() {}

void MultiScaleAnchorGenerator::destroy()
{
    delete this;
}

IPluginV2Ext* MultiScaleAnchorGenerator::clone() const
{
    IPluginV2Ext* plugin = new MultiScaleAnchorGenerator(mParam.data(), mNumLayers);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

MultiScaleAnchorPluginCreator::MultiScaleAnchorPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("minLevel", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("maxLevel", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("aspectRatios", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("featureMapShapes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("numLayers", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("scalesPerOctave", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("anchorScale", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MultiScaleAnchorPluginCreator::getPluginName() const
{
    return MULTI_SCALE_ANCHOR_PLUGIN_NAME;
}

const char* MultiScaleAnchorPluginCreator::getPluginVersion() const
{
    return MULTI_SCALE_ANCHOR_PLUGIN_VERSION;
}

const PluginFieldCollection* MultiScaleAnchorPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* MultiScaleAnchorPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    int minLevel = 3, maxLevel = 7;
    int numLayers = 5;
    int scalesPerOctave = 2;
    float anchorScale = 4.0;
    std::vector<float> aspectRatios;
    std::vector<int> fMapShapes;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "numLayers"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            numLayers = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "minLevel"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            minLevel = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "maxLevel"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            maxLevel = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "anchorScale"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            anchorScale = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "aspectRatios"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            aspectRatios.reserve(size);
            const auto* aR = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                aspectRatios.push_back(*aR);
                aR++;
            }
        }
        else if (!strcmp(attrName, "featureMapShapes"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            int size = fields[i].length;
            fMapShapes.reserve(size);
            const int* fMap = static_cast<const int*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                fMapShapes.push_back(*fMap);
                fMap++;
            }
        }
        else if (!strcmp(attrName, "scalesPerOctave"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            scalesPerOctave = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
    }
    // Reducing the number of boxes predicted by the first layer.
    // This is in accordance with the standard implementation.
    std::vector<float> firstLayerAspectRatios;

    ASSERT(numLayers > 0);
    ASSERT((int) fMapShapes.size() == numLayers);

    // A comprehensive list of box parameters that are required by anchor generator
    std::vector<MultiScaleAnchorParameters> boxParams(numLayers);

    // One set of box parameters for one layer
    for (int i = 0; i < numLayers; i++)
    {
        boxParams[i] = {
            minLevel, maxLevel, 
            aspectRatios.data(), (int) aspectRatios.size(), 
            fMapShapes[i], fMapShapes[i],
            anchorScale, scalesPerOctave,
            pow(2, i + minLevel),
             {1, 1, 1, 1} // variance no use
            };
    }

    MultiScaleAnchorGenerator* obj = new MultiScaleAnchorGenerator(boxParams.data(), numLayers);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2Ext* MultiScaleAnchorPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call MultiScaleAnchor::destroy()
    MultiScaleAnchorGenerator* obj = new MultiScaleAnchorGenerator(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
