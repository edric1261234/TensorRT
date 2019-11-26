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

#include "batchedNMSPlugin.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::BatchedNMSPlugin;
using nvinfer1::plugin::BatchedNMSPluginCreator;
using nvinfer1::plugin::NMSParameters;

#include "bboxUtils.h"
#include "cuda_runtime_api.h"
#include "gatherNMSOutputs.h"
#include "kernel.h"
#include "nmsUtils.h"
pluginStatus_t nmsInference(cudaStream_t stream, const int N, const int perBatchBoxesSize, const int perBatchScoresSize,
	const bool shareLocation, const int backgroundLabelId, const int numPredsPerClass, const int numClasses,
	const int topK, const int keepTopK, const float scoreThreshold, const float iouThreshold, const DataType DT_BBOX,
	const void* locData, const DataType DT_SCORE, const void* confData, void* keepCount, void* nmsedBoxes,
	void* nmsedScores, void* nmsedClasses, void* workspace, bool isNormalized, bool confSigmoid, bool clipBoxes)
{
	// locCount = batch_size * number_boxes_per_sample * 4
	const int locCount = N * perBatchBoxesSize;
	/*
	* shareLocation
	* Bounding box are shared among all classes, i.e., a bounding box could be classified as any candidate class.
	* Otherwise
	* Bounding box are designed for specific classes, i.e., a bounding box could be classified as one certain class or
	* not (binary classification).
	*/
	const int numLocClasses = shareLocation ? 1 : numClasses;

	size_t bboxDataSize = detectionForwardBBoxDataSize(N, perBatchBoxesSize, DataType::kFLOAT);
	void* bboxDataRaw = workspace;
	cudaMemcpyAsync(bboxDataRaw, locData, bboxDataSize, cudaMemcpyDeviceToDevice, stream);
	pluginStatus_t status;

	/*
	* bboxDataRaw format:
	* [batch size, numPriors (per sample), numLocClasses, 4]
	*/
	// float for now
	void* bboxData;
	size_t bboxPermuteSize = detectionForwardBBoxPermuteSize(shareLocation, N, perBatchBoxesSize, DataType::kFLOAT);
	void* bboxPermute = nextWorkspacePtr((int8_t*)bboxDataRaw, bboxDataSize);

	/*
	* After permutation, bboxData format:
	* [batch_size, numLocClasses, numPriors (per sample) (numPredsPerClass), 4]
	* This is equivalent to swapping axis
	*/
	if (!shareLocation)
	{
		status = permuteData(
			stream, locCount, numLocClasses, numPredsPerClass, 4, DataType::kFLOAT, false, bboxDataRaw, bboxPermute);
		ASSERT_FAILURE(status == STATUS_SUCCESS);
		bboxData = bboxPermute;
	}
	/*
	* If shareLocation, numLocClasses = 1
	* No need to permute data on linear memory
	*/
	else
	{
		bboxData = bboxDataRaw;
	}

	/*
	* Conf data format
	* [batch size, numPriors * param.numClasses, 1, 1]
	*/
	const int numScores = N * perBatchScoresSize;
	size_t totalScoresSize = detectionForwardPreNMSSize(N, perBatchScoresSize);
	void* scores = nextWorkspacePtr((int8_t*)bboxPermute, bboxPermuteSize);

	// need a conf_scores
	/*
	* After permutation, bboxData format:
	* [batch_size, numClasses, numPredsPerClass, 1]
	*/
	status = permuteData(
		stream, numScores, numClasses, numPredsPerClass, 1, DataType::kFLOAT, confSigmoid, confData, scores);
	ASSERT_FAILURE(status == STATUS_SUCCESS);

	size_t indicesSize = detectionForwardPreNMSSize(N, perBatchScoresSize);
	void* indices = nextWorkspacePtr((int8_t*)scores, totalScoresSize);

	size_t postNMSScoresSize = detectionForwardPostNMSSize(N, numClasses, topK);
	size_t postNMSIndicesSize = detectionForwardPostNMSSize(N, numClasses, topK);
	void* postNMSScores = nextWorkspacePtr((int8_t*)indices, indicesSize);
	void* postNMSIndices = nextWorkspacePtr((int8_t*)postNMSScores, postNMSScoresSize);

	void* sortingWorkspace = nextWorkspacePtr((int8_t*)postNMSIndices, postNMSIndicesSize);
	// Sort the scores so that the following NMS could be applied.
	status = sortScoresPerClass(stream, N, numClasses, numPredsPerClass, backgroundLabelId, scoreThreshold,
		DataType::kFLOAT, scores, indices, sortingWorkspace);

	ASSERT_FAILURE(status == STATUS_SUCCESS);

	// This is set to true as the input bounding boxes are of the format [ymin,
	// xmin, ymax, xmax]. The default implementation assumes [xmin, ymin, xmax, ymax]
	bool flipXY = true;
	// NMS
	status = allClassNMS(stream, N, numClasses, numPredsPerClass, topK, iouThreshold, shareLocation, isNormalized,
		DataType::kFLOAT, DataType::kFLOAT, bboxData, scores, indices, postNMSScores, postNMSIndices, flipXY);
	ASSERT_FAILURE(status == STATUS_SUCCESS);

	// Sort the bounding boxes after NMS using scores
	status = sortScoresPerImage(stream, N, numClasses * topK, DataType::kFLOAT, postNMSScores, postNMSIndices, scores,
		indices, sortingWorkspace);

	ASSERT_FAILURE(status == STATUS_SUCCESS);

	// Gather data from the sorted bounding boxes after NMS
	status = gatherNMSOutputs(stream, shareLocation, N, numPredsPerClass, numClasses, topK, keepTopK, DataType::kFLOAT,
		DataType::kFLOAT, indices, scores, bboxData, keepCount, nmsedBoxes, nmsedScores, nmsedClasses, clipBoxes);
	ASSERT_FAILURE(status == STATUS_SUCCESS);

	return STATUS_SUCCESS;
}

namespace
{
const char* NMS_PLUGIN_VERSION{"1"};
const char* NMS_PLUGIN_NAME{"BatchedNMS_TRT"};
} // namespace

PluginFieldCollection BatchedNMSPluginCreator::mFC{};
std::vector<PluginField> BatchedNMSPluginCreator::mPluginAttributes;

BatchedNMSPlugin::BatchedNMSPlugin(NMSParameters params)
    : param(params)
{
}

BatchedNMSPlugin::BatchedNMSPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    param = read<NMSParameters>(d);
    boxesSize = read<int>(d);
    scoresSize = read<int>(d);
    numPriors = read<int>(d);
    mClipBoxes = read<bool>(d);
    ASSERT(d == a + length);
}

int BatchedNMSPlugin::getNbOutputs() const
{
    return 4;
}

int BatchedNMSPlugin::initialize()
{
    return STATUS_SUCCESS;
}

void BatchedNMSPlugin::terminate() {}

Dims BatchedNMSPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims == 2);
    ASSERT(index >= 0 && index < this->getNbOutputs());
    ASSERT(inputs[0].nbDims == 3);
    ASSERT(inputs[1].nbDims == 2);
    // boxesSize: number of box coordinates for one sample
    boxesSize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
    // scoresSize: number of scores for one sample
    scoresSize = inputs[1].d[0] * inputs[1].d[1];
    // num_detections
    if (index == 0)
    {
        Dims dim0{};
        dim0.nbDims = 0;
        return dim0;
    }
    // nmsed_boxes
    if (index == 1)
    {
        return DimsHW(param.keepTopK, 4);
    }
    // nmsed_scores or nmsed_classes
    Dims dim1{};
    dim1.nbDims = 1;
    dim1.d[0] = param.keepTopK;
    return dim1;
}

size_t BatchedNMSPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return detectionInferenceWorkspaceSize(param.shareLocation, maxBatchSize, boxesSize, scoresSize, param.numClasses,
        numPriors, param.topK, DataType::kFLOAT, DataType::kFLOAT);
}

int BatchedNMSPlugin::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const void* const locData = inputs[0];
    const void* const confData = inputs[1];

    void* keepCount = outputs[0];
    void* nmsedBoxes = outputs[1];
    void* nmsedScores = outputs[2];
    void* nmsedClasses = outputs[3];

    pluginStatus_t status = nmsInference(stream, batchSize, boxesSize, scoresSize, param.shareLocation,
        param.backgroundLabelId, numPriors, param.numClasses, param.topK, param.keepTopK, param.scoreThreshold,
        param.iouThreshold, DataType::kFLOAT, locData, DataType::kFLOAT, confData, keepCount, nmsedBoxes, nmsedScores,
        nmsedClasses, workspace, param.isNormalized, false, mClipBoxes);
    ASSERT(status == STATUS_SUCCESS);
    return 0;
}

size_t BatchedNMSPlugin::getSerializationSize() const
{
    // NMSParameters, boxesSize,scoresSize,numPriors
    return sizeof(NMSParameters) + sizeof(int) * 3 + sizeof(bool);
}

void BatchedNMSPlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, param);
    write(d, boxesSize);
    write(d, scoresSize);
    write(d, numPriors);
    write(d, mClipBoxes);
    ASSERT(d == a + getSerializationSize());
}

void BatchedNMSPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize)
{
    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 4);
    ASSERT(inputDims[0].nbDims == 3);
    ASSERT(inputDims[1].nbDims == 2);
    ASSERT(std::none_of(inputIsBroadcast, inputIsBroadcast + nbInputs, [](bool b) { return b; }));
    ASSERT(std::none_of(outputIsBroadcast, outputIsBroadcast + nbInputs, [](bool b) { return b; }));

    boxesSize = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2];
    scoresSize = inputDims[1].d[0] * inputDims[1].d[1];
    // num_boxes
    numPriors = inputDims[0].d[0];
    const int numLocClasses = param.shareLocation ? 1 : param.numClasses;
    // Third dimension of boxes must be either 1 or num_classes
    ASSERT(inputDims[0].d[1] == numLocClasses);
    ASSERT(inputDims[0].d[2] == 4);
}

bool BatchedNMSPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return ((type == DataType::kFLOAT || type == DataType::kINT32) && format == PluginFormat::kNCHW);
}
const char* BatchedNMSPlugin::getPluginType() const
{
    return NMS_PLUGIN_NAME;
}

const char* BatchedNMSPlugin::getPluginVersion() const
{
    return NMS_PLUGIN_VERSION;
}

void BatchedNMSPlugin::destroy()
{
    delete this;
}

IPluginV2Ext* BatchedNMSPlugin::clone() const
{
    auto* plugin = new BatchedNMSPlugin(param);
    plugin->boxesSize = boxesSize;
    plugin->scoresSize = scoresSize;
    plugin->numPriors = numPriors;
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->setClipParam(mClipBoxes);
    return plugin;
}

void BatchedNMSPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* BatchedNMSPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

nvinfer1::DataType BatchedNMSPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    if (index == 0)
    {
        return nvinfer1::DataType::kINT32;
    }
    return inputTypes[0];
}

void BatchedNMSPlugin::setClipParam(bool clip)
{
    mClipBoxes = clip;
}

bool BatchedNMSPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool BatchedNMSPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

BatchedNMSPluginCreator::BatchedNMSPluginCreator()
    : params{}
{
    mPluginAttributes.emplace_back(PluginField("shareLocation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("backgroundLabelId", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("numClasses", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keepTopK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("scoreThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("iouThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("isNormalized", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("clipBoxes", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* BatchedNMSPluginCreator::getPluginName() const
{
    return NMS_PLUGIN_NAME;
}

const char* BatchedNMSPluginCreator::getPluginVersion() const
{
    return NMS_PLUGIN_VERSION;
}

const PluginFieldCollection* BatchedNMSPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* BatchedNMSPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    mClipBoxes = true;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "shareLocation"))
        {
            params.shareLocation = *(static_cast<const bool*>(fields[i].data));
        }
        else if (!strcmp(attrName, "backgroundLabelId"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.backgroundLabelId = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "numClasses"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.numClasses = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "topK"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.topK = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "keepTopK"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.keepTopK = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "scoreThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.scoreThreshold = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "iouThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.iouThreshold = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "isNormalized"))
        {
            params.isNormalized = *(static_cast<const bool*>(fields[i].data));
        }
        else if (!strcmp(attrName, "clipBoxes"))
        {
            mClipBoxes = *(static_cast<const bool*>(fields[i].data));
        }
    }

    BatchedNMSPlugin* plugin = new BatchedNMSPlugin(params);
    plugin->setClipParam(mClipBoxes);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* BatchedNMSPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call NMS::destroy()
    BatchedNMSPlugin* plugin = new BatchedNMSPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
