/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/common.h"

namespace tensorrt_llm
{
namespace kernels
{
using SizeType32 = tensorrt_llm::runtime::SizeType32;

template <typename T>
size_t invokeComputeCumsumLastDimWorkspaceSize(SizeType32 inputLength);

template <typename T>
void invokeCumsumLastDim(SizeType32 batchSize, SizeType32 inputLength, void const* __restrict__ input,
    void* __restrict__ output, void* workspace, size_t tempStorageBytes, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
