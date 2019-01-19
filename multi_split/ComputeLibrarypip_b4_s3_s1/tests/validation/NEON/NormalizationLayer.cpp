/*
 * Copyright (c) 2017-2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NENormalizationLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/NormalizationTypesDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/NormalizationLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance for float operations */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
constexpr AbsoluteTolerance<float> tolerance_f16(0.001f);
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
constexpr AbsoluteTolerance<float> tolerance_f32(0.00001f);
/** Tolerance for fixed point operations */
constexpr AbsoluteTolerance<int8_t>  tolerance_qs8(2);
constexpr AbsoluteTolerance<int16_t> tolerance_qs16(4);

/** Input data set. */
const auto NormalizationDatasetQS = combine(combine(combine(combine(datasets::TinyShapes(), datasets::NormalizationTypes()), framework::dataset::make("NormalizationSize", 3, 9, 2)),
                                                    framework::dataset::make("Beta", { 0.5f, 1.f, 2.f })),
                                            framework::dataset::make("IsScaled", { true }));
const auto NormalizationDataset = combine(combine(combine(combine(datasets::SmallShapes(), datasets::NormalizationTypes()), framework::dataset::make("NormalizationSize", 3, 9, 2)),
                                                  framework::dataset::make("Beta", { 0.5f, 1.f, 2.f })),
                                          framework::dataset::make("IsScaled", { true }));
const auto NormalizationDatasetFP32 = combine(combine(combine(combine(datasets::SmallShapes(), datasets::NormalizationTypes()), framework::dataset::make("NormalizationSize", 3, 9, 2)),
                                                      framework::dataset::make("Beta", { 0.5f, 1.f, 2.f })),
                                              framework::dataset::make("IsScaled", { true, false }));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(NormalizationLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
    framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0), // Mismatching data type input/output
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0), // Mismatching shapes
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0), // Even normalization
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0), // Non implemented IN_MAP_2D
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QS8, 4), // Mismatching fixed point position
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0), // Window shrink
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32, 0),
                                          }),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16, 0),
                                            TensorInfo(TensorShape(27U, 11U, 2U), 1, DataType::F32, 0),
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0),
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0),
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QS8, 3),
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0),
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32, 0),
                                          })),
    framework::dataset::make("NormInfo",  { NormalizationLayerInfo(NormType::IN_MAP_1D, 5),
                                            NormalizationLayerInfo(NormType::IN_MAP_1D, 5),
                                            NormalizationLayerInfo(NormType::IN_MAP_1D, 4),
                                            NormalizationLayerInfo(NormType::IN_MAP_2D, 5),
                                            NormalizationLayerInfo(NormType::IN_MAP_1D, 5),
                                            NormalizationLayerInfo(NormType::IN_MAP_1D, 5),
                                            NormalizationLayerInfo(NormType::CROSS_MAP, 1),
                                           })),
    framework::dataset::make("Expected", { false, false, false, false, false, false, true })),
    input_info, output_info, norm_info, expected)
{
    bool is_valid = bool(NENormalizationLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), norm_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NENormalizationLayerFixture = NormalizationValidationFixture<Tensor, Accessor, NENormalizationLayer, T>;

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NENormalizationLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(NormalizationDataset, framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NENormalizationLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(NormalizationDataset, framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END()
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NENormalizationLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(NormalizationDatasetFP32, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NENormalizationLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(NormalizationDatasetFP32, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using NENormalizationLayerFixedPointFixture = NormalizationValidationFixedPointFixture<Tensor, Accessor, NENormalizationLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QS8)
// Testing for fixed point position [1,6) as reciprocal limits the maximum fixed point position to 5
FIXTURE_DATA_TEST_CASE(RunTiny, NENormalizationLayerFixedPointFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(NormalizationDatasetQS, framework::dataset::make("DataType",
                       DataType::QS8)),
                       framework::dataset::make("FractionalBits", 1, 6)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qs8);
}
FIXTURE_DATA_TEST_CASE(RunSmall, NENormalizationLayerFixedPointFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(NormalizationDataset, framework::dataset::make("DataType",
                       DataType::QS8)),
                       framework::dataset::make("FractionalBits", 1, 6)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qs8);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
// Testing for fixed point position [1,14) as reciprocal limits the maximum fixed point position to 14
FIXTURE_DATA_TEST_CASE(RunTiny, NENormalizationLayerFixedPointFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(NormalizationDatasetQS, framework::dataset::make("DataType",
                       DataType::QS16)),
                       framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qs16);
}
FIXTURE_DATA_TEST_CASE(RunSmall, NENormalizationLayerFixedPointFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(NormalizationDataset, framework::dataset::make("DataType",
                       DataType::QS16)),
                       framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qs16);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
