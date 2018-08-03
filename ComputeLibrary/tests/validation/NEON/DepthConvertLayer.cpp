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
 * OUT OF OR IN CONCLCTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEDepthConvertLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DepthConvertLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Input data sets **/
const auto DepthConvertLayerU8toU16Dataset             = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::U16));
const auto DepthConvertLayerU8toS16Dataset             = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::S16));
const auto DepthConvertLayerU8toS32Dataset             = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::S32));
const auto DepthConvertLayerU16toU8Dataset             = combine(framework::dataset::make("DataType", DataType::U16), framework::dataset::make("DataType", DataType::U8));
const auto DepthConvertLayerU16toU32Dataset            = combine(framework::dataset::make("DataType", DataType::U16), framework::dataset::make("DataType", DataType::U32));
const auto DepthConvertLayerS16toU8Dataset             = combine(framework::dataset::make("DataType", DataType::S16), framework::dataset::make("DataType", DataType::U8));
const auto DepthConvertLayerS16toS32Dataset            = combine(framework::dataset::make("DataType", DataType::S16), framework::dataset::make("DataType", DataType::S32));
const auto DepthConvertLayerQS8toFP32Dataset           = combine(framework::dataset::make("DataType", DataType::QS8), framework::dataset::make("DataType", DataType::F32));
const auto DepthConvertLayerQS16toFP32Dataset          = combine(framework::dataset::make("DataType", DataType::QS16), framework::dataset::make("DataType", DataType::F32));
const auto DepthConvertLayerFP32toQS8Dataset           = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::QS8));
const auto DepthConvertLayerFP32toQS16Dataset          = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::QS16));
const auto DepthConvertLayerShiftDataset               = framework::dataset::make("Shift", 0, 7);
const auto DepthConvertLayerFixedPointQuantizedDataset = framework::dataset::make("FractionalBits", 1, 7);
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(DepthConvertLayer)
template <typename T>
using NEDepthConvertLayerToU16Fixture = DepthConvertLayerValidationFixture<Tensor, Accessor, NEDepthConvertLayer, T, uint16_t>;
template <typename T>
using NEDepthConvertLayerToS16Fixture = DepthConvertLayerValidationFixture<Tensor, Accessor, NEDepthConvertLayer, T, int16_t>;
template <typename T>
using NEDepthConvertLayerToS32Fixture = DepthConvertLayerValidationFixture<Tensor, Accessor, NEDepthConvertLayer, T, int32_t>;
template <typename T>
using NEDepthConvertLayerToU8Fixture = DepthConvertLayerValidationFixture<Tensor, Accessor, NEDepthConvertLayer, T, uint8_t>;
template <typename T>
using NEDepthConvertLayerToU32Fixture = DepthConvertLayerValidationFixture<Tensor, Accessor, NEDepthConvertLayer, T, uint32_t>;
template <typename T>
using NEDepthConvertLayerToFP32FixedPointFixture = DepthConvertLayerValidationFractionalBitsFixture<Tensor, Accessor, NEDepthConvertLayer, T, float>;
template <typename T>
using NEDepthConvertLayerToQS8FixedPointFixture = DepthConvertLayerValidationFractionalBitsFixture<Tensor, Accessor, NEDepthConvertLayer, T, int8_t>;
template <typename T>
using NEDepthConvertLayerToQS16FixedPointFixture = DepthConvertLayerValidationFractionalBitsFixture<Tensor, Accessor, NEDepthConvertLayer, T, int16_t>;

TEST_SUITE(U8_to_U16)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDataset),
               shape, policy, shift)
{
    int fixed_point_position = 0;

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::U8, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape, DataType::U16, 1, fixed_point_position);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToU16Fixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU8toU16Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToU16Fixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU8toU16Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(U8_to_S16)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDataset),
               shape, policy, shift)
{
    int fixed_point_position = 0;

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::U8, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape, DataType::S16, 1, fixed_point_position);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToS16Fixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU8toS16Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToS16Fixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU8toS16Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE(U8_to_S32)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDataset),
               shape, policy, shift)
{
    int fixed_point_position = 0;

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::U8, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape, DataType::S32, 1, fixed_point_position);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToS32Fixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU8toS32Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToS32Fixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU8toS32Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(U16_to_U8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDataset),
               shape, policy, shift)
{
    int fixed_point_position = 0;

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::U16, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape, DataType::U8, 1, fixed_point_position);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToU8Fixture<uint16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU16toU8Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToU8Fixture<uint16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU16toU8Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(U16_to_U32)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDataset),
               shape, policy, shift)
{
    int fixed_point_position = 0;

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::U16, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape, DataType::U32, 1, fixed_point_position);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToU32Fixture<uint16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerU16toU32Dataset),
                                                                                                                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                       DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToU32Fixture<uint16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerU16toU32Dataset),
                                                                                                                     framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                     DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(S16_to_U8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDataset),
               shape, policy, shift)
{
    int fixed_point_position = 0;

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::S16, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape, DataType::U8, 1, fixed_point_position);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToU8Fixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerS16toU8Dataset),
                                                                                                                     framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                     DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToU8Fixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerS16toU8Dataset),
                                                                                                                   framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                   DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(S16_to_S32)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerShiftDataset),
               shape, policy, shift)
{
    int fixed_point_position = 0;

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::S16, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape, DataType::S32, 1, fixed_point_position);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEDepthConvertLayerToS32Fixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), DepthConvertLayerS16toS32Dataset),
                                                                                                                      framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                      DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDepthConvertLayerToS32Fixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), DepthConvertLayerS16toS32Dataset),
                                                                                                                    framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                                                                    DepthConvertLayerShiftDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(Quantized_to_FP32)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("DataType", { DataType::QS8, DataType::QS16 })),
                                                                           framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerFixedPointQuantizedDataset),
               shape, dt, policy, fixed_point_position)
{
    int shift = 0;

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, dt, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape, DataType::F32, 1, fixed_point_position);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}
FIXTURE_DATA_TEST_CASE(RunTinyQS8, NEDepthConvertLayerToFP32FixedPointFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::TinyShapes(),
                       DepthConvertLayerQS8toFP32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       DepthConvertLayerFixedPointQuantizedDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunTinyQS16, NEDepthConvertLayerToFP32FixedPointFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::TinyShapes(),
                       DepthConvertLayerQS16toFP32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       DepthConvertLayerFixedPointQuantizedDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunSmallQS8, NEDepthConvertLayerToFP32FixedPointFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SmallShapes(),
                       DepthConvertLayerQS8toFP32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       DepthConvertLayerFixedPointQuantizedDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunSmallQS16, NEDepthConvertLayerToFP32FixedPointFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SmallShapes(),
                       DepthConvertLayerQS16toFP32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       DepthConvertLayerFixedPointQuantizedDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(FP32_to_Quantized)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("DataType", { DataType::QS8, DataType::QS16 })),
                                                                           framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                                                                   DepthConvertLayerFixedPointQuantizedDataset),
               shape, dt, policy, fixed_point_position)
{
    int shift = 0;

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::F32, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape, dt, 1, fixed_point_position);

    // Create and Configure function
    NEDepthConvertLayer depth_convert;
    depth_convert.configure(&src, &dst, policy, shift);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}
FIXTURE_DATA_TEST_CASE(RunTinyQS8, NEDepthConvertLayerToQS8FixedPointFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::TinyShapes(),
                       DepthConvertLayerFP32toQS8Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       DepthConvertLayerFixedPointQuantizedDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunTinyQS16, NEDepthConvertLayerToQS16FixedPointFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::TinyShapes(),
                       DepthConvertLayerFP32toQS16Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       DepthConvertLayerFixedPointQuantizedDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunSmallQS8, NEDepthConvertLayerToQS8FixedPointFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SmallShapes(),
                       DepthConvertLayerFP32toQS8Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       DepthConvertLayerFixedPointQuantizedDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunSmallQS16, NEDepthConvertLayerToQS16FixedPointFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SmallShapes(),
                       DepthConvertLayerFP32toQS16Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       DepthConvertLayerFixedPointQuantizedDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
