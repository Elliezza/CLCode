/*
 * Copyright (c) 2017 ARM Limited.
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

#ifndef __ARM_COMPUTE_GCDROPOUTLAYERKERNEL_H__
#define __ARM_COMPUTE_GCDROPOUTLAYERKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"

namespace arm_compute
{
class IGCTensor;

/** Interface for the dropout layer kernel.
 *
 * Dropout is used to improve over-fit on neural networks.
 *
 */
class GCDropoutLayerKernel : public IGCKernel
{
public:
    /** Default constructor */
    GCDropoutLayerKernel();

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCDropoutLayerKernel(const GCDropoutLayerKernel &) = delete;

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCDropoutLayerKernel &operator=(const GCDropoutLayerKernel &) = delete;

    /** Allow instances of this class to be moved */
    GCDropoutLayerKernel(GCDropoutLayerKernel &&) = default;

    /** Allow instances of this class to be moved */
    GCDropoutLayerKernel &operator=(GCDropoutLayerKernel &&) = default;

    /** Set the input and output of the kernel.
     *
     * @param[in]  input   The input tensor for this op. Data types supported: F16/F32
     * @param[out] mask    The mask tensor. Data types supported: Same as @p input
     * @param[out] output  The output tensor. Data types supported: Same as @p input
     * @param[in]  ratio   Dropout ratio
     * @param[in]  forward Forward or backward propagation
     *
     */
    void configure(const IGCTensor *input, IGCTensor *mask, IGCTensor *output, float ratio, bool forward);

    // Inherited methods overridden:
    void run(const Window &window) override;

private:
    const IGCTensor *_input;
    IGCTensor       *_mask;
    IGCTensor       *_output;
    unsigned int     _num_elems_processed_per_iteration;
};
}

#endif /*__ARM_COMPUTE_GCDROPOUTLAYERKERNEL_H__ */
