/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf_network.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  A network that first processes 3D position to density and
 *          subsequently direction to color.
 */

#pragma once

#include <typeinfo>
#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/network.h>

#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <vector>

#include <neural-graphics-primitives/debug.h>

NGP_NAMESPACE_BEGIN

template <typename T>
__global__ void extract_density(
	const uint32_t n_elements, //max_samples 262144 * 16
	const uint32_t density_stride, //1
	const uint32_t rgbd_stride, //16
	const T* __restrict__ density,
	T* __restrict__ rgbd
) {



	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// printf("\n i: %d, n_elements: %d,  density_stride: %d, rgbd_stride: %d \n", i, n_elements, density_stride, rgbd_stride);
	rgbd[i * rgbd_stride] = density[i * density_stride];
}

template <typename T>
__global__ void extract_mask(
	const uint32_t n_labels,
	const uint32_t n_elements, //max_samples 262144 * 16 
	const uint32_t seg_stride, //1
	const uint32_t rgbd_stride, //16
	const T* __restrict__ rgbd,
	T* __restrict__ seg
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	for (uint32_t j = 0; j < n_labels; ++j) {
		seg[i*seg_stride + j] = rgbd[i*rgbd_stride + j];
		// seg[i*seg_stride + j] = 0.0f;
	}
}

template <typename T>
__global__ void pack_mask_with_density(
	const uint32_t n_labels,
	const uint32_t n_elements, //max_samples 262144 * 16
	const uint32_t seg_stride, //1
	const uint32_t rgbd_stride, //16
	const T* __restrict__ seg,
	T* __restrict__ rgbd
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// const uint32_t elem_idx = i / n_labels;

	for (uint32_t j = 0; j < n_labels; ++j) {
		// rgbd[i*rgbd_stride + j] = 0.0f;
		rgbd[i*rgbd_stride + j] = seg[i*seg_stride + j];
	}
}

template <typename T>
__global__ void extract_rgb(
	const uint32_t n_elements, // 262144 * 3
	const uint32_t rgb_stride, //16
	const uint32_t output_stride, //16
	const T* __restrict__ rgbd,
	T* __restrict__ rgb
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	rgb[elem_idx*rgb_stride + dim_idx] = rgbd[elem_idx*output_stride + dim_idx];
}

template <typename T>
__global__ void add_density_gradient(
	const uint32_t n_elements,
	const uint32_t rgbd_stride,
	const T* __restrict__ rgbd,
	const uint32_t density_stride,
	T* __restrict__ density
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	density[i * density_stride] += rgbd[i * rgbd_stride + 3];
}

template <typename T>
__global__ void add_density_gradient_and_seg_gradient(
	const uint32_t n_elements,
	const uint32_t seg_stride,
	const T* __restrict__ dL_dseg,
	const uint32_t density_stride,
	T* __restrict__ dL_density
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	dL_density[i * density_stride] += dL_dseg[i * seg_stride];
}

template <typename T>
__global__ void add_mask_gradient(
	const uint32_t n_elements,
	const uint32_t rgbd_stride,
	const T* __restrict__ rgbd,
	const uint32_t density_stride,
	T* __restrict__ density
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	density[i * density_stride + 1] += rgbd[i * rgbd_stride + 4];
}

template <typename T>
class NerfNetwork : public tcnn::Network<float, T> {
public:
	using json = nlohmann::json;

	NerfNetwork(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims, uint32_t dir_offset, const json& pos_encoding, const json& dir_encoding, const json& density_network, const json& rgb_network, const json& seg_network) : m_n_pos_dims{n_pos_dims}, m_n_dir_dims{n_dir_dims}, m_dir_offset{dir_offset}, m_n_extra_dims{n_extra_dims} {
		m_pos_encoding.reset(tcnn::create_encoding<T>(n_pos_dims, pos_encoding, density_network.contains("otype") && (tcnn::equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || tcnn::equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u));
		uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);

		m_dir_encoding.reset(tcnn::create_encoding<T>(m_n_dir_dims + m_n_extra_dims, dir_encoding, rgb_alignment));

		json local_density_network_config = density_network;
		local_density_network_config["n_input_dims"] = m_pos_encoding->padded_output_width();
		if (!density_network.contains("n_output_dims")) {
			local_density_network_config["n_output_dims"] = 16;
		}
		m_density_network.reset(tcnn::create_network<T>(local_density_network_config));



		m_rgb_network_input_width = tcnn::next_multiple(m_dir_encoding->padded_output_width() + m_density_network->padded_output_width(), rgb_alignment);

		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
		local_rgb_network_config["n_output_dims"] = 3;
		m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));

		// Segmentation Network
		uint32_t seg_alignment = tcnn::minimum_alignment(seg_network);
		// m_seg_network_input_width = tcnn::next_multiple(m_dir_encoding->padded_output_width() + m_density_network->padded_output_width(), seg_alignment);

		// If take output from the density network rather than position encoding
		m_seg_network_input_width = tcnn::next_multiple(m_density_network->padded_output_width(), seg_alignment);

		// json local_seg_network_config = seg_network;
		// local_seg_network_config["n_input_dims"] = m_seg_network_input_width;
		// local_seg_network_config["n_output_dims"] = 16;
		// m_seg_network.reset(tcnn::create_network<T>(local_seg_network_config));
	}

	virtual ~NerfNetwork() { }

	void inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {
		uint32_t batch_size = input.n();

		tcnn::GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		tcnn::GPUMatrixDynamic<T> rgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

		tcnn::GPUMatrixDynamic<T> density_network_output = rgb_network_input.slice_rows(0, m_density_network->padded_output_width());

		// RGB network output already has output pointer address so no need to extract RGB right now. 
		// output automatically gets updated.
		tcnn::GPUMatrixDynamic<T> rgb_network_output{output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};
		// tcnn::GPUMatrixDynamic<T> seg_network_output{m_seg_network->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

		m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			density_network_input,
			use_inference_params
		);

		m_density_network->inference_mixed_precision(stream, density_network_input, density_network_output, use_inference_params);

		auto dir_out = rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
		m_dir_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
			dir_out,
			use_inference_params
		);

		m_rgb_network->inference_mixed_precision(stream, rgb_network_input, rgb_network_output, use_inference_params);

		// Inference for segmentation network
		// if (train_seg){
		// 	m_seg_network->inference_mixed_precision(stream, density_network_output, seg_network_output, use_inference_params);
		// }

		// Getting the densiy from the density network output with stride of 1 RM layout
		tcnn::linear_kernel(extract_density<T>, 0, stream,
			batch_size,
			density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
			output.layout() == tcnn::AoS ? padded_output_width() : 1,
			density_network_output.data(),
			output.data() + 3 * (output.layout() == tcnn::AoS ? 1 : batch_size)
		);

		// tcnn::linear_kernel(pack_mask_with_density<T>, 0, stream,
		// 	n_labels,
		// 	batch_size,
		// 	density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
		// 	output.layout() == tcnn::AoS ? padded_output_width() : 1,
			// seg_network_output.data(),
		// 	output.data() + 4 * (output.layout() == tcnn::AoS ? 1 : batch_size)
		// );

	}

	uint32_t padded_density_output_width() const {
		return m_density_network->padded_output_width();
	}

	std::unique_ptr<tcnn::Context> forward_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override {
		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->density_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		forward->rgb_network_input = tcnn::GPUMatrixDynamic<T>{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

		forward->pos_encoding_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			&forward->density_network_input, //  output of pos encoding is input of density network
			use_inference_params,
			prepare_input_gradients
		);

		// output of the density network is input of the rgb network
		forward->density_network_output = forward->rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, prepare_input_gradients);

		auto dir_out = forward->rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
		forward->dir_encoding_ctx = m_dir_encoding->forward(
			stream,
			input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
			&dir_out,
			use_inference_params,
			prepare_input_gradients
		);

		//TODO Need to Check for Seg

		if (output) {
			forward->rgb_network_output = tcnn::GPUMatrixDynamic<T>{output->data(), m_rgb_network->padded_output_width(), batch_size, output->layout()};
		}

		forward->rgb_network_ctx = m_rgb_network->forward(stream, forward->rgb_network_input, output ? &forward->rgb_network_output : nullptr, use_inference_params, prepare_input_gradients);

		// forward->seg_network_output = tcnn::GPUMatrixDynamic<T>{output->data(), m_seg_network->padded_output_width(), batch_size, output->layout()};
		// if (train_seg){
		// 	forward->seg_network_input = tcnn::GPUMatrixDynamic<T>{m_seg_network_input_width, batch_size, stream, m_pos_encoding->preferred_output_layout()};
		// 	forward->seg_network_ctx = m_seg_network->forward(stream, forward->density_network_output, output ? &forward->seg_network_output : nullptr, use_inference_params, prepare_input_gradients); 
		// }

		if (output) {
			tcnn::linear_kernel(extract_density<T>, 0, stream,
				batch_size, 
				m_dir_encoding->preferred_output_layout() == tcnn::AoS ? forward->density_network_output.stride() : 1, 
				padded_output_width(), 
				forward->density_network_output.data(), 
				output->data()+3
			);

			//TODO: Check this
			if (train_seg){
				tcnn::linear_kernel(pack_mask_with_density<T>, 0, stream,
					n_labels,
					batch_size,
					m_dir_encoding->preferred_output_layout() == tcnn::AoS ? forward->density_network_output.stride() : 1, 
					padded_output_width(),
					forward->seg_network_output.data(),
					output->data() + 4 
				);
			}

		}

		return forward;
	}

	void backward_impl(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input, // coord matrix
		const tcnn::GPUMatrixDynamic<T>& output, // rgbsigma_matrix
		const tcnn::GPUMatrixDynamic<T>& dL_doutput, // gradient matrix
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr, //coords gradient matrix
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite,
		bool use_mask_gradients = false 
	) override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		tcnn::GPUMatrix<T> dL_drgb{m_rgb_network->padded_output_width(), batch_size, stream};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream));
		tcnn::linear_kernel(extract_rgb<T>, 0, stream,
			batch_size*3, dL_drgb.m(), dL_doutput.m(), dL_doutput.data(), dL_drgb.data()
		);

		const tcnn::GPUMatrixDynamic<T> rgb_network_output{(T*)output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};
		tcnn::GPUMatrixDynamic<T> dL_drgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};
		m_rgb_network->backward(stream, *forward.rgb_network_ctx, forward.rgb_network_input, rgb_network_output, dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode);

		// Backprop through dir encoding if it is trainable or if we need input gradients
		if (m_dir_encoding->n_params() > 0 || dL_dinput) {
			tcnn::GPUMatrixDynamic<T> dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			tcnn::GPUMatrixDynamic<float> dL_ddir_encoding_input;
			if (dL_dinput) {
				dL_ddir_encoding_input = dL_dinput->slice_rows(m_dir_offset, m_dir_encoding->input_width());
			}

			m_dir_encoding->backward(
				stream,
				*forward.dir_encoding_ctx,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				forward.rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width()),
				dL_ddir_encoding_output,
				dL_dinput ? &dL_ddir_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}

		// Backprop through segmentation network
		// tcnn::GPUMatrix<T> dL_dseg{m_seg_network->padded_output_width(), batch_size, stream};
		// CUDA_CHECK_THROW(cudaMemsetAsync(dL_dseg.data(), 0, dL_dseg.n_bytes(), stream));

		// tcnn::GPUMatrixDynamic<T> dL_dseg_network_input{m_seg_network_input_width, batch_size, stream, m_pos_encoding->preferred_output_layout()};
		// const tcnn::GPUMatrixDynamic<T> seg_network_output{(T*)output.data(), m_seg_network->padded_output_width(), batch_size, output.layout()};

		// printf("\n layout of seg_network_output %d\n", seg_network_output.layout() == tcnn::AoS ? 1 : 0);
		// printf("\n layout of dL_dseg_network_input %d\n", dL_dseg_network_input.layout() == tcnn::AoS ? 1 : 0);
		// printf("\n layout of dL_dseg %d\n", dL_dseg.layout() == tcnn::AoS ? 1 : 0);
		// printf("\n layout of forward.rgb_network_input %d\n", forward.rgb_network_input.layout() == tcnn::AoS ? 1 : 0);
		// printf("\n layout of rgb_network_output %d\n", rgb_network_output.layout() == tcnn::AoS ? 1 : 0);
		// printf("\n layout of dL_drgb %d\n", dL_drgb.layout() == tcnn::AoS ? 1 : 0);
		// printf("\n layout of dL_drgb_network_input %d\n", dL_drgb_network_input.layout() == tcnn::AoS ? 1 : 0);
		// debug_print<network_precision_t>(dL_doutput, dL_doutput.n_bytes(), 0, 1, 16);

		// if (train_seg){
			// First extract mask
			// tcnn::linear_kernel(extract_mask<T>, 0, stream,
			// 	n_labels,
			// 	batch_size, 
			// 	dL_drgb.m(),  //16 
			// 	dL_doutput.m(),  //16
			// 	dL_doutput.data() + 4, 
			// 	dL_dseg.data() 
			// );

			// Layouts:
			// dL_doutput: AoS and CM
			// dL_dseg: AoS and CM
			// dL_drgb: AoS and CM

			// Getting the derivatives of dL/dseg_network_input
			// m_seg_network->backward(stream, *forward.seg_network_ctx, forward.seg_network_input, forward.seg_network_output, dL_dseg, &dL_dseg_network_input, use_inference_params, param_gradients_mode);
		// }


		// debug_print<network_precision_t>(dL_dseg_network_input, dL_dseg_network_input.n_bytes(), 0, dL_dseg_network_input.cols(), 16);
		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_output = dL_drgb_network_input.slice_rows(0, m_density_network->padded_output_width());

		tcnn::linear_kernel(add_density_gradient<T>, 0, stream,
			batch_size,
			dL_doutput.m(), //16
			dL_doutput.data(),
			dL_ddensity_network_output.layout() == tcnn::RM ? 1 : dL_ddensity_network_output.stride(), //1
			dL_ddensity_network_output.data()
		);

		// if (train_seg){
		// 	// Add the dseg_network_input to the dL_ddensity_network_output
		// 	tcnn::linear_kernel(add_density_gradient_and_seg_gradient<T>, 0, stream,
		// 		batch_size,
		// 		dL_dseg_network_input.layout() == tcnn::RM ? 1 : dL_dseg_network_input.m(), //16
		// 		dL_dseg_network_input.data(), //
		// 		dL_ddensity_network_output.layout() == tcnn::RM ? 1 : dL_ddensity_network_output.stride(),
		// 		dL_ddensity_network_output.data()
		// 	);

			//Layouts
			// dL_doutput: CM
			// dL_dseg_network_input: RM
			// dL_ddensity_network_output: RM

			// printf("\n dL_dsegnetwork input m %d\n", dL_dseg_network_input.m());
			// printf("\n dL_doutput m %d\n", dL_doutput.m());

			// printf("\n layout of dL_dseg_network_input %d\n", dL_dseg_network_input.layout() == tcnn::RM ? 1 : 0);
			// printf("\n layout of dL_ddensity_network_output %d\n", dL_ddensity_network_output.layout() == tcnn::RM ? 1 : 0);
			// printf("\n layou of dL_doutput %d\n", dL_doutput.layout() == tcnn::RM ? 1 : 0);
			// printf("\n dL_ddensity_network_output stride %d\n", dL_ddensity_network_output.stride());
			// exit(0);
		// }

		//TODO: Add the dL_drgb_network_input to the dL_dseg_network_input which will be dL_ddensity_network_input
		// Backprop through density network
		// dL_ddensity_network_output is RM / SoA
		// dL_drgb_network_input is RM / SoA
		// dL_drgb_network_input is CM / AoS

		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, forward.density_network_output, dL_ddensity_network_output, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);

		// Backprop through pos encoding if it is trainable or if we need input gradients
		if (dL_ddensity_network_input.data()) {
			tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
			}

			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_network_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}
	}

	void density(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("NerfNetwork::density input must be in column major format.");
		}

		uint32_t batch_size = output.n();
		tcnn::GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

		m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			density_network_input,
			use_inference_params
		);

		m_density_network->inference_mixed_precision(stream, density_network_input, output, use_inference_params);
	}

	// std::unique_ptr<tcnn::Context> density_forward(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) {
	// 	if (input.layout() != tcnn::CM) {
	// 		throw std::runtime_error("NerfNetwork::density_forward input must be in column major format.");
	// 	}

	// 	// Make sure our temporary buffers have the correct size for the given batch size
	// 	uint32_t batch_size = input.n();

	// 	auto forward = std::make_unique<ForwardContext>();

	// 	forward->density_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

	// 	forward->pos_encoding_ctx = m_pos_encoding->forward(
	// 		stream,
	// 		input.slice_rows(0, m_pos_encoding->input_width()),
	// 		&forward->density_network_input,
	// 		use_inference_params,
	// 		prepare_input_gradients
	// 	);

	// 	if (output) {
	// 		forward->density_network_output = tcnn::GPUMatrixDynamic<T>{output->data(), m_density_network->padded_output_width(), batch_size, output->layout()};
	// 	}

	// 	forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, output ? &forward->density_network_output : nullptr, use_inference_params, prepare_input_gradients);

	// 	return forward;
	// }

	// void density_backward(
	// 	cudaStream_t stream,
	// 	const tcnn::Context& ctx,
	// 	const tcnn::GPUMatrixDynamic<float>& input,
	// 	const tcnn::GPUMatrixDynamic<T>& output,
	// 	const tcnn::GPUMatrixDynamic<T>& dL_doutput,
	// 	tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
	// 	bool use_inference_params = false,
	// 	tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	// ) {
	// 	if (input.layout() != tcnn::CM || (dL_dinput && dL_dinput->layout() != tcnn::CM)) {
	// 		throw std::runtime_error("NerfNetwork::density_backward input must be in column major format.");
	// 	}

	// 	const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

	// 	// Make sure our temporary buffers have the correct size for the given batch size
	// 	uint32_t batch_size = input.n();

	// 	tcnn::GPUMatrixDynamic<T> dL_ddensity_network_input;
	// 	if (m_pos_encoding->n_params() > 0 || dL_dinput) {
	// 		dL_ddensity_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
	// 	}

	// 	m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, output, dL_doutput, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);

	// 	// Backprop through pos encoding if it is trainable or if we need input gradients
	// 	if (dL_ddensity_network_input.data()) {
	// 		tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
	// 		if (dL_dinput) {
	// 			dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
	// 		}

	// 		m_pos_encoding->backward(
	// 			stream,
	// 			*forward.pos_encoding_ctx,
	// 			input.slice_rows(0, m_pos_encoding->input_width()),
	// 			forward.density_network_input,
	// 			dL_ddensity_network_input,
	// 			dL_dinput ? &dL_dpos_encoding_input : nullptr,
	// 			use_inference_params,
	// 			param_gradients_mode
	// 		);
	// 	}
	// }

	void set_params_impl(T* params, T* inference_params, T* gradients) override {
		size_t offset = 0;
		m_density_network->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_density_network->n_params();

		m_rgb_network->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_rgb_network->n_params();

		m_pos_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_pos_encoding->n_params();

		m_dir_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_dir_encoding->n_params();

		// if (train_seg){
		// 	m_seg_network->set_params(params + offset, inference_params + offset, gradients + offset);
		// 	offset += m_seg_network->n_params();
		// }

	}

	void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, float scale = 1) override {
		m_density_network->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_density_network->n_params();

		m_rgb_network->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_rgb_network->n_params();

		m_pos_encoding->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_pos_encoding->n_params();

		m_dir_encoding->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_dir_encoding->n_params();

		// if (train_seg){
		// 	m_seg_network->initialize_params(rnd, params_full_precision, scale);
		// 	params_full_precision += m_seg_network->n_params();
		// }
	}

	size_t n_params() const override {
		if (train_seg) {
			// return m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params() + m_seg_network->n_params();
		}
		else 
		{
			return m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params(); 
		}
		
	}

	uint32_t padded_output_width() const override {
		return std::max(m_rgb_network->padded_output_width(), (uint32_t)4);
	}

	uint32_t input_width() const override {
		return m_dir_offset + m_n_dir_dims + m_n_extra_dims;
	}

	uint32_t output_width() const override {
		return 4;
	}

	uint32_t n_extra_dims() const {
		return m_n_extra_dims;
	}

	uint32_t required_input_alignment() const override {
		return 1; // No alignment required due to encoding
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		auto layers = m_density_network->layer_sizes();
		auto rgb_layers = m_rgb_network->layer_sizes();
		layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
		return layers;
	}

	uint32_t width(uint32_t layer) const override {
		if (layer == 0) {
			return m_pos_encoding->padded_output_width();
		} else if (layer < m_density_network->num_forward_activations() + 1) {
			return m_density_network->width(layer - 1);
		} else if (layer == m_density_network->num_forward_activations() + 1) {
			return m_rgb_network_input_width;
		} else {
			return m_rgb_network->width(layer - 2 - m_density_network->num_forward_activations());
		}
	}

	uint32_t num_forward_activations() const override {
		return m_density_network->num_forward_activations() + m_rgb_network->num_forward_activations() + 2;
	}

	std::pair<const T*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		if (layer == 0) {
			return {forward.density_network_input.data(), m_pos_encoding->preferred_output_layout()};
		} else if (layer < m_density_network->num_forward_activations() + 1) {
			return m_density_network->forward_activations(*forward.density_network_ctx, layer - 1);
		} else if (layer == m_density_network->num_forward_activations() + 1) {
			return {forward.rgb_network_input.data(), m_dir_encoding->preferred_output_layout()};
		} else {
			return m_rgb_network->forward_activations(*forward.rgb_network_ctx, layer - 2 - m_density_network->num_forward_activations());
		}
	}

	const std::shared_ptr<tcnn::Encoding<T>>& pos_encoding() const {
		return m_pos_encoding;
	}

	const std::shared_ptr<tcnn::Encoding<T>>& dir_encoding() const {
		return m_dir_encoding;
	}

	const std::shared_ptr<tcnn::Network<T>>& density_network() const {
		return m_density_network;
	}

	const std::shared_ptr<tcnn::Network<T>>& rgb_network() const {
		return m_rgb_network;
	}

	tcnn::json hyperparams() const override {
		json density_network_hyperparams = m_density_network->hyperparams();
		density_network_hyperparams["n_output_dims"] = m_density_network->padded_output_width();
		return {
			{"otype", "NerfNetwork"},
			{"pos_encoding", m_pos_encoding->hyperparams()},
			{"dir_encoding", m_dir_encoding->hyperparams()},
			{"density_network", density_network_hyperparams},
			{"rgb_network", m_rgb_network->hyperparams()},
		};
	}

private:
	std::shared_ptr<tcnn::Network<T>> m_density_network;
	// std::shared_ptr<tcnn::Network<T>> m_seg_network;
	std::shared_ptr<tcnn::Network<T>> m_rgb_network;
	std::shared_ptr<tcnn::Encoding<T>> m_pos_encoding;
	std::shared_ptr<tcnn::Encoding<T>> m_dir_encoding;

	uint32_t m_rgb_network_input_width;
	uint32_t m_seg_network_input_width;
	uint32_t m_n_pos_dims;
	uint32_t m_n_dir_dims;
	uint32_t m_n_extra_dims; // extra dimensions are assumed to be part of a compound encoding with dir_dims
	uint32_t m_dir_offset;

	bool train_seg = false;
	const uint32_t n_labels = 1;

	// // Storage of forward pass data
	struct ForwardContext : public tcnn::Context {
		tcnn::GPUMatrixDynamic<T> density_network_input;
		tcnn::GPUMatrixDynamic<T> seg_network_input;
		tcnn::GPUMatrixDynamic<T> seg_network_output;
		tcnn::GPUMatrixDynamic<T> density_network_output;
		tcnn::GPUMatrixDynamic<T> rgb_network_input;
		tcnn::GPUMatrix<T> rgb_network_output;

		std::unique_ptr<Context> pos_encoding_ctx;
		std::unique_ptr<Context> dir_encoding_ctx;

		std::unique_ptr<Context> density_network_ctx;
		std::unique_ptr<Context> rgb_network_ctx;
		std::unique_ptr<Context> seg_network_ctx;
	};
};

NGP_NAMESPACE_END
