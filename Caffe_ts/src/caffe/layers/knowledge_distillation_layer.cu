#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>
#include <stdio.h>

#include "caffe/layers/knowledge_distillation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	
template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* soft_label, Dtype* loss_data, Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    loss_data[index] = -soft_label[index] * (log(max(prob_data[index], Dtype(FLT_MIN)))-log(max(soft_label[index], Dtype(FLT_MIN))));
    counts[index] = 1;
  }
}

template <typename Dtype>
void KnowledgeDistillationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Both logits are divided by the temperature T.
  caffe_gpu_memcpy(bottom[0]->count() * sizeof(Dtype), bottom[0]->gpu_data(), s_logit_.mutable_gpu_data());
  caffe_gpu_scal(bottom[0]->count(), Dtype(1)/T, s_logit_.mutable_gpu_data());
  caffe_gpu_memcpy(bottom[1]->count() * sizeof(Dtype), bottom[1]->gpu_data(), t_logit_.mutable_gpu_data());
  caffe_gpu_scal(bottom[0]->count(), Dtype(1)/T, t_logit_.mutable_gpu_data());
  // The forward pass computes the softmax prob values.
  s_softmax_layer_->Forward(s_softmax_bottom_vec_, s_softmax_top_vec_);
  t_softmax_layer_->Forward(t_softmax_bottom_vec_, t_softmax_top_vec_);
  const Dtype* prob_data = s_prob_.gpu_data();
  const Dtype* soft_label = t_prob_.gpu_data();
  int dim = s_prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  int pos;
  // Compute KL divergence.
  if (bottom.size() == 2) { // label inputs or ignore_label are not given.
    count = outer_num_ * inner_num_;
	
	Dtype* loss_data = bottom[0]->mutable_gpu_diff();
	Dtype* counts = s_prob_.mutable_gpu_diff();
	const int nthreads = (outer_num_ - 1) * dim + (bottom[0]->shape(softmax_axis_) - 1) * inner_num_ + inner_num_;
	SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, soft_label, loss_data, counts);
	caffe_gpu_asum(nthreads, loss_data, &loss);
  }

  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
}

/*
template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}
*/

template <typename Dtype>
void KnowledgeDistillationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = s_prob_.gpu_data();
    caffe_gpu_memcpy(s_prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* soft_label = t_prob_.gpu_data();
    int dim = s_prob_.count() / outer_num_;
    int count = outer_num_ * inner_num_;
    // The gradients here are multiplied by T,
    // which is T^2 (as suggested in the paper) * 1/T (logits divided by T).
    caffe_gpu_axpby<Dtype>(outer_num_*dim, -T, soft_label, T, bottom_diff);

    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_gpu_scal(s_prob_.count(), loss_weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(KnowledgeDistillationLayer);

}  // namespace caffe
