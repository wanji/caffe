// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

#include <sys/time.h>

namespace caffe {

static struct timeval g_start;

void tic() {
  gettimeofday(&g_start, NULL);
}

double toc() {
  struct timeval current;
  gettimeofday(&current, NULL);
  return (current.tv_sec - g_start.tv_sec) * 1000.0 +
    (current.tv_usec - g_start.tv_usec) / 1000.0;
}

template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  for (int n = 0; n < num_; ++n) {
    // First, im2col
    im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_,
                      width_, kernel_size_, pad_, stride_, col_data);
    // Second, innerproduct with groups
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
        (Dtype)0., top_data + (*top)[0]->offset(n) + top_offset * g);
    }
    // third, add bias
    if (bias_term_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
          N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
          (Dtype)1., top_data + (*top)[0]->offset(n));
    }
  }
  return Dtype(0.);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
LOG(INFO) << "\t" << this->layer_param_.name() << "(gpu) - begin";
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* col_diff = col_buffer_.mutable_gpu_diff();
  // bias gradient if necessary
  Dtype* bias_diff = NULL;

  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    CUDA_CHECK(cudaMemset(bias_diff, 0,
        sizeof(Dtype) * this->blobs_[1]->count()));
    for (int n = 0; n < num_; ++n) {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
          1., top_diff + top[0]->offset(n),
          reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
          1., bias_diff);
    }
  }

  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  CUDA_CHECK(cudaMemset(weight_diff, 0,
      sizeof(Dtype) * this->blobs_[0]->count()));
tic();
double time[] = {0, 0, 0, 0, 0};
  for (int n = 0; n < num_; ++n) {
time[0] += toc(); tic();
    // since we saved memory in the forward pass by not storing all col data,
    // we will need to recompute them.
    im2col_gpu(bottom_data + (*bottom)[0]->offset(n), channels_, height_,
                      width_, kernel_size_, pad_, stride_, col_data);
time[1] += toc(); tic();
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., top_diff + top[0]->offset(n) + top_offset * g,
        col_data + col_offset * g, (Dtype)1.,
        weight_diff + weight_offset * g);
    }
time[2] += toc(); tic();
    // gradient w.r.t. bottom data, if necessary
    if (propagate_down) {
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight + weight_offset * g,
          top_diff + top[0]->offset(n) + top_offset * g,
          (Dtype)0., col_diff + col_offset * g);
      }
time[3] += toc(); tic();
      // col2im back to the data
      col2im_gpu(col_diff, channels_, height_, width_, kernel_size_, pad_,
          stride_, bottom_diff + (*bottom)[0]->offset(n));
time[4] += toc(); tic();
    }
  }
LOG(INFO) << "\t" << this->layer_param_.name() << "(gpu) - end"
  << "\t" << time[0]
  << "\t" << time[1]
  << "\t" << time[2]
  << "\t" << time[3]
  << "\t" << time[4];
}


INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
