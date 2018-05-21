/***********************************************************/
//  context.cu is part of the Fibulator 3D atrial fibrillation
//  simulation software
//  copyright Shahriar Iravanian (siravan@emory.edu) 2016
//
//  common.h defined Config and the base methods for the ionic models
//  static inheritance via templates
//
//**********************************************************/

#pragma once

#include <stdexcept>

struct Config {
  float dt;
  float diff;
  float apd;
  float vmin;
  float vmax;
  int alpha;
  int beta;
  const float *table;

  float base_apd;
  float grad_apd;
  float3 normal;
  bool chronic;
  bool cluster;
};

template <typename T>
__device__ __host__ void init_cell(float *v, float *w, int stim) {}

template <typename T>
__device__ __host__ void deriv(float *state, float *rate, const Config &cfg) {}

template <typename T>
__host__ void init_table(float *table) {
  throw std::runtime_error("init_table without ionic model");
}
