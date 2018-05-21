/***********************************************************/
//  ionic.h is part of the Fibulator 3D atrial fibrillation
//  simulation software
//  copyright Shahriar Iravanian (siravan@emory.edu) 2016-2017
//
//  ionic.h contains class definitions for the ionic models
//
//**********************************************************/

#pragma once

struct Fenton {
  static constexpr int VARIABLES = 4;

#ifdef ADAMS_BASHFORTH
  static constexpr int PARAMS = 2 * VARIABLES;
#else
  static constexpr int PARAMS = VARIABLES;
#endif

  static constexpr float UPSTROKE = 0.5;
  static constexpr float ERROR_LEVEL = 0.02;
  static constexpr float DIFF = 0.5f;  // diffusion coefficient
  static constexpr float APD = 1.0f;   // APD ratio
  static constexpr float VMIN = 0.0f;
  static constexpr float VMAX = 1.0f;
  static constexpr int TABLE_ROWS = 1;
  static constexpr int TABLE_COLS = 1;
  static constexpr bool NEEDS_NORMAL = false;
  static constexpr bool CACHE_TABLE = false;
};

struct Courtemanche {
  static constexpr int VARIABLES = 21;

#ifdef ADAMS_BASHFORTH
  static constexpr int PARAMS = 2 * VARIABLES;
#else
  static constexpr int PARAMS = VARIABLES;
#endif

  static constexpr float UPSTROKE = -40.0;
  static constexpr float ERROR_LEVEL = 0.001;
  static constexpr float DIFF = 1.0f;  // diffusion coefficient
  static constexpr float APD = 1.0f;   // APD ratio
  static constexpr float VMIN = -80.0f;
  static constexpr float VMAX = +20.0f;
  static constexpr int TABLE_ROWS = 150;
  static constexpr int TABLE_COLS = 30;
  static constexpr bool NEEDS_NORMAL = true;
  static constexpr bool CACHE_TABLE = false;
};
