/***********************************************************/
//  courtemanche.h is part of the Fibulator 3D atrial fibrillation
//  simulation software
//  copyright Shahriar Iravanian (siravan@emory.edu) 2016
//
//  courtemanche.cu/h implements Courtemanche, Ramirez, Nattel, 1998
//
//**********************************************************/

#pragma once

#include "common.h"
#include "ionic.h"

/*
   There are a total of 75 entries in the algebraic variable array.
   There are a total of 21 entries in each of the rate and state variable
   arrays.
   There are a total of 49 entries in the constant variable array.
*/

template <int N>
__device__ __host__ float Pow(float x) {
  return powf(x, N);
}

template <>
__device__ __host__ inline float Pow<0>(float x) {
  return 1.0;
}

template <>
__device__ __host__ inline float Pow<1>(float x) {
  return x;
}

template <>
__device__ __host__ inline float Pow<2>(float x) {
  return x * x;
}

template <>
__device__ __host__ inline float Pow<3>(float x) {
  return x * x * x;
}

template <>
__device__ __host__ inline float Pow<-1>(float x) {
  return 1.0 / x;
}

template <>
__device__ __host__ inline float Pow<-2>(float x) {
  return 1.0 / (x * x);
}

template <>
__device__ __host__ inline void init_cell<Courtemanche>(float *v, float *w,
                                                        int stim) {
  *v = stim ? 20.0 : -81.18;  // state[0] is V in membrane (millivolt).
  w[0] = 1.117e+01;  // state[_Na_i_] is intracellular_ion_concentrations
                     // (millimolar).
  w[1] = 2.98e-3;    // state[_m_] is m in fast_sodium_current_m_gate
                     // (dimensionless).
  w[2] = 9.649e-1;   // state[_h_] is h in fast_sodium_current_h_gate
                     // (dimensionless).
  w[3] = 9.775e-1;   // state[_j_] is j in fast_sodium_current_j_gate
                     // (dimensionless).
  w[4] = 1.39e+02;   // state[_K_i_] is intracellular_ion_concentrations
                     // (millimolar).
  w[5] = 3.043e-2;   // state[_oa_] is oa in
                     // transient_outward_K_current_oa_gate (dimensionless).
  w[6] = 9.992e-1;   // state[_oi_] is oi in
                     // transient_outward_K_current_oi_gate (dimensionless).
  w[7] = 4.966e-3;   // state[_ua_] is ua in
                     // ultrarapid_delayed_rectifier_K_current_ua_gate
                     // (dimensionless).
  w[8] = 9.986e-1;   // state[_ui_] is ui in
                     // ultrarapid_delayed_rectifier_K_current_ui_gate
                     // (dimensionless).
  w[9] = 3.296e-5;   // state[_xr_] is xr in
  // rapid_delayed_rectifier_K_current_xr_gate (dimensionless).
  w[10] = 1.869e-2;  // state[_xs_] is xs in
  // slow_delayed_rectifier_K_current_xs_gate (dimensionless).
  w[11] = 1.013e-4;  // state[_Ca_i_] is Ca_i in
                     // intracellular_ion_concentrations (millimolar).
  w[12] = 1.367e-4;  // state[_d_] is d in L_type_Ca_channel_d_gate
                     // (dimensionless).
  w[13] = 9.996e-1;  // state[_f_] is f in L_type_Ca_channel_f_gate
                     // (dimensionless).
  w[14] = 7.755e-1;  // state[_f_Ca_] is f_Ca in
                     // L_type_Ca_channel_f_Ca_gate (dimensionless).
  w[15] = 1.488;     // state[_Ca_rel_] is Ca_rel in
                     // intracellular_ion_concentrations (millimolar).
  w[16] = 0.0;       // state[_u_] is u in Ca_release_current_from_JSR_u_gate
                     // (dimensionless).
  w[17] = 1;         // state[_v_] is v in Ca_release_current_from_JSR_v_gate
                     // (dimensionless).
  w[18] = 0.9992;    // state[_w_] is w in
                     // Ca_release_current_from_JSR_w_gate (dimensionless).
  w[19] = 1.488;     // state[_Ca_up_] is Ca_up in
                     // intracellular_ion_concentrations (millimolar).
}

#define d_infinity inter[0]
#define f_infinity inter[1]
#define tau_w inter[2]
#define tau_d inter[3]
#define tau_f inter[4]
#define w_infinity inter[5]
#define m_inf inter[6]
#define h_inf inter[7]
#define j_inf inter[8]
#define tau_oa inter[9]
#define tau_oi inter[10]
#define tau_ua inter[11]
#define tau_ui inter[12]
#define tau_xr inter[13]
#define tau_xs inter[14]
#define tau_m inter[15]
#define tau_h inter[16]
#define tau_j inter[17]
#define oa_infinity inter[18]
#define oi_infinity inter[19]
#define ua_infinity inter[20]
#define ui_infinity inter[21]
#define xr_infinity inter[22]
#define xs_infinity inter[23]
#define g_Kur inter[24]
#define f_NaK inter[25]
#define i_NaCaa inter[26]
#define i_NaCab inter[27]
#define i_K1a inter[28]
#define i_Kra inter[29]

enum States {
  _Na_i_ = 1,
  _m_,
  _h_,
  _j_,
  _K_i_,
  _oa_,
  _oi_,
  _ua_,
  _ui_,
  _xr_,
  _xs_,
  _Ca_i_,
  _d_,
  _f_,
  _f_Ca_,
  _Ca_rel_,
  _u_,
  _v_,
  _w_,
  _Ca_up_
};

__device__ __host__ inline void calc_inter(float V, float *inter) {
  constexpr float R = 8.3143;   // R in membrane (joule/mole_kelvin).
  constexpr float T = 310;      // T in (kelvin).
  constexpr float F = 96.4867;  // F in membrane (coulomb/millimole).
  constexpr float Cm = 100;     //  Cm in membrane (picoF).
  constexpr float Na_o = 140;   //  Na_o (millimolar).
  constexpr float g_K1 = 0.09;  //  g_K1 (nanoS/picoF).
  constexpr float K_Q10 = 3;    //  transient_outward_K_current (dimensionless).
  constexpr float g_Kr =
      0.029411765;  //  rapid_delayed_rectifier_K_current (nanoS/picoF).
  constexpr float Ca_o = 1.8;         //  (millimolar).
  constexpr float I_NaCa_max = 1600;  //  Na_Ca_exchanger_current(picoA/picoF).
  constexpr float K_mNa = 87.5;       //  Na_Ca_exchanger_current (millimolar).
  constexpr float K_mCa = 1.38;       //  Na_Ca_exchanger_current (millimolar).
  constexpr float K_sat = 0.1;    //  Na_Ca_exchanger_current (dimensionless).
  constexpr float gamma_ = 0.35;  //  Na_Ca_exchanger_current (dimensionless).
  constexpr float sigma = 1.0;    //  sodium_potassium_pump (dimensionless).

  d_infinity = Pow<-1>(1.0 + expf((V + 10.0) / -8.0));
  tau_d =
      (fabsf(V + 10.0) < 1.0e-10
           ? 4.579 / (1.0 + expf((V + 10.0) / -6.24))
           : (1.0 - expf((V + 10.0) / -6.24)) /
                 (0.0350000 * (V + 10.0) * (1.0 + expf((V + 10.0) / -6.24))));

  f_infinity = expf(-(V + 28.0) / 6.9) / (1.0 + expf(-(V + 28.0) / 6.9));
  tau_f = 9.0 *
          Pow<-1>(0.0197000 * expf(-Pow<2>(0.0337) * Pow<2>(V + 10.0)) + 0.02);

  tau_w = (fabsf(V - 7.9) < 1.0e-10
               ? (6.0 * 0.2) / 1.3
               : (6.0 * (1.0 - expf(-(V - 7.9) / 5.0))) /
                     ((1.0 + 0.3 * expf(-(V - 7.9) / 5.0)) * 1.0 * (V - 7.9)));
  w_infinity = 1.0 - Pow<-1>(1.0 + expf(-(V - 40.0) / 17.0));

  float alpha_m =
      (fabsf(V - -47.13) < 0.001 ? 3.2 : (0.32 * (V + 47.13)) /
                                             (1.0 - expf(-0.1 * (V + 47.13))));
  float beta_m = 0.08 * expf(-V / 11.0);
  m_inf = alpha_m / (alpha_m + beta_m);
  tau_m = 1.0 / (alpha_m + beta_m);

  float alpha_h = (V < -40.0 ? 0.135 * expf((V + 80.0) / -6.8) : 0.0);
  float beta_h = (V < -40.0 ? 3.56 * expf(0.079 * V) + 310000. * expf(0.35 * V)
                            : 1.0 / (0.13 * (1.0 + expf((V + 10.66) / -11.1))));
  h_inf = alpha_h / (alpha_h + beta_h);
  tau_h = 1.0 / (alpha_h + beta_h);

  float alpha_j =
      (V < -40.0
           ? ((-127140. * expf(0.2444 * V) - 3.474e-05 * expf(-0.04391 * V)) *
              (V + 37.78)) /
                 (1.0 + expf(0.311 * (V + 79.23)))
           : 0.0);
  float beta_j =
      (V < -40.0
           ? (0.1212 * expf(-0.01052 * V)) / (1.0 + expf(-0.1378 * (V + 40.14)))
           : (0.3 * expf(-2.535e-07 * V)) / (1.0 + expf(-0.1 * (V + 32.0))));
  j_inf = alpha_j / (alpha_j + beta_j);
  tau_j = 1.0 / (alpha_j + beta_j);

  float alpha_oa = 0.65 * Pow<-1>(expf((V - -10.0) / -8.5) +
                                  expf(((V - -10.0) - 40.0) / -59.0));
  float beta_oa = 0.65 * Pow<-1>(2.5 + expf(((V - -10.0) + 72.0) / 17.0));
  tau_oa = Pow<-1>(alpha_oa + beta_oa) / K_Q10;
  oa_infinity = Pow<-1>(1.0 + expf(((V - -10.0) + 10.47) / -17.54));

  float alpha_oi = Pow<-1>(18.53 + 1.0 * expf(((V - -10.0) + 103.7) / 10.95));
  float beta_oi = Pow<-1>(35.56 + 1.0 * expf(((V - -10.0) - 8.74) / -7.44));
  tau_oi = Pow<-1>(alpha_oi + beta_oi) / K_Q10;
  oi_infinity = Pow<-1>(1.0 + expf(((V - -10.0) + 33.1) / 5.3));

  float alpha_ua = 0.65 * Pow<-1>(expf((V - -10.0) / -8.5) +
                                  expf(((V - -10.0) - 40.0) / -59.0));
  float beta_ua = 0.65 * Pow<-1>(2.5 + expf(((V - -10.0) + 72.0) / 17.0));
  tau_ua = Pow<-1>(alpha_ua + beta_ua) / K_Q10;
  ua_infinity = Pow<-1>(1.0 + expf(((V - -10.0) + 20.3) / -9.6));

  float alpha_ui = Pow<-1>(21.0 + 1.0 * expf(((V - -10.0) - 195.000) / -28.0));
  float beta_ui = 1.0 / expf(((V - -10.0) - 168.0) / -16.0);
  tau_ui = Pow<-1>(alpha_ui + beta_ui) / K_Q10;
  ui_infinity = Pow<-1>(1.0 + expf(((V - -10.0) - 109.45) / 27.48));

  float alpha_xr =
      (fabsf(V + 14.1) < 1.0e-10
           ? 0.0015
           : (0.0003 * (V + 14.1)) / (1.0 - expf((V + 14.1) / -5.0)));
  float beta_xr =
      (fabsf(V - 3.3328) < 1.0e-10
           ? 0.000378361
           : (7.3898e-05 * (V - 3.3328)) / (expf((V - 3.3328) / 5.1237) - 1.0));
  tau_xr = Pow<-1>(alpha_xr + beta_xr);
  xr_infinity = Pow<-1>(1.0 + expf((V + 14.1) / -6.5));

  float alpha_xs =
      (fabsf(V - 19.9) < 1.0e-10
           ? 0.00068
           : (4.0e-05 * (V - 19.9)) / (1.0 - expf((V - 19.9) / -17.0)));
  float beta_xs =
      (fabsf(V - 19.9) < 1.0e-10
           ? 0.000315
           : (3.5e-05 * (V - 19.9)) / (expf((V - 19.9) / 9.0) - 1.0));
  tau_xs = 0.5 * Pow<-1>(alpha_xs + beta_xs);
  xs_infinity = sqrt(Pow<-1>(1.0 + expf((V - 19.9) / -12.7)));

  g_Kur = 0.005 + 0.05 / (1.0 + expf((V - 15.0) / -13.0));

  f_NaK = Pow<-1>(1.0 + 0.1245 * expf((-0.1 * F * V) / (R * T)) +
                  0.0365 * sigma * expf((-F * V) / (R * T)));

  float i_NaCad = (Pow<3>(K_mNa) + Pow<3>(Na_o)) * (K_mCa + Ca_o) *
                  (1.0 + K_sat * expf(((gamma_ - 1.0) * V * F) / (R * T)));

  // Pow<3>(state[_Na_i_])
  i_NaCaa =
      (Cm * I_NaCa_max * (expf((gamma_ * F * V) / (R * T)) * Ca_o)) / i_NaCad;

  // state[_Ca_i_]
  i_NaCab = (Cm * I_NaCa_max *
             (expf(((gamma_ - 1.0) * F * V) / (R * T)) * Pow<3>(Na_o))) /
            i_NaCad;

  i_K1a = (Cm * g_K1) / (1.0 + expf(0.07 * (V + 80.0)));  // * (V - E_K)

  i_Kra = (Cm * g_Kr) /
          (1.0 + expf((V + 15.0) / 22.4));  // * state[_xr_] * (V - E_K)
}

__device__ __host__ inline float integrate_gate(float g, float g_inf, float tau,
                                                const Config &cfg) {
  // return (g_inf - g) / tau;
  // implenting Non-Standard-Finite-Difference (NSFD) method
  return (g - g_inf) * expm1(-cfg.dt / tau) / cfg.dt;
}

template <>
__device__ __host__ inline void deriv<Courtemanche>(float *state, float *rate,
                                                    const Config &cfg) {
  constexpr float R = 8.3143;     // (joule/mole_kelvin).
  constexpr float T = 310;        // (kelvin).
  constexpr float F = 96.4867;    // (coulomb/millimole).
  constexpr float Cm = 100;       //  Cm is Cm in membrane (picoF).
  constexpr float g_Na = 7.8;     //  fast_sodium_current (nanoS/picoF).
  constexpr float Na_o = 140;     //  (millimolar).
  constexpr float K_o = 5.4;      //  (millimolar).
  constexpr float g_to = 0.1652;  //  transient_outward_K_current (nanoS/picoF).
  constexpr float g_Ks =
      0.12941176;  //  slow_delayed_rectifier_K_current (nanoS/picoF).
  constexpr float g_Ca_L = 0.12375;  //  L_type_Ca_channel (nanoS/picoF).
  constexpr float Km_Na_i = 10;      //  sodium_potassium_pump (millimolar).
  constexpr float Km_K_o = 1.5;      //  sodium_potassium_pump (millimolar).
  constexpr float i_NaK_max =
      0.59933874;  //  sodium_potassium_pump (picoA/picoF).
  constexpr float i_CaP_max =
      0.275;  // sarcolemmal_calcium_pump_current (picoA/picoF).
  constexpr float g_B_Na = 0.0006744375;  //  background_currents (nanoS/picoF).
  constexpr float g_B_Ca = 0.001131;      //  background_currents (nanoS/picoF).
  constexpr float g_B_K = 0;              //  background_currents (nanoS/picoF).
  constexpr float Ca_o = 1.8;             //  (millimolar).
  constexpr float K_rel =
      30;  //  Ca_release_current_from_JSR (per_millisecond).
  constexpr float tau_tr =
      180;  //  transfer_current_from_NSR_to_JSR (millisecond).
  constexpr float I_up_max =
      0.005;  //  Ca_uptake_current_by_the_NSR (millimolar/millisecond).
  constexpr float K_up =
      0.00092;  //  Ca_uptake_current_by_the_NSR (millimolar).
  constexpr float Ca_up_max = 15;  //  Ca_leak_current_by_the_NSR (millimolar).
  constexpr float CMDN_max =
      0.05;  //  CMDN_max is CMDN_max in Ca_buffers (millimolar).
  constexpr float TRPN_max =
      0.07;  //  TRPN_max is TRPN_max in Ca_buffers (millimolar).
  constexpr float CSQN_max =
      10;  //  CSQN_max is CSQN_max in Ca_buffers (millimolar).
  constexpr float Km_CMDN =
      0.00238;  //  Km_CMDN is Km_CMDN in Ca_buffers (millimolar).
  constexpr float Km_TRPN =
      0.0005;  //  Km_TRPN is Km_TRPN in Ca_buffers (millimolar).
  constexpr float Km_CSQN =
      0.8;  //  Km_CSQN is Km_CSQN in Ca_buffers (millimolar).
  constexpr float V_cell =
      20100;  //  intracellular_ion_concentrations (micrometre_3).
  constexpr float V_i =
      V_cell * 0.68;  //  intracellular_ion_concentrations (micrometre_3).
  constexpr float tau_f_Ca =
      2.0;  //  L_type_Ca_channel_f_Ca_gate (millisecond).
  constexpr float tau_u =
      8.0;  //  Ca_release_current_from_JSR_u_gate (millisecond).
  constexpr float V_rel =
      0.00480000 * V_cell;  // intracellular_ion_concentrations (micrometre_3).
  constexpr float V_up =
      0.0552000 * V_cell;  // intracellular_ion_concentrations (micrometre_3).

  float V = state[0];

  int i = static_cast<int>(V + 100);
  i = i < 0 ? 0 : (i >= Courtemanche::TABLE_ROWS ? Courtemanche::TABLE_ROWS - 1
                                                 : i);
  const float *inter = &cfg.table[i * Courtemanche::TABLE_COLS];

  float f_Ca_infinity = Pow<-1>(1.0 + state[_Ca_i_] / 0.00035);

  rate[_d_] = integrate_gate(state[_d_], d_infinity, tau_d, cfg);
  rate[_f_Ca_] = integrate_gate(state[_f_Ca_], f_Ca_infinity, tau_f_Ca, cfg);
  rate[_f_] = integrate_gate(state[_f_], f_infinity, tau_f, cfg);
  rate[_w_] = integrate_gate(state[_w_], w_infinity, tau_w, cfg);
  rate[_m_] = integrate_gate(state[_m_], m_inf, tau_m, cfg);
  rate[_h_] = integrate_gate(state[_h_], h_inf, tau_h, cfg);
  rate[_j_] = integrate_gate(state[_j_], j_inf, tau_j, cfg);
  rate[_oa_] = integrate_gate(state[_oa_], oa_infinity, tau_oa, cfg);
  rate[_oi_] = integrate_gate(state[_oi_], oi_infinity, tau_oi, cfg);
  rate[_ua_] = integrate_gate(state[_ua_], ua_infinity, tau_ua, cfg);
  rate[_ui_] = integrate_gate(state[_ui_], ui_infinity, tau_ui, cfg);
  rate[_xr_] = integrate_gate(state[_xr_], xr_infinity, tau_xr, cfg);
  rate[_xs_] = integrate_gate(state[_xs_], xs_infinity, tau_xs, cfg);

  float E_K = ((R * T) / F) * log(K_o / state[_K_i_]);
  float i_K1 = i_K1a * (V - E_K);
  /*
    chronic model based on "Patient-derived models link re-entrant driver
    localization in atrial fibrillation to fibrosis spatial pattern."
    Cardiovasc Res. 2016 Jun 1;110(3):443-54.
  */
  float i_to = (cfg.chronic ? 0.5 : 1.0) * Cm * g_to * Pow<3>(state[_oa_]) *
               state[_oi_] * (V - E_K);
  float i_Kur = (cfg.chronic ? 0.5 : 1.0) * Cm * g_Kur * Pow<3>(state[_ua_]) *
                state[_ui_] * (V - E_K);
  float i_Kr = i_Kra * state[_xr_] * (V - E_K);
  float i_Ks = Cm * g_Ks * Pow<2>(state[_xs_]) * (V - E_K);
  float i_NaK = (((Cm * i_NaK_max * f_NaK * 1.0) /
                  (1.0 + sqrt(Pow<3>(Km_Na_i / state[_Na_i_])))) *
                 K_o) /
                (K_o + Km_K_o);
  float i_B_K = Cm * g_B_K * (V - E_K);
  rate[_K_i_] =
      (2.0 * i_NaK - (i_K1 + i_to + i_Kur + i_Kr + i_Ks + i_B_K)) / (V_i * F);

  float E_Na = ((R * T) / F) * log(Na_o / state[_Na_i_]);
  float i_Na =
      Cm * g_Na * Pow<3>(state[_m_]) * state[_h_] * state[_j_] * (V - E_Na);
  float i_NaCa = i_NaCaa * Pow<3>(state[_Na_i_]) - i_NaCab * state[_Ca_i_];
  float i_B_Na = Cm * g_B_Na * (V - E_Na);
  rate[_Na_i_] = (-3.0 * i_NaK - (3.0 * i_NaCa + i_B_Na + i_Na)) / (V_i * F);

  float i_st = 0.0;
  float i_Ca_L = (cfg.chronic ? 0.3 : 1.0) * Cm * g_Ca_L * state[_d_] *
                 state[_f_] * state[_f_Ca_] * (V - 65.0);
  float i_CaP = (Cm * i_CaP_max * state[_Ca_i_]) / (0.0005 + state[_Ca_i_]);
  float E_Ca = ((R * T) / (2.0 * F)) * log(Ca_o / state[_Ca_i_]);
  float i_B_Ca = Cm * g_B_Ca * (V - E_Ca);
  rate[0] = -(i_Na + i_K1 + i_to + i_Kur + i_Kr + i_Ks + i_B_Na + i_B_Ca +
              i_NaK + i_CaP + i_NaCa + i_Ca_L + i_st) /
            Cm;

  float i_rel = K_rel * Pow<2>(state[_u_]) * state[_v_] * state[_w_] *
                (state[_Ca_rel_] - state[_Ca_i_]);
  float i_tr = (state[_Ca_up_] - state[_Ca_rel_]) / tau_tr;
  rate[_Ca_rel_] =
      (i_tr - i_rel) *
      Pow<-1>(1.0 + (CSQN_max * Km_CSQN) / Pow<2>(state[_Ca_rel_] + Km_CSQN));

  float Fn = 1000.0 * (1.0e-15 * V_rel * i_rel -
                       (1.0e-15 / (2.0 * F)) * (0.5 * i_Ca_L - 0.2 * i_NaCa));
  float u_infinity = Pow<-1>(1.0 + expf(-(Fn - 3.4175e-13) / 1.367e-15));
  // rate[_u_] = (u_infinity - state[_u_]) / tau_u;
  rate[_u_] = integrate_gate(state[_u_], u_infinity, tau_u, cfg);

  float tau_v = 1.91 + 2.09 * u_infinity;
  float v_infinity = 1.0 - Pow<-1>(1.0 + expf(-(Fn - 6.835e-14) / 1.367e-15));
  // rate[_v_] = (v_infinity - state[_v_]) / tau_v;
  rate[_v_] = integrate_gate(state[_v_], v_infinity, tau_v, cfg);

  float i_up = I_up_max / (1.0 + K_up / state[_Ca_i_]);
  float i_up_leak = (I_up_max * state[_Ca_up_]) / Ca_up_max;
  rate[_Ca_up_] = i_up - (i_up_leak + (i_tr * V_rel) / V_up);

  float B1 = (2.0 * i_NaCa - (i_CaP + i_Ca_L + i_B_Ca)) / (2.0 * V_i * F) +
             (V_up * (i_up_leak - i_up) + i_rel * V_rel) / V_i;
  float B2 = 1.0 + (TRPN_max * Km_TRPN) / Pow<2>(state[_Ca_i_] + Km_TRPN) +
             (CMDN_max * Km_CMDN) / Pow<2>(state[_Ca_i_] + Km_CMDN);
  rate[_Ca_i_] = B1 / B2;
}

#undef d_infinity
#undef f_infinity
#undef tau_w
#undef tau_d
#undef tau_f
#undef w_infinity
#undef m_inf
#undef h_inf
#undef j_inf
#undef tau_oa
#undef tau_oi
#undef tau_ua
#undef tau_ui
#undef tau_xr
#undef tau_xs
#undef tau_m
#undef tau_h
#undef tau_j
#undef oa_infinity
#undef oi_infinity
#undef ua_infinity
#undef ui_infinity
#undef xr_infinity
#undef xs_infinity
#undef g_Kur
#undef f_NaK
#undef i_NaCaa
#undef i_NaCab
#undef i_K1a
#undef i_Kra

template <>
inline void init_table<Courtemanche>(float *table) {
  for (int i = 0; i < Courtemanche::TABLE_ROWS; i++) {
    calc_inter(static_cast<float>(i - 100),
               &table[i * Courtemanche::TABLE_COLS]);
  }
}
