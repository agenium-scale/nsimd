#ifndef RENAMEAVX_H
#define RENAMEAVX_H

/* ------------------------------------------------------------------------- */
/* Naming of functions avx */

#ifdef NSIMD_AVX

#ifdef DETERMINISTIC

#define xsin nsimd_sleef_sin_u35d_avx_f64
#define xsinf nsimd_sleef_sin_u35d_avx_f32
#define xcos nsimd_sleef_cos_u35d_avx_f64
#define xcosf nsimd_sleef_cos_u35d_avx_f32
#define xsincos nsimd_sleef_sincos_u35d_avx_f64
#define xsincosf nsimd_sleef_sincos_u35d_avx_f32
#define xtan nsimd_sleef_tan_u35d_avx_f64
#define xtanf nsimd_sleef_tan_u35d_avx_f32
#define xasin nsimd_sleef_asin_u35d_avx_f64
#define xasinf nsimd_sleef_asin_u35d_avx_f32
#define xacos nsimd_sleef_acos_u35d_avx_f64
#define xacosf nsimd_sleef_acos_u35d_avx_f32
#define xatan nsimd_sleef_atan_u35d_avx_f64
#define xatanf nsimd_sleef_atan_u35d_avx_f32
#define xatan2 nsimd_sleef_atan2_u35d_avx_f64
#define xatan2f nsimd_sleef_atan2_u35d_avx_f32
#define xlog nsimd_sleef_log_u35d_avx_f64
#define xlogf nsimd_sleef_log_u35d_avx_f32
#define xcbrt nsimd_sleef_cbrt_u35d_avx_f64
#define xcbrtf nsimd_sleef_cbrt_u35d_avx_f32
#define xsin_u1 nsimd_sleef_sin_u10d_avx_f64
#define xsinf_u1 nsimd_sleef_sin_u10d_avx_f32
#define xcos_u1 nsimd_sleef_cos_u10d_avx_f64
#define xcosf_u1 nsimd_sleef_cos_u10d_avx_f32
#define xsincos_u1 nsimd_sleef_sincos_u10d_avx_f64
#define xsincosf_u1 nsimd_sleef_sincos_u10d_avx_f32
#define xtan_u1 nsimd_sleef_tan_u10d_avx_f64
#define xtanf_u1 nsimd_sleef_tan_u10d_avx_f32
#define xasin_u1 nsimd_sleef_asin_u10d_avx_f64
#define xasinf_u1 nsimd_sleef_asin_u10d_avx_f32
#define xacos_u1 nsimd_sleef_acos_u10d_avx_f64
#define xacosf_u1 nsimd_sleef_acos_u10d_avx_f32
#define xatan_u1 nsimd_sleef_atan_u10d_avx_f64
#define xatanf_u1 nsimd_sleef_atan_u10d_avx_f32
#define xatan2_u1 nsimd_sleef_atan2_u10d_avx_f64
#define xatan2f_u1 nsimd_sleef_atan2_u10d_avx_f32
#define xlog_u1 nsimd_sleef_log_u10d_avx_f64
#define xlogf_u1 nsimd_sleef_log_u10d_avx_f32
#define xcbrt_u1 nsimd_sleef_cbrt_u10d_avx_f64
#define xcbrtf_u1 nsimd_sleef_cbrt_u10d_avx_f32
#define xexp nsimd_sleef_exp_u10d_avx_f64
#define xexpf nsimd_sleef_exp_u10d_avx_f32
#define xpow nsimd_sleef_pow_u10d_avx_f64
#define xpowf nsimd_sleef_pow_u10d_avx_f32
#define xsinh nsimd_sleef_sinh_u10d_avx_f64
#define xsinhf nsimd_sleef_sinh_u10d_avx_f32
#define xcosh nsimd_sleef_cosh_u10d_avx_f64
#define xcoshf nsimd_sleef_cosh_u10d_avx_f32
#define xtanh nsimd_sleef_tanh_u10d_avx_f64
#define xtanhf nsimd_sleef_tanh_u10d_avx_f32
#define xsinh_u35 nsimd_sleef_sinh_u35d_avx_f64
#define xsinhf_u35 nsimd_sleef_sinh_u35d_avx_f32
#define xcosh_u35 nsimd_sleef_cosh_u35d_avx_f64
#define xcoshf_u35 nsimd_sleef_cosh_u35d_avx_f32
#define xtanh_u35 nsimd_sleef_tanh_u35d_avx_f64
#define xtanhf_u35 nsimd_sleef_tanh_u35d_avx_f32
#define xfastsin_u3500 nsimd_sleef_fastsin_u3500d_avx_f64
#define xfastsinf_u3500 nsimd_sleef_fastsin_u3500d_avx_f32
#define xfastcos_u3500 nsimd_sleef_fastcos_u3500d_avx_f64
#define xfastcosf_u3500 nsimd_sleef_fastcos_u3500d_avx_f32
#define xfastpow_u3500 nsimd_sleef_fastpow_u3500d_avx_f64
#define xfastpowf_u3500 nsimd_sleef_fastpow_u3500d_avx_f32
#define xasinh nsimd_sleef_asinh_u10d_avx_f64
#define xasinhf nsimd_sleef_asinh_u10d_avx_f32
#define xacosh nsimd_sleef_acosh_u10d_avx_f64
#define xacoshf nsimd_sleef_acosh_u10d_avx_f32
#define xatanh nsimd_sleef_atanh_u10d_avx_f64
#define xatanhf nsimd_sleef_atanh_u10d_avx_f32
#define xexp2 nsimd_sleef_exp2_u10d_avx_f64
#define xexp2f nsimd_sleef_exp2_u10d_avx_f32
#define xexp2_u35 nsimd_sleef_exp2_u35d_avx_f64
#define xexp2f_u35 nsimd_sleef_exp2_u35d_avx_f32
#define xexp10 nsimd_sleef_exp10_u10d_avx_f64
#define xexp10f nsimd_sleef_exp10_u10d_avx_f32
#define xexp10_u35 nsimd_sleef_exp10_u35d_avx_f64
#define xexp10f_u35 nsimd_sleef_exp10_u35d_avx_f32
#define xexpm1 nsimd_sleef_expm1_u10d_avx_f64
#define xexpm1f nsimd_sleef_expm1_u10d_avx_f32
#define xlog10 nsimd_sleef_log10_u10d_avx_f64
#define xlog10f nsimd_sleef_log10_u10d_avx_f32
#define xlog2 nsimd_sleef_log2_u10d_avx_f64
#define xlog2f nsimd_sleef_log2_u10d_avx_f32
#define xlog2_u35 nsimd_sleef_log2_u35d_avx_f64
#define xlog2f_u35 nsimd_sleef_log2_u35d_avx_f32
#define xlog1p nsimd_sleef_log1p_u10d_avx_f64
#define xlog1pf nsimd_sleef_log1p_u10d_avx_f32
#define xsincospi_u05 nsimd_sleef_sincospi_u05d_avx_f64
#define xsincospif_u05 nsimd_sleef_sincospi_u05d_avx_f32
#define xsincospi_u35 nsimd_sleef_sincospi_u35d_avx_f64
#define xsincospif_u35 nsimd_sleef_sincospi_u35d_avx_f32
#define xsinpi_u05 nsimd_sleef_sinpi_u05d_avx_f64
#define xsinpif_u05 nsimd_sleef_sinpi_u05d_avx_f32
#define xcospi_u05 nsimd_sleef_cospi_u05d_avx_f64
#define xcospif_u05 nsimd_sleef_cospi_u05d_avx_f32
#define xldexp nsimd_sleef_ldexp_avx_f64
#define xldexpf nsimd_sleef_ldexp_avx_f32
#define xilogb nsimd_sleef_ilogb_avx_f64
#define xilogbf nsimd_sleef_ilogb_avx_f32
#define xfma nsimd_sleef_fma_avx_f64
#define xfmaf nsimd_sleef_fma_avx_f32
#define xsqrt nsimd_sleef_sqrt_avx_f64
#define xsqrtf nsimd_sleef_sqrt_avx_f32
#define xsqrt_u05 nsimd_sleef_sqrt_u05d_avx_f64
#define xsqrtf_u05 nsimd_sleef_sqrt_u05d_avx_f32
#define xsqrt_u35 nsimd_sleef_sqrt_u35d_avx_f64
#define xsqrtf_u35 nsimd_sleef_sqrt_u35d_avx_f32
#define xhypot_u05 nsimd_sleef_hypot_u05d_avx_f64
#define xhypotf_u05 nsimd_sleef_hypot_u05d_avx_f32
#define xhypot_u35 nsimd_sleef_hypot_u35d_avx_f64
#define xhypotf_u35 nsimd_sleef_hypot_u35d_avx_f32
#define xfabs nsimd_sleef_fabs_avx_f64
#define xfabsf nsimd_sleef_fabs_avx_f32
#define xcopysign nsimd_sleef_copysign_avx_f64
#define xcopysignf nsimd_sleef_copysign_avx_f32
#define xfmax nsimd_sleef_fmax_avx_f64
#define xfmaxf nsimd_sleef_fmax_avx_f32
#define xfmin nsimd_sleef_fmin_avx_f64
#define xfminf nsimd_sleef_fmin_avx_f32
#define xfdim nsimd_sleef_fdim_avx_f64
#define xfdimf nsimd_sleef_fdim_avx_f32
#define xtrunc nsimd_sleef_trunc_avx_f64
#define xtruncf nsimd_sleef_trunc_avx_f32
#define xfloor nsimd_sleef_floor_avx_f64
#define xfloorf nsimd_sleef_floor_avx_f32
#define xceil nsimd_sleef_ceil_avx_f64
#define xceilf nsimd_sleef_ceil_avx_f32
#define xround nsimd_sleef_round_avx_f64
#define xroundf nsimd_sleef_round_avx_f32
#define xrint nsimd_sleef_rint_avx_f64
#define xrintf nsimd_sleef_rint_avx_f32
#define xnextafter nsimd_sleef_nextafter_avx_f64
#define xnextafterf nsimd_sleef_nextafter_avx_f32
#define xfrfrexp nsimd_sleef_frfrexp_avx_f64
#define xfrfrexpf nsimd_sleef_frfrexp_avx_f32
#define xexpfrexp nsimd_sleef_expfrexp_avx_f64
#define xexpfrexpf nsimd_sleef_expfrexp_avx_f32
#define xfmod nsimd_sleef_fmod_avx_f64
#define xfmodf nsimd_sleef_fmod_avx_f32
#define xremainder nsimd_sleef_remainder_avx_f64
#define xremainderf nsimd_sleef_remainder_avx_f32
#define xmodf nsimd_sleef_modf_avx_f64
#define xmodff nsimd_sleef_modf_avx_f32
#define xlgamma_u1 nsimd_sleef_lgamma_u10d_avx_f64
#define xlgammaf_u1 nsimd_sleef_lgamma_u10d_avx_f32
#define xtgamma_u1 nsimd_sleef_tgamma_u10d_avx_f64
#define xtgammaf_u1 nsimd_sleef_tgamma_u10d_avx_f32
#define xerf_u1 nsimd_sleef_erf_u10d_avx_f64
#define xerff_u1 nsimd_sleef_erf_u10d_avx_f32
#define xerfc_u15 nsimd_sleef_erfc_u15d_avx_f64
#define xerfcf_u15 nsimd_sleef_erfc_u15d_avx_f32
#define xgetInt nsimd_sleef_getInt_avx_f64
#define xgetIntf nsimd_sleef_getInt_avx_f32
#define xgetPtr nsimd_sleef_getPtr_avx_f64
#define xgetPtrf nsimd_sleef_getPtr_avx_f32

#else

#define xsin nsimd_sleef_sin_u35_avx_f64
#define xsinf nsimd_sleef_sin_u35_avx_f32
#define xcos nsimd_sleef_cos_u35_avx_f64
#define xcosf nsimd_sleef_cos_u35_avx_f32
#define xsincos nsimd_sleef_sincos_u35_avx_f64
#define xsincosf nsimd_sleef_sincos_u35_avx_f32
#define xtan nsimd_sleef_tan_u35_avx_f64
#define xtanf nsimd_sleef_tan_u35_avx_f32
#define xasin nsimd_sleef_asin_u35_avx_f64
#define xasinf nsimd_sleef_asin_u35_avx_f32
#define xacos nsimd_sleef_acos_u35_avx_f64
#define xacosf nsimd_sleef_acos_u35_avx_f32
#define xatan nsimd_sleef_atan_u35_avx_f64
#define xatanf nsimd_sleef_atan_u35_avx_f32
#define xatan2 nsimd_sleef_atan2_u35_avx_f64
#define xatan2f nsimd_sleef_atan2_u35_avx_f32
#define xlog nsimd_sleef_log_u35_avx_f64
#define xlogf nsimd_sleef_log_u35_avx_f32
#define xcbrt nsimd_sleef_cbrt_u35_avx_f64
#define xcbrtf nsimd_sleef_cbrt_u35_avx_f32
#define xsin_u1 nsimd_sleef_sin_u10_avx_f64
#define xsinf_u1 nsimd_sleef_sin_u10_avx_f32
#define xcos_u1 nsimd_sleef_cos_u10_avx_f64
#define xcosf_u1 nsimd_sleef_cos_u10_avx_f32
#define xsincos_u1 nsimd_sleef_sincos_u10_avx_f64
#define xsincosf_u1 nsimd_sleef_sincos_u10_avx_f32
#define xtan_u1 nsimd_sleef_tan_u10_avx_f64
#define xtanf_u1 nsimd_sleef_tan_u10_avx_f32
#define xasin_u1 nsimd_sleef_asin_u10_avx_f64
#define xasinf_u1 nsimd_sleef_asin_u10_avx_f32
#define xacos_u1 nsimd_sleef_acos_u10_avx_f64
#define xacosf_u1 nsimd_sleef_acos_u10_avx_f32
#define xatan_u1 nsimd_sleef_atan_u10_avx_f64
#define xatanf_u1 nsimd_sleef_atan_u10_avx_f32
#define xatan2_u1 nsimd_sleef_atan2_u10_avx_f64
#define xatan2f_u1 nsimd_sleef_atan2_u10_avx_f32
#define xlog_u1 nsimd_sleef_log_u10_avx_f64
#define xlogf_u1 nsimd_sleef_log_u10_avx_f32
#define xcbrt_u1 nsimd_sleef_cbrt_u10_avx_f64
#define xcbrtf_u1 nsimd_sleef_cbrt_u10_avx_f32
#define xexp nsimd_sleef_exp_u10_avx_f64
#define xexpf nsimd_sleef_exp_u10_avx_f32
#define xpow nsimd_sleef_pow_u10_avx_f64
#define xpowf nsimd_sleef_pow_u10_avx_f32
#define xsinh nsimd_sleef_sinh_u10_avx_f64
#define xsinhf nsimd_sleef_sinh_u10_avx_f32
#define xcosh nsimd_sleef_cosh_u10_avx_f64
#define xcoshf nsimd_sleef_cosh_u10_avx_f32
#define xtanh nsimd_sleef_tanh_u10_avx_f64
#define xtanhf nsimd_sleef_tanh_u10_avx_f32
#define xsinh_u35 nsimd_sleef_sinh_u35_avx_f64
#define xsinhf_u35 nsimd_sleef_sinh_u35_avx_f32
#define xcosh_u35 nsimd_sleef_cosh_u35_avx_f64
#define xcoshf_u35 nsimd_sleef_cosh_u35_avx_f32
#define xtanh_u35 nsimd_sleef_tanh_u35_avx_f64
#define xtanhf_u35 nsimd_sleef_tanh_u35_avx_f32
#define xfastsin_u3500 nsimd_sleef_fastsin_u3500_avx_f64
#define xfastsinf_u3500 nsimd_sleef_fastsin_u3500_avx_f32
#define xfastcos_u3500 nsimd_sleef_fastcos_u3500_avx_f64
#define xfastcosf_u3500 nsimd_sleef_fastcos_u3500_avx_f32
#define xfastpow_u3500 nsimd_sleef_fastpow_u3500_avx_f64
#define xfastpowf_u3500 nsimd_sleef_fastpow_u3500_avx_f32
#define xasinh nsimd_sleef_asinh_u10_avx_f64
#define xasinhf nsimd_sleef_asinh_u10_avx_f32
#define xacosh nsimd_sleef_acosh_u10_avx_f64
#define xacoshf nsimd_sleef_acosh_u10_avx_f32
#define xatanh nsimd_sleef_atanh_u10_avx_f64
#define xatanhf nsimd_sleef_atanh_u10_avx_f32
#define xexp2 nsimd_sleef_exp2_u10_avx_f64
#define xexp2f nsimd_sleef_exp2_u10_avx_f32
#define xexp2_u35 nsimd_sleef_exp2_u35_avx_f64
#define xexp2f_u35 nsimd_sleef_exp2_u35_avx_f32
#define xexp10 nsimd_sleef_exp10_u10_avx_f64
#define xexp10f nsimd_sleef_exp10_u10_avx_f32
#define xexp10_u35 nsimd_sleef_exp10_u35_avx_f64
#define xexp10f_u35 nsimd_sleef_exp10_u35_avx_f32
#define xexpm1 nsimd_sleef_expm1_u10_avx_f64
#define xexpm1f nsimd_sleef_expm1_u10_avx_f32
#define xlog10 nsimd_sleef_log10_u10_avx_f64
#define xlog10f nsimd_sleef_log10_u10_avx_f32
#define xlog2 nsimd_sleef_log2_u10_avx_f64
#define xlog2f nsimd_sleef_log2_u10_avx_f32
#define xlog2_u35 nsimd_sleef_log2_u35_avx_f64
#define xlog2f_u35 nsimd_sleef_log2_u35_avx_f32
#define xlog1p nsimd_sleef_log1p_u10_avx_f64
#define xlog1pf nsimd_sleef_log1p_u10_avx_f32
#define xsincospi_u05 nsimd_sleef_sincospi_u05_avx_f64
#define xsincospif_u05 nsimd_sleef_sincospi_u05_avx_f32
#define xsincospi_u35 nsimd_sleef_sincospi_u35_avx_f64
#define xsincospif_u35 nsimd_sleef_sincospi_u35_avx_f32
#define xsinpi_u05 nsimd_sleef_sinpi_u05_avx_f64
#define xsinpif_u05 nsimd_sleef_sinpi_u05_avx_f32
#define xcospi_u05 nsimd_sleef_cospi_u05_avx_f64
#define xcospif_u05 nsimd_sleef_cospi_u05_avx_f32
#define xldexp nsimd_sleef_ldexp_avx_f64
#define xldexpf nsimd_sleef_ldexp_avx_f32
#define xilogb nsimd_sleef_ilogb_avx_f64
#define xilogbf nsimd_sleef_ilogb_avx_f32
#define xfma nsimd_sleef_fma_avx_f64
#define xfmaf nsimd_sleef_fma_avx_f32
#define xsqrt nsimd_sleef_sqrt_avx_f64
#define xsqrtf nsimd_sleef_sqrt_avx_f32
#define xsqrt_u05 nsimd_sleef_sqrt_u05_avx_f64
#define xsqrtf_u05 nsimd_sleef_sqrt_u05_avx_f32
#define xsqrt_u35 nsimd_sleef_sqrt_u35_avx_f64
#define xsqrtf_u35 nsimd_sleef_sqrt_u35_avx_f32
#define xhypot_u05 nsimd_sleef_hypot_u05_avx_f64
#define xhypotf_u05 nsimd_sleef_hypot_u05_avx_f32
#define xhypot_u35 nsimd_sleef_hypot_u35_avx_f64
#define xhypotf_u35 nsimd_sleef_hypot_u35_avx_f32
#define xfabs nsimd_sleef_fabs_avx_f64
#define xfabsf nsimd_sleef_fabs_avx_f32
#define xcopysign nsimd_sleef_copysign_avx_f64
#define xcopysignf nsimd_sleef_copysign_avx_f32
#define xfmax nsimd_sleef_fmax_avx_f64
#define xfmaxf nsimd_sleef_fmax_avx_f32
#define xfmin nsimd_sleef_fmin_avx_f64
#define xfminf nsimd_sleef_fmin_avx_f32
#define xfdim nsimd_sleef_fdim_avx_f64
#define xfdimf nsimd_sleef_fdim_avx_f32
#define xtrunc nsimd_sleef_trunc_avx_f64
#define xtruncf nsimd_sleef_trunc_avx_f32
#define xfloor nsimd_sleef_floor_avx_f64
#define xfloorf nsimd_sleef_floor_avx_f32
#define xceil nsimd_sleef_ceil_avx_f64
#define xceilf nsimd_sleef_ceil_avx_f32
#define xround nsimd_sleef_round_avx_f64
#define xroundf nsimd_sleef_round_avx_f32
#define xrint nsimd_sleef_rint_avx_f64
#define xrintf nsimd_sleef_rint_avx_f32
#define xnextafter nsimd_sleef_nextafter_avx_f64
#define xnextafterf nsimd_sleef_nextafter_avx_f32
#define xfrfrexp nsimd_sleef_frfrexp_avx_f64
#define xfrfrexpf nsimd_sleef_frfrexp_avx_f32
#define xexpfrexp nsimd_sleef_expfrexp_avx_f64
#define xexpfrexpf nsimd_sleef_expfrexp_avx_f32
#define xfmod nsimd_sleef_fmod_avx_f64
#define xfmodf nsimd_sleef_fmod_avx_f32
#define xremainder nsimd_sleef_remainder_avx_f64
#define xremainderf nsimd_sleef_remainder_avx_f32
#define xmodf nsimd_sleef_modf_avx_f64
#define xmodff nsimd_sleef_modf_avx_f32
#define xlgamma_u1 nsimd_sleef_lgamma_u10_avx_f64
#define xlgammaf_u1 nsimd_sleef_lgamma_u10_avx_f32
#define xtgamma_u1 nsimd_sleef_tgamma_u10_avx_f64
#define xtgammaf_u1 nsimd_sleef_tgamma_u10_avx_f32
#define xerf_u1 nsimd_sleef_erf_u10_avx_f64
#define xerff_u1 nsimd_sleef_erf_u10_avx_f32
#define xerfc_u15 nsimd_sleef_erfc_u15_avx_f64
#define xerfcf_u15 nsimd_sleef_erfc_u15_avx_f32
#define xgetInt nsimd_sleef_getInt_avx_f64
#define xgetIntf nsimd_sleef_getInt_avx_f32
#define xgetPtr nsimd_sleef_getPtr_avx_f64
#define xgetPtrf nsimd_sleef_getPtr_avx_f32

#endif

#define rempi nsimd_sleef_rempi_avx
#define rempif nsimd_sleef_rempif_avx
#define rempisub nsimd_sleef_rempisub_avx
#define rempisubf nsimd_sleef_rempisubf_avx
#define gammak nsimd_gammak_avx
#define gammafk nsimd_gammafk_avx

#endif

#endif
