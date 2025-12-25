/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::ScaletError;
use crate::spetrum_arith::SpectrumArithmeticFactory;
use num_traits::{AsPrimitive, MulAdd, Num, Zero};
use pxfm::{
    f_exp, f_exp2, f_exp2f, f_expf, f_log2, f_log2f, f_pow, f_powf, f_rsqrt, f_rsqrtf, f_sincos,
    f_sincosf,
};
use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};
use std::sync::Arc;
use zaft::{FftDirection, FftExecutor, Zaft};

pub trait CwtSample:
    MulAdd<Self, Output = Self>
    + AddAssign
    + MulAssign
    + 'static
    + Copy
    + Clone
    + Send
    + Sync
    + Num
    + Default
    + Neg<Output = Self>
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Debug
    + Display
    + Zero
    + PartialOrd
    + AsPrimitive<usize>
    + AsPrimitive<isize>
    + AsPrimitive<f32>
    + SpectrumArithmeticFactory
{
    fn pow(self, other: Self) -> Self;
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn rsqrt(self) -> Self;
    fn log2(self) -> Self;
    fn ceil(self) -> Self;
    fn floor(self) -> Self;
    fn fract(self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn copysign(self, other: Self) -> Self;
    fn make_fft(
        length: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<Self> + Send + Sync>, ScaletError>;
    fn sincos(self) -> (Self, Self);
    const NEG_INFINITY: Self;
    const INFINITY: Self;
    const PI: Self;
    const FRAC_1_PI: Self;
    const TWO_PI: Self;
    const TWO_SQRT_BY_PI_POWER_0_25: Self;
    const TWO_S2_OVER_3_PI_POWER_M0_25: Self;
    const TWO_OVER_5_SQ_PI_POWER_M0_25: Self;
}

impl CwtSample for f32 {
    #[inline]
    fn pow(self, other: Self) -> Self {
        f_powf(self, other)
    }

    #[inline]
    fn exp(self) -> Self {
        f_expf(self)
    }

    #[inline]
    fn exp2(self) -> Self {
        f_exp2f(self)
    }

    #[inline]
    fn rsqrt(self) -> Self {
        f_rsqrtf(self)
    }

    #[inline]
    fn log2(self) -> Self {
        f_log2f(self)
    }

    #[inline]
    fn ceil(self) -> Self {
        f32::ceil(self)
    }

    #[inline]
    fn floor(self) -> Self {
        f32::floor(self)
    }

    #[inline]
    fn fract(self) -> Self {
        f32::fract(self)
    }

    #[inline]
    fn abs(self) -> Self {
        f32::abs(self)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }

    #[inline]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }

    #[inline]
    fn copysign(self, other: Self) -> Self {
        f32::copysign(self, other)
    }

    #[inline]
    fn sincos(self) -> (Self, Self) {
        f_sincosf(self)
    }

    fn make_fft(
        length: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<Self> + Send + Sync>, ScaletError> {
        match fft_direction {
            FftDirection::Forward => {
                Zaft::make_forward_fft_f32(length).map_err(|x| ScaletError::FftError(x.to_string()))
            }
            FftDirection::Inverse => {
                Zaft::make_inverse_fft_f32(length).map_err(|x| ScaletError::FftError(x.to_string()))
            }
        }
    }

    const INFINITY: Self = f32::INFINITY;

    const NEG_INFINITY: Self = f32::NEG_INFINITY;

    const PI: Self = f32::from_bits(0x40490fdb);

    const FRAC_1_PI: Self = f32::from_bits(0x3ea2f983);

    // Computed in SageMath:
    // R = RealField(256)
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    // float_to_hex(float(R.pi() * 2))
    const TWO_PI: Self = f32::from_bits(0x40c90fdb); // accurate PI*2

    // Computed in SageMath:
    // R = RealField(256)
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    // float_to_hex(float(R(2).sqrt() * R.pi() ** 0.25))
    const TWO_SQRT_BY_PI_POWER_0_25: Self = f32::from_bits(0x3ff0ff58); // accurate 2.sqrt() * PI^(0.25)

    // Computed in SageMath:
    // R = RealField(256)
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    // v = R(2) * (R(2)/R(3)).sqrt() * R.pi()**(-1/4)
    // float_to_hex(float(v))
    const TWO_S2_OVER_3_PI_POWER_M0_25: Self = f32::from_bits(0x3f9d00ab);

    // Computed in SageMath:
    // R = RealField(256)
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    // v = R(2) / R(5).sqrt() * R.pi()**(-1/4)
    // float_to_hex(float(v))
    const TWO_OVER_5_SQ_PI_POWER_M0_25: Self = f32::from_bits(0x3f2bfcdd);
}

impl CwtSample for f64 {
    #[inline]
    fn pow(self, other: Self) -> Self {
        f_pow(self, other)
    }
    #[inline]
    fn exp(self) -> Self {
        f_exp(self)
    }

    #[inline]
    fn exp2(self) -> Self {
        f_exp2(self)
    }

    #[inline]
    fn rsqrt(self) -> Self {
        f_rsqrt(self)
    }

    #[inline]
    fn log2(self) -> Self {
        f_log2(self)
    }

    #[inline]
    fn ceil(self) -> Self {
        f64::ceil(self)
    }

    #[inline]
    fn floor(self) -> Self {
        f64::floor(self)
    }

    #[inline]
    fn fract(self) -> Self {
        f64::fract(self)
    }

    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }

    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    #[inline]
    fn copysign(self, other: Self) -> Self {
        f64::copysign(self, other)
    }

    #[inline]
    fn sincos(self) -> (Self, Self) {
        f_sincos(self)
    }

    fn make_fft(
        length: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<Self> + Send + Sync>, ScaletError> {
        match fft_direction {
            FftDirection::Forward => {
                Zaft::make_forward_fft_f64(length).map_err(|x| ScaletError::FftError(x.to_string()))
            }
            FftDirection::Inverse => {
                Zaft::make_inverse_fft_f64(length).map_err(|x| ScaletError::FftError(x.to_string()))
            }
        }
    }

    const INFINITY: Self = f64::INFINITY;

    const NEG_INFINITY: Self = f64::NEG_INFINITY;

    const FRAC_1_PI: Self = f64::from_bits(0x3fd45f306dc9c883);

    const PI: Self = f64::from_bits(0x400921fb54442d18);

    // Computed in SageMath:
    // R = RealField(256)
    // def double_to_hex(f):
    //     packed = struct.pack('>d', float(f))
    //     return '0x' + packed.hex()
    // double_to_hex(float(R.pi() * 2))
    const TWO_PI: Self = f64::from_bits(0x401921fb54442d18); // accurate PI*2

    // Computed in SageMath:
    // R = RealField(256)
    // def double_to_hex(f):
    //     packed = struct.pack('>d', float(f))
    //     return '0x' + packed.hex()
    // double_to_hex(float(R(2).sqrt() * R.pi() ** 0.25))
    const TWO_SQRT_BY_PI_POWER_0_25: Self = f64::from_bits(0x3ffe1feb0eafec2d); // accurate 2.sqrt() * PI^(0.25)

    // Computed in SageMath:
    // R = RealField(256)
    // def double_to_hex(f):
    //     packed = struct.pack('>d', float(f))
    //     return '0x' + packed.hex()
    // v = R(2) * (R(2)/R(3)).sqrt() * R.pi()**(-1/4)
    // double_to_hex(float(v))
    const TWO_S2_OVER_3_PI_POWER_M0_25: Self = f64::from_bits(0x3ff3a0155e202ded);

    // Computed in SageMath:
    // R = RealField(256)
    // def double_to_hex(f):
    //     packed = struct.pack('>d', float(f))
    //     return '0x' + packed.hex()
    // v = R(2) / R(5).sqrt() * R.pi()**(-1/4)
    // double_to_hex(float(v))
    const TWO_OVER_5_SQ_PI_POWER_M0_25: Self = f64::from_bits(0x3fe57f9b91b1c6bb);
}
