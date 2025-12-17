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
use crate::mla::fmla;
use crate::sample::CwtSample;
use num_complex::Complex;
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

#[inline(always)]
#[allow(unused)]
pub(crate) fn c_mul_fast_conj<T: CwtSample>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    let re = fmla(a.re, b.re, a.im * b.im);
    let im = fmla(a.re, -b.im, a.im * b.re);
    Complex::new(re, im)
}

pub trait SpectrumArithmetic<T> {
    // input * other.conj() * normalize_value
    fn mul_by_b_conj_normalize(
        &self,
        dst: &mut [Complex<T>],
        input: &[Complex<T>],
        other: &[Complex<T>],
        normalize_value: T,
    );
}

#[allow(unused)]
#[derive(Debug, Default)]
pub(crate) struct CommonSpectrumArithmetic<T: Default> {
    phantom: PhantomData<T>,
}

#[allow(unused)]
impl<T: CwtSample> SpectrumArithmetic<T> for CommonSpectrumArithmetic<T> {
    fn mul_by_b_conj_normalize(
        &self,
        dst: &mut [Complex<T>],
        input: &[Complex<T>],
        other: &[Complex<T>],
        normalize_value: T,
    ) {
        for ((v_dst, &signal), &wavelet) in dst.iter_mut().zip(input.iter()).zip(other.iter()) {
            *v_dst = c_mul_fast_conj(signal, wavelet) * normalize_value;
        }
    }
}

pub trait SpectrumArithmeticFactory {
    fn spectrum_arithmetic() -> Arc<dyn SpectrumArithmetic<Self> + Send + Sync>;
}

impl SpectrumArithmeticFactory for f32 {
    fn spectrum_arithmetic() -> Arc<dyn SpectrumArithmetic<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn SpectrumArithmetic<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    use crate::avx::AvxSpectrumF32;
                    return Arc::new(AvxSpectrumF32::default());
                }
            }
            #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
            {
                if std::arch::is_x86_feature_detected!("sse4.2") {
                    use crate::sse::Sse42SpectrumF32;
                    return Arc::new(Sse42SpectrumF32::default());
                }
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                #[cfg(feature = "fcma")]
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::FcmaSpectrumF32;
                    return Arc::new(FcmaSpectrumF32::default());
                }
                use crate::neon::NeonSpectrumF32;
                Arc::new(NeonSpectrumF32::default())
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                Arc::new(CommonSpectrumArithmetic::default())
            }
        })
        .clone()
    }
}

impl SpectrumArithmeticFactory for f64 {
    fn spectrum_arithmetic() -> Arc<dyn SpectrumArithmetic<Self> + Send + Sync> {
        static Q: OnceLock<Arc<dyn SpectrumArithmetic<f64> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    use crate::avx::AvxSpectrumF64;
                    return Arc::new(AvxSpectrumF64::default());
                }
            }
            #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
            {
                if std::arch::is_x86_feature_detected!("sse4.2") {
                    use crate::sse::Sse42SpectrumF64;
                    return Arc::new(Sse42SpectrumF64::default());
                }
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                #[cfg(feature = "fcma")]
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::FcmaSpectrumF64;
                    return Arc::new(FcmaSpectrumF64::default());
                }
                use crate::neon::NeonSpectrumF64;
                Arc::new(NeonSpectrumF64::default())
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                Arc::new(CommonSpectrumArithmetic::default())
            }
        })
        .clone()
    }
}
