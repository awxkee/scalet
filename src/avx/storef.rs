/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
use num_complex::Complex;
use std::arch::x86_64::*;
use std::ops::Mul;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct AvxStoreF {
    pub(crate) v: __m256,
}

impl AvxStoreF {
    #[inline]
    pub(crate) fn raw(v: __m256) -> Self {
        Self { v }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn dup(v: f32) -> Self {
        Self {
            v: _mm256_set1_ps(v),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn load(data: &[Complex<f32>]) -> Self {
        unsafe { Self::raw(_mm256_loadu_ps(data.as_ptr().cast())) }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn load2(data: &[Complex<f32>]) -> Self {
        unsafe { Self::raw(_mm256_castps128_ps256(_mm_loadu_ps(data.as_ptr().cast()))) }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn load1(data: &Complex<f32>) -> Self {
        unsafe {
            Self::raw(_mm256_castps128_ps256(_mm_castsi128_ps(_mm_loadu_si64(
                data as *const Complex<f32> as *const _,
            ))))
        }
    }

    // a * b.conj()
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn mul_by_conj_b(self, other: Self) -> Self {
        let ar = _mm256_moveldup_ps(self.v); // duplicate even lanes (re parts)
        let ai = _mm256_movehdup_ps(self.v); // duplicate odd lanes (im parts)

        // Swap real/imag of b for cross terms
        let bswap = _mm256_shuffle_ps::<0b10110001>(other.v, other.v); // [im, re, im, re, ...]

        // re = ar*br - ai*bi
        // im = ar*bi + ai*br
        Self::raw(_mm256_fmsubadd_ps(ai, bswap, _mm256_mul_ps(ar, other.v)))
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn write(&self, data: &mut [Complex<f32>]) {
        unsafe { _mm256_storeu_ps(data.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn write2(&self, data: &mut [Complex<f32>]) {
        unsafe { _mm_storeu_ps(data.as_mut_ptr().cast(), _mm256_castps256_ps128(self.v)) }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn write1(&self, data: &mut Complex<f32>) {
        unsafe {
            _mm_storeu_si64(
                data as *mut Complex<f32> as *mut _,
                _mm_castps_si128(_mm256_castps256_ps128(self.v)),
            )
        }
    }
}

impl Mul<AvxStoreF> for AvxStoreF {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: AvxStoreF) -> Self::Output {
        unsafe { Self::raw(_mm256_mul_ps(self.v, rhs.v)) }
    }
}

#[cfg(test)]
mod tests {
    use crate::avx::storef::AvxStoreF;
    use num_complex::Complex;

    /// Scalar reference: (a_re + i*a_im) * conj(b_re + i*b_im)
    ///   = (a_re*b_re + a_im*b_im) + i*(a_im*b_re - a_re*b_im)
    fn ref_mul_conj(a: Complex<f32>, b: Complex<f32>) -> Complex<f32> {
        a * b.conj()
    }

    /// Pack 4 complex numbers into an AvxStoreF, run mul_by_conj_b, unpack results.
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn avx_mul_conj(a: &[Complex<f32>; 4], b: &[Complex<f32>; 4]) -> [Complex<f32>; 4] {
        let va = AvxStoreF::load(a.as_slice());
        let vb = AvxStoreF::load(b.as_slice());
        let vc = va.mul_by_conj_b(vb);
        let mut out = [Complex::new(0.0f32, 0.0); 4];
        vc.write(out.as_mut_slice());
        out
    }

    fn assert_complex_approx(got: Complex<f32>, want: Complex<f32>, label: &str) {
        let eps = 1e-5_f32;
        assert!(
            (got.re - want.re).abs() < eps && (got.im - want.im).abs() < eps,
            "{label}: got ({}, {}i), want ({}, {}i)",
            got.re,
            got.im,
            want.re,
            want.im
        );
    }

    fn run(a: [Complex<f32>; 4], b: [Complex<f32>; 4]) {
        let result = unsafe { avx_mul_conj(&a, &b) };
        for i in 0..4 {
            let want = ref_mul_conj(a[i], b[i]);
            assert_complex_approx(result[i], want, &format!("lane {i}"));
        }
    }

    #[test]
    fn test_mul_conj_mixed_lanes() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        let a = [
            Complex::new(1., 2.),
            Complex::new(3., -1.),
            Complex::new(0., 5.),
            Complex::new(-2., -3.),
        ];
        let b = [
            Complex::new(4., 1.),
            Complex::new(-1., 2.),
            Complex::new(3., 3.),
            Complex::new(1., -1.),
        ];
        run(a, b);
    }

    #[test]
    fn test_mul_conj_b_is_real() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        // conj of a real number is itself, so a*conj(b_re) = a * b_re
        let a = [Complex::new(2., 3.); 4];
        let b = [Complex::new(5., 0.); 4]; // pure real
        run(a, b);
    }

    #[test]
    fn test_mul_conj_a_is_real() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        // a real * conj(b)  →  re part conjugates b
        let a = [Complex::new(4., 0.); 4];
        let b = [Complex::new(1., 2.); 4];
        run(a, b);
    }

    #[test]
    fn test_mul_conj_zero() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        let a = [Complex::new(0., 0.); 4];
        let b = [Complex::new(3., 4.); 4];
        run(a, b);

        let a2 = [Complex::new(3., 4.); 4];
        let b2 = [Complex::new(0., 0.); 4];
        run(a2, b2);
    }

    #[test]
    fn test_mul_conj_negatives() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        let a = [
            Complex::new(-1., -1.),
            Complex::new(-3., 2.),
            Complex::new(1., -4.),
            Complex::new(-2., -5.),
        ];
        let b = [
            Complex::new(-2., 3.),
            Complex::new(4., -1.),
            Complex::new(-3., -2.),
            Complex::new(1., 1.),
        ];
        run(a, b);
    }

    #[test]
    fn test_mul_conj_large_values() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        // Check no overflow / catastrophic cancellation at scale
        let a = [Complex::new(1e3_f32, -1e3_f32); 4];
        let b = [Complex::new(1e3_f32, 1e3_f32); 4];
        run(a, b);
    }

    #[test]
    fn test_mul_conj_self_gives_magnitude_squared() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        // a * conj(a) should be purely real: re = |a|², im = 0
        let a = Complex::new(3., 4.);
        let result = unsafe { avx_mul_conj(&[a; 4], &[a; 4]) };
        for lane in &result {
            assert_complex_approx(*lane, Complex::new(25., 0.), "self-conj");
        }
    }
}
