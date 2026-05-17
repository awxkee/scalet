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
pub(crate) struct AvxStoreD {
    pub(crate) v: __m256d,
}

impl AvxStoreD {
    #[inline]
    pub(crate) fn raw(v: __m256d) -> Self {
        Self { v }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn dup(v: f64) -> Self {
        Self {
            v: _mm256_set1_pd(v),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn load(data: &[Complex<f64>]) -> Self {
        unsafe { Self::raw(_mm256_loadu_pd(data.as_ptr().cast())) }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn load1(data: &Complex<f64>) -> Self {
        unsafe {
            Self::raw(_mm256_castpd128_pd256(_mm_loadu_pd(
                data as *const Complex<f64> as *const f64,
            )))
        }
    }

    // a * b.conj()
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn mul_by_conj_b(self, other: Self) -> Self {
        // Swap real and imaginary parts of 'a' for FMA
        let a_yx = _mm256_shuffle_pd::<0b0101>(self.v, self.v); // [a_im, a_re, b_im, b_re]

        // Duplicate real and imaginary parts of 'b'
        let b_xx = _mm256_shuffle_pd::<0b0000>(other.v, other.v); // [c_re, c_re, d_re, d_re]
        let b_yy = _mm256_shuffle_pd::<0b1111>(other.v, other.v); // [c_im, c_im, d_im, d_im]

        Self::raw(_mm256_fmsubadd_pd(self.v, b_xx, _mm256_mul_pd(a_yx, b_yy)))
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn write(&self, data: &mut [Complex<f64>]) {
        unsafe { _mm256_storeu_pd(data.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn write1(&self, data: &mut Complex<f64>) {
        unsafe {
            _mm_storeu_pd(
                data as *mut Complex<f64> as *mut _,
                _mm256_castpd256_pd128(self.v),
            )
        }
    }
}

impl Mul<AvxStoreD> for AvxStoreD {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: AvxStoreD) -> Self::Output {
        unsafe { Self::raw(_mm256_mul_pd(self.v, rhs.v)) }
    }
}

#[cfg(test)]
mod tests {
    use crate::avx::stored::AvxStoreD;
    use num_complex::Complex;

    /// Scalar reference: (a_re + i*a_im) * conj(b_re + i*b_im)
    ///   = (a_re*b_re + a_im*b_im) + i*(a_im*b_re - a_re*b_im)
    fn ref_mul_conj(a: Complex<f64>, b: Complex<f64>) -> Complex<f64> {
        a * b.conj()
    }

    /// Pack 4 complex numbers into an AvxStoreF, run mul_by_conj_b, unpack results.
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn avx_mul_conj(a: &[Complex<f64>; 2], b: &[Complex<f64>; 2]) -> [Complex<f64>; 2] {
        let va = AvxStoreD::load(a.as_slice());
        let vb = AvxStoreD::load(b.as_slice());
        let vc = va.mul_by_conj_b(vb);
        let mut out = [Complex::new(0.0f64, 0.0); 2];
        vc.write(out.as_mut_slice());
        out
    }

    fn assert_complex_approx(got: Complex<f64>, want: Complex<f64>, label: &str) {
        let eps = 1e-5_f64;
        assert!(
            (got.re - want.re).abs() < eps && (got.im - want.im).abs() < eps,
            "{label}: got ({}, {}i), want ({}, {}i)",
            got.re,
            got.im,
            want.re,
            want.im
        );
    }

    fn run(a: [Complex<f64>; 2], b: [Complex<f64>; 2]) {
        let result = unsafe { avx_mul_conj(&a, &b) };
        for i in 0..2 {
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
        let a = [Complex::new(1., 2.), Complex::new(-2., -3.)];
        let b = [Complex::new(4., 1.), Complex::new(1., -1.)];
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
        let a = [Complex::new(2., 3.); 2];
        let b = [Complex::new(5., 0.); 2]; // pure real
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
        let a = [Complex::new(4., 0.); 2];
        let b = [Complex::new(1., 2.); 2];
        run(a, b);
    }

    #[test]
    fn test_mul_conj_zero() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        let a = [Complex::new(0., 0.); 2];
        let b = [Complex::new(3., 4.); 2];
        run(a, b);

        let a2 = [Complex::new(3., 4.); 2];
        let b2 = [Complex::new(0., 0.); 2];
        run(a2, b2);
    }

    #[test]
    fn test_mul_conj_negatives() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        let a = [Complex::new(-1., -1.), Complex::new(-3., 2.)];
        let b = [Complex::new(-2., 3.), Complex::new(4., -1.)];
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
        let a = [Complex::new(1e3_f64, -1e3_f64); 2];
        let b = [Complex::new(1e3_f64, 1e3_f64); 2];
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
        let result = unsafe { avx_mul_conj(&[a; 2], &[a; 2]) };
        for lane in &result {
            assert_complex_approx(*lane, Complex::new(25., 0.), "self-conj");
        }
    }
}
