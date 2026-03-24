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
use crate::cwt_executor::CommonCwtExecutor;
use crate::err::try_vec;
use crate::sample::CwtSample;
use crate::scale_bounds::find_min_max_scales;
use crate::scales::{linear_scales, log_piecewise_scales};
use crate::{CwtExecutor, CwtOptions, CwtWavelet, ScaleType, ScaletError};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;
use zaft::FftDirection;

pub(crate) fn gen_psi<T: CwtSample>(points: usize) -> Result<Vec<T>, ScaletError>
where
    usize: AsPrimitive<T>,
    f64: AsPrimitive<T>,
    isize: AsPrimitive<T>,
{
    let mut psih = try_vec![T::zero(); points];
    let recip_points = 1f64.as_() / points.as_();
    for (i, v) in psih.iter_mut().enumerate() {
        let idx = if i < points / 2 {
            i as isize
        } else {
            i as isize - points as isize
        };
        let w = idx.as_() * T::TWO_PI * recip_points;
        *v = w;
    }
    Ok(psih)
}

pub(crate) fn create_cwt<T: CwtSample>(
    wavelet: Arc<dyn CwtWavelet<T> + Send + Sync>,
    filter_size: usize,
    scale_type: ScaleType,
    options: CwtOptions,
) -> Result<Arc<dyn CwtExecutor<T> + Send + Sync>, ScaletError>
where
    usize: AsPrimitive<T>,
    f64: AsPrimitive<T>,
    isize: AsPrimitive<T>,
{
    if filter_size == 0 {
        return Err(ScaletError::ZeroBaseSized);
    }
    let scale_bounds = find_min_max_scales(wavelet.clone(), -0.5f64.as_())?;
    let scales = match scale_type {
        ScaleType::Log => {
            log_piecewise_scales(scale_bounds.min, scale_bounds.max, options.nv.as_())?
        }
        ScaleType::Linear => linear_scales(scale_bounds.min, scale_bounds.max, options.nv.as_())?,
    };
    let fft_forward = T::make_fft(filter_size, FftDirection::Forward)?;
    let fft_inverse = T::make_fft(filter_size, FftDirection::Inverse)?;
    let psi = gen_psi(filter_size)?;
    let scratch_length = fft_inverse
        .scratch_length()
        .max(fft_forward.scratch_length());
    let cached_wavelet: Vec<Complex<T>> = if options.full_cache {
        let mut built_wavelets: Vec<Complex<T>> = Vec::new();
        built_wavelets
            .try_reserve_exact(filter_size * scales.len())
            .map_err(|_| ScaletError::Allocation(filter_size * scales.len()))?;
        let mut current_psi = vec![T::zero(); filter_size];
        for &scale in scales.iter() {
            for (dst, &psi) in current_psi.iter_mut().zip(psi.iter()) {
                *dst = psi * scale;
            }
            let wavelet_fft = wavelet.make_wavelet(&current_psi)?;
            built_wavelets.extend_from_slice(&wavelet_fft);
        }
        built_wavelets
    } else {
        vec![]
    };
    Ok(Arc::new(CommonCwtExecutor {
        wavelet,
        fft_forward,
        fft_inverse,
        scales,
        psi,
        execution_length: filter_size,
        l1_norm: options.l1_norm,
        complex_arithmetic: T::spectrum_arithmetic(),
        scratch_length,
        built_wavelets: cached_wavelet,
    }))
}
