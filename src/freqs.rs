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
use crate::err::try_vec;
use crate::factory::gen_psi;
use crate::sample::CwtSample;
use crate::{CwtWavelet, ScaletError};
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) fn scale_to_frequencies_impl<T: CwtSample>(
    wavelet: Arc<dyn CwtWavelet<T> + Send + Sync>,
    scales: &[T],
    sampling_frequency: T,
    filter_length: usize,
) -> Result<Vec<T>, ScaletError>
where
    usize: AsPrimitive<T>,
    f64: AsPrimitive<T>,
    isize: AsPrimitive<T>,
{
    if filter_length == 0 {
        return Err(ScaletError::ZeroBaseSized);
    }
    let psi = gen_psi(filter_length)?;
    let mut max_indices = try_vec![0usize; scales.len()];
    let mut current_psi = try_vec![T::zero(); filter_length];
    for (index, &scale) in max_indices.iter_mut().zip(scales.iter()) {
        for (dst, &psi) in current_psi.iter_mut().zip(psi.iter()) {
            *dst = psi * scale;
        }

        // Generate the final complex FFT filter for the current scale 'a'.
        let wavelet_fft = wavelet.make_wavelet(&current_psi)?;

        if wavelet_fft.len() != filter_length {
            return Err(ScaletError::WaveletInvalidSize(
                filter_length,
                wavelet_fft.len(),
            ));
        }
        let idx = wavelet_fft
            .iter()
            .enumerate() // gives (index, &value)
            .max_by(|a, b| a.1.re.partial_cmp(&b.1.re).unwrap()) // compare values
            .map(|(idx, _)| idx);

        *index = idx.unwrap_or(0);
    }

    let mut freqs = try_vec![T::zero(); scales.len()];
    let idx_scale = sampling_frequency / filter_length.as_();
    for (dst, idx) in freqs.iter_mut().zip(&max_indices) {
        *dst = idx.as_() * idx_scale;
    }

    Ok(freqs)
}
