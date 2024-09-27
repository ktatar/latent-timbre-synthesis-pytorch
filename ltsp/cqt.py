import torch
import numpy as np
import scipy

# copied from the fork: 
# https://github.com/d-dawg78/audio/blob/main/src/torchaudio/transforms/_transforms.py

class CQT(torch.nn.Module):
    r"""Create the constant Q-transform for a raw audio signal.

    .. devices:: CPU CUDA

    .. properties:: Autograd

    Sources
        * https://librosa.org/doc/main/generated/librosa.cqt.html
        * https://www.aes.org/e-lib/online/browse.cfm?elib=17112
        * https://newt.phys.unsw.edu.au/jw/notes.html

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        hop_length (int, optional): Length of hop between CQT windows. (Default: ``400``)
        f_min (float, optional): Minimum frequency, which corresponds to first note.
            (Default: ``32.703``, or the frequency of C1 in Hz)
        n_bins (int, optional): Number of CQT frequency bins, starting at ``f_min``. (Default: ``84``)
        bins_per_octave (int, optional): Number of bins per octave. (Default: ``12``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        resampling_method (str, optional): The resampling method to use.
            Options: [``sinc_interp_hann``, ``sinc_interp_kaiser``] (Default: ``"sinc_interp_hann"``)
        dtype (torch.device, optional):
            Determines the precision that kernels are pre-computed and cached in. Note that complex
            bases are either cfloat or cdouble depending on provided precision.
            Options: [``torch.float``, ``torch.double``] (Default: ``torch.float``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.CQT(sample_rate)
        >>> cqt = transform(waveform)  # (..., n_bins, time)
    """
    __constants__ = ["transform"]

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 256,
        f_min: float = 32.703,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        resampling_method: str = "sinc_interp_hann",
        dtype: torch.dtype = torch.float,
    ) -> None:
        super(CQT, self).__init__()
        torch._C._log_api_usage_once("torchaudio.transforms.CQT")
        
        # CQT corresponds to a VQT with gamma set to 0
        self.transform = VQT(
            sample_rate=sample_rate,
            hop_length=hop_length,
            f_min=f_min,
            n_bins=n_bins,
            gamma=0.,
            bins_per_octave=bins_per_octave,
            window_fn=window_fn,
            resampling_method=resampling_method,
            dtype=dtype,
        )
    
    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., channels, time).
                2D or 3D; batch dimension is optional.

        Returns:
            Tensor: constant-Q transform spectrogram of size (..., channels, ``n_bins``, time).
        """
        return self.transform(waveform)
    

class InverseCQT(torch.nn.Module):
    r"""Compute the inverse constant Q-transform.

    .. devices:: CPU CUDA

    .. properties:: Autograd

    Sources
        * https://librosa.org/doc/main/generated/librosa.icqt.html
        * https://www.aes.org/e-lib/online/browse.cfm?elib=17112
        * https://newt.phys.unsw.edu.au/jw/notes.html

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        hop_length (int, optional): Length of hop between CQT windows. (Default: ``400``)
        f_min (float, optional): Minimum frequency, which corresponds to first note.
            (Default: ``32.703``, or the frequency of C1 in Hz)
        n_bins (int, optional): Number of CQT frequency bins, starting at ``f_min``. (Default: ``84``)
        bins_per_octave (int, optional): Number of bins per octave. (Default: ``12``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        resampling_method (str, optional): The resampling method to use.
            Options: [``sinc_interp_hann``, ``sinc_interp_kaiser``] (Default: ``"sinc_interp_hann"``)
        dtype (torch.device, optional):
            Determines the precision that kernels are pre-computed and cached in. Note that complex
            bases are either cfloat or cdouble depending on provided precision.
            Options: [``torch.float``, ``torch.double``] (Default: ``torch.float``)

    Example
        >>> transform = transforms.InverseCQT()
        >>> waveform = transform(cqt)  # (..., time)
    """
    __constants__ = ["sample_rate", "resampling_method", "forward_params"]

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 256,
        f_min: float = 32.703,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        resampling_method: str = "sinc_interp_hann",
        dtype: torch.dtype = torch.float,
    ) -> None:
        super(InverseCQT, self).__init__()
        torch._C._log_api_usage_once("torchaudio.transforms.InverseCQT")
        
        self.sample_rate = sample_rate
        n_filters = min(bins_per_octave, n_bins)
        frequencies, n_octaves = F.frequency_set(f_min, n_bins, bins_per_octave, dtype=dtype)
        alpha = F.relative_bandwidths(frequencies, n_bins, bins_per_octave)
        freq_lengths, _ = F.wavelet_lengths(frequencies, self.sample_rate, alpha, 0.)
        
        self.resampling_method = resampling_method
        
        # Get sample rates and hop lengths used during CQT downsampling
        sample_rates = []
        hop_lengths = []
        temp_sr, temp_hop = float(self.sample_rate), hop_length
        
        for _ in range(n_octaves - 1, -1, -1):
            sample_rates.append(temp_sr)
            hop_lengths.append(temp_hop)
            
            if temp_hop % 2 == 0:
                temp_sr /= 2.
                temp_hop //= 2
        
        sample_rates.reverse()
        hop_lengths.reverse()
        
        # Now pre-compute what's needed for forward loop
        self.forward_params = []
        
        for oct_index, (temp_sr, temp_hop) in enumerate(zip(sample_rates, hop_lengths)):
            # Slice out correct octave
            indices = slice(n_filters * oct_index, n_filters * (oct_index + 1))
            
            octave_freqs = frequencies[indices]
            octave_alphas = alpha[indices]
            
            # Compute wavelet filterbanks
            basis, lengths = F.wavelet_fbank(octave_freqs, temp_sr, octave_alphas, 0., window_fn, dtype=dtype)
            n_fft = basis.shape[1]
            
            # Normalize wrt FFT window length
            factors = lengths.unsqueeze(1) / float(n_fft)
            basis *= factors
            
            # Wavelet basis FFT
            fft_basis = torch.fft.fft(basis, n=n_fft, dim=1)[:, :(n_fft//2) + 1]
            
            # Transpose basis
            basis_inverse = fft_basis.H
            
            # Compute filter power spectrum
            squared_mag = torch.abs(basis_inverse)**2
            frequency_pow = 1 / squared_mag.sum(dim=0)
            
            # Adjust by normalizing with lengths
            frequency_pow *= n_fft / freq_lengths[indices]
            
            self.register_buffer(f"basis_inverse_{oct_index}", basis_inverse)
            self.register_buffer(f"frequency_pow_{oct_index}", frequency_pow)
            self.forward_params.append((temp_sr, temp_hop, indices))

        # Create ones on the correct device in the forward pass
        self.ones = lambda x: torch.ones(x, dtype=dtype, device=self.basis_inverse_0.device)

    def forward(self, cqt: Tensor) -> Tensor:
        r"""
        Args:
            cqt (Tensor): Constant-q transform tensor of dimension (..., channels, ``n_bins``, time).
                3D or 4D; batch dimension is optional.

        Returns:
            Tensor: waveform of size (..., channels, time).
        """
        # Iterate down the octaves
        for buffer_index, (temp_sr, temp_hop, indices) in enumerate(self.forward_params):
            # Inverse project the basis
            temp_proj = torch.einsum(
                'fc,c,...ct->...ft',
                getattr(self, f"basis_inverse_{buffer_index}"),
                getattr(self, f"frequency_pow_{buffer_index}"),
                cqt[..., indices, :],
            )
            n_fft = 2 * (temp_proj.shape[-2] - 1)
            
            if temp_proj.ndim == 4:
                # torch istft does not support 4D computation yet
                # iterate through channels for stft computation
                for channel in range(temp_proj.shape[1]):
                    channel_waveform = torch.istft(
                        temp_proj[:, channel, :, :],
                        n_fft=n_fft,
                        hop_length=temp_hop,
                        window=self.ones(n_fft),
                    )
                
                    if channel == 0:
                        temp_waveform = channel_waveform.unsqueeze(1)
                    else:
                        temp_waveform = torch.cat([temp_waveform, channel_waveform.unsqueeze(1)], dim=1)
            
            else:
                temp_waveform = torch.istft(
                    temp_proj, n_fft=n_fft, hop_length=temp_hop, window=self.ones(n_fft),
                )
            
            temp_waveform = F.resample(
                temp_waveform,
                orig_freq=1,
                new_freq=self.sample_rate//temp_sr,
                resampling_method=self.resampling_method,
            )
            
            if buffer_index == 0:
                waveform = temp_waveform
            else:
                waveform[..., :temp_waveform.shape[-1]] += temp_waveform
        
        return waveform
