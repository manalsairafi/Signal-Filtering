# Signal-Filtering
Python program made for designing detector filters to optimize energy resolutions

The program takes as input the signal and the noise from the detector as ```np.float32``` raw files - this should be changed depending on the type of file used. 
The file considered is a string of pulses, each pulse is of length $2^{14}$, this may also be changed wherever ```np.array_split()``` is used.

Then the type of sample in the detector (used for energy calibration) is chosen. The default is an Fe55 sample. This is only relevant if the energy spectrum is to be produced and the resolutions calculated. The FFTs of the signal as well as of the noise are displayed to aid choosing a filter. The frequency response of the chosen filter is displayed (for checking purposes). Then you are given the option to fit the filtered signal. The filtered signals are saved using the name of the original file and the filter chosen. The  user is prompted with ```again?``` if yes, a different filter can be applied on the same original signal. If the spectrum was found and fit, the fitting parameters are saved in a .csv file.

The filters $F(\omega)$ are designed in the frequency domain and applied to the signal pulses via convolution

$$ Y(\omega) = F(\omega) X(\omega)$$

here $Y(\omega)$ is the Fourier transform of the filtered pulse and $X(\omega)$ is that of the unfiltered pulse. The filtered pulses can then be returned by and inverse transform.

## The filters
note: in the following I use the frequency $f=\omega/2\pi$ and $f_0$ is the frequency cutoff, $n$ is the filter order

**(1) Simple filter**

Low pass:
$$F(f)= \left( \frac{1}{1+i(f/f_0)}\right)^n $$

High pass:
$$F(f)= \left( \frac{i(f/f_0)}{1+i(f/f_0)}\right)^n $$

**(2) Butterworth filter**

Low pass:
$$F(f)= \frac{1}{\sqrt{1+(f/f_0)^{2n}}} $$

High pass:
$$F(f)= \frac{(|f|/f_0)^n}{\sqrt{1+(|f|/f_0)^{2n}}} $$

**(3) Chebychev filters**

Only works as a low pass filter

Chebychev type1:
$$F(f) = \frac{1}{\sqrt{1+\varepsilon^2 T_n(f/f_0)^2}}$$

Chebychev type2:
$$F(f) = \sqrt{\frac{\varepsilon^2 T_n(f/f_0)^2}{1+\varepsilon^2 T_n(f/f_0)^2}}$$

$T_n(f/f_0)$ is the $n$th order Chebychev polynomial, $\varepsilon$ is the ripple factor. The Chebychev polynomials are calculated by the recurrence relation:
$$T_0(x) = 1$$
$$T_1(x) = x$$
$$T_{n+1} = 2x T_n(x) - T_{n-1}(x)$$

**(4) Matched filter**

*This is expected to be the optimum*
$$F(f) = \frac{Y(f)^*}{|N(f)|^2}$$
$N(f)$ is the average Fourier transform of the noise signal

## The spectrum

The spectrum is found by taking a histogram of the pulse heights, fitting to a gaussian of two peaks. The peaks are calibrated to the sample, then the gaussian fits are repeated in the energy domain.
