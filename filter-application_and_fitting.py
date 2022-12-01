# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 16:25:03 2022

@author: manal
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import csv
from scipy.optimize import curve_fit
import copy

#the filter functions
def simple_filter(f, f0, n, btype = 'low'):
    '''
    Parameters
    ----------
    f : list or 1darray
        list of frequencies.
    f0 : float
        cutoff frequency.
    n : integer
        order of filter.
    btype : string, optional
        either 'low'or 'high' for lowpass filter or highpassfilter. The default is 'low'.

    Raises
    ------
    ValueError
        if f0 is not one value, if n is not integer, and if type of filter is not low or high.

    Returns
    -------
    list or 1darray
        filter frequency response.

    ''' 
    if type(n) != int:
        raise ValueError('n (order of filter) must be an integer')
        
    if btype not in ['low', 'high']:
        raise ValueError('btype must be \'low\' (for lowpass) or \'high\' (for highpass)')
    else:
        if btype == 'low':
            return  ( 1/(1+1j*(f/f0)) )**n
    
        if btype == 'high':
            return (1j*(f/f0)/(1+1j*(f/f0)))**n
    
def butter_filter(f, f0, n, btype = 'low'):
    '''
    Parameters
    ----------
    f : list or 1darray
        list of frequencies.
    f0 : float
        cutoff frequency.
    n : integer
        order of filter.
    btype : string, optional
        either 'low'or 'high' for lowpass filter or highpassfilter. The default is 'low'.

    Raises
    ------
    ValueError
        if f0 is not one value, if n is not integer, and if type of filter is not low or high.

    Returns
    -------
    list or 1darray
        filter frequency response.

    '''
    if type(n) != int:
        raise ValueError('n (order of filter) must be an integer')
        
    if btype not in ['low', 'high']:
        raise ValueError('btype must be \'low\' (for lowpass) or \'high\' (for highpass)')
    else:
        if btype == 'low':
            return 1/np.sqrt(1+(f/f0)**(2*n))
    
        if btype == 'high':
            posf = (f[:len(f)//2]/f0)**n/np.sqrt(1+(f[:len(f)//2]/f0)**(2*n))
            negf = (-f[len(f)//2:]/f0)**n/np.sqrt((1+(-f[len(f)//2:]/f0)**(2*n)))
            full = np.array([])
            full = np.append(full, posf)
            full = np.append(full, negf)
            return full
        
def cheby_polynomial(x, n):
    ts = n * [0]
    ts[0] = 1
    
    if n > 0:
        ts[1] = x
        i = 2
        while i <= n-1 :
            ts[i] = 2*x* ts[i-1] - ts[i-2]
            i += 1
            
    return ts[-1]

def cheby1_filter(f, f0, eps, n):
    '''
    Parameters
    ----------
    f : list or 1darray
        list of frequencies.
    f0 : float
        cutoff frequency.
    delta : float
        passband ripple, eps is the ripple factor.
     n : integer
         order of filter.

    Returns
    -------
    list or 1darray
        filter frequency response.

    '''
    if type(n) != int:
        raise ValueError('n (order of filter) must be an integer')
        
    tn = cheby_polynomial(np.abs(f)/f0,n)
    
    return 1/np.sqrt(1+eps**2 * tn**2)

def cheby2_filter(f, f0, eps, n):
    '''
    Parameters
    ----------
    f : list or 1darray
        list of frequencies.
    f0 : float
        cutoff frequency.
    gamma : float
        stopband attenuation.
     n : integer
         order of filter.

    Returns
    -------
    list or 1darray
        filter frequency response.

    '''
    if type(n) != int:
        raise ValueError('n (order of filter) must be an integer')
    
    tn = cheby_polynomial(f0/np.abs(f),n)
    
    tn[0] = tn[1]
    
    return np.sqrt(eps**2 * tn**2/ (1 + eps**2 * tn**2))

#applying the filter        
def apply_filter(sig, filt):
    '''

    Parameters
    ----------
    sig : array_like
        signal file in time space.
    filt : array_like
        frequency response of filter.

    Returns
    -------
    array_like
        filtered signal.

    '''
    sigf = fft(sig)
    sigfiltf = np.multiply(sigf, filt)
    return np.float32(ifft(sigfiltf))

#fitting function
def two_point_gaussian(x, A1, s1, mu1, A2, s2, mu2):
    return A1*np.exp(-(x-mu1)**2/(2*s1**2))+A2*np.exp(-(x-mu2)**2/(2*s2**2))

#linear function
def linear(x, m, b):
    return m*x + b


#getting the files
file_signal = input('Signal File: ')
file_noise = input('Noise File: ')

element = input('Element (Fe55, else): ')
if element == 'Fe55':
    E_peaks = [5.9, 6.5]
if element == 'else':
    E_peaks = [input('peak1 (keV): '), input('peak2 (keV): ')]
    
signals = np.fromfile(file_signal, dtype=np.float32)
noise = np.fromfile(file_noise, dtype=np.float32)

times = np.linspace(0, 2**(14)*40e-6, 2**14) #defining the time steps

#shifting down pulses

#first splitting into traces
signals_split = np.array_split(signals, len(signals)/2**(14)) 

#shifting
signals_normal = []
for i in signals_split:
    x = i - min(i)
    signals_normal.append(x)

#same for noise
noise_split = np.array_split(noise, len(noise)/2**(14))

noise_normal = []
for i in noise_split:
    x = i - min(i)
    noise_normal.append(x)
        
signals_split = copy.deepcopy(signals_normal)
noise_split = copy.deepcopy(noise_normal)

#for the ffts
N = 2**(14) #length of each trace
T = 40e-6 #time step

#creating the template
template = signals_split[0]

for i in signals_split[1:]:
    template += i
    
template /= len(signals_split)

avg_signal = fft(template) #this is the FFT of the signal

#getting fft of noise 
avg_noise = np.abs(fft(noise_split[0]))

for i in noise_split[1:]:
    fft_i = np.abs(fft(i))
    for j in range(len(avg_noise)):
        avg_noise[j] += (fft_i[j])

avg_noise = avg_noise/len(noise_split) #fft of noise - should be real (abs)

#frequencies
xf = fftfreq(N, T) 

plt.plot(xf[:N//2], np.abs(avg_signal[:N//2]), alpha=0.8, label = 'source')
plt.plot(xf[:N//2], np.abs(avg_noise[:N//2]), alpha=0.8, label = 'noise')
plt.xscale('log')
plt.yscale('log')

plt.legend()
plt.grid()

plt.show()


#setting up for filtering
again = 'yes'

while again == 'yes':
    #making copies so the original is unchanged
    save_name = copy.copy(file_signal[:-4])
    signal_filt = copy.deepcopy(signals_normal)
    
    filter_type = input('Filter type  (simple/ butter/ cheby1/ cheby2/ matched: ')
    
    if filter_type not in ['simple', 'butter', 'matched', 'CC', 'cheby1', 'cheby2']:
        raise ValueError('filter type must be either \'simple\', \'butter\', \'cheby1\',\
                         \'cheby2\' or \'matched\'')
        
        
    filtered = []

    """__________________________filters designed based on ffts__________________________"""
    
    if filter_type in ['simple', 'butter', 'matched', 'cheby1', 'cheby2']:
        
        if filter_type == "simple" or filter_type == "butter":
            cutoff = (input('Cutoff frequency: '))
            order = int(input('Filter order: '))
            bandpass = input('Highpass or lowpass: ')
            
            if cutoff == 'auto':
                cutoff_name = 'auto'
                
                opt_curve = avg_signal/(avg_noise**2)
                
                peak_index = np.where(opt_curve == max(opt_curve[:N//2]))
                
                cutoff = xf[:N//2][peak_index]
                
            else:
                cutoff_name = str((cutoff))
                cutoff = float(cutoff)
            
            save_name += '_'+filter_type +'_'+cutoff_name+'_n'+str(order)+'_'+bandpass+'.raw'
            
            if filter_type == 'simple':
                filt = simple_filter(xf, cutoff, order, btype = bandpass)
            
            if filter_type == 'butter':
                filt = butter_filter(xf, cutoff, order, btype = bandpass)
                
        if filter_type == 'cheby1' or filter_type == 'cheby2':
           cutoff = (input('Cutoff frequency: '))
           order = int(input('Filter order: '))
           
           if cutoff == 'auto':
               cutoff_name = 'auto'
               
               opt_curve = avg_signal/(avg_noise**2)
               
               peak_index = np.where(opt_curve == max(opt_curve[:N//2]))
               
               cutoff = xf[:N//2][peak_index]
               
           else:
               cutoff_name = str(int(cutoff))
               cutoff = float(cutoff)
           
           if filter_type == 'cheby1':
                add = input('Passband ripple: ')
                filt = cheby1_filter(xf, cutoff, float(add), order)
            
           if filter_type == 'cheby2':
                add = input('Stopband attenuation: ')
                filt = cheby2_filter(xf, cutoff, float(add), order)
                
           save_name += '_'+filter_type +'_'+cutoff_name+'_n'+str(order)+'_'+add+'.raw'
             
        if filter_type == 'matched':
            save_name += '_'+filter_type+'.raw'
            filt = np.conjugate(avg_signal)/(avg_noise**2)
            
        plt.plot(xf[:N//2], np.abs(filt[:N//2]))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('frequency [Hz]')
        plt.title('Frequency response of chosen filter')
        plt.grid()
        plt.show()
        
        for i in signal_filt:
            filtered.append(np.real(apply_filter(i, filt)))
    
    to_fit = input('Fit? ')
    if to_fit == 'yes':
        max_peaks = []
        for i in filtered:
            max_peaks.append(max(i)-min(i))
        
        #getting the spectrum
        spectrum, _, _ = plt.hist(max_peaks, bins =500)
        plt.close()
        
        rows = []
        for i in range(len(spectrum)):
            rows.append([i, spectrum[i]])

        save_name = save_name[:-4]+"_spectrum.csv"
        Details = ['channel', 'counts']  

        with open(save_name, 'w+', newline ='') as f: 
            write = csv.writer(f) 
            write.writerow(Details) 
            write.writerows(rows)
        
        channels, spectrum = np.loadtxt(save_name,unpack = True, delimiter=',', skiprows = 1, dtype = float)
        
        plt.plot(channels, spectrum)
        plt.show()

        print('initial guesses:')
        p1 = float(input('A1: ')) 
        p2 = float(input('s1: ') )
        p3 = float(input('mu1: ')) 
        p4 = float(input('A2: ') )
        p5 = float(input('s2: ') )
        p6 = float(input('mu2: ')) 


        fit_params, pcov = curve_fit(two_point_gaussian, channels, spectrum,
                                     p0 = [p1,p2,p3,p4,p5,p6],
                                 bounds=((0, 0, 0, 0, 0, p6-20), (np.inf, np.inf, p3+20, np.inf, np.inf, p6+50)))

        perr = np.sqrt(np.diag(pcov))

        fit = two_point_gaussian(channels, fit_params[0], fit_params[1], fit_params[2], 
                                 fit_params[3], fit_params[4], fit_params[5])
        
            
        calib_params, _ = curve_fit(linear, [fit_params[2], fit_params[5]], E_peaks)

        energies = linear(channels, calib_params[0], calib_params[1])

        fit_params_en, pcov_en = curve_fit(two_point_gaussian, energies, spectrum,
                                     p0 = [fit_params[0], fit_params[1]*calib_params[0] + calib_params[1], fit_params[2]*calib_params[0] + calib_params[1],
                                           fit_params[3], fit_params[4]*calib_params[0] + calib_params[1], fit_params[5]*calib_params[0] + calib_params[1]],
                                     bounds=((0, 0, 5.7, 0, 0, 6.4), (np.inf, np.inf, 6, np.inf, np.inf, 6.6)),maxfev =5000)
                                           

        perr_en = np.sqrt(np.diag(pcov_en))

        fit_en = two_point_gaussian(energies, fit_params_en[0], fit_params_en[1], fit_params_en[2], 
                                 fit_params_en[3], fit_params_en[4], fit_params_en[5])

        plt.scatter(energies, spectrum)
        plt.plot(energies, fit_en, c= 'black')
        plt.show()

        fits = [['A1', fit_params_en[0], perr_en[0]],
                ['s1', fit_params_en[1], perr_en[1]],
                ['Res 5.9keV percent', 100*2*np.sqrt(2*np.log(2))*fit_params_en[1]/fit_params_en[2], 100*2*np.sqrt(2*np.log(2))*np.sqrt((perr_en[1]/fit_params_en[2])**2 + (fit_params_en[1]*perr_en[2]/fit_params_en[2]**2)**2)],
                ['mu1', fit_params_en[2], perr_en[2]],
                ['A2', fit_params_en[3], perr_en[3]],
                ['s2', fit_params_en[4], perr_en[4]],
                ['Res 6.5keV percent', 100*2*np.sqrt(2*np.log(2))*fit_params_en[4]/fit_params_en[5], 100*2*np.sqrt(2*np.log(2))*np.sqrt((perr_en[4]/fit_params_en[5])**2 + (fit_params_en[4]*perr_en[5]/fit_params_en[5]**2)**2)],
                ['mu2', fit_params_en[5], perr_en[5]]]
        
        print(fits[2][1], fits[6][1])
        
        file = save_name[:-4] + '_fit.csv'
        Details = ['', 'fit', 'error']  


        with open(file, 'w+', newline ='') as f: 
            write = csv.writer(f) 
            write.writerow(Details) 
            write.writerows(fits) 
    again = input('again? ')
