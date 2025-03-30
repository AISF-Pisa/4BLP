"""
Implementetion of DFT and FFT
"""
import time
import numpy as np
import matplotlib.pyplot as plt


def DFT(x, anti=-1):
    '''
    Compute the discrete Fourier Transform of the 1D array x

    Parameters
    ----------
    x : 1darray
        data to transform
    anti : int, optional
        -1 trasform
         1 anti trasform

    Return
    ------
    dft : 1d array
        dft or anti dft of x
    '''

    N = len(x)        # length of array
    n = np.arange(N)  # array from 0 to N
    k = n[:, None]    # transposed of n written as a Nx1 matrix
    # is equivalent to k = np.reshape(n, (N, 1))
    # so k * n will be a N x N matrix

    M = np.exp(anti * 2j * np.pi * k * n / N)
    dft = M @ x

    if anti == 1:
        return dft/N
    else:
        return dft


def FFT(x, anti=-1):
    '''
    Compute the Fast Fourier Transform of the 1D array x.
    Using non recursive Cooley-Tukey FFT.
    In recursive FFT implementation, at the lowest
    recursion level we must perform  N/N_min DFT.
    The efficiency of the algorithm would benefit by
    computing these matrix-vector products all at once
    as a single matrix-matrix product.
    At each level of recursion, we also perform
    duplicate operations which can be vectorized.

    Parameters
    ----------
    x : 1darray
        data to transform
    anti : int, optional
        -1 trasform
         1 anti trasform

    Return
    ------
    fft : 1d array
        fft or anti fft of x

    '''
    N = len(x)

    if np.log2(N) % 1 > 0:
        msg_err = "The size of x must be a power of 2"
        raise ValueError(msg_err)

    # stop criterion
    N_min = min(N, 2**2)

    #  DFT on all length-N_min sub-problems
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(anti * 2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    while X.shape[0] < N:
        # first  part of the matrix, the one on the left
        X_even = X[:,:X.shape[1] // 2 ] # all rows, first  X.shape[1]//2 columns
        # second part of the matrix, the one on the right
        X_odd  = X[:, X.shape[1] // 2:] # all rows, second X.shape[1]//2 columns

        f = np.exp(anti * 1j * np.pi * np.arange(X.shape[0]) / X.shape[0])[:, None]
        X = np.vstack([X_even + f*X_odd, X_even - f*X_odd]) # re-merge the matrix

    fft = X.ravel() # flattens the array
    # from  matrix Nx1 to array with length N

    if anti == 1:
        return fft/N
    else :
        return fft


def traditional_iterative_fft(x, anti=-1):
    """
    Compute the FFT of a 1D array using an iterative in-place approach.
    Optimized to minimize memory usage.

    Parameters:
    -----------
    x : np.array
        Input signal (length must be a power of 2).
    anti : int, optional, dafult -1
        -1 trasform
         1 anti trasform

    Returns:
    --------
    x : np.array
        The transformed array.
    """
    N = len(x)
    if np.log2(N) % 1 > 0:
        raise ValueError("The input size must be a power of 2.")

    x = np.array(x, dtype=np.complex128)  # Ensure complex type

    # Step 1: Bit-reversal reordering
    j = 0
    for i in range(1, N):
        bit = N >> 1     # bit-shift right
        while j & bit:   # bitwise and
            j ^= bit     # xor
            bit >>= 1    # bit-shift right
        j ^= bit         # xor
        if i < j:
            x[i], x[j] = x[j], x[i]  # Swap elements in-place

    # Step 2: Iterative FFT computation
    m = 2
    while m <= N:
        wm = np.exp(anti * 2j * np.pi / m)  # Root of unity
        for k in range(0, N, m):
            w = 1
            for j in range(m // 2):
                t = w * x[k + j + m // 2]  # Odd term
                u = x[k + j]               # Even term
                x[k + j] = u + t           # update in-place
                x[k + j + m // 2] = u - t  # update in-place
                w *= wm                    # Update twiddle factor
        m *= 2  # Double the segment size

    if anti == 1:
        x /= N  # Normalize for inverse FFT

    return x


def RFFT(x, anti=-1):
    '''
    Compute the fft for real value using FFT
    only values corresponding to positive
    frequencies are returned.

    For the forward transform (anti=-1):
    -------------------------------------
    1) The real signal is converted into a complex sequence by combining
       even and odd samples: z[n] = x[2n] + j * x[2n+1].
    2) A standard FFT is applied to this reduced sequence of length N/2.
    3) The symmetric spectrum required for the real transform is reconstructed.
    4) Take only the first N/2+1 values of the spectrum, for positive frequencies.

    For the inverse transform (anti=1):
    ------------------------------------
    1) Given only the RFFT coefficients, the full spectrum is reconstructed 
      using conjugate symmetry.
    2) The even and odd components of the signal in the frequency domain are separated.
    3) An iFFT of length N/2 is applied to obtain a complex signal Z.
    4) The real part of Z gives the even-indexed samples, the imaginary part the odd ones.

    Parameters
    ----------
    x : 1darray
        data to transform
    anti : int, optional
        -1 trasform
         1 anti trasform

    Return
    ------
    rfft : 1d array
        rfft or anti rfft of x
    '''
    if anti == -1 :
        z  = x[0::2] + 1j * x[1::2]  # Splitting odd and even
        Zf = FFT(z)
        Zc = np.array([Zf[-k] for k in range(len(z))]).conj()
        Zx =  0.5  * (Zf + Zc)
        Zy = -0.5j * (Zf - Zc)

        N = len(x)
        W = np.exp(- 2j * np.pi * np.arange(N//2) / N)
        Z = np.concatenate([Zx + W*Zy, Zx - W*Zy])

        return Z[:N//2+1]

    if anti == 1 :
       
        N = 2 * (len(x) - 1)  # Length of the original signal
        k = np.arange(N//2)   # Index until N/2-1

        # Reconstruction of the full spectrum
        X_full = np.zeros(N, dtype=complex)
        X_full[:len(x)] = x
        X_full[len(x):] = np.conj(x[-2:0:-1])  # Simmetria coniugata

        Xe = 0.5 * (X_full[k] + np.conj(X_full[N//2 - k]))
        Xo = 0.5 * (X_full[k] - np.conj(X_full[N//2 - k])) * np.exp(2j * np.pi * k / N)
        # N/2 IFFT
        Z = Xe + 1j * Xo
        z = FFT(Z, anti=1)

        # Reconstruction of the original signal
        x_n = np.zeros(N, dtype=float)
        x_n[0::2] = np.real(z)
        x_n[1::2] = np.imag(z)

        return x_n


def fft_freq(n, d, real):
    '''
    Return the Discrete Fourier Transform sample frequencies.
    if real = False then:
    f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
    f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd
    else :
    f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
    f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd

    Parameters
    ----------
    n : int
        length of array that you transform

    d : float
        Sample spacing (inverse of the sampling rate).
        If the data array is in seconds
        the frequencies will be in hertz
    real : bool
        false for fft
        true for rfft

    Returns
    -------
    f: 1d array
        Array of length n containing the sample frequencies.
    '''
    if not real:
        if n%2 == 0:
            f1 = np.array([i for i in range(0, n//2)])
            f2 = np.array([i for i in range(-n//2,0)])
            return np.concatenate((f1, f2))/(d*n)
        else :
            f1 = np.array([i for i in range((n-1)//2 + 1)])
            f2 = np.array([i for i in range(-(n-1)//2, 0)])
            return np.concatenate((f1, f2))/(d*n)
    if real:
        if n%2 == 0:
            f1 = np.array([i for i in range(0, n//2 +1)])
            return f1 / (d*n)
        else :
            f1 = np.array([i for i in range((n-1)//2 +1)])
            return f1 / (d*n)


if __name__ == '__main__':

    # data
    x = np.linspace(0, 10, 2**11) # no more 2**14 for DFT
    y = 1 + 3*np.sin(2*(2*np.pi)*x) + np.sin(5*(2*np.pi)*x) + 0.5*np.sin(8*(2*np.pi)*x) + 0.001*np.sin(40*(2*np.pi)*x)
    noise = np.array([np.random.random() for _ in x])
    noise = 2 * noise - 1 # from [0, 1] to [-1, 1]
    intensity = 0.0
    y = y + intensity * noise
    #y = y - np.mean(y)
    #y = y*np.sin(40*(2*np.pi)*x)

    # DFT
    t0 = time.time()
    dft_m_i  = DFT(y)
    dt = time.time() - t0
    print(dt)
    anti_dft = np.real(DFT(dft_m_i, anti=1))

    #FFT
    t0 = time.time()
    fft_m_i = FFT(y)
    #fft_m_i = traditional_iterative_fft(y)
    dt = time.time()-t0
    print(dt)
    anti_fft = np.real(FFT(fft_m_i, anti=1))
    #anti_fft = np.real(traditional_iterative_fft(fft_m_i, anti=1))
    
    #RFFT
    t0 = time.time()
    fft_m_r   = RFFT(y)
    dt = time.time()-t0
    print(dt)
    anti_rfft = np.real(RFFT(fft_m_r, anti=1))
    
    #numpy FFT
    t0 = time.time()
    fft_n_i = np.fft.fft(y)
    dt = time.time()-t0
    print(dt)
    fft_n_r = np.fft.rfft(y)

    freq_m_i = fft_freq(len(y), x[-1]/len(y), real=False)
    freq_n_i = np.fft.fftfreq(len(y), x[-1]/len(y))
    freq_m_r = fft_freq(len(y), x[-1]/len(y), real=True)
    freq_n_r = np.fft.rfftfreq(len(y), x[-1]/len(y))

    ##Plot

    plt.figure(1)
    plt.title('Original Signal', fontsize=15)
    plt.xlabel('x [s]', fontsize=15)
    plt.ylabel('y [a.u.]', fontsize=15)
    plt.plot(x, y, 'b')
    plt.grid()

    plt.figure(2)
    plt.title('total fft, dft', fontsize=15)
    plt.xlabel('frequencies [Hz]', fontsize=15)
    plt.ylabel('abs(spectrum)', fontsize=15)
    plt.plot(freq_m_i, abs(dft_m_i), 'b', label='dft')
    plt.plot(freq_n_i, abs(fft_n_i), 'k', label='fft')
    plt.plot(freq_m_i, abs(fft_m_i), 'r', label='numpy')
    plt.legend(loc='best')
    plt.grid()

    plt.figure(3)
    plt.title('real fft', fontsize=15)
    plt.xlabel('frequencies [Hz]',fontsize=15)
    plt.ylabel('abs(spectrum)', fontsize=15)
    plt.plot(freq_n_r, abs(fft_n_r), 'k', label='numpy')
    plt.plot(freq_m_r, abs(fft_m_r), 'r', label='fft')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.grid()

    plt.figure(4)
    plt.subplot(221)
    plt.title("abs(FFT - np.fft.fft)")
    plt.plot(abs(abs(fft_m_i)-abs(fft_n_i)), 'b')
    plt.yscale('log')
    plt.grid()

    plt.subplot(222)
    plt.title("abs(DFT - np.fft.fft)")
    plt.plot(abs(abs(dft_m_i)-abs(fft_n_i)), 'b')
    plt.yscale('log')
    plt.grid()

    plt.subplot(223)
    plt.title(r"signal - $\mathcal{F}^{-1}(\mathcal{F}(signal))$")
    plt.plot(x, abs(anti_fft-y), 'r',  label='fft')
    plt.plot(x, abs(anti_rfft-y), 'b',  label='rfft')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.grid()

    plt.subplot(224)
    plt.title(r"signal - $\mathcal{F}^{-1}(\mathcal{F}(signal))$")
    plt.plot(x, abs(anti_dft-y), 'b',  label='dft')
    plt.plot(x, abs(anti_fft-y), 'r',  label='fft')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.grid()

    plt.show()
