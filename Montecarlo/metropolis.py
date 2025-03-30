import time
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def target(x, mu=0.0, sigma=1.0):
    '''
    Target distribution, i.e. the distribution
    from which we want to sample
    
    Parameters
    ----------
    x : float
        indipendent variable
    mu, sigma : float, otipnal, dafult 0, 1
        Gaussian parameters
    
    Returns
    -------
    log_g : float
        logaritm of the target distribution
    '''
    
    log_g = ((x - mu)/sigma)**2
    
    return -0.5*log_g


def uniform_proposal(x0, rng, step=0.5):
    '''
    Distribution that we know how to sample
    
    Parameters
    ----------
    x0 : float
        piont of the chain at 'time' i
    rng : Generator(PCG64)
        random number generator fomr numpy
    step : float, optional, dafult 0.5
        step of the uniform distribution
        
    Returns
    ----------
    uniform distribution centered in x0
    '''
   
    return x0 + step*rng.uniform(-1, 1)
    

def metropolis_hastings(target, proposal, rng, n=int(1e3), step=0.5, args=()):
    '''
    Metropolis hastings algorithm
    
    Parameters
    ----------
    target : callable
        Distribution to sampling
    proposal : callable
        Distribution that we know how to sample
    rng : Generator(PCG64)
        random number generator fomr numpy
    n : int
        number of iteration
    step : float, optional, dafult 0.5
        step for proposal distribution
    args : tuple, optional, default ()
        extra arguments for target distribution
    
    Returns
    -------
    samples : 1darray
        array of montecarlo chain
    
    '''
    
    samples = np.zeros(n)
    
    x0    = rng.uniform(-10, 10)
    logP0 = target(x0, *args)
    
    
    for i in range(n):
        
        x_pr = proposal(x0, rng, step=step)    # Sample from proposal
        logP = target(x_pr, *args)             # Compute log of target distribution
        logr = logP - logP0                    # Compute log of the ratio

        if np.log(rng.uniform(0, 1)) < logr:
            x0 = x_pr
            logP0 = logP
            samples[i] = x_pr
           
        else:
            samples[i] = x0
        
    return samples


def autocorrelation(chain):
    '''
    compute autocorrelation of the chain
    
    Paramateres
    -----------
    chain : 1darray
        array of montecarlo chain
    
    Returns
    -------
    auto_corr : 1darray
        array with auto-correlation of chain
    time : int
        auto-correlation time
    '''

    m = np.mean(chain) # Mean
    s = np.var(chain)  # Variance
    
    xhat = chain - m
    auto_corr = np.correlate(xhat, xhat, 'full')[len(xhat)-1:]
    
    auto_corr = auto_corr/s/len(xhat) # Normalization

    # Auto-correlation time
    time = np.where(abs(auto_corr) < 0.01)[0][0]
    
    return auto_corr, time


if __name__ == "__main__":
    
    start = time.time()
    
    rng = np.random.default_rng(69420)
    
    misure = int(1e5)  # Number of samples
    termal = int(5e2)  # Number of thermalization steps
    
    n = misure + termal       # Total number of MCMC steps
    target_args = (2.0, 0.5)  # Gaussian parameters
    
    samples = metropolis_hastings(
                target, uniform_proposal, rng, n=n,
                step=0.5, args=target_args)
    
              
    acf, ac_time = autocorrelation(samples)

    plt.figure(1, figsize=(14, 6))
    plt.suptitle('Output data')
    
    plt.subplot(221)
    plt.plot(np.linspace(0, n, n), samples, 'b.')
    plt.ylabel('Chain', fontsize=12)
    plt.grid()

    samples = samples[termal:]
    clean_samples = samples[::ac_time]

    plt.subplot(223)
    plt.plot(np.linspace(0, n, n)[termal:][::ac_time], clean_samples, 'b.')
    plt.xlabel('iteration', fontsize=12)
    plt.ylabel('Cleaned chain', fontsize=12)
    plt.grid()
    
    plt.subplot(122)
    plt.plot(ac_time, acf[ac_time], 'ro')
    plt.plot(acf[:200], 'b')
    plt.xlabel('iteration', fontsize=12)
    plt.ylabel('auto-correlation function', fontsize=12)
    plt.grid()

    
    print(f'{np.mean(samples):.3f}')
    print(f'{np.mean(clean_samples):.3f}')

    print(f'{np.std(samples)/np.sqrt(len(samples)-1):.3f}')
    print(f'{np.std(clean_samples)/np.sqrt(len(clean_samples)-1):.3f}')
    
    #distribution
    pdf = norm(*target_args)
                
    x = np.linspace(target_args[0] - 6*target_args[1], 
                    target_args[0] + 6*target_args[1], 1000)
    
    plt.figure(3)
    plt.title('Gaussian distribution')
    plt.hist(samples, density=True, bins=50, histtype='step', color='b', label='samples')
    plt.plot(x, pdf.pdf(x),'-k', label='pdf')
    plt.xlabel("x", fontsize=15)
    plt.ylabel("pdf(x)", fontsize=15)
    plt.grid()
    
    end = time.time() - start
    print(f'Elapsed time = {end:.3f} seconds')
    
    plt.show()