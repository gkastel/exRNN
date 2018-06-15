import matplotlib.pyplot as plt
import numpy as np
import sys


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y



datadir = "./data/"
alphapar=0.0
betapar=0.0

def plotSamples():
	plotstart=0
	for rseed in range(0,10):
		plt.figure()
		l1 = np.load(datadir+"/loss_%d_0.npy"%(rseed ));
		l2 = np.load(datadir+"/loss_%d_1.npy"%(rseed ));
		#l3 = np.load(datadir+"/loss_%d_2.npy"%(rseed ));

		l1 = l1[plotstart:];
		l2 = l2[plotstart:];
		#l3 = l3[0:];

		plt.plot(l1, 'b', alpha=0.2)
		plt.plot(l2, 'r', alpha=0.2)


		#plt.plot(l3, 'm', alpha=0.5)

		l1 = smooth(l1, window_len=101);
		l2 = smooth(l2, window_len=101);
		#l3 = smooth(l3, window_len=101);

		plt.plot(l1, color='blue' )
		plt.plot(l2, color='red');
		plt.plot(l1-l2, 'k', alpha=0.3)
		ax = plt.gca()
		ax.grid(True)

		#plt.xlim([0,10000]);
		plt.xlabel('Iteration');
		plt.ylabel('Cross-Entropy Loss');
		plt.title("g %d"%(rseed))
		plt.ylim([-50,200])
		plt.savefig("./data/plt_%d.pdf"%(rseed))



def plotDifs():
	m1 , m2 = [], []
	for rseed in range(0,50):
		l1 = np.load(datadir+"/loss_%d_0_0.000000_0.000000.npy"%(rseed ));
		l2 = np.load(datadir+"/loss_%d_1_%f_%f.npy"%(rseed, alphapar, betapar ));

		#l1 = np.load(datadir+"/loss_%d_0.npy"%(rseed ));
		#l2 = np.load(datadir+"/loss_%d_1.npy"%(rseed ));

		m1.append( np.sum(l1))
		m2.append( np.sum(l2))
	
	plt.figure()
	a1 = np.array(m1);
	a2 = np.array(m2);

	plt.plot(a1, 'b')
	plt.plot(a2, 'r')
	plt.plot((a1 - a2), 'k');
	#print np.sum(a1-a2) / (np.sum(a1) + np.sum(a2))
	print np.sum(a1-a2)


	#plt.figure()
	#plt.boxplot(a1,  a2)

def plotPars():


	ms, sts = [], []
	for  par in [-0.2, -0.1, 0, 0.1 , 0.2]:
		alphapar = 0
		betapar=par
		m1 = np.zeros((50));
		for rseed in range(0,50):
			l1 = np.load(datadir+"/loss_%d_0_0.000000_0.000000.npy"%(rseed ));
			l2 = np.load(datadir+"/loss_%d_1_%f_%f.npy"%(rseed, alphapar, betapar ));

			s1 = ( np.sum(l1))
			s2 = ( np.sum(l2))
			m1[rseed] = ((s1-s2)/(s2))
		ms.append(np.mean(m1))
		sts.append(np.std(m1)/np.sqrt(50))
	
	print ms
	print sts
	plt.figure()
	#xs = [0.1, 0.2, 0.3, 0.4, 0.5]
	xs = [0.5, 0.6, 0.7, 0.8, 0.9]

	plt.errorbar(xs, 100.*np.array(ms), yerr=100*np.array(sts))
	plt.xlabel('beta')
	plt.ylabel('Cumulative loss reduction (%)')

	plt.savefig("./data/cum_beta.pdf")


	if (0):
		plt.figure()
		a1 = np.array(m1);
		plt.plot(100.*a1, 'bd')
		print np.mean(m1)
		plt.ylabel('Cumulative loss reduction (%)');
		plt.xlabel('Trial #')
		plt.gca().yaxis.grid(True);
		plt.savefig("./data/cumloss_%d.pdf")




plotSamples()
plotPars()

plt.show()
