import scipy.signal as sc_sig
import numpy as np
import scipy
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.integrate import simps
import h5py
import pandas as pd
import warnings

class ClasseDataset:

    def __init__(self):

        self.centrato = False               # indicates whether the time window has been cut and centered (bool)
        self.demeaned = False               # indicates whether the mean has been removed. Two types of means: will be string, "noise" or "total"
        self.normalized = False             # indicates whether it is normalized. Two types: will be string, "Max_track" or "Threshold_Num_samples_track"
        self.sismogramma = np.array([])     # np.array (,)
        self.metadata = {}                  # dictionary of lists, not np.array

    def leggi_custom_dataset(self, percorsohdf5, percorsocsv):
        """
        Read ALL traces
        to be saved only vertical component
        """
        
        filehdf5 = h5py.File(percorsohdf5, 'r')
        self.sismogramma = filehdf5.get("dataset1")
        self.sismogramma = np.array(self.sismogramma)
        self.metadata = 0
        self.metadata = pd.read_csv(percorsocsv)

        self.centrato = self.metadata["centrato"][1]
        self.demeaned = self.metadata["demeaned"][1]
        if "normalized" in self.metadata.keys():
            self.normalized = self.metadata["normalized"][1]
        filehdf5.close()

    def finestra(self, semiampiezza=0):
        """
            cut and center the window
            semiampiezza: number of samples (semiamplitude of the window)
        """
        sismogramma = [0 for _ in range(len(self.sismogramma))]

        if self.centrato:
            if len(self.sismogramma[0]) > 2 * semiampiezza:
                centro = len(self.sismogramma[0]) // 2
                for i in range(len(self.sismogramma)):
                    sismogramma[i] = self.sismogramma[i][centro - semiampiezza:
                                                         centro + semiampiezza]
                self.sismogramma = np.array(sismogramma)
            else:
                print("\nAlredy centered with a smaller window")
                print("I do nothing\n")

        else:
            for i in range(len(self.sismogramma)):
                if self.metadata["trace_P_arrival_sample"][i] > semiampiezza:
                    sismogramma[i] = self.sismogramma[i][int(self.metadata["trace_P_arrival_sample"][i]) - semiampiezza:
                                                         int(self.metadata["trace_P_arrival_sample"][i]) + semiampiezza]
                else:
                    print(f"short waveform! {i}")
                    stringa = "#"
                    for _ in range(100):
                        stringa = stringa + "#"
                    warnings.warn("\n"+stringa+"\nWARNING: CHOSE A SMALLER WINDOW,"
                                               "I DO NOTHING\n"+stringa)
                    print("Semiamplitude = ", semiampiezza, "P_arrival = ", self.metadata["trace_P_arrival_sample"][i])
                    input()
        

            self.sismogramma = np.array(sismogramma)

            self.centrato = True

    def demean(self, metodo: str = 'rumore', semiamp: int = 80):
        """
            method "totale" -> mean computed on whole trace
            method "rumore" -> mean computed on some samples before P arrival
        """
        if metodo == "totale":
            self.demeaned = "totale"
            self.sismogramma = self.sismogramma - np.mean(self.sismogramma, axis=1).reshape(len(self.sismogramma),1)

        if metodo == "rumore":
            self.demeaned = "rumore"
            if self.centrato:
                lung = len(self.sismogramma[0])
                self.sismogramma = self.sismogramma - np.mean(self.sismogramma[ : , lung//2-semiamp : lung//2-5], axis=1).reshape(len(self.sismogramma),1)

            else:
                for i in range(len(self.sismogramma)):
                    start_5 = self.metadata["trace_P_arrival_sample"][i] - 5
                    self.sismogramma[i] = self.sismogramma[i] - \
                                          np.mean(self.sismogramma[i][start_5 - semiamp : start_5])

        if metodo != "rumore" and metodo != "totale":
            print("NOT ALLOWED METHOD\n rumore or totale?")
            metodo = input()
            if metodo == "rumore" or metodo == "totale":
                self.demean(metodo)

    def normalizza(self, soglia=20.):
       
        if soglia == "None" or soglia == None:
            print("Normalize to the 'max'")
            
            self.sismogramma = self.sismogramma * 1.0                
            self.sismogramma = self.sismogramma / np.max([np.max(self.sismogramma,axis=1),-np.min(self.sismogramma,axis = 1)], axis = 0).reshape(len(self.sismogramma),1)

            self.normalized = f"Max_traccia_{len(self.sismogramma[0])}_di_samples"
        else:
            lung_traccia = len(self.sismogramma[0])
            self.sismogramma = self.sismogramma * 1.0             
            sism_0_arr = self.sismogramma[:,0:lung_traccia//2-5]
            self.sismogramma = self.sismogramma / (soglia * np.max([np.max(sism_0_arr,axis=1),-np.min(sism_0_arr,axis = 1)], axis = 0).reshape(len(sism_0_arr),1))
            self.sismogramma[self.sismogramma > 1.0] = 1.0
            self.sismogramma[self.sismogramma < -1.0] = -1.0
            self.sismogramma = self.sismogramma / (np.max([np.max(self.sismogramma,axis=1),-np.min(self.sismogramma,axis = 1)], axis = 0).reshape(len(self.sismogramma),1))
            self.normalized = f"Soglia={soglia}_traccia_di_{len(self.sismogramma[0])}_samples"
            tmp = np.max([np.max(sism_0_arr,axis=1),-np.min(sism_0_arr,axis = 1)], axis = 0).reshape(len(sism_0_arr),1)
            print(np.where(tmp==0))



def freq_filter(signal,sf,freqs,type_filter="bandpass", order_filter=2):
    """ 
    sf: sampling rate of the input waveform
    freqs: list of frequences (e.g. 2 for bandpass), or single float (e.g. for highpass)
        sf sampling frequence  """
    # type_filter: ‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’

    freqs=np.array(freqs)
    filt_b1,filt_a1=sc_sig.butter(order_filter,freqs/(sf/2),btype=type_filter)
    filtered_sig=sc_sig.filtfilt(filt_b1,filt_a1,sc_sig.detrend(signal))
    return filtered_sig

def sliding_window_view(arr, window_shape, steps):
    # -*- coding: utf-8 -*-
    """
    Created on 4/6/2021
    @author: Ryan meowklaski
    """

    """ Produce a view from a sliding, striding window over `arr`.
        The window is only placed in 'valid' positions - no overlapping
        over the boundary.
        Parameters
        ----------
        arr : numpy.ndarray, shape=(...,[x, (...), z])
            The array to slide the window over.
        window_shape : Sequence[int]
            The shape of the window to raster: [Wx, (...), Wz],
            determines the length of [x, (...), z]
        steps : Sequence[int]
            The step size used when applying the window
            along the [x, (...), z] directions: [Sx, (...), Sz]
        Returns
        -------
        view of `arr`, shape=([X, (...), Z], ..., [Wx, (...), Wz])
            Where X = (x - Wx) // Sx + 1
        Notes
        -----
        In general, given
          `out` = sliding_window_view(arr,
                                      window_shape=[Wx, (...), Wz],
                                      steps=[Sx, (...), Sz])
           out[ix, (...), iz] = arr[..., ix*Sx:ix*Sx+Wx,  (...), iz*Sz:iz*Sz+Wz]
         Examples
         --------
         >>> import numpy as np
         >>> x = np.arange(9).reshape(3,3)
         >>> x
         array([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
         >>> y = sliding_window_view(x, window_shape=(2, 2), steps=(1, 1))
         >>> y
         array([[[[0, 1],
                  [3, 4]],
                 [[1, 2],
                  [4, 5]]],
                [[[3, 4],
                  [6, 7]],
                 [[4, 5],
                  [7, 8]]]])
        >>> np.shares_memory(x, y)
         True
        # Performing a neural net style 2D conv (correlation)
        # placing a 4x4 filter with stride-1
        >>> data = np.random.rand(10, 3, 16, 16)  # (N, C, H, W)
        >>> filters = np.random.rand(5, 3, 4, 4)  # (F, C, Hf, Wf)
        >>> windowed_data = sliding_window_view(data,
        ...                                     window_shape=(4, 4),
        ...                                     steps=(1, 1))
        >>> conv_out = np.tensordot(filters,
        ...                         windowed_data,
        ...                         axes=[[1,2,3], [3,4,5]])
        # (F, H', W', N) -> (N, F, H', W')
        >>> conv_out = conv_out.transpose([3,0,1,2])
         """
    import numpy as np
    from numpy.lib.stride_tricks import as_strided
    in_shape = np.array(arr.shape[-len(steps):])  # [x, (...), z]
    window_shape = np.array(window_shape)  # [Wx, (...), Wz]
    steps = np.array(steps)  # [Sx, (...), Sz]
    nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

    # number of per-byte steps to take to fill window
    window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
    # number of per-byte steps to take to place window
    step_strides = tuple(window_strides[-len(steps):] * steps)
    # number of bytes to step to populate sliding window view
    strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

    outshape = tuple((in_shape - window_shape) // steps + 1)
    # outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
    outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
    return as_strided(arr, shape=outshape, strides=strides, writeable=False)

def get_hos(data, window_size, func):
    """
    @param data: waveform
    @param window_size: the moving window size of hos function
    @param func: function of hos
    @return: hos
    """

    # detrend the waveform
    data = scipy.signal.detrend(data)

    # get a sliding window view of given np array
    slid_view =  sliding_window_view(data, (window_size,), (1,))

    # apply the function of slid_view along axis 1
    return func(slid_view,axis=1)



"""Higher Order Statistics"""
def S_1(data, **kwargs):
    return np.mean(data,**kwargs)

def S_2(data,**kwargs):
    return np.std(data,ddof=1,**kwargs)

def S_3(data,**kwargs):
    return scipy.stats.skew(data,**kwargs)

def S_4(data,axis=1,**kwargs):
    return scipy.stats.kurtosis(data,axis=axis,**kwargs)

def S_6(data, axis=1):
    # can be slower
    return np.sum((data-np.mean(data,axis=axis)[:,None])**6,axis=axis)/(data.shape[1]-1)/np.std(data,ddof=1,axis=axis)**6-15

def get_onset_4(waveform,window_size=100, threshold=[0.1], statistics=S_6, origin_sample=0, sampling_rate=200):
    # Origin sample refers to "event orignin time". Constrain to be near the origin time!
    # Number of threshold -> arbitrary

    # get hos, here we use S_3, S_4, ...
    hos = get_hos(waveform, window_size, statistics)
    # smooth the S4
    hos = np.convolve(hos, np.ones(3)/3, mode='valid')
    # get first derivative
    diff = np.diff(hos)
    # narrow the search range to a region near the maximum
    pre_window = 200 * sampling_rate//200


    # Search near the max of HOS
    lower_bound = np.argmax(hos) - pre_window
    if lower_bound < 0:
        lower_bound = 0
    upper_bound = lower_bound + pre_window +  window_size

    onsets = []
    for i in range(len(threshold)):
        try:
            # find the onset larger than threshold[i] * maximum of diff
            onsets.append(np.where(diff[lower_bound:upper_bound] > threshold[i] * np.max(diff))[0][0] + lower_bound + window_size)
        except:
            # use trigger position when nothing found
            onsets.append(-100000)

    try:
        # find the onset corresponding to the maximum of diff
        onset_max = np.argmax(diff[lower_bound:upper_bound]) + lower_bound + window_size
    except:
        onset_max = -100000
    return onsets, diff, onset_max, lower_bound, hos

def cluster_agg_max_distance(picks, dmax=300):
    # agglomerative 1D hierarchical clustering with conditions on max intra-cluster distance

    # starting fromm all single elements representing separate clusters,
    # agglomerate Reciprocal Nearest Neighbour,
    #     if diameter new cluster < dmax, accept new cluster
    # I End the cycle when no new cluster is born

    # picks have to be a sorted list!
    pic_M = [[i] for i in picks]
    Z = linkage(pic_M,"complete")       # "compute" the clustering procedure. returns the "rappresentation of the dendrogram"
    crit = Z[:, 2]                      # distance between clusters at each step
    flat_clusters = fcluster(Z, t=dmax, criterion='monocrit', monocrit=crit) # "stops" the clustering procedure based on criteria inside crit (distance<=dmax)

    # sclust -> index of the starting point of each cluster (refered to the sorted list "picks")
    # eclust -> index of the ending point of each cluster (refered to the sorted list "picks")
    sclust=[0]
    eclust=[]
    for i in range(len(flat_clusters)):
        if i !=0:
            if flat_clusters[i-1] != flat_clusters[i]:
                sclust.append(i)
                eclust.append(i-1)
    eclust.append(len(flat_clusters)-1)
    return sclust, eclust   

def accept_cluster(startclust:list,endclust:list):
    # to be used in the case where two or more clusters are present!
    """
    startclust[i]: index of starting point of i-th cluster
    endtclust[i]: index of starting point of i-th cluster
    e.g. i have the picks [1,2,3,50,51,100], grouped as: [[1,2,3], [50,51], 100]. We have:
            startclust = [0,3,5]
            endclust = [2,4,5]
    """
    # Starting from startclust and endclust, the function returns the index of the "accepted" cluster.
    # Denoting with p1 and p2 the numbers of onsets present in the two most populated clusters, P the number of total clusters,
    # we accept the most popolous iif: 
        # 1) cluster maggiore comprende almeno metà dei pick (metà nel senso di //) o se ha 4 punti o di più
        # 2) p1 >= 3 
        # 3) p1 >= 2 * p2 

    if len(startclust) != len(endclust):
        raise Exception("Len startclust does not match endclust")
    
    if len(startclust) == 1:       # Accept if only 1 cluster is provided
        return 0
    
    index_ok = -1
    size = np.array(endclust) - np.array(startclust) + 1 # diff contains the sizes of the clusters
    ssort = np.sort(size)
    if (ssort[-1] > (endclust[-1]+1)//2 ) and (ssort[-1] >= 3) and (ssort[-1] >= 2*ssort[-2]): 
        #Accept most popolous cluster!
        index_ok = np.argmax(size)

    return index_ok

def semblance(u):
    # https://doi.org/10.1093/gji/ggu311 eq. (4)
    # u have to be:
    #   1) each row aligned for its own arrival time
    #   2) demeaned
    u = np.array(u)
    Num = np.sum(u, axis=0)**2
    Den = np.sum(u*u, axis=0)
    return simps(Num)/simps(Den)/len(u)

def semblance_normalized_tracess(u):
    # https://doi.org/10.1093/gji/ggu311 eq. (4)
    # u have to be:
    #   1) each row aligned for its own arrival time
    #   2) demeaned
    u = np.array(u)
    # u = u - np.mean(u, axis=1).reshape(len(u),1) (not necessary if demeaned before (it is better) )
    u = u / np.max([np.max(u,axis=1),-np.min(u,axis = 1)], axis = 0).reshape(len(u),1)
    Num = np.sum(u, axis=0)**2
    Den = np.sum(u*u, axis=0)
    return simps(Num)/simps(Den)/len(u)

def SNR2(Data, arrival, amp=4*200, source_sample=None, equalsize=False,freq=None):
    # Calc the signal to noise ratio
    if source_sample is None:
        source_sample = Data.shape[1]//2
    sig = []
    noise = []
    leng = Data.shape[1]
    # select signal window      [arrival - shift ; arrival + amp -shift ]
    # select noise window       [origin_time - amp; origin_time]
    shift = max(amp//10,20)
    for i in range(len(Data)):
        arrivo = arrival[i]
        if amp+1 < arrivo < leng-amp-1:
            if freq is None: 
                sig.append(Data[i,arrivo-shift:arrivo+amp-shift])
                noise.append(Data[i,source_sample-amp:source_sample])
            else:
                sig.append(freq_filter(Data[i,arrivo-shift:arrivo+amp-shift],200,freq,type_filter="highpass"))
                noise.append(freq_filter(Data[i,source_sample-amp:source_sample],200,freq,type_filter="highpass"))
        elif equalsize:             # to create fictious data, in order to insert snr in catalogue
            sig.append([0 for _ in range(amp)])
            noise.append([_ for _ in range(amp)])
    sig = np.array(sig)
    noise = np.array(noise)
    res = np.std(sig,axis=1)/np.std(noise,axis=1)
    print(res.shape)
    return res

def semblance_for_array(D,time, key, s_=50,s=50, ntraces=-1, normalize=True, filter_frequencies=None, semi_amp_filt = 500):
    # return the values of semblance (in a list) for each event at fixed array.
    """
    D:          dataset (complete)
    time:       arrival times (pd.dataframe)
    key:        key of the dataframe (to retrive info of arrivals)
    ntraces:    if > 1, calc semblance only if number of traces for array == ntraces
    """
    semblance_arr = []

    event_list = np.array([s[:12] for s in time["trace_name"]])
    event_uniq = list(set(event_list))
    event_uniq.sort()

    for ev in event_uniq:
        tmp = time[(event_list==ev)]                                # select a single event
        arr_list = np.array([s[16:18] for s in tmp["trace_name"]])  # select a single array for each event (tipical arr_list=["01", "01", "10"..] )
        arr_uniq = list(set(arr_list))
        arr_uniq.sort()
        for arr in arr_uniq:
            tmp_2 = tmp[(arr_list==arr)]
            if len(tmp_2) > 1:                                      # can't calculate semblance for 1 lonely trace
                if len(tmp_2)==ntraces or ntraces==-1 or (ntraces==10 and len(tmp_2)==11):
                    
                    if filter_frequencies is None:
                        u = [D.sismogramma[i][tmp_2[key][i]-s_:tmp_2[key][i]+s] - np.mean(D.sismogramma[i][tmp_2[key][i]-150:tmp_2[key][i]-10]) for j,i in enumerate(tmp_2.index)]
                    else:
                        try:
                            u = [freq_filter(D.sismogramma[i][tmp_2[key][i]-semi_amp_filt:tmp_2[key][i]+semi_amp_filt],200,filter_frequencies,type_filter="bandpass")[semi_amp_filt-s_:semi_amp_filt+s] for i in tmp_2.index]
                        except Exception as e: 
                            print(f"Error: {e},{tmp_2.index},{key}")
                            continue
                        u=np.array(u)
                        u=u-np.mean(u[:, 0:45], axis=1).reshape(len(u),1)
                    cond=True
                    for io in u:
                        if len(io) != s_+s:
                            cond = False

                    if  cond:
                        if normalize:
                            semblance_arr.append(semblance_normalized_tracess(u)- 1/len(u))
                        else:
                            semblance_arr.append(semblance(u)- 1/len(u))
                    else:
                        print(f"Length of traces is not ok!, {tmp_2.index},{key}")

    semblance_arr = np.array(semblance_arr)
    return semblance_arr
