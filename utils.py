import math
import numpy as np
import tensorflow as tf
from scipy import signal
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import mutual_info_classif

'''
from mutual_information import mutual_information 
        https://gist.github.com/elsonidoq/4230222
        doesn't make any deffrence in output    
'''
import math
import time
import pyriemann.utils.mean as rie_mean
from scipy.signal import hilbert
from scipy import linalg



def gevd(x1,x2,no_pairs):
    '''
    Solve generalized eigenvalue decomposition
    Keyword arguments:
        x1 -- numpy array of size [NO_channels, NO_samples]
        x2 -- numpy array of size [NO_channels, NO_samples]
        no_pairs -- number of pairs of eigenvectors to be returned 
    Return:
        numpy array of 2*No_pairs eigenvectors 
    '''
    ev,vr= linalg.eig(x1,x2,right=True) 
    evAbs = np.abs(ev)
    sort_indices = np.argsort(evAbs)
    chosen_indices = np.zeros(2*no_pairs).astype(int)
    chosen_indices[0:no_pairs] = sort_indices[0:no_pairs]
    chosen_indices[no_pairs:2*no_pairs] = sort_indices[-no_pairs:]
    
    w = vr[:,chosen_indices] # ignore nan entries 
    return w

def csp_one_one(cov_matrix,NO_csp ,NO_pairs):
    '''
    calculate spatial filter for class (1,2) (1,3) (1,4) (2,3) (2,4) (3,4)
    Keyword arguments:
    cov_matrix -- numpy array of size [N_classes , NO_channels, NO_channels]
    NO_csp -- number of spatial filters
    Return:	spatial filter numpy array of size [NO_channels, NO_csp] 
    '''
    N, _ = cov_matrix[0].shape # N is number of channel
    w = np.zeros((N,NO_csp))
    kk = 0 # internal counter
    for cc1 in range(0,4):
        for cc2 in range(cc1+1,4):
            #generalized eigen value decompositon between classess
            w[:,NO_pairs*2*(kk):NO_pairs*2*(kk+1)] = gevd(cov_matrix[cc1], cov_matrix[cc2],NO_pairs)
            kk +=1
    return w

def csp_one_all(cov_matrix,NO_csp ,NO_pairs):
    '''
    calculate spatial filter for class (1 vs other ) (2 vs other ) (3 vs other ) (4 vs other ) 
    Keyword arguments:
    cov_matrix -- numpy array of size [N_classes ,NO_channels, NO_channels]
    NO_csp -- number of spatial filters
    Return:	spatial filter numpy array of size [ NO_channels ,NO_csp] 
    '''
    N_class , N, _ = cov_matrix.shape # N is number of channel
    
    w = np.zeros((N,NO_csp))
    kk = 0 # internal counter
    for classCov in range(0,4):
        #covariance average of other
        covAvg = rie_mean.mean_covariance(cov_matrix[ np.arange(0,N_class) != classCov,:,:], metric = 'euclid')
        w[:,NO_pairs*2*(kk):NO_pairs*2*(kk+1)] = gevd(cov_matrix[classCov], covAvg ,NO_pairs)
        kk +=1
    return w

def generate_projection(data,
                        class_vec,
                        f_bands_nom,
                        NO_weights,
                        NO_class,
                        OnevsOne): 
    '''
    generate spatial filters for every frequancy band and return weight matrix
    
    Keyword arguments:
    data         -- numpy array of size [NO_trials,channels,time_samples]
    class_vec    -- containing the class labels, numpy array of size [NO_trials]
    NO_weights   -- number of weights ,
    NO_class     -- number of classes,
    f_bands_nom  -- numpy array [[start_freq1,end_freq1],...,[start_freqN,end_freqN]]
    time_windows -- numpy array [[start_time1,end_time1],...,[start_timeN,end_timeN]] 

    Return: spatial filter numpy array of size [NO_timewindows,NO_freqbands,22,NO_csp] 
    '''
    NO_bands = len(f_bands_nom)
    NO_channels = len(data[0,:,0])
    NO_trials = class_vec.size
    if(OnevsOne):
        NO_csp = np.int(((NO_class * (NO_class-1))/2) * NO_weights * 2)
    else :
        NO_csp = np.int(NO_class * NO_weights * 2)
    # Initialize spatial filter: 
    w = np.zeros((NO_bands,NO_channels,NO_csp))
    # iterate through all time windows 
    
    # get start and end point of current time window 
    # *must get value in arguments
    t_start = int(2.5 * 250)
    t_end = int(6 * 250)

    # iterate through all frequency bandwids 
    print("Calculate filter for band : ")
    for subband in range(0,NO_bands): 
        print(subband,",", end=" ")
        cov      = np.zeros((4, NO_trials, NO_channels, NO_channels)) # sum of covariance depending on the class
        cov_avg  = np.zeros((4, NO_channels, NO_channels))
        cov_cntr = np.zeros(4).astype(int) # counter of class occurence 

        #go thrugh all trials and estimate covariance matrix of every class 
        for trial in range(0,NO_trials):
            #frequency band of every channel
            data_filter = bandpass_filter(data[trial,:,t_start:t_end],f_bands_nom[subband])
            # must calculae indecies of class label for automation
            cur_class_idx = int(class_vec[trial]-1)

            # caclulate current covariance matrix 
            cov[cur_class_idx,cov_cntr[cur_class_idx],:,:] = np.dot(data_filter,np.transpose(data_filter))

            # update covariance matrix and class counter 
            cov_cntr[cur_class_idx] += 1

        # calculate average of covariance matrix 
        for clas in range(0,4):
            cov_avg[clas,:,:] = rie_mean.mean_covariance(cov[clas,:cov_cntr[clas],:,:], metric = 'euclid')
        if(OnevsOne):
            w[subband,:,:] = csp_one_one(cov_avg, NO_csp, NO_weights)
        else :
            w[subband,:,:] = csp_one_all(cov_avg, NO_csp, NO_weights)
    return w


def extract_feature(data,w,f_bands_nom):
    '''
    calculate log variance features using the precalculated spatial filters
    
    Keyword arguments:
    data         -- numpy array of size [NO_trials,channels,time_samples]
    w            -- spatial filters, numpy array of size [NO_timewindows,NO_freqbands,22,NO_csp]
    f_bands_nom  -- numpy array [[start_freq1,end_freq1],...,[start_freqN,end_freqN]]
    time_windows -- numpy array [[start_time1,end_time1],...,[start_timeN,end_timeN]] 

    Return: features, numpy array of size [NO_trials,(NO_csp*NO_bands*NO_time_windows)]
    
    '''
    NO_csp = len(w[0,0,:])
    NO_bands = len(f_bands_nom)
    NO_trials = len(data[:,0,0])
    NO_features = NO_csp * NO_bands
    feature_mat = np.zeros(( NO_trials, NO_bands, NO_csp))
    
    # initialize feature vector
    feat = np.zeros((NO_bands,NO_csp))

    # go through all trials 
    t_start = int(2.5 * 250)
    t_end = int(6 * 250)
    
    for trial in range(0,NO_trials):
        for subband in range(0,NO_bands):
            #Apply spatial Filter to data 
            cur_data_s = np.dot(np.transpose(w[subband]),data[trial,:,t_start:t_end])

            #frequency filtering  
            cur_data_f_s = bandpass_filter(cur_data_s,f_bands_nom[subband])

            # calculate variance of all channels 
            feat[subband] = np.var(cur_data_f_s,axis=1)
        # calculate log10 of normalized feature vector 

        for subband in range(0,NO_bands):
            feat[subband] = np.log10(feat[subband]/np.sum(feat[subband]))
            #feat[subband] = (feat[subband]/np.sum(feat[subband]))

         # store feature in list
        feature_mat[trial,:,:] = feat
    return feature_mat
    #return np.reshape(feature_mat,(NO_trials,-1))

def select_feature_class(feature_mat,label,N_pair,N_selection,N_class):
    '''
    find best index with mutual information based feature selection regarding each class
    without concatenation
    
    Keyword arguments:
    feature_mat   -- numpy array of size [NO_trials,Bands,classes * 2 * pair]
    N_selection   -- number of channel to select
    N_pair        -- number of pair
    N_class       -- number of classes
    
    Return: selected feature for each class , array of size [Number of Classes,(Selected index)] 
    '''
    channels = feature_mat.shape[1] * feature_mat.shape[2]
    feature_mat =  np.swapaxes(feature_mat,1,2)
    
    bins = np.arange(0,channels+1,2*N_pair)

    class_selected = [[] for _ in range(N_class)]
    trans = {}

    for i , j in zip(range(2*N_pair) , reversed(range(2*N_pair))):
        trans[i] = j
        
    for i_class in range(N_class):
        data = feature_mat[:,(i_class * 2 * N_pair) : ((i_class + 1) * 2 * N_pair),:]
        data = np.reshape(data,(data.shape[0],-1))
        c_label = label == i_class+1
        Mi = mutual_info_classif(data, c_label.ravel() ,discrete_features = False,n_neighbors = 50)
        selected = np.argsort(-Mi)
        binSelected = np.digitize(selected,bins)
        finalSelected = set()
        counter = 0
        i_selected = 0
        while(counter < N_selection and i_selected < binSelected.shape[0] ):
            if(selected[i_selected] not in finalSelected):
                finalSelected.add(selected[i_selected])
                pair = trans[selected[i_selected] - bins[binSelected[i_selected]-1]] + bins[binSelected[i_selected]-1]
                finalSelected.add(pair)
                counter += 1
            i_selected +=1

        class_selected[i_class] = list(finalSelected)
    return class_selected

def select_feature_all(feature_mat,label,N_pair,N_selection,N_class):
    '''
    find best index with mutual information based feature selection after concatenating all of
    the feature extracted from bands
    
    Keyword arguments:
    feature_mat  -- numpy array of size [NO_trials,Bands,classes * 2 * pair]
    N_selection  -- number of channel to select
    N_pair       -- number of pair
    N_class      -- number of classes
    
    Return: selected feature for each class,array of size [Number of Classes,(Selected index)] 
    
    '''
    channels = feature_mat.shape[1] * feature_mat.shape[2]
    feature_mat =  np.swapaxes(feature_mat,1,2)
    
    bins = np.arange(0,channels+1,2*N_pair)

    trans = {}
    for i , j in zip(range(2*N_pair) , reversed(range(2*N_pair))):
        trans[i] = j
        
    #data = feature_mat[:,(i_class * 2 * N_pair) : ((i_class + 1) * 2 * N_pair),:]
    
    data = np.reshape(feature_mat,(feature_mat.shape[0],-1))
    Mi = mutual_info_classif(data,label.ravel() ,discrete_features = False,n_neighbors = 50)
    
    selected = np.argsort(-Mi)
    print(selected)
    binSelected = np.digitize(selected,bins)
    finalSelected = set()
    
    counter = 0
    i_selected = 0
    
    while(counter < N_selection and i_selected < binSelected.shape[0] ):
        if(selected[i_selected] not in finalSelected):
            finalSelected.add(selected[i_selected])
            pair = trans[selected[i_selected] - bins[binSelected[i_selected]-1]] + bins[binSelected[i_selected]-1]
            finalSelected.add(pair)
            counter += 1
        i_selected +=1

    return list(finalSelected)

def reduce_feature_class(feature_mat,class_selected,N_pair):
    '''
    generate new data set based on feature selected regarding class
    
    Keyword arguments:
    feature_mat     -- numpy array of size [NO_trials,Bands,classes * 2 * pair]
    class_selected  -- list of size [ N_class , Selected Index]
    N_pair          -- number of pair
    
    Return: new data
    ''' 
    feature_mat =  np.swapaxes(feature_mat,1,2)
    new_feature_mat = np.zeros((feature_mat.shape[0],0))
    for i_class in range(len(class_selected)):
        data = feature_mat[:,(i_class * 2 * N_pair) : ((i_class + 1) * 2 * N_pair),:]
        data = np.reshape(data,(data.shape[0],-1))
        new_feature_mat = np.hstack((new_feature_mat,data[:,class_selected[i_class]]))
    return new_feature_mat

def reduce_feature_all(feature_mat,selected,N_pair):
    '''
    generate new data set based on feature selected after concatenation
    
    Keyword arguments:
    feature_mat     -- numpy array of size [NO_trials,Bands,classes * 2 * pair]
    selected        -- list of size [ Selected Index]
    N_pair          -- number of pair
    
    Return: new data
    ''' 
    feature_mat = np.swapaxes(feature_mat,1,2)
    feature_mat = np.reshape(feature_mat,(feature_mat.shape[0],-1))
    feature_mat = feature_mat[:,selected]
    return feature_mat

def transform(data,w,f_bands_nom):
    '''
    tramsform input data into csp space when no feature selection is performed
    
    Keyword arguments:
    data         -- numpy array of size [NO_trials,channels,time_samples]
    w            -- spatial filters, numpy array of size [NO_timewindows,NO_freqbands,22,NO_csp]
    f_bands_nom  -- numpy array [[start_freq1,end_freq1],...,[start_freqN,end_freqN]]

    Return: features, numpy array of size [NO_trials,(NO_csp*NO_bands)] 
    '''
    t_start = int(2.5 * 250)
    t_end = int(6 * 250)
    NO_csp = len(w[0,0,:])
    NO_bands = len(f_bands_nom)
    NO_trials = len(data[:,0,0])
    NO_features = NO_csp*NO_bands
    feature_mat = np.zeros(( NO_trials, NO_bands, NO_csp,t_end - t_start))
    
    # initialize feature vector
    feat = np.zeros(( NO_bands, NO_csp, t_end-t_start))

    # go through all trials 
    t_start = int(2.5 * 250)
    t_end = int(6 * 250)
    
    for trial in range(0,NO_trials):
        for subband in range(0,NO_bands):
            #Apply spatial Filter to data 
            cur_data_s = np.dot(np.transpose(w[subband]),data[trial,:,t_start:t_end])

            #frequency filtering  
            feat[subband] = bandpass_filter(cur_data_s,f_bands_nom[subband])

         # store feature in list
        feature_mat[trial,:,:,:] = feat
    feature_mat = np.reshape(feature_mat,(feature_mat.shape[0],-1,feature_mat.shape[-1]))
    return feature_mat
        
def transform_class(data,w,f_bands_nom,class_selected,N_selection,N_pair):
    '''
    tramsform input data into csp space when feature selection is performed regarding each class
    
    Keyword arguments:
    data         -- numpy array of size [NO_trials,channels,time_samples]
    w            -- spatial filters, numpy array of size [NO_timewindows,NO_freqbands,22,NO_csp]
    f_bands_nom  -- numpy array [[start_freq1,end_freq1],...,[start_freqN,end_freqN]]

    Return: features, numpy array of size [NO_trials,(sum of selected index *NO_bands)] 
    '''
    t_start = int(2.5 * 250)
    t_end = int(6 * 250)
    NO_csp = len(w[0,0,:])
    NO_bands = len(f_bands_nom)
    NO_trials = len(data[:,0,0])
    NO_features = NO_csp*NO_bands
    feature_mat = np.zeros(( NO_trials, NO_bands, NO_csp,t_end - t_start))
    
    # initialize feature vector
    feat = np.zeros(( NO_bands, NO_csp, t_end-t_start))

    # go through all trials 
    t_start = int(2.5 * 250)
    t_end = int(6 * 250)
    
    for trial in range(0,NO_trials):
        for subband in range(0,NO_bands):
            #Apply spatial Filter to data 
            cur_data_s = np.dot(np.transpose(w[subband]),data[trial,:,t_start:t_end])

            #frequency filtering  
            feat[subband] = bandpass_filter(cur_data_s,f_bands_nom[subband])

         # store feature in list
        feature_mat[trial,:,:,:] = feat
    
    N_class = len(class_selected)
    feature_mat =  np.swapaxes(feature_mat,1,2)
    new_feature_mat = np.zeros((feature_mat.shape[0],N_class * 2 * N_selection,t_end - t_start))
    
    for i_class in range(N_class):
        data = feature_mat[:,(i_class * 2 * N_pair) : ((i_class + 1) * 2 * N_pair),:,:]
        data = np.reshape(data,(data.shape[0],data.shape[1] * data.shape[2] , data.shape[3]))
        new_feature_mat[:,((i_class)*2 * N_selection) : ((i_class + 1)*2 * N_selection) ,:] = data[:,class_selected[i_class],:]

    return new_feature_mat

def transform_all(data,w,f_bands_nom,selected,N_selection,N_pair):
    '''
    tramsform input data into csp space when feature selection is performed after concatenation of feature
    
    Keyword arguments:
    data         -- numpy array of size [NO_trials,channels,time_samples]
    w            -- spatial filters, numpy array of size [NO_timewindows,NO_freqbands,22,NO_csp]
    f_bands_nom  -- numpy array [[start_freq1,end_freq1],...,[start_freqN,end_freqN]]

    Return: features, numpy array of size [NO_trials,(sum of selected index *NO_bands)] 
    '''
    t_start = int(2.5 * 250)
    t_end = int(6 * 250)
    NO_csp = len(w[0,0,:])
    NO_bands = len(f_bands_nom)
    NO_trials = len(data[:,0,0])
    NO_features = NO_csp*NO_bands
    feature_mat = np.zeros(( NO_trials, NO_bands, NO_csp,t_end - t_start))
    
    # initialize feature vector
    feat = np.zeros(( NO_bands, NO_csp, t_end-t_start))

    # go through all trials 
    t_start = int(2.5 * 250)
    t_end = int(6 * 250)
    
    for trial in range(0,NO_trials):
        for subband in range(0,NO_bands):
            #Apply spatial Filter to data 
            cur_data_s = np.dot(np.transpose(w[subband]),data[trial,:,t_start:t_end])

            #frequency filtering  
            feat[subband] = bandpass_filter(cur_data_s,f_bands_nom[subband])

         # store feature in list
        feature_mat[trial,:,:,:] = feat
    
    
    
    feature_mat = np.swapaxes(feature_mat,1,2)
    feature_mat = np.reshape(feature_mat,(feature_mat.shape[0],-1,feature_mat.shape[-1]))
    feature_mat = feature_mat[:,selected,:]
    
    return feature_mat

def bandpass_filter(signal_in,f_band_nom):
    '''
    Filter N channels with cheby type 2 filter of order 4

    Keyword arguments:
    signal_in  -- numpy array of size [NO_channels, NO_samples]
    f_band_nom -- normalized frequency band [freq_start, freq_end]

    Return: filtered signal 
    '''
    NO_channels ,NO_samples = signal_in.shape
    sig_filt = np.zeros((NO_channels ,NO_samples))
    b, a = signal.cheby2(4, 40, f_band_nom, 'bandpass')
    for channel in range(0,NO_channels):
        # used filtfile for preventing phase delay
        sig_filt[channel] = signal.filtfilt(b,a,signal_in[channel,:])

    return sig_filt

def load_bands(Bands,Fs):
    '''
    Normalizng the Bandwidth
    Keyword arguments:
    bandwith -- numpy array containing bandwiths ex. [2,4,8,16,32]
    f_s      -- sampling frequency

    Return: numpy array of normalized frequency bands
    '''
    for i_bands in range(len(Bands)):
        Bands[i_bands] = [ float(Bands[i_bands][0])/(Fs/2) , float(Bands[i_bands][1])/(Fs/2) ]

    return Bands


def hilbert_transform(signal):
    '''
    perform hilbert transform and return envelope of signal
    keyword arguments:
        signal -- signal in shape of (N_channel , Time_samples )
    Return :
        envelope of input signal (N_channel , Time_samples)
    '''
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope
