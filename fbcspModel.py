from utils import *
import time
from sklearn.svm import LinearSVC, SVC
from get_data import get_data
from copy import deepcopy

class FBCSP_Model:
    def __init__(self ,
                 Path = '../datasets/BNCI_IV_2A/',
                 Fs = 250,
                 Svm_kernel = 'linear',
                 Svm_c = 0.05,
                 NO_channels = 22,
                 NO_subjects = 9,
                 NO_weights = 2,
                 NO_class = 4,
                 selection = True,
                 NO_selection = 4,
                 class_selection_base = True,
                 Output = 'csp-space',
                 Bands = [[4,8],[8,12],[12,16],[16,20],[20,24],[24,28]],
                 OneVersuseOne = True,
                ):
        '''
        filterbank csp model
        arguments :
            Path                 -- path to where data set is saved 'String'
            Fs                   -- sampling frequency 'Integer'
            Svm_kernel           -- svm kernel for classification 'String'
            Svm_c                -- svm cost parameter 'Float'
            NO_channels          -- number of channel 'Integer'
            NO_subjects          -- number of subject in dataset 'Integer'
            NO_weights           -- parameter M in csp algorithm 'Integer'
            NO_class             -- number of classess 'Integer'
            selection            -- whether or not feature selection will perform  'Boolean'
            NO_selection         -- number of feature to be selected after transforming into csp-space 'Integer'
            class_selection_base -- whether or not feature selection perform with regarding class 'Boolean'
            Output               -- what will be output of run_csp function / 'power': accuracy  'csp-space' : transformed signal
            Bands                -- Filter Bank bandwidth
            OneVersuseOne        -- whether or not perform one versuse one or one versuse all 'Boolean'
            
        '''
        self.data_path = Path
        self.svm_kernel = Svm_kernel #'sigmoid'#'linear' # 'sigmoid', 'rbf', 'poly'
        self.svm_c = Svm_c # 0.05 for linear, 20 for rbf
        self.fs = Fs # sampling frequency 
        self.NO_channels = NO_channels # number of EEG channels 
        self.NO_subjects = NO_subjects
        self.NO_weights  = NO_weights # Total number of CSP feature per band
        self.NO_selection = NO_selection # number of feature to be selected 
        self.NO_class    = NO_class
        self.class_selection_base = class_selection_base
        self.bw = deepcopy(Bands) # bandwidth of filtered signals 
        self.Output = Output
        self.OnevsOne = OneVersuseOne 
        self.selection = selection
        self.f_bands_nom = load_bands(self.bw,self.fs) # get normalized bands 

        self.NO_bands = len(self.f_bands_nom)
        #self.NO_time_windows = int(self.time_windows.size/2)
        if(self.OnevsOne):
            self.NO_features = 2 * self.NO_weights * self.NO_bands * ((self.NO_class * (self.NO_class-1))/2)
        else :
            self.NO_features = 2 * self.NO_weights * self.NO_bands * self.NO_class
        
        self.train_time = 0
        self.train_trials = 0
        self.eval_time = 0
        self.eval_trials = 0

    def run_csp(self):
        '''
        start to train filter bank csp model
        '''
        ########### Training ############
        start_train = time.time()
        #  Apply CSP to bands to get spatial filter 
        w = generate_projection(self.train_data,
                                self.train_label,
                                self.f_bands_nom,
                                self.NO_weights,
                                self.NO_class,
                                self.OnevsOne
                               )
        #  Extract feature after transforming signal with calculated filter in variable "w"
        feature_mat = extract_feature(self.train_data,w,self.f_bands_nom)
        
        #  check whether feature selection should be performed or not
        if(self.selection):
            if(self.class_selection_base):
                self.selected_features = select_feature_class(feature_mat,
                                                  self.train_label,
                                                  self.NO_weights,
                                                  self.NO_selection,
                                                  self.NO_class)
                
                print("Number of selected feature ",sum(len(x) for x in self.selected_features))
            else:
                self.selected_features = select_feature_all(feature_mat,
                                      self.train_label,
                                      self.NO_weights,
                                      self.NO_selection,
                                      self.NO_class)
                
                print("Number of selected feature ",len(self.selected_features))
            
        if(self.Output == 'power'):
            print("")
            print("extracting feature and classification")
            if(self.selection):
                # select selected feature if feature selection is performed earlier
                if(self.class_selection_base):
                    feature_mat = reduce_feature_class(feature_mat,
                                                 self.selected_features,
                                                 self.NO_weights)
                else:
                    feature_mat = reduce_feature_all(feature_mat,
                                                 self.selected_features,
                                                 self.NO_weights)
            
            else :
                # flattening signal if no feature selection is performed 
                feature_mat = feature_mat.reshape((feature_mat.shape[0],-1))
            
            
            # 3. Stage Train SVM Model 
            if self.svm_kernel == 'linear' : 
                clf = LinearSVC(C = self.svm_c,
                                intercept_scaling=1,
                                loss='hinge',
                                max_iter=1000,
                                multi_class='ovr',
                                penalty='l2',
                                random_state=1,
                                tol=0.00001)
            else:
                clf = SVC(self.svm_c,
                          self.svm_kernel, 
                          degree=10, 
                          gamma='auto',
                          coef0=0.0,
                          tol=0.001, 
                          cache_size=10000, 
                          max_iter=-1, 
                          decision_function_shape='ovr')
            clf.fit(feature_mat,self.train_label) 
            end_train = time.time()
            self.train_time += end_train-start_train
            self.train_trials += len(self.train_label)

            ################################# Evaluation ###################################################
            start_eval = time.time()
            eval_feature_mat = extract_feature(self.eval_data,w,self.f_bands_nom)
            if(self.selection):
                if(self.class_selection_base):
                    eval_feature_mat = reduce_feature_class(eval_feature_mat,
                                                      self.selected_features,
                                                      self.NO_weights)
                else :
                    eval_feature_mat = reduce_feature_all(eval_feature_mat,
                                                      self.selected_features,
                                                      self.NO_weights)
            else :
                eval_feature_mat = eval_feature_mat.reshape((eval_feature_mat.shape[0],-1))
            
            success_rate     = clf.score(eval_feature_mat,self.eval_label)

            end_eval = time.time()

            print("Time for one Evaluation " + str((end_eval-start_eval)/len(self.eval_label)) )

            self.eval_time += end_eval-start_eval
            self.eval_trials += len(self.eval_label)
            
            return success_rate
        
        else :
            # set output to be csp-space if "self.Output" is not power
            if(self.selection):
                # if selection is performed earlier
                # how feature selection is performed ? with regards or concatenation
                if(self.class_selection_base):
                    train_transformed = transform_class(self.train_data,
                                                  w,
                                                  self.f_bands_nom,
                                                  self.selected_features,
                                                  self.NO_selection,
                                                  self.NO_weights)

                    eval_transformed = transform_class(self.eval_data,
                                                 w,
                                                 self.f_bands_nom,
                                                 self.selected_features,
                                                 self.NO_selection,
                                                 self.NO_weights)
                else :
                    train_transformed = transform_all(self.train_data,
                                                  w,
                                                  self.f_bands_nom,
                                                  self.selected_features,
                                                  self.NO_selection,
                                                  self.NO_weights)

                    eval_transformed = transform_all(self.eval_data,
                                                 w,
                                                 self.f_bands_nom,
                                                 self.selected_features,
                                                 self.NO_selection,
                                                 self.NO_weights)
            else :
                # if no feature selection is performed
                train_transformed = transform(self.train_data,
                                              w,
                                              self.f_bands_nom)
                
                eval_transformed = transform(self.eval_data,
                                             w,
                                             self.f_bands_nom)
                
            return [train_transformed,self.train_label,eval_transformed,self.eval_label]


    def load_data(self):
        '''
        load data from "self.path" variable
        '''
        self.train_data,self.train_label = get_data(self.subject,True,self.data_path)
        #self.train_data  = self.train_data[0:20]
        #self.train_label = self.train_label[0:20]
        self.eval_data,self.eval_label = get_data(self.subject,False,self.data_path)
        
        
        
        
        
        