import pandas as pd
import numpy as np
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from statsmodels.stats.outliers_influence import variance_inflation_factor


class MISTRImputer:
    def __init__(self, M=50, K=5, L=3, nmin=6, SupLogRank=1, tao=25,
                 A=200, discrete_sampling=True, vif_thresh=10,
                 h=None, random_state=None, verbose=True
                 ):
        '''
        RIST related parameters:
            M: number of trees in each fold, default is 50
            K: number of covariate considered per spilt, usually sqrt(d) or d/3, d is the number of covariates
            L: number of folds, 1-5 are recommended.
            nmin: minimum number of observed data in each node, default is 6.
            tao: lengh of study. Must be larger than all survival times. As long as it is 
                 larger than all survival times, the value of it will not make any difference.
            SupLogRank: "1" is default, using log-rank test to find best split. 
                        "2" is using sup log-rank test to find best split. This could be time consuming.
                        "0" is using t-test to compare two groups, not recommended.

        MISTR related parameters:
            A: number of imputations generated per censored cases
            discrete_sampling: if True, the sampled time points are discrete (taken from the observed time points)
                               otherwise, continuous (uniformly sampled from the time interval)
            h: max horizon
        '''
        self.M = M
        self.K = K
        self.L = L
        self.nmin = nmin
        self.SupLogRank = SupLogRank
        self.tao = tao
        self.A = A
        self.discrete_sampling = discrete_sampling
        self.vif_thresh = vif_thresh
        self.h = h
        self.random_state = random_state
        self.verbose = verbose

    def fit_transform(self, X_train, W_train, Y_train, 
                            X_test, W_test, Y_test):
        '''
        X_train, W_train, Y_train: type pd.DataFrame
        Y_train: (event indicator, time)
        '''

        df_train = self._prepare_df_for_RIST(X_train, W_train, Y_train, is_train=True)
        df_test = self._prepare_df_for_RIST(X_test, W_test, Y_test, is_train=False)

        # fit the RIST model
        df_surv_train, df_surv_test = self.run_RIST(df_train, df_test)

        # draw A samples only for the censored rows
        mask_train = df_train['censor'].values    # True = uncensored
        times_train = df_train['obtime'].values
        df_train_imputed = self._draw_samples(df_surv_train,
                                              uncensor_mask=mask_train,
                                              obtimes=times_train)

        mask_test  = df_test['censor'].values
        times_test = df_test['obtime'].values
        df_test_imputed  = self._draw_samples(df_surv_test,
                                              uncensor_mask=mask_test,
                                              obtimes=times_test)
        
        return df_train_imputed, df_test_imputed
        

    def run_RIST(self, df_train, df_test):
        
        # activate the pandas<->R DataFrame converter
        pandas2ri.activate()
        
        # source your RIST functions
        _ = robjects.r.source('../models_causal_impute_meta/RISTfunctions.r')
        robjects.r(f'K <- {self.K}')
        # grab the R function
        Muti_ERT_fit     = robjects.globalenv['Muti_ERT_fit']
        Muti_ERT_Predict = robjects.globalenv['Muti_ERT_Predict']
        

        # convert pandas DataFrame to an R data.frame
        r_df_train = pandas2ri.py2rpy(df_train)
        r_df_test = pandas2ri.py2rpy(df_test)
        # fit
        fit = Muti_ERT_fit(r_df_train, M=self.M, K=self.K, L=self.L, 
                           nmin=self.nmin, SupLogRank=self.SupLogRank,
                           tao=self.tao, impute="random")
        
        # predict using the last fold
        R_forest = fit.rx2('Forest_seq')[self.L-1]
        R_survmat = fit.rx2('SurvMat_seq')[self.L-1]
        time_int  = fit.rx2('time_intrest')
        pred_train = Muti_ERT_Predict(r_df_train, R_forest, R_survmat, time_int)
        pred_test = Muti_ERT_Predict(r_df_test, R_forest, R_survmat, time_int)

        # pull out the raw R vectors/matrices
        r_surv_train   = pred_train.rx2('Surv_predict')
        r_times_train  = pred_train.rx2('time_intrest')
        # force into NumPy arrays
        surv_mat_train = np.array(r_surv_train)
        time_grid_train = list(r_times_train)
        # convert to pandas DataFrame
        colnames = [t for t in time_grid_train]
        df_surv_train  = pd.DataFrame(surv_mat_train, columns=colnames)

        r_surv_test = pred_test.rx2('Surv_predict')
        r_times_test = pred_test.rx2('time_intrest')
        # force into NumPy arrays
        surv_mat_test = np.array(r_surv_test)
        time_grid_test = list(r_times_test)
        # convert to pandas DataFrame
        colnames = [t for t in time_grid_test]
        df_surv_test  = pd.DataFrame(surv_mat_test, columns=colnames)
        
        return df_surv_train, df_surv_test

        
    def _prepare_df_for_RIST(self, X, W, Y, is_train=True):
        '''
        X: pd.DataFrame
        W: pd.DataFrame
        Y: pd.DataFrame, in the order of (time, event indicator)
        '''
        X = X
        W = W.reshape(-1, 1)
        Y = Y
        X = np.concatenate((X, W), axis=1) # use treatment indicator as a covariate
        # construct column names required by RIST function
        cols = [f'X{i+1}' for i in range(X.shape[1])]
        df = pd.DataFrame(data=X, columns=cols)
        # remove collinearity among X
        df = self._remove_high_vif_features(df, verbose=self.verbose, is_train=is_train)
        df['censor'] = Y[:, 1]
        df['obtime'] = Y[:, 0]
        df['censor'] = df['censor'].astype(bool)
        return df
    
    def _remove_high_vif_features(self, X, verbose=True, is_train=True):
        """
        Iteratively remove columns with VIF above threshold (during training),
        and apply the same column removals (during testing).
        
        Parameters:
            X       : DataFrame with numeric, non-missing data
            verbose : If True, print each removal step
            is_train: If True, compute and store columns to drop; else, reuse stored list
        
        Returns:
            DataFrame with reduced multicollinearity
        """
        X = X.select_dtypes(include=[np.number]).copy()

        if is_train:
            self.dropped_vif_cols = []  # initialize during training

            while True:
                vif = pd.Series(
                    [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                    index=X.columns
                )
                max_vif = vif.max()
                if max_vif > self.vif_thresh:
                    feature_to_drop = vif.idxmax()
                    if verbose:
                        print(f"\tDropping '{feature_to_drop}' with VIF={max_vif:.2f}")
                    self.dropped_vif_cols.append(feature_to_drop)
                    X = X.drop(columns=[feature_to_drop])
                else:
                    break
            
            if self.dropped_vif_cols:
                self.K = int(np.sqrt(X.shape[1]))
        else:
            # Drop the same columns from test data
            for col in getattr(self, 'dropped_vif_cols', []):
                if col in X.columns:
                    X = X.drop(columns=[col])

        return X

    
    def _draw_samples(self, df_surv, uncensor_mask, obtimes):
        '''
        Draw A imputed times *only* for those with uncensor_mask=False.
+       Uncensored (uncensor_mask=True) rows get their original obtime repeated.
        '''
        n, m = df_surv.shape
        S0 = np.ones((n, 1)) # add time zeros with survival probability 1
        S = df_surv.values
        S = np.concatenate((S0, S), axis=1)
        p = S[:, :-1] - S[:, 1:] 
        p = p / p.sum(axis=1, keepdims=True) # normalize
        time_grid = df_surv.columns.values
        # prepend zero to align bins:
        extended_grid = np.concatenate(([0.], time_grid))  # length m+1
        
       # prepare output array
        sampled_times = np.empty((n, self.A))
        for i in range(n):
            if uncensor_mask[i]:
                # uncensored: just repeat the observed time
                sampled_times[i, :] = obtimes[i]
            else:
                # censored: sample A times from the discrete dist
                draws = np.random.choice(m, size=self.A, p=p[i])
                if self.discrete_sampling:
                    sampled_times[i, :] = time_grid[draws]
                else:
                    # continuous uniform within the chosen bin
                    u = np.random.rand(self.A)
                    left  = extended_grid[draws]
                    right = extended_grid[draws + 1]
                    sampled_times[i, :] = left + u * (right - left)
        cols = [f"draw_{a}" for a in range(self.A)]
        return pd.DataFrame(sampled_times, columns=cols)        