import numpy as np
import pandas as pd
import os
import math
from scipy.stats import multivariate_normal
from scipy.stats import beta, bernoulli

class SyntheticDataGenerator:
    """
    Generate modularized synthetic data scenarios
    """
    def __init__(self, scenario='A', n_samples=5000, n_features=5,
                 random_state=None, dataset_name=None, 
                 RCT=True, treatment_proportion=0.5, unobserved=False, overlap=True,
                 informative_censoring=False, info_censor_baseline=1.0, info_censor_alpha=0.1):
        '''
        unobserved: bool
            only used if RCT == False
            If True, add unobserved confounders to the dataset, i.e. propensity score e(X, U).
            If RCT == False and unobserved == False, then propensity score is e(X).
        informative_censoring: bool
            If True, ignore the scenario's original C-generation and
            instead draw
                C_i ~ Exp(rate = info_censor_baseline + info_censor_alpha * T_i)
        info_censor_baseline: float
            lambda_0 in the rate
        info_censor_alpha: float
            alpha in the rate
        '''
        
        self.scenario = scenario
        assert self.scenario in ['A', 'B', 'C', 'D', 'E'], f"Scenario mapped must be A to E, got {self.scenario}"
        self.n = n_samples
        self.p = n_features
        self.random_state = random_state
        self.dataset_name = dataset_name or f"Scenario_{scenario}"
        self.rct = RCT
        self.treatment_proportion = treatment_proportion # only used if RCT=True
        self.overlap = overlap # only used if RCT=False, overlap assumption holding
        self.unobserved = unobserved
        self.informative_censoring = informative_censoring
        self.info_censor_baseline = info_censor_baseline
        self.info_censor_alpha = info_censor_alpha
        np.random.seed(self.random_state)
        self._meta = {
            'dataset_name': self.dataset_name,
            'scenario': self.scenario,
            'n_samples': self.n,
            'n_features': self.p,
            'RCT': self.rct,
            'unobserved': self.unobserved,
            'overlap': self.overlap,
            'informative_censoring': self.informative_censoring,
            'random_state': self.random_state,
        }

    def _simulate_cox(self, linpred, baseline, params):
        """
        Inverse-transform sampling from a Cox model:
          S(t)=exp(-H0(t) e^{linpred}),  H0 for exponential or Weibull.
        """
        U = np.random.uniform(size=len(linpred))
        if baseline == 'exponential':
            lam0 = params['lambda']
            return -np.log(U) / (lam0 * np.exp(linpred))
        elif baseline == 'weibull':
            lam0 = params['lambda']
            k = params['k']
            return lam0 * ((-np.log(U) / np.exp(linpred)) ** (1.0 / k))
        else:
            raise ValueError("Unsupported baseline for Cox")

    def _gen_X(self, size):
        df = pd.DataFrame(
            np.random.rand(size, self.p),
            columns=[f'X{i+1}' for i in range(self.p)]
        )
        # unobserved confounders
        U = np.random.rand(size, 2)
        U = 1 - U
        df['U1'], df['U2'] = U[:,0], U[:,1]
        return df

    def _add_treatment(self, df):
        df = df.copy()
        if self.rct:
            # Randomized treatment assignment
            p = self.treatment_proportion
            W = np.random.binomial(1, p, size=len(df))
            self._meta['propensity_type'] = 'RCT'
            self._meta['treatment_proportion'] = p,
        else:
            X = df[[c for c in df.columns if c.startswith('X')]].values
            if not self.unobserved:
                if self.overlap:
                    # e(X) via Beta PDF on X[:,0]
                    pdf_vals = beta.pdf(X[:, 0], 2, 4)
                    e = (1 + pdf_vals) / 4 # propensity score e(X)
                    self._meta['propensity_type'] = 'e(X)'
                else:
                    # overlap assumption not holding:
                    e = np.where(X[:, 0] > 0.8, 1, np.where(X[:, 0] < 0.2, 0, 0.5))
                    self._meta['propensity_type'] = 'e(X)_no_overlap'
            else:
                # Unobserved confounders: e(X,U) via Beta PDF on (0.3*X[:,0] + 0.7*U1)
                U = df[['U1', 'U2']].values
                lin = 0.3 * X[:, 0] + 0.7 * U[:, 0]
                pdf_vals = beta.pdf(lin, 2, 4)
                e = (1 + pdf_vals) / 4 # propensity score e(X)
                self._meta['propensity_type'] = 'e(X,U)'
            self._meta['propensity_params'] = {'beta_a': 2, 'beta_b': 4}
            W = bernoulli.rvs(e, size=X.shape[0])
        df['W'] = W
        return df

    def _simulate_T(self, df, W_forced=None):
        if W_forced is not None:
            df = df.copy()
            df['W'] = W_forced
        X = df[[c for c in df.columns if c.startswith('X')]].values
        W = df['W'].values
        s = self.scenario
        eps = np.random.normal(size=len(df)) # TODO: why some with eps, some without?

        U = df[['U1', 'U2']].values
        U_counfounder = (U[:, 0] - X[:, 1]) if self.unobserved else np.zeros(len(U))

        # Cox-based T for scenario A
        if s == 'A':
            # linpred = X1 + (-0.5+X2)*W
            linpred = X[:,0] + (-0.5 + X[:,1]) * W
            linpred +=  0.5 * U_counfounder
            self._meta['T_distribution'] = 'Cox'
            return self._simulate_cox(linpred, 'weibull', {'lambda':1.0, 'k':0.5})
        
        # AFT scenario B
        if s == 'B':
            lp = -1.85 -0.8*(X[:,0]<0.5) +0.7*np.sqrt(X[:,1]) +0.2*X[:,2]
            lp += (0.7 -0.4*(X[:,0]<0.5) -0.4*np.sqrt(X[:,1])) * W
            lp += 0.5 * U_counfounder
            self._meta['T_distribution'] = 'AFT'
            return np.exp(lp + eps)
        
        # Poisson-based
        if s in ['C', 'E']:
            lam = X[:,1]**2 + X[:,2] + 6 + 2*(np.sqrt(X[:,0]) - 0.3) * W
            lam += 0.5 * U_counfounder
            if s in [7,8]:
                lam += 1
            self._meta['T_distribution'] = 'Poisson'
            return np.random.poisson(lam)
        
        # AFT scenario D
        if s == 'D':
            lp = 0.3 -0.5*(X[:,0]<0.5) +0.5*np.sqrt(X[:,1]) +0.2*X[:,2]
            lp += (1 -0.8*(X[:,0]<0.5) -0.8*np.sqrt(X[:,1])) * W
            lp += 0.5 * U_counfounder
            self._meta['T_distribution'] = 'AFT'
            return np.exp(lp + eps)
        raise ValueError("Unsupported scenario for T (event time)")

    def _simulate_C(self, df):
        X = df[[c for c in df.columns if c.startswith('X')]].values
        W = df['W'].values
        T_true = df['T'].values
        if self.informative_censoring:
            self._meta['info_censor_baseline'] = self.info_censor_baseline
            self._meta['info_censor_alpha'] = self.info_censor_alpha
            # informative censoring: rate depends on T_true
            lam0  = self.info_censor_baseline
            alpha = self.info_censor_alpha
            rates = lam0 + alpha * T_true
            # Exponential(scale=1/rates)
            return np.random.exponential(scale=1.0/rates, size=len(df))
        else:
            s = self.scenario
            # Cox-based C for scenarios B & D
            if s in ('B', 'D'):
                if s == 1:
                    lpC = -1.75 -0.5*np.sqrt(X[:,1]) +0.2*X[:,2]
                    lpC += (1.15 +0.5*(X[:,0]<0.5) -0.3*np.sqrt(X[:,1])) * W
                else:  # s==9
                    lpC = -0.9 +2*np.sqrt(X[:,1]) +2*X[:,2]
                    lpC += (1.15 +0.5*(X[:,0]<0.5) -0.3*np.sqrt(X[:,1])) * W
                return self._simulate_cox(lpC, 'weibull', {'lambda':1.0, 'k':2.0})
            # Uniform scenario A
            if s == 'A':
                return np.random.uniform(0,3, size=len(df))
            # piecewise uniform C
            if s == 'C':
                u = np.random.rand(len(df)) # Uniform(0,1)
                # if s < 0.6, return inf
                return np.where(u < 0.6, np.inf, 1 + (X[:,3] < 0.5).astype(int))
            # Poisson E
            if s == 'E':
                lam = 3 + np.log1p(np.exp(2*X[:,1] + X[:,2]))
                return np.random.poisson(lam, size=len(df))
            raise ValueError("Unsupported scenario for C (censoring time)")

    def generate_datasets(self):
        def build_df(n):
            Xdf = self._gen_X(n)
            Xdf = self._add_treatment(Xdf)
            df = Xdf.copy()
            # true potential outcomes
            T0 = self._simulate_T(Xdf, W_forced=np.zeros(len(Xdf), dtype=int))
            T1 = self._simulate_T(Xdf, W_forced=np.ones(len(Xdf), dtype=int))
            df['T0'] = T0
            df['T1'] = T1
            # factual event time
            T_f = np.where(Xdf['W']==1, T1, T0)
            df['T'] = T_f
            # factual censoring time
            C  = self._simulate_C(df)
            obs = np.minimum(T_f, C)
            df['C'] = C
            # observed time and event indicator
            df['observed_time'] = obs
            df['event'] = (T_f <= C).astype(int)
            df['id'] = np.arange(len(df))
            df = df[['id', 'observed_time', 'event', 'W'] + [c for c in df.columns if c not in ['id', 'observed_time', 'event', 'W']]]
            return df
        
        df = build_df(self.n)

        self._meta['treatment_proportion'] = df['W'].mean()
        self._meta['censoring_rate'] = 1 - df['event'].mean()


        return {'data':df, 
                'metadata': self._meta}


def load_data(dataset_type='synthetic', data_dir='./data/synthetic/'):
    experiment_setups = {}
    if dataset_type == 'synthetic':
        idx_split_file_path = os.path.join(data_dir, 'idx_split.csv')
        experiment_repeat_setups = pd.read_csv(idx_split_file_path).set_index("idx")
        for causal_config in ["RCT-50",
                            "RCT-5",
                            "OBS-CPS",
                            "OBS-UConf",
                            "OBS-NoPos",
                            "OBS-CPS-IC",
                            "OBS-UConf-IC",
                            "OBS-NoPos-IC"]:
            data_path = os.path.join(data_dir, f'{causal_config}.h5')
            scenario_dict = {}
            for survival_scenario in ['A', 'B', 'C', 'D', 'E']:
                dataset_key = f'Scenario_{survival_scenario}/data'
                with pd.HDFStore(data_path, mode='r') as store:
                    df = store[dataset_key]
                    metadata = store.get_storer(dataset_key).attrs.metadata
                summary_characteristics = {
                    # rates
                    'censoring_rate': 1-df['event'].mean(),
                    'treatment_rate': df['W'].mean(),

                    # event times
                    'event_time_min': df['T'].min(),
                    'event_time_25pct': df['T'].quantile(0.25),
                    'event_time_median': df['T'].median(),
                    'event_time_75pct': df['T'].quantile(0.75),
                    'event_time_max': df['T'].max(),
                    'event_time_mean': df['T'].mean(),
                    'event_time_std': df['T'].std(),

                    # censoring times
                    'censoring_time_min': df['C'].min(),
                    'censoring_time_median': df['C'].median(),
                    'censoring_time_max': df['C'].max(),
                    'censoring_time_mean': df['C'].mean(),
                    'censoring_time_std': df['C'].std(),

                    # treatment effects
                    'ate': (df['T1']-df['T0']).mean(),
                    'cate_min': (df['T1']-df['T0']).min(),
                    'cate_median': (df['T1']-df['T0']).median(),
                    'cate_max': (df['T1']-df['T0']).max()
                    }
                result = {"dataset": df, 
                        "summary": summary_characteristics, 
                        "metadata": metadata}
                scenario_dict[f"Scenario_{survival_scenario}"] = result
            
            experiment_setups[causal_config] = scenario_dict
        
    else:
        raise NotImplementedError
    
    return experiment_setups, experiment_repeat_setups



def prepare_data_split(dataset_df, experiment_repeat_setups, 
                       num_repeats=10, 
                       dataset_type='synthetic',
                       cate_true_col=None,
                       train_size=5000,
                       val_size=2500,
                       test_size=2500):
    """
    Returns dictionary:
        split_results[repeat_id] = {
            "train": (X_train, W_train, Y_train, cate_train_true),
            "val":   (X_val,   W_val,   Y_val,   cate_val_true),
            "test":  (X_test,  W_test,  Y_test,  cate_test_true)
        }
    """
    # --- train/val/test size ---
    if all(0 <= x <= 1 for x in (train_size, val_size, test_size)):
        assert math.isclose(train_size + val_size + test_size, 1.0,
                            rel_tol=1e-9, abs_tol=1e-9), \
            f"Sizes must sum to 1. Got {train_size + val_size + test_size}"
        length = len(experiment_repeat_setups)
        train_size = int(length * train_size)
        val_size = int(length * val_size)
        test_size = length - train_size - val_size
    else:
        # assume absolute counts were passed in
        total = train_size + val_size + test_size
        assert total <= len(dataset_df), \
            f"Total {total} exceeds dataset size {len(dataset_df)}"
    
    # --- dataset-specific schema ---
    if dataset_type == 'synthetic':
        X_cols = [c for c in dataset_df.columns if c.startswith("X") and c[1:].isdigit()]
        W_col = 'W'
        y_cols = ['observed_time', 'event']
    elif dataset_type in ('actg', 'actgL'):
        X_bi_cols = ['gender', 'race', 'hemo', 'homo', 'drugs', 'str2', 'symptom']
        X_cont_cols = ['age', 'wtkg',  'karnof', 'cd40', 'cd80']
        U = ['z30']
        X_cols = X_bi_cols + X_cont_cols
        y_cols = ['observed_time_month', 'effect_non_censor']
        W_col = 'trt'
        cate_true_col = 'cate_base'
    elif dataset_type == 'mimicSyn':
        X_cols = ['Anion gap', 'Bicarbonate', 'Calcium total', 'Chloride', 'Creatinine',
                    'Glucose', 'Magnesium', 'Phosphate', 'Potassium', 'Sodium',
                    'Urea nitrogen', 'Hematocrit', 'Hemoglobin', 'MCH', 'MCHC', 'MCV',
                    'Platelet count', 'RDW', 'Red blood cells', 'White blood cells',
                    'Admission age', 'Sex:Male', 'Race:Black', 'Race:Hispanic',
                    'Race:Other', 'Race:White', 'Insurance:Medicare', 'Insurance:Other',
                    'Marital status:Married', 'Marital status:Single',
                    'Marital status:Widowed', 'Direct emergency:Yes', 'Night admission:Yes',
                    'Previous admission this month:Yes', 'Admissions number:2',
                    'Admissions number:3+']
        W_col = 'W'
        y_cols = ['observed_time', 'event']
        cate_true_col = 'true_cate'
    elif dataset_type == 'actgSyn':
        X_cols = ['age', 'wtkg', 'hemo', 'homo', 'drugs', 'karnof', 'oprior', 'z30', 'zprior', 'preanti', 
                'race', 'gender', 'str2', 'strat', 'symptom', 'treat', 'offtrt', 
                'cd40', 'cd420', 'cd496', 'r', 'cd80', 'cd820']
        W_col = 'W'
        y_cols = ['observed_time', 'event']
        cate_true_col = 'true_cate'
    elif dataset_type == 'twin':
        X_binary_cols = ['anemia', 'cardiac', 'lung', 'diabetes', 'herpes', 'hydra',
                        'hemo', 'chyper', 'phyper', 'eclamp', 'incervix', 'pre4000', 'preterm',
                        'renal', 'rh', 'uterine', 'othermr', 
                        'gestat', 'dmage', 'dmeduc', 'dmar', 'nprevist', 'adequacy']
        X_num_cols = ['dtotord', 'cigar', 'drink', 'wtgain']
        X_ohe_cols = ['pldel_2', 'pldel_3', 'pldel_4', 'pldel_5', 'resstatb_2', 'resstatb_3', 'resstatb_4', 
                    'mpcb_1', 'mpcb_2', 'mpcb_3', 'mpcb_4', 'mpcb_5', 'mpcb_6', 'mpcb_7', 'mpcb_8', 'mpcb_9']
        X_cols = X_binary_cols + X_num_cols + X_ohe_cols
        W_col = 'W'
        y_cols = ['observed_time', 'event']
        cate_true_col = 'true_cate'
    else:
        raise NotImplementedError(f"Dataset type {dataset_type} not implemented")

    # --- splitting loop ---
    split_results = {}
    for rand_idx in range(num_repeats):
        random_idx_vals = experiment_repeat_setups[f'random_idx{rand_idx}'].values
        train_ids = random_idx_vals[:train_size]
        val_ids   = random_idx_vals[train_size:train_size+val_size]
        test_ids  = random_idx_vals[train_size+val_size:train_size+val_size+test_size]

        def _subset(ids):
            df = dataset_df[dataset_df['id'].isin(ids)]
            X = df[X_cols].to_numpy()
            W = df[W_col].to_numpy()
            Y = df[y_cols].to_numpy()
            if cate_true_col is not None and cate_true_col in df.columns:
                cate_true = df[cate_true_col].to_numpy()
            else:
                cate_true = (df["T1"] - df["T0"]).to_numpy()
            return X, W, Y, cate_true
        
        split_results[rand_idx] = {
            "train": _subset(train_ids),
            "val":   _subset(val_ids),
            "test":  _subset(test_ids)
        }

    return split_results


class SyntheticDataGeneratorPlus:
    """
    Generate synthetic survival data scenarios (1-10) with Cox-based event and censoring times where appropriate.

    Each dataset includes:
      - train, test_random, test_quantiles
      - metadata dictionary

    All other scenarios remain as before.
    """
    def __init__(self, scenario=1, n_samples=5000, n_features=5,
                 random_state=None, dataset_name=None, 
                 RCT=True, treatment_proportion=0.5, unobserved=False, overlap=True,
                 informative_censoring=False, info_censor_baseline=1.0, info_censor_alpha=0.1):
        '''
        unobserved: bool
            only used if RCT == False
            If True, add unobserved confounders to the dataset, i.e. propensity score e(X, U).
            If RCT == False and unobserved == False, then propensity score is e(X).
        informative_censoring: bool
            If True, ignore the scenario's original C-generation and
            instead draw
                C_i ~ Exp(rate = info_censor_baseline + info_censor_alpha * T_i)
        info_censor_baseline: float
            lambda_0 in the rate
        info_censor_alpha: float
            alpha in the rate
        '''
        
        self.scenario_alphabetical = scenario
        self.scenario = self.map_scenario(self.scenario_alphabetical)
        assert 1 <= self.scenario <= 10, "Scenario mapped must be 1-10" # 1-10 refer to Meir et al. (2025)
        self.n = n_samples
        self.p = n_features
        self.random_state = random_state
        self.dataset_name = dataset_name or f"scenario_{scenario}"
        self.rct = RCT
        self.treatment_proportion = treatment_proportion # only used if RCT=True
        self.overlap = overlap # only used if RCT=False, overlap assumption holding
        self.unobserved = unobserved
        self.informative_censoring = informative_censoring
        self.info_censor_baseline = info_censor_baseline
        self.info_censor_alpha = info_censor_alpha
        np.random.seed(self.random_state)
        self._meta = {
            'dataset_name': self.dataset_name,
            'scenario': self.scenario_alphabetical,
            'n_samples': self.n,
            'n_features': self.p,
            'RCT': self.rct,
            'unobserved': self.unobserved,
            'overlap': self.overlap,
            'informative_censoring': self.informative_censoring,
            'random_state': self.random_state,
        }

    def map_scenario(self, scenario):
        if  scenario == 'A':
            return 2
        elif scenario == 'B':
            return 1
        elif scenario == 'C':
            return 5
        elif scenario == 'D':
            return 9
        elif scenario == 'E':
            return 8
        else:
            raise ValueError("Unsupported scenario")

    def _simulate_cox(self, linpred, baseline, params):
        """
        Inverse-transform sampling from a Cox model:
          S(t)=exp(-H0(t) e^{linpred}),  H0 for exponential or Weibull.
        """
        U = np.random.uniform(size=len(linpred))
        if baseline == 'exponential':
            lam0 = params['lambda']
            return -np.log(U) / (lam0 * np.exp(linpred))
        elif baseline == 'weibull':
            lam0 = params['lambda']
            k = params['k']
            return lam0 * ((-np.log(U) / np.exp(linpred)) ** (1.0 / k))
        else:
            raise ValueError("Unsupported baseline for Cox")

    def _gen_X(self, size):
        df = pd.DataFrame(
            np.random.rand(size, self.p),
            columns=[f'X{i+1}' for i in range(self.p)]
        )
        # unobserved confounders
        U = np.random.rand(size, 2)
        U = 1 - U
        df['U1'], df['U2'] = U[:,0], U[:,1]
        return df

    def _add_treatment(self, df):
        df = df.copy()
        if self.rct:
            # Randomized treatment assignment
            p = self.treatment_proportion
            W = np.random.binomial(1, p, size=len(df))
            self._meta['propensity_type'] = 'RCT'
            self._meta['treatment_proportion'] = p,
        else:
            X = df[[c for c in df.columns if c.startswith('X')]].values
            if not self.unobserved:
                if self.overlap:
                    # e(X) via Beta PDF on X[:,0]
                    pdf_vals = beta.pdf(X[:, 0], 2, 4)
                    e = (1 + pdf_vals) / 4 # propensity score e(X)
                    self._meta['propensity_type'] = 'e(X)'
                else:
                    # overlap assumption not holding:
                    e = np.where(X[:, 0] > 0.8, 1, np.where(X[:, 0] < 0.2, 0, 0.5))
                    self._meta['propensity_type'] = 'e(X)_no_overlap'
            else:
                # Unobserved confounders: e(X,U) via Beta PDF on (0.3*X[:,0] + 0.7*U1)
                U = df[['U1', 'U2']].values
                lin = 0.3 * X[:, 0] + 0.7 * U[:, 0]
                pdf_vals = beta.pdf(lin, 2, 4)
                e = (1 + pdf_vals) / 4 # propensity score e(X)
                self._meta['propensity_type'] = 'e(X,U)'
            self._meta['propensity_params'] = {'beta_a': 2, 'beta_b': 4}
            W = bernoulli.rvs(e, size=X.shape[0])
        df['W'] = W
        return df

    def _simulate_T(self, df, W_forced=None):
        if W_forced is not None:
            df = df.copy()
            df['W'] = W_forced
        X = df[[c for c in df.columns if c.startswith('X')]].values
        W = df['W'].values
        s = self.scenario
        eps = np.random.normal(size=len(df)) # TODO: why some with eps, some without?

        U = df[['U1', 'U2']].values
        U_counfounder = (U[:, 0] - X[:, 1]) if self.unobserved else np.zeros(len(U))

        # Cox-based T for scenarios 2 & 10
        if s in (2, 10):
            # linpred = X1 + (-0.5+X2)*W
            linpred = X[:,0] + (-0.5 + X[:,1]) * W
            linpred +=  0.5 * U_counfounder
            self._meta['T_distribution'] = 'Cox'
            return self._simulate_cox(linpred, 'weibull', {'lambda':1.0, 'k':0.5})
        # AFT scenario 1
        if s == 1:
            lp = -1.85 -0.8*(X[:,0]<0.5) +0.7*np.sqrt(X[:,1]) +0.2*X[:,2]
            lp += (0.7 -0.4*(X[:,0]<0.5) -0.4*np.sqrt(X[:,1])) * W
            lp += 0.5 * U_counfounder
            self._meta['T_distribution'] = 'AFT'
            return np.exp(lp + eps)
        # Poisson-based
        if s in (3,5,6,7,8):
            lam = X[:,1]**2 + X[:,2] + 6 + 2*(np.sqrt(X[:,0]) - 0.3) * W
            lam += 0.5 * U_counfounder
            if s in [7,8]:
                lam += 1
            self._meta['T_distribution'] = 'Poisson'
            return np.random.poisson(lam)
            
        # Poisson variant 4
        if s == 4:
            lam = X[:,1] + X[:,2] + np.maximum(0, X[:,0] - 0.3) * W + 1e-3 # small constant added for stability
            lam += 0.5 * U_counfounder
            self._meta['T_distribution'] = 'Poisson'
            return np.random.poisson(lam)
        # AFT scenario 9
        if s == 9:
            lp = 0.3 -0.5*(X[:,0]<0.5) +0.5*np.sqrt(X[:,1]) +0.2*X[:,2]
            lp += (1 -0.8*(X[:,0]<0.5) -0.8*np.sqrt(X[:,1])) * W
            lp += 0.5 * U_counfounder
            self._meta['T_distribution'] = 'AFT'
            return np.exp(lp + eps)
        raise ValueError("Unsupported scenario for T")

    def _simulate_C(self, df):
        X = df[[c for c in df.columns if c.startswith('X')]].values
        W = df['W'].values
        T_true = df['T'].values
        if self.informative_censoring:
            self._meta['info_censor_baseline'] = self.info_censor_baseline
            self._meta['info_censor_alpha'] = self.info_censor_alpha
            # informative censoring: rate depends on T_true
            lam0  = self.info_censor_baseline
            alpha = self.info_censor_alpha
            rates = lam0 + alpha * T_true
            # Exponential(scale=1/rates)
            return np.random.exponential(scale=1.0/rates, size=len(df))
        else:
            s = self.scenario
            # Cox-based C for scenarios 1 & 9
            if s in (1, 9):
                if s == 1:
                    lpC = -1.75 -0.5*np.sqrt(X[:,1]) +0.2*X[:,2]
                    lpC += (1.15 +0.5*(X[:,0]<0.5) -0.3*np.sqrt(X[:,1])) * W
                else:  # s==9
                    lpC = -0.9 +2*np.sqrt(X[:,1]) +2*X[:,2]
                    lpC += (1.15 +0.5*(X[:,0]<0.5) -0.3*np.sqrt(X[:,1])) * W
                return self._simulate_cox(lpC, 'weibull', {'lambda':1.0, 'k':2.0})
            # Uniform scenario 2
            if s == 2:
                return np.random.uniform(0,3, size=len(df))
            # Poisson scenarios 3,4
            if s in (3,4):
                lam = (12 if s == 3 else 1) + np.log1p(np.exp(X[:,2]))
                return np.random.poisson(lam)
            # piecewise uniform 5
            if s == 5:
                u = np.random.rand(len(df)) # Uniform(0,1)
                # if s < 0.6, return inf
                return np.where(u < 0.6, np.inf, 1 + (X[:,3] < 0.5).astype(int))
            # Poisson 6,7,8
            if s in (6,7,8):
                if s == 6: # switching 8 and 6 of the original
                    # lam = 3 + np.log1p(np.exp(2*X[:,1] + X[:,2]))
                    lam = 3
                elif s == 7: # unknown mechanism censoring
                    U = df[['U1','U2']].values
                    lam = 3 + 4*U[:,0] + 2*U[:,1]
                else: # switching 8 and 6 of the original
                    # lam = 3
                    lam = 3 + np.log1p(np.exp(2*X[:,1] + X[:,2]))
                return np.random.poisson(lam, size=len(df))
            # interval-censor 10
            if s == 10:
                u = np.random.rand(len(df))
                c1 = np.inf
                c2 = np.random.uniform(0, 0.05,size=len(df))
                return np.where(u<0.1, c1, c2)
            raise ValueError("Unsupported scenario for C")

    def generate_datasets(self):
        def build_df(n):
            Xdf = self._gen_X(n)
            Xdf = self._add_treatment(Xdf)
            df = Xdf.copy()
            # true potential outcomes
            T0 = self._simulate_T(Xdf, W_forced=np.zeros(len(Xdf), dtype=int))
            T1 = self._simulate_T(Xdf, W_forced=np.ones(len(Xdf), dtype=int))
            df['T0'] = T0
            df['T1'] = T1
            # factual event time
            T_f = np.where(Xdf['W']==1, T1, T0)
            df['T'] = T_f
            # factual censoring time
            C  = self._simulate_C(df)
            obs = np.minimum(T_f, C)
            df['C'] = C
            # observed time and event indicator
            df['observed_time'] = obs
            df['event'] = (T_f <= C).astype(int)
            df['id'] = np.arange(len(df))
            df = df[['id', 'observed_time', 'event', 'W'] + [c for c in df.columns if c not in ['id', 'observed_time', 'event', 'W']]]
            return df
        
        df = build_df(self.n)

        self._meta['treatment_proportion'] = df['W'].mean()
        self._meta['censoring_rate'] = 1 - df['event'].mean()


        return {'data':df, 
                'metadata': self._meta}