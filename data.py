import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.stats import beta, bernoulli


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