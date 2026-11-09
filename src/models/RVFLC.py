"""edRVFL (Shi et al. 2021) with per-head ridge readouts and conformal scoring.

  H_j = sigma([X, H_{j-1}] W_j + b_j)      (H_0 = sigma(X W_0 + b_0))
  head j: ridge readout on [X, H_j]         (j = 0 .. L-1)

L heads sit on L hidden layers; every head sees X plus its own layer's features.

head_mode:
  "edrvfl"  L heads on [X, H_j]                     (default)
  "concat"  one readout on [X, H_0, ..., H_{L-1}]   (dRVFL-style single readout)
  "linear"  one head on X
"""
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder


class HeadedDeepRVFL3:
    def __init__(self, n_layers=3, n_hidden=50, lamda=0.1, input_scaling=1.0, bias_scaling=1.0,
                 activation="sigmoid", device="cuda", seed=0, dtype="float64",
                 head_mode="edrvfl"):
        assert head_mode in ("edrvfl", "concat", "linear")
        self.L = n_layers
        self.n_hidden = n_hidden
        self.lam = lamda
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dt = torch.float64 if dtype == "float64" else torch.float32
        self.head_mode = head_mode
        self.seed = seed
        self.act = torch.sigmoid if activation == "sigmoid" else torch.relu

    # ------------------------------------------------------------------ helpers
    def _designs_new(self, Xn):
        """Return the list of head designs for new inputs, in head order."""
        if self.head_mode == "linear":
            return [Xn]
        H_prev = None
        Hs = []
        for j in range(self.L):
            A = Xn if H_prev is None else torch.cat([Xn, H_prev], dim=1)
            H_prev = self.act(A @ self.Wr[j] + self.br[j])
            Hs.append(H_prev)
        if self.head_mode == "concat":
            return [torch.cat([Xn] + Hs, dim=1)]
        return [torch.cat([Xn, H], dim=1) for H in Hs]      # edrvfl: [X, H_j]

    # ---------------------------------------------------------------------- fit
    def fit(self, X, y):
        self.gen = torch.Generator(device=self.device).manual_seed(self.seed)
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float64), device=self.device, dtype=self.dt)
        y1 = np.asarray(y)
        assert y1.ndim == 1 and y1.dtype.kind in "iu", "classification only"
        self.enc = OneHotEncoder(sparse_output=False)
        Y = self.enc.fit_transform(y1.reshape(-1, 1))
        self.classes_ = self.enc.categories_[0]
        self.Yk = torch.as_tensor(Y, dtype=self.dt, device=self.device)
        self.X_train = Xt

        Wr, br = [], []
        if self.head_mode != "linear":
            H_prev = None
            for j in range(self.L):
                A = Xt if H_prev is None else torch.cat([Xt, H_prev], dim=1)
                n_in = A.shape[1]
                W32 = (torch.rand(n_in, self.n_hidden, device=self.device,
                                  generator=self.gen) * 2 - 1) * self.input_scaling
                b32 = (torch.rand(self.n_hidden, device=self.device,
                                  generator=self.gen) * 2 - 1) * self.bias_scaling
                W, b = W32.to(self.dt), b32.to(self.dt)
                Wr.append(W); br.append(b)
                H_prev = self.act(A @ W + b)
        self.Wr, self.br = Wr, br

        designs = self._designs_new(Xt)
        self.n_heads = len(designs)
        self.w = torch.full((self.n_heads,), 1.0 / self.n_heads, dtype=self.dt, device=self.device)

        Y_t = self.Yk
        self.heads = []
        for A in designs:
            G = A.T @ A + self.lam * torch.eye(A.shape[1], device=self.device, dtype=self.dt)
            Lc = torch.linalg.cholesky(G)
            beta = torch.cholesky_solve(A.T @ Y_t, Lc)
            Sid = torch.cholesky_solve(torch.eye(A.shape[1], device=self.device, dtype=self.dt), Lc)
            AD = A @ Sid
            hd = (AD * A).sum(dim=1)
            fit = A @ beta
            r = (Y_t - fit) / (1.0 - hd).clamp_min(1e-8).unsqueeze(1)        # LOO residual
            self.heads.append(dict(beta=beta, Sid=Sid, A=A, hd=hd, r=r, fit=fit))

        idx = Y_t.argmax(1)
        s_head, s = [], torch.zeros(Xt.shape[0], dtype=self.dt, device=self.device)
        for j, h in enumerate(self.heads):
            f_loo = h["fit"] - h["hd"].unsqueeze(1) * h["r"]                  # own-point LOO outputs
            P = torch.softmax(f_loo, dim=1)
            sj = 1.0 - P.gather(1, idx.unsqueeze(1)).squeeze(1)
            s_head.append(sj)
            s = s + self.w[j] * sj
        self.s_head = torch.stack(s_head)      # (H, n)
        self.s_cal = s                          # (n,)
        return self

    # ------------------------------------------------------------ p-values
    def head_pvalues(self, X, batch_elems=1e8, score_dtype=None, per_head=True):
        """Cross-conformal p-values: per head (H, n_te, K) and aggregated (n_te, K).

        batch_elems: target number of elements in the (nb, n_cal, K) working tensor.
        score_dtype: precision used for the softmax/compare stage. The fit and the LOO algebra
            stay in self.dt; only the scoring pass is cast. "float32" is ~1.7x faster and produced
            identical sets on Connect-4, Shuttle and Adult.
        per_head: when False, skip the per-head p-values (only the aggregate is needed). Saves one
            compare+sum pass per head; used for methods that only consume the aggregate.
        """
        dt2 = self.dt
        if score_dtype is not None:
            dt2 = torch.float32 if str(score_dtype).endswith("float32") else torch.float64
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float64), device=self.device, dtype=self.dt)
        n_cal = self.s_cal.shape[0]
        K = self.Yk.shape[1]
        B = max(1, int(batch_elems // (n_cal * K)))
        s_cal = self.s_cal.to(dt2)
        s_head = [x.to(dt2) for x in self.s_head]
        out_h, out_a = [], []
        with torch.no_grad():
            for start in range(0, Xt.shape[0], B):
                xb = Xt[start:start + B]
                nb = xb.shape[0]
                As = self._designs_new(xb)
                T = torch.zeros(nb, n_cal, K, dtype=dt2, device=self.device)
                ph = []
                for j, h in enumerate(self.heads):
                    A_t = As[j]
                    u = ((A_t @ h["Sid"]) @ h["A"].T).to(dt2)                       # nb x n_cal
                    f_full = (A_t @ h["beta"]).to(dt2)                             # nb x K
                    r = h["r"].to(dt2)                                             # n_cal x K
                    f_loo = f_full[:, None, :] - u[:, :, None] * r[None, :, :]     # nb x n_cal x K
                    Tj = 1.0 - torch.softmax(f_loo, dim=2)
                    if per_head:
                        cnt_j = (Tj <= s_head[j][None, :, None]).sum(dim=1)
                        ph.append((cnt_j.to(self.dt) + 1.0) / (n_cal + 1.0))
                    T.add_(Tj, alpha=float(self.w[j]))      # weighted accumulate, in place
                    del f_loo, u, Tj
                cnt = (T <= s_cal[None, :, None]).sum(dim=1)
                out_a.append((cnt.to(self.dt) + 1.0) / (n_cal + 1.0))
                if per_head:
                    out_h.append(torch.stack(ph))
                del T
        P_h = torch.cat(out_h, dim=1).cpu().numpy() if per_head else None
        P_a = torch.cat(out_a).cpu().numpy()
        return P_h, P_a

    # --------------------------------------------------------------- probabilities
    def predict_proba(self, X):
        """Per-head and aggregated softmax probabilities: (H, n, K) and (n, K)."""
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float64), device=self.device, dtype=self.dt)
        with torch.no_grad():
            As = self._designs_new(Xt)
            P_h = torch.stack([torch.softmax(As[j] @ h["beta"], dim=1) for j, h in enumerate(self.heads)])
            P_a = (self.w[:, None, None] * P_h).sum(0)
        return P_h.cpu().numpy(), P_a.cpu().numpy()

    def slim(self):
        """Drop everything needed only for leave-one-out; keep what forward passes need."""
        for h in self.heads:
            for k in ("A", "Sid", "hd", "r", "fit"):
                h.pop(k, None)
        self.s_head = self.s_cal = None
        return self


class CVPlusConformal:
    """K-fold CV+ conformal scores (Barber et al. 2021, Section 3, Theorem 4).

    Each of the K folds is held out in turn; the model fitted on the rest scores the held-out
    points. Coverage guarantee: >= 1 - 2a - min{2(1-1/K)/(n/K+1), (1-K/n)/(K+1)} >= 1 - 2a - sqrt(2/n).
    Unlike the leave-one-out path, every test point is scored by only K models, so the p-value
    step costs O(K log(n/K)) comparisons instead of O(n_cal) element operations. The price is K
    model fits and slightly larger sets, because each model trains on a (1 - 1/K) fraction.
    """

    def __init__(self, n_folds=10, n_layers=3, n_hidden=50, lamda=0.1, device="cuda", seed=0,
                 head_mode="edrvfl", dtype="float64"):
        self.K = n_folds
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lam = lamda
        self.device = device
        self.seed = seed
        self.head_mode = head_mode
        self.dtype = dtype

    def fit(self, X, y):
        from sklearn.model_selection import StratifiedKFold
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        try:
            splitter = StratifiedKFold(self.K, shuffle=True, random_state=self.seed % (2 ** 31))
            folds = list(splitter.split(X, y))
        except ValueError:                       # a class with fewer members than folds
            from sklearn.model_selection import KFold
            folds = list(KFold(self.K, shuffle=True, random_state=self.seed % (2 ** 31)).split(X))
        n = len(y)
        self.classes_ = np.unique(y)
        self.models, self.sorted_agg, self.sorted_head = [], [], []
        s_agg = np.zeros(n)
        H = None
        s_head = None
        for tr, ho in folds:
            m = HeadedDeepRVFL3(n_layers=self.n_layers, n_hidden=self.n_hidden, lamda=self.lam,
                                device=self.device, seed=self.seed, dtype=self.dtype,
                                head_mode=self.head_mode).fit(X[tr], y[tr])
            Q_h, Q_a = m.predict_proba(X[ho])
            if H is None:
                H = Q_h.shape[0]
                s_head = np.zeros((H, n))
            yi = np.searchsorted(m.classes_, y[ho])
            seen = m.classes_[np.clip(yi, 0, len(m.classes_) - 1)] == y[ho]
            yi = np.where(seen, yi, 0)
            idx = np.arange(len(ho))
            # a label the fold model never saw gets probability 0, i.e. nonconformity 1
            s_agg[ho] = np.where(seen, 1.0 - Q_a[idx, yi], 1.0)
            for j in range(H):
                s_head[j, ho] = np.where(seen, 1.0 - Q_h[j][idx, yi], 1.0)
            self.sorted_agg.append(np.sort(s_agg[ho]))
            self.sorted_head.append([np.sort(s_head[j, ho]) for j in range(H)])
            self.models.append(m.slim())
        self.s_agg = s_agg
        self.s_head = s_head
        self.n_heads = H
        self.n_cal = n
        return self

    def pvalues(self, X, per_head=True):
        """(H, n_te, K) per-head CV+ p-values and (n_te, K) aggregated p-values.

        Fold models may have seen fewer classes than the full training fold; their output is
        scattered into the global class order, and a class a fold model never saw gets
        probability 0 (nonconformity 1), matching the convention used when fitting.
        """
        n_te = len(X)
        Kcls = len(self.classes_)
        counts = np.zeros((n_te, Kcls))
        counts_h = np.zeros((self.n_heads, n_te, Kcls)) if per_head else None
        for k, m in enumerate(self.models):
            Q_h, Q_a = m.predict_proba(X)
            pos = np.searchsorted(self.classes_, m.classes_)
            full_a = np.zeros((n_te, Kcls))
            full_a[:, pos] = Q_a
            sa = self.sorted_agg[k]
            counts += len(sa) - np.searchsorted(sa, 1.0 - full_a, side="left")
            if per_head:
                for j in range(self.n_heads):
                    full_h = np.zeros((n_te, Kcls))
                    full_h[:, pos] = Q_h[j]
                    sh = self.sorted_head[k][j]
                    counts_h[j] += len(sh) - np.searchsorted(sh, 1.0 - full_h, side="left")
        p_agg = (counts + 1.0) / (self.n_cal + 1.0)
        p_head = (counts_h + 1.0) / (self.n_cal + 1.0) if per_head else None
        return p_head, p_agg

    def predict_proba(self, X):
        """Bagged prediction: average the K fold models' probabilities (global class order)."""
        n_te = len(X)
        Kcls = len(self.classes_)
        acc_a = np.zeros((n_te, Kcls))
        acc_h = None
        for m in self.models:
            Q_h, Q_a = m.predict_proba(X)
            pos = np.searchsorted(self.classes_, m.classes_)
            full_a = np.zeros((n_te, Kcls))
            full_a[:, pos] = Q_a
            acc_a += full_a
            if acc_h is None:
                acc_h = np.zeros((self.n_heads, n_te, Kcls))
            for j in range(self.n_heads):
                acc_h[j][:, pos] += Q_h[j]
        acc_a /= len(self.models)
        acc_h /= len(self.models)
        return acc_h, acc_a
