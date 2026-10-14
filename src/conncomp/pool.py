"""A cache of good forms that stays spread out: niches by angle, and lineages.

Adaptive searches (conncomp.search, conncomp.subspace_search) perturb cached forms. A plain "most recent forms at the
maximum" cache is taken over by the descendants of the first form that reaches a new maximum, so the search ends up
exploring a small cap around one ancestor. DiverseCache prevents this:

* niches: two cached forms are at least `min_angle` apart, measured between descriptors (the normalized Hessian
  curve H(f) in Hessian mode, the normalized form otherwise; terms of f of degree <= 1 do not change H, so f itself
  would be the wrong descriptor). A new form inside the niche of a member replaces it only if it has more ovals;
  otherwise it takes a new slot, evicting a member with the fewest ovals when the cache is full;
* lineages: every form carries the id of the uniform sample it descends from; `parent()` draws a lineage uniformly
  among the cached ones, then a member of it (weighted towards high counts), so one lineage cannot monopolize the
  search even when it holds most of the members.
"""

import math
import random

import numpy as np
import torch

from conncomp.hessian import hessian


def descriptors(coefs, deg, use_hessian):
    """Unit descriptor vectors (B, D') of forms (D, B) on the GPU: normalized H(f), or the normalized form."""
    c = hessian(deg, coefs) if use_hessian else coefs
    return (c / c.norm(dim=0, keepdim=True)).T


class DiverseCache:
    def __init__(self, maxlen=1000, min_angle_deg=10.0, rng=None):
        self.maxlen = maxlen
        self.cos = math.cos(math.radians(min_angle_deg))
        self.forms, self.keys = [], None  # numpy forms; torch (n, D') descriptors
        self.score, self.lineage = [], []
        self.rng = rng or random.Random()

    def __len__(self):
        return len(self.forms)

    def lineages(self):
        return len(set(self.lineage))

    def add(self, coefs, keys, scores, lineage):
        """Offer forms coefs (D, B) with descriptors keys (B, D') and oval counts scores (B,); lineage: one id for
        all of them (perturbations of one parent) or a sequence of ids (unrelated forms)."""
        if coefs.shape[1] == 0:
            return 0
        lins = [lineage] * coefs.shape[1] if np.isscalar(lineage) or isinstance(lineage, tuple) else list(lineage)
        order = np.argsort(-np.asarray(scores), kind="stable")
        cpu = coefs.T.cpu().numpy()
        added = 0
        for j in order:
            s = int(scores[j])
            k = keys[j]
            if self.keys is not None and len(self.forms):
                cos = (self.keys @ k).abs()
                m = int(cos.argmax())
                if float(cos[m]) >= self.cos:  # inside the niche of member m
                    if s > self.score[m]:
                        self._put(m, cpu[j], k, s, lins[j])
                        added += 1
                    continue
            if len(self.forms) < self.maxlen:
                self._append(cpu[j], k, s, lins[j])
                added += 1
            else:
                low = min(self.score)
                if s >= low:
                    victims = [i for i, x in enumerate(self.score) if x == low]
                    self._put(self.rng.choice(victims), cpu[j], k, s, lins[j])
                    added += 1
        return added

    def _append(self, f, k, s, lin):
        self.forms.append(f)
        self.keys = k[None].clone() if self.keys is None else torch.cat([self.keys, k[None]])
        self.score.append(s)
        self.lineage.append(lin)

    def _put(self, i, f, k, s, lin):
        self.forms[i], self.score[i], self.lineage[i] = f, s, lin
        self.keys[i] = k

    def parent(self, bias=4.0):
        """A cached form: a lineage uniformly at random, then a member with weight bias^(score - lineage best).
        Returns (form, lineage id)."""
        lin = self.rng.choice(sorted(set(self.lineage)))
        idx = [i for i, x in enumerate(self.lineage) if x == lin]
        best = max(self.score[i] for i in idx)
        w = [bias ** (self.score[i] - best) for i in idx]
        i = self.rng.choices(idx, weights=w)[0]
        return self.forms[i], lin

    def best(self):
        """Cached forms sorted by decreasing oval count (ties in insertion order)."""
        order = sorted(range(len(self.forms)), key=lambda i: -self.score[i])
        return [self.forms[i] for i in order], [self.score[i] for i in order]

    def histogram(self):
        h = {}
        for s in self.score:
            h[s] = h.get(s, 0) + 1
        return dict(sorted(h.items(), reverse=True))


def spread(forms, deg, use_hessian, min_angle_deg, device=None):
    """Greedy subset of the forms (list of numpy vectors, in order of preference) with descriptors pairwise at least
    min_angle_deg apart."""
    if not forms:
        return []
    c = torch.as_tensor(np.stack(forms, axis=1), device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    K = descriptors(c, deg, use_hessian)
    cos = math.cos(math.radians(min_angle_deg))
    keep = []
    for i in range(len(forms)):
        if not keep or float((K[keep] @ K[i]).abs().max()) < cos:
            keep.append(i)
    return [forms[i] for i in keep]


class Collector:
    """Collect all counted forms with at least `min_ovals` ovals; dump them periodically, appending to one file per
    oval count: <stem>_ovals<n>.txt next to `path` (one coefficient vector per line, as conncomp.io.save_coefs).

    Thinning (min_angle_deg > 0): a form is kept only if its descriptor (normalized H(f), or the normalized form) is
    at least min_angle_deg away from all forms already kept with the same oval count. The kept descriptors stay on
    the GPU (float32); candidates are compared in chunks with the kept set and with the earlier candidates of their
    chunk (a candidate close to any earlier one is dropped, even if that one was dropped too: slightly conservative,
    the kept set is still pairwise separated).
    """

    def __init__(self, path, min_ovals, every=60.0, min_angle_deg=0.0, deg=None, use_hessian=True, chunk=4096):
        import time

        self.path, self.min_ovals, self.every = path, min_ovals, every
        self.cos = math.cos(math.radians(min_angle_deg)) if min_angle_deg > 0 else None
        self.deg, self.use_hessian, self.chunk = deg, use_hessian, chunk
        self.buf, self.total, self.seen, self.keys = {}, {}, {}, {}
        self._time = time.time
        self.last = self._time()
        path.parent.mkdir(parents=True, exist_ok=True)

    def file(self, n):
        return self.path.with_name(f"{self.path.stem}_ovals{n}.txt")

    def _thin(self, n, c):
        """Indices of the columns of c (D, m) to keep for oval count n (updates the kept descriptors)."""
        K = descriptors(c, self.deg, self.use_hessian).float()
        kept = []
        for s0 in range(0, K.shape[0], self.chunk):
            k = K[s0 : s0 + self.chunk]
            ok = torch.ones(k.shape[0], dtype=torch.bool, device=k.device)
            old = self.keys.get(n)
            if old is not None:
                for t0 in range(0, old.shape[0], 65536):
                    ok &= (k @ old[t0 : t0 + 65536].T).abs().amax(dim=1) < self.cos
            close = (k @ k.T).abs() >= self.cos
            ok &= ~torch.tril(close, diagonal=-1).any(dim=1)
            idx = torch.nonzero(ok)[:, 0]
            if idx.numel():
                self.keys[n] = k[idx] if old is None else torch.cat([old, k[idx]])
                kept.append(idx + s0)
        return torch.cat(kept) if kept else torch.zeros(0, dtype=torch.long, device=c.device)

    def add(self, coefs, ovals):
        """Forms coefs (D, B) with oval counts ovals (B,); dumps if `every` seconds have passed."""
        ovals = np.asarray(ovals)
        for n in np.unique(ovals[ovals >= self.min_ovals]).tolist():
            sel = torch.as_tensor(np.nonzero(ovals == n)[0], device=coefs.device)
            c = coefs[:, sel]
            self.seen[n] = self.seen.get(n, 0) + c.shape[1]
            if self.cos is not None:
                c = c[:, self._thin(n, c)]
            if c.shape[1]:
                self.buf.setdefault(n, []).extend(c.T.cpu().numpy())
                self.total[n] = self.total.get(n, 0) + c.shape[1]
        if self._time() - self.last >= self.every:
            self.dump()

    def dump(self):
        for n, rows in self.buf.items():
            if rows:
                with open(self.file(n), "a") as fh:
                    for x in rows:
                        fh.write(" ".join(str(y) for y in x) + "\n")
        self.buf = {}
        self.last = self._time()

    def summary(self):
        if self.cos is None:
            return ", ".join(f"{n}: {c:,}" for n, c in sorted(self.total.items(), reverse=True))
        return ", ".join(f"{n}: {self.total.get(n, 0):,}/{c:,}" for n, c in sorted(self.seen.items(), reverse=True))
