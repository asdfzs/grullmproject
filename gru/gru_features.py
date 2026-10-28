"""
Shared by gru/train_gru.py (offline training) and team_code/intervention_agent.py (online trigger).
Keeping featurize() in ONE place guarantees the runtime GRU sees exactly the features it was trained on.
"""
import numpy as np

FRONT_CLASSES = ['none', 'vehicle_stopped', 'vehicle_moving', 'pedestrian']
NONE = 999.0
WIN = 60          # 3 s at 20 Hz
DT = 0.05

FEATURE_NAMES = [
    'ego_speed', 'lon_accel', 'yaw_rate', 'stop_dur',
    'front_dist', 'front_ok', 'front_rel_speed', 'front_speed', 'front_width', 'front_hazard', 'front_stopped_dur',
    'front2_dist', 'front2_ok', 'front2_speed',
    'left_occ', 'left_s', 'left_ok', 'left_rel_speed', 'left_count', 'left_mean_speed',
    'right_occ', 'right_s', 'right_ok', 'right_rel_speed', 'right_count', 'right_mean_speed',
    'tl_red', 'tl_dist', 'tl_ok', 'at_junction',
] + ['cls_' + c for c in FRONT_CLASSES]

DROP_GROUPS = {
    'front2': lambda n: n.startswith('front2_'),
    'hazard': lambda n: n == 'front_hazard',
    'side':   lambda n: n.startswith('left_') or n.startswith('right_'),
    'light':  lambda n: n.startswith('tl_'),
}


def keep_indices(drop):
    for g in drop:
        if g not in DROP_GROUPS:
            raise SystemExit(f'unknown --drop group: {g}  (choose from {list(DROP_GROUPS)})')
    return [i for i, n in enumerate(FEATURE_NAMES) if not any(DROP_GROUPS[g](n) for g in drop)]


def featurize(r):
    """r: one CSV row as dict (strings or numbers). Returns the full 34-dim feature list."""
    f = lambda k: float(r[k])
    capd = lambda v, c: (min(v, c), 1.0) if v < NONE - 1 else (c, 0.0)   # (value, present_flag)
    fd, fd_ok = capd(f('front_dist_m'), 60.0)
    f2, f2_ok = capd(f('front2_dist_m'), 60.0)
    ls, ls_ok = capd(abs(f('left_nearest_rel_s_m')), 60.0)
    rs, rs_ok = capd(abs(f('right_nearest_rel_s_m')), 60.0)
    tl, tl_ok = capd(f('traffic_light_dist_m'), 100.0)
    x = [
        f('ego_speed_kmh'), np.clip(f('lon_accel_ms2'), -10, 10), np.clip(f('yaw_rate_degs'), -90, 90),
        min(f('stop_duration_s'), 60.0),
        fd, fd_ok, f('front_rel_speed_ms'), f('front_speed_ms'), f('front_width_m'),
        f('front_hazard_on'), min(f('front_stopped_duration_s'), 60.0),
        f2, f2_ok, f('front2_speed_ms'),
        f('left_occupied'), ls, ls_ok, f('left_rel_speed_ms'), f('left_count'), f('left_mean_speed_ms'),
        f('right_occupied'), rs, rs_ok, f('right_rel_speed_ms'), f('right_count'), f('right_mean_speed_ms'),
        f('traffic_light_red'), tl, tl_ok, f('at_junction'),
    ]
    x += [1.0 if r['front_class'] == c else 0.0 for c in FRONT_CLASSES]
    return x


def make_model(n_in):
    import torch.nn as nn

    class DeadlockGRU(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.gru = nn.GRU(n_in, 32, num_layers=2, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1]).squeeze(-1)

    return DeadlockGRU(n_in)


class OnlineGRU:
    """Streaming GRU for the driving agent: push one row per tick, get P(call LLM)."""

    def __init__(self, model_dir):
        import json, os, torch
        self.torch = torch
        meta = json.load(open(os.path.join(model_dir, 'norm.json')))
        self.mu = np.array(meta['mean'], dtype=np.float32)
        self.sd = np.array(meta['std'], dtype=np.float32)
        self.keep = meta.get('keep', list(range(len(FEATURE_NAMES))))
        self.win = meta.get('win', WIN)          # samples per window
        self.step = meta.get('step', 1)          # ticks between samples
        self.model = make_model(len(self.keep))
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt'), map_location='cpu'))
        self.model.eval()
        self.buf = []
        self.label = meta.get('label', 'p')

    def push(self, row):
        x = np.array(featurize(row), dtype=np.float32)[self.keep]
        x = (x - self.mu) / self.sd
        n = (self.win - 1) * self.step + 1       # ticks covered by one window
        if not self.buf:
            self.buf = [x] * (n - 1)
        self.buf.append(x)
        self.buf = self.buf[-n:]
        seq = np.stack(self.buf[::self.step])    # oldest ... newest, every `step` ticks
        with self.torch.no_grad():
            logit = self.model(self.torch.tensor(seq[None], dtype=self.torch.float32))
        return float(self.torch.sigmoid(logit)[0])
