import torch
from torch import nn

from common.sample_norm import sample_norm


class AutoencPrtclBasedEvntIn(nn.Module):

    def __init__(self,
                 prtcl_ltnt_size: int,
                 prtcl_in_evnt_cnt: int,
                 evnt_ltnt_size: int,
                 device):
        super(AutoencPrtclBasedEvntIn, self).__init__()

        # self.autoenc_prtcl = autoenc_prtcl
        self.prtcl_ltnt_size = prtcl_ltnt_size
        self.prtcl_in_evnt_cnt = prtcl_in_evnt_cnt
        self.evnt_ltnt_size = evnt_ltnt_size
        self.device = device

        pre_ltnt = 128

        self.net = nn.Sequential(
            nn.Linear(prtcl_ltnt_size * prtcl_in_evnt_cnt, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, pre_ltnt),
            # nn.Tanh(),
        ).to(device)

        self.mu = nn.Sequential(
            nn.Linear(pre_ltnt, pre_ltnt),
            nn.Tanh(),
            nn.Linear(pre_ltnt, evnt_ltnt_size),
            # nn.Tanh()
        ).to(device=device)

        self.var = nn.Sequential(
            nn.Linear(pre_ltnt, pre_ltnt),
            nn.Tanh(),
            nn.Linear(pre_ltnt, evnt_ltnt_size),
            # nn.Tanh()
        ).to(device=device)

    def forward(self, prtcls_ltnt: torch.Tensor):
        # batch_size = len(prtcls_ltnt)

        pre_ltnt = self.net(prtcls_ltnt)

        lat_mu = self.mu(pre_ltnt)
        lat_ver = self.var(pre_ltnt)

        return lat_mu, lat_ver


class AutoencPrtclBasedEvntOut(nn.Module):

    def __init__(self,
                 emb_size: int,
                 prtcl_ltnt_size: int,
                 prtcl_in_evnt_cnt: int,
                 evnt_ltnt_size: int,
                 device):
        super(AutoencPrtclBasedEvntOut, self).__init__()

        self.emb_size = emb_size
        self.prtcl_ltnt_size = prtcl_ltnt_size
        self.prtcl_in_evnt_cnt = prtcl_in_evnt_cnt
        self.evnt_ltnt_size = evnt_ltnt_size

        self.device = device

        self.net = nn.Sequential(
            nn.Linear(evnt_ltnt_size, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, prtcl_ltnt_size * prtcl_in_evnt_cnt),
            # nn.Tanh(),
        ).to(device)

    def forward(self, ltnt: torch.Tensor):
        out_prtcl_ltnts = self.net(ltnt)

        return out_prtcl_ltnts


'''
Takes in a batch of events, each of size:
PADDING x FEATURES

Returns a batch of events, each of size:
PADDING x EMB_FEATURES
'''
class AutoencPrtclBasedEvnt(nn.Module):
    def __init__(self,
                 auto_in: AutoencPrtclBasedEvntIn,
                 auto_out: AutoencPrtclBasedEvntOut):
        super(AutoencPrtclBasedEvnt, self).__init__()

        self.auto_in = auto_in
        self.auto_out = auto_out

    @staticmethod
    def create(
            emb_size: int,
            prtcl_ltnt_size: int,
            prtcl_in_evnt_cnt: int,
            evnt_ltnt_size: int,
            device):
        auto_in = AutoencPrtclBasedEvntIn(
            prtcl_ltnt_size=prtcl_ltnt_size,
            prtcl_in_evnt_cnt=prtcl_in_evnt_cnt,
            evnt_ltnt_size=evnt_ltnt_size,
            device=device)

        auto_out = AutoencPrtclBasedEvntOut(
            emb_size=emb_size,
            prtcl_ltnt_size=prtcl_ltnt_size,
            prtcl_in_evnt_cnt=prtcl_in_evnt_cnt,
            evnt_ltnt_size=evnt_ltnt_size,
            device=device)

        return auto_in, auto_out

    def forward(self, prtcls_ltnt: torch.Tensor, ):
        batch_size = len(prtcls_ltnt)
        prtcls_ltnt = prtcls_ltnt.flatten(start_dim=1)

        lat_mu, lat_var = self.auto_in(prtcls_ltnt)

        lat_vec = sample_norm(lat_mu, lat_var)

        out_prtcl_ltnts: torch.Tensor = self.auto_out(lat_vec)
        out_prtcl_ltnts = out_prtcl_ltnts.reshape(batch_size, self.auto_out.prtcl_in_evnt_cnt,
                                                  self.auto_out.prtcl_ltnt_size)
        return out_prtcl_ltnts, lat_mu, lat_var