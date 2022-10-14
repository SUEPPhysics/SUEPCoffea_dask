from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear
from torch_geometric.nn import DataParallel

class Net(nn.Module):
    def __init__(self, out_dim=1, hidden_dim=16):
        super(Net, self).__init__()
        
        # hidden_dim = 16
        # out_dim = 2
        
        self.pf_encode = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        
        self.conv = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
            k=8
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ELU(),
            nn.Linear(8, 4),
            nn.ELU(),
            nn.Linear(4, out_dim)
            #nn.Sigmoid()    
            )
        
    def forward(self,
                x_pf,
                batch_pf):

        x_pf_enc = self.pf_encode(x_pf)
        
        # create a representation of PFs to PFs
        feats1 = self.conv(x=(x_pf_enc, x_pf_enc), batch=(batch_pf, batch_pf))
        feats2 = self.conv(x=(feats1, feats1), batch=(batch_pf, batch_pf))

        batch = batch_pf
        out, batch = avg_pool_x(batch, feats2, batch)
        
        out = self.output(out)
        
        return out, batch
