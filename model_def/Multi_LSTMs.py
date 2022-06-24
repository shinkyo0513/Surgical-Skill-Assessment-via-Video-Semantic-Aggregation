import torch
from torch import nn

class Multi_LSTMs (nn.Module):
    def __init__ (self, num_lstms, num_input_chnals, num_output_chanls, bidirectional, detach_hidden_state, share_param):
        super(Multi_LSTMs, self).__init__()
        self.num_lstms = num_lstms
        self.num_input_chnals = num_input_chnals
        self.num_output_chanls = num_output_chanls
        self.bidirectional = bidirectional
        self.detach_hidden_state = detach_hidden_state

        if detach_hidden_state:
            if bidirectional:
                if share_param:
                    self.lstms_forw = nn.ModuleList([nn.LSTMCell(input_size=num_input_chnals,
                                                                hidden_size=num_output_chanls//2)
                                                    ] * num_lstms)
                    self.lstms_back = nn.ModuleList([nn.LSTMCell(input_size=num_input_chnals,
                                                                hidden_size=num_output_chanls//2)
                                                    ] * num_lstms)
                else:
                    self.lstms_forw = nn.ModuleList([nn.LSTMCell(input_size=num_input_chnals,
                                                                hidden_size=num_output_chanls//2) for k in range(num_lstms)])
                    self.lstms_back = nn.ModuleList([nn.LSTMCell(input_size=num_input_chnals,
                                                                hidden_size=num_output_chanls//2) for k in range(num_lstms)])
            else:
                if share_param:
                    self.lstms = nn.ModuleList([nn.LSTMCell(input_size=num_input_chnals,
                                                            hidden_size=num_output_chanls)
                                                ] * num_lstms)
                else:
                    self.lstms = nn.ModuleList([nn.LSTMCell(input_size=num_input_chnals,
                                                            hidden_size=num_output_chanls) for k in range(num_lstms)])
        else:
            if share_param:
                self.lstms = nn.ModuleList([nn.LSTM(input_size=num_input_chnals, 
                                                    hidden_size=num_output_chanls, 
                                                    bidirectional=bidirectional,
                                                    batch_first=True)] * num_lstms)
            else:
                self.lstms = nn.ModuleList([nn.LSTM(input_size=num_input_chnals, 
                                                    hidden_size=num_output_chanls, 
                                                    bidirectional=bidirectional,
                                                    batch_first=True) for k in range(num_lstms)])
        
    def forward (self, inputs):
        device = inputs.device
        bs, nt, ch, k = inputs.shape
        assert ch == self.num_input_chnals
        assert k == self.num_lstms

        if self.detach_hidden_state:
            if self.bidirectional:
                h_forw = [torch.zeros((bs, self.num_output_chanls//2)).to(device), ] * k
                c_forw = [torch.zeros((bs, self.num_output_chanls//2)).to(device), ] * k
                h_back = [torch.zeros((bs, self.num_output_chanls//2)).to(device), ] * k
                c_back = [torch.zeros((bs, self.num_output_chanls//2)).to(device), ] * k
                hs_forw = [list() for kidx in range(k)]
                hs_back = [list() for kidx in range(k)]
                for kidx in range(k):
                    for tidx in range(nt):
                        h_forw[kidx], c_forw[kidx] = self.lstms_forw[kidx](inputs[:,tidx,:,kidx], (h_forw[kidx].detach(), c_forw[kidx].detach()))
                        h_back[kidx], c_back[kidx] = self.lstms_back[kidx](inputs[:,nt-tidx-1,:,kidx], (h_back[kidx].detach(), c_back[kidx].detach()))
                        hs_forw[kidx].append(h_forw[kidx].clone())
                        hs_back[kidx].append(h_back[kidx].clone())
                    hs_forw[kidx] = torch.stack(hs_forw[kidx], dim=1)
                    hs_back[kidx].reverse()
                    hs_back[kidx] = torch.stack(hs_back[kidx], dim=1)
                hs_forw = torch.stack(hs_forw, dim=3)
                hs_back = torch.stack(hs_back, dim=3)
                outputs = torch.cat([hs_forw, hs_back], dim=2)  # bs x nt x ch x k
            else:
                h = [torch.zeros((bs, self.num_output_chanls)).to(device), ] * k
                c = [torch.zeros((bs, self.num_output_chanls)).to(device), ] * k
                hs = [list() for kidx in range(k)]
                for kidx in range(k):
                    for tidx in range(nt):
                        h[kidx], c[kidx] = self.lstms[kidx](inputs[:,tidx,:,kidx], (h[kidx].detach(), c[kidx].detach()))
                        hs[kidx].append(h[kidx].clone())
                    hs[kidx] = torch.stack(hs[kidx], dim=1)
                hs = torch.stack(hs, dim=3)
                outputs = hs  # bs x nt x ch x k
        else:
            outputs = [self.lstms[kidx](inputs[:,:,:,kidx])[0] for kidx in range(k)] # list of BxL'x128, length K
            outputs = torch.stack(outputs, dim=3).contiguous()   # bs x nt x ch x k

        return outputs
