import torch
import torch.nn.functional as F
import numpy as np 

# Backward hook
observ_grad_ = []
def backward_hook(m, i_grad, o_grad): 
    global observ_grad_
    observ_grad_.insert(0, o_grad[0].detach())

# Forward hook
observ_actv_ = []
def forward_hook(m, i, o):
    global observ_actv_
    observ_actv_.append(o.detach())

def grad_cam_rnn (inputs, labels, model, device, layer_name, norm_vis=True):
    model.eval()   # Set model to evaluate mode
    
    bs, ch, nt, h, w = inputs.shape
    assert ch == 3
    assert labels.shape[0] == bs

    # Backward hook
    backward_hook = lambda li: lambda m, i_grad, o_grad: li.insert(0, o_grad[0].detach()) # layer7: 4x1024x7x7
    # backward_hook = lambda li: lambda m, i_grad, o_grad: li.insert(0, o_grad.detach())
    # Forward hook
    forward_hook = lambda li: lambda m, i, o: li.append(o.detach()) # layer7: 4x1024x7x7
    # forward_hook = lambda li: lambda m, i, o: li.append(o.detach())
    observ_layer = model
    for name in layer_name:
        # print(dict(observ_layer.named_children()).keys())
        observ_layer = dict(observ_layer.named_children())[name]

    observ_grad_ = []
    bh = observ_layer.register_backward_hook(backward_hook(observ_grad_))
    # observ_layer.register_backward_hook(backward_hook)
    observ_actv_ = []
    fh = observ_layer.register_forward_hook(forward_hook(observ_actv_))
    # observ_layer.register_forward_hook(forward_hook)

    inputs = inputs.to(device)
    labels = labels.to(dtype=torch.long)

    # Forward pass
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    observ_actv = torch.stack(observ_actv_, dim=2)   # N x 512 x num_f x14x14
    # print(observ_actv.shape)
    # print(f'actv: min: {observ_actv.min()}, max: {observ_actv.max()}')

    # backward pass
    backward_signals = torch.zeros_like(outputs, device=device)
    for bidx in range(bs):
        backward_signals[bidx, labels[bidx].cpu().item()] = 1.0
    outputs.backward(backward_signals)

    observ_grad = torch.stack(observ_grad_, dim=2)   # N x 512 x num_f x14x14
    # print(f'grad: {observ_grad.shape}; actv: {observ_actv.shape}')
    # print(f'actv: min: {observ_grad.min()}, max: {observ_grad.max()}')

    observ_grad_w = observ_grad.mean(dim=4, keepdim=True).mean(dim=3, keepdim=True) # N x 512 x num_f x1x1
    out_masks = F.relu( (observ_grad_w*observ_actv).sum(dim=1, keepdim=True) ) # N x 1 x num_f x14x14
    out_masks = out_masks.detach().cpu()
    # print(f'min: {out_masks.min()}, max: {out_masks.max()}')

    fh.remove()
    bh.remove()

    if norm_vis:
        normed_masks = out_masks.view(bs, -1)
        mins = torch.min(normed_masks, dim=1, keepdim=True)[0]
        maxs = torch.max(normed_masks, dim=1, keepdim=True)[0]
        normed_masks = (normed_masks - mins) / (maxs - mins)    
        out_masks = normed_masks.reshape(out_masks.shape)

    return out_masks

def grad_cam (inputs, labels, model, device, norm_vis=True):
    print('Use GradCAM to explain...')
    model.eval()   # Set model to evaluate mode
    
    inp_bs, inp_nt, inp_ch, inp_h, inp_w = inputs.shape     # ch = num_crop * num_frame * 3

    # layer_dict = dict(model.model.named_children())
    # assert layer_name in layer_dict, \
    #     f'Given layer ({layer_name}) is not in model. {list(layer_dict.keys())}'
    # observ_layer = layer_dict[layer_name]

    # observ_layer = model
    # for name in layer_name:
    #     assert name in dict(observ_layer.named_children())
    #     observ_layer = dict(observ_layer.named_children())[name]

    observ_layer = model.featmap_extractor[3]

    observ_layer.register_full_backward_hook(backward_hook)
    observ_layer.register_forward_hook(forward_hook)

    inputs = inputs.to(device)
    print(f'inputs: {inputs.shape}')
    labels = labels.to(dtype=torch.long)

    # Forward pass
    outputs = model(inputs)[0]
    # print(f'outputs: {outputs.shape}')

    observ_actv = observ_actv_[0]   # 1 x C x nt x h' x w'
    # observ_actv = torch.stack(observ_actv_, dim=2)   # N x 512 x num_f x14x14
    # print('observ_actv:', observ_actv.shape)
    # observ_actv = torch.repeat_interleave(observ_actv, int(nt/observ_actv.shape[2]), dim=2)

    # backward pass
    backward_signals = torch.zeros_like(outputs, device=device)
    for bidx in range(inp_bs):
        backward_signals[bidx, 0] = labels[bidx].item()
        # backward_signals[bidx, 0] = 1.0
    outputs.backward(backward_signals)

    observ_grad = observ_grad_[0]   # 1 x C x nt x h' x w'
    # observ_grad = torch.stack(observ_grad_, dim=2)   # N x 512 x num_f x14x14
    # print('observ_grad:', observ_grad.shape)

    observ_grad_w = observ_grad.mean(dim=4, keepdim=True).mean(dim=3, keepdim=True) # 1 x C x nt x 1x1
    out_masks = F.relu( (observ_grad_w*observ_actv).sum(dim=1, keepdim=True) ) # 1 x 1 x nt x h' x w'

    mask_bs, mask_ch, mask_nt, mask_h, mask_w = out_masks.shape

    if norm_vis:
        out_masks_reshape = out_masks.reshape(mask_bs, mask_ch, -1)
        masks_min = out_masks_reshape.min(dim=2, keepdim=True)[0]
        # masks_max = out_masks_reshape.max(dim=2, keepdim=True)[0]
        masks_max = torch.quantile(out_masks_reshape, 0.98, dim=2, keepdim=True)
        out_masks_reshape = (out_masks_reshape - masks_min) / (masks_max - masks_min)
        out_masks_reshape = torch.clamp(out_masks_reshape, min=0, max=1)
        out_masks = out_masks_reshape.reshape(out_masks.shape)

    if mask_nt != inp_nt:
        out_masks = torch.repeat_interleave(out_masks, int(inp_nt/mask_nt), dim=2)  # 1 x 1 x nt x h' x w'

    out_masks = out_masks.permute(0, 2, 1, 3, 4)
    return out_masks


# def full_layers_grad_cam (inputs, labels, model, device, num_frame=8, norm_vis=True):
#     model.eval()   # Set model to evaluate mode
    
#     inp_bs, inp_ch, inp_h, inp_w = inputs.shape     # ch = num_crop * num_frame * 3

#     layer_dict = {name: layer for name, layer in dict(model.model.named_children()).items() if name in ['s1', 's2', 's3', 's4', 's5']}
#     fhs = []
#     bhs = []
#     for layer_name, layer in layer_dict.items():
#         fhs.append(layer.register_forward_hook(forward_hook))
#         bhs.append(layer.register_backward_hook(backward_hook))

#     inputs = inputs.to(device)
#     labels = labels.to(dtype=torch.long)

#     global observ_grad_, observ_actv_
#     observ_grad_ = []
#     observ_actv_ = []

#     # Forward pass
#     outputs = model(inputs)
#     _, preds = torch.max(outputs, 1)
#     # backward pass
#     backward_signals = torch.zeros_like(outputs, device=device)
#     for bidx in range(inp_bs):
#         backward_signals[bidx, labels[bidx].item()] = 1.0
#     outputs.backward(backward_signals)

#     masks_dict = {}
#     for layer_idx, layer_name in enumerate(layer_dict.keys()):
#         layer_grad = observ_grad_[layer_idx]    # bs' x C x nt' x h' x w'
#         layer_actv = observ_actv_[layer_idx]    # bs' x C x nt' x h' x w'
#         # print(f'{layer_name}: grad: {layer_grad.shape}, actv: {layer_actv.shape}')

#         layer_grad_w = layer_grad.mean(dim=4, keepdim=True).mean(dim=3, keepdim=True) # bs' x C x nt x 1 x 1
#         layer_masks = F.relu( (layer_grad_w*layer_actv).sum(dim=1, keepdim=True) ) # bs' x 1 x nt x h' x w'

#         mask_bs, mask_ch, mask_nt, mask_h, mask_w = layer_masks.shape
#         # print(f'mask_bs: {mask_bs}, inp_bs: {inp_bs}')

#         if norm_vis:
#             layer_masks_reshape = layer_masks.reshape(mask_bs, mask_ch, -1)
#             masks_min = layer_masks_reshape.min(dim=2, keepdim=True)[0]
#             masks_max = layer_masks_reshape.max(dim=2, keepdim=True)[0]
#             layer_masks_reshape = (layer_masks_reshape - masks_min) / (masks_max - masks_min)
#             layer_masks = layer_masks_reshape.reshape(layer_masks.shape)

#         num_crop = mask_bs // inp_bs
#         layer_masks = torch.stack(torch.split(layer_masks, inp_bs, dim=0), dim=1)   # bs x num_crop x 1 x nt x h' x w'
#         if mask_nt != num_frame:
#             layer_masks = torch.repeat_interleave(layer_masks, int(num_frame/mask_nt), dim=3)
#         masks_dict[layer_name] = layer_masks

#     for bh in bhs:
#         bh.remove()
#     for fh in fhs:
#         fh.remove()

#     return masks_dict



