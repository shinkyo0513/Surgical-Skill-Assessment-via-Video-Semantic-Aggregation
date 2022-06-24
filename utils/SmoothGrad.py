import torch
import numpy as np 

def smooth_grad (inputs, labels, model, device, 
                    nsamples=25, variant=None, stdev_spread=0.15):
    # model.eval()   # Set model to evaluate mode


    bs, nt, ch, h, w = inputs.shape
    assert ch == 3
    assert labels.shape[0] == bs
    assert variant in [None, 'square', 'variance']

    inputs = inputs.to(device)
    labels = labels.to(device)
    # labels = labels.to(dtype=torch.long)

    inputs_max = torch.max(inputs.view(bs, -1), dim=1)[0]
    inputs_min = torch.min(inputs.view(bs, -1), dim=1)[0]
    stdev = stdev_spread * (inputs_max - inputs_min)    # bs
    stdev = stdev.view(-1, 1).expand(-1, ch*nt*h*w)

    outputs = model(inputs)[0]

    backward_signals = torch.zeros_like(outputs, device=device)
    for bidx in range(bs):
        backward_signals[bidx, 0] = labels[bidx].item()
    # backward_signals = labels

    all_grads = []
    for i in range(nsamples):
        noise = torch.normal(0.0, stdev).to(device).reshape(bs, nt, ch, h, w)
        # print(noise.shape, noise.min(), noise.max())
        noisy_inputs = inputs + noise
        noisy_inputs.requires_grad_()

        # Forward
        outputs = model(noisy_inputs)[0]

        # Backward
        outputs.backward(backward_signals, retain_graph=True)
        noisy_grads = noisy_inputs.grad.cpu()
        all_grads.append(noisy_grads)
    all_grads = torch.stack(all_grads, dim=1)   # bs x nsamples x nt x ch x h x w
    
    if variant == None:
        smth_grad = torch.mean(all_grads, dim=1)
    elif variant == 'square':
        smth_grad = torch.mean(all_grads ** 2, dim=1)
    elif variant == 'variance':
        smth_grad = torch.var(all_grads, dim=1)
    smth_grad = smth_grad.numpy()   # NxTx3xHxW

    normed_grad = np.sum(np.abs(smth_grad), axis=2, keepdims=True)   # NxTx1xHxW
    normed_grad = normed_grad.reshape((bs, -1))
    vmax = np.percentile(normed_grad, 99.0, axis=1, keepdims=True)
    vmin = np.min(normed_grad, axis=1, keepdims=True)
    normed_grad = torch.from_numpy(np.clip((normed_grad - vmin) / (vmax - vmin), 0, 1))   # N x 1*T*H*W
    normed_grad = normed_grad.reshape((bs, nt, 1, h, w))
    # grad_show = torch.repeat_interleave(normed_grad, 3, dim=1)
    return normed_grad

