
"""fuhao7i 2021.12.06"""

weights_path = ''
print('+---------------------------------------------+')
print('+   Loading weights into model state dict..   +')
print('+---------------------------------------------+')
model_dict = model.state_dict()

for k in model_dict.keys():
    print('model.keys ==>', k)

pretrained_dict = torch.load(weights_path)

for k in pretrained_dict['state_dict'].keys():
    print('pretrained.keys ==>', k)

momo_dict = {}

for k, v in pretrained_dict['state_dict'].items(): 
    if k in model_dict.keys():
        if pretrained_dict['state_dict'][k].size() == model_dict[k].size():
            momo_dict.update({k: v})

for k, v in momo_dict.items():
    print('model load => ', k)
    
model_dict.update(momo_dict)
model.load_state_dict(model_dict)
print('+---------------------------------------------+')
print('+                 Finished！                  +')
print('+---------------------------------------------+')