import argparse
parser = argparse.ArgumentParser(description="Replace the decoder parameters of checkpoint1 with the encoder parameters of checkpoint2")
parser.add_argument('--checkpoint1', help='the first checkpoint')
parser.add_argument('--checkpoint2', help='the second checkpoint')
parser.add_argument('--save-checkpoint', default='checkpoint_last.pt', help='the name of saved checkpoint')
parser.add_argument('--checkpoint-init', help='you know it !!!')
args = parser.parse_args()
print('*+----+'*9)
print('+ checkpoint 1 : {0:<44} +'.format(args.checkpoint1))
print('+ checkpoint 2 : {0:<44} +'.format(args.checkpoint2))
print('+ checkpoint res : {0:<42} +'.format(args.save_checkpoint))
if args.checkpoint_init != None:
    print('+ checkpoint init : {0:<41} +'.format(args.checkpoint_init))
print('*+----+'*9)

import torch
en_point = torch.load(args.checkpoint1, map_location=torch.device('cpu'))
de_point = torch.load(args.checkpoint2, map_location=torch.device('cpu'))
en_model = en_point['model']
de_model = de_point['model']
if args.checkpoint_init != None:
    init_point = torch.load(args.checkpoint_init, map_location=torch.device('cpu'))
    init_model = init_point['model']

replace_list = ['self_attn.k_proj.weight',
                'self_attn.k_proj.bias',
                'self_attn.v_proj.weight',
                'self_attn.v_proj.bias',
                'self_attn.q_proj.weight',
                'self_attn.q_proj.bias',
                'self_attn.out_proj.weight',
                'self_attn.out_proj.bias',
                'self_attn_layer_norm.weight',
                'self_attn_layer_norm.bias',
                'fc1.weight',
                'fc1.bias',
                'fc2.weight',
                'fc2.bias',
                'final_layer_norm.weight',
                'final_layer_norm.bias']

if args.checkpoint_init == None:
    for layer in range(6):
        for param in replace_list:
            en_model['decoder.layers.{0}.{1}'.format(layer, param)] = de_model['encoder.layers.{0}.{1}'.format(layer, param)]
    en_point['model'] = en_model
    torch.save(en_point, args.save_checkpoint)

else:
    for layer in range(6):
        for param in replace_list:
            init_model['encoder.layers.{0}.{1}'.format(layer, param)] = en_model['encoder.layers.{0}.{1}'.format(layer, param)]
            init_model['decoder.layers.{0}.{1}'.format(layer, param)] = de_model['encoder.layers.{0}.{1}'.format(layer, param)]
    init_point['model'] = init_model
    torch.save(init_point, args.save_checkpoint)
