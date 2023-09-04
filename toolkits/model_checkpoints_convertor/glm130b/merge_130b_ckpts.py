import torch
import pdb
output_state_dict = {}
rank_00 = torch.load('/mnt/glm-ckpts/glm-130b-sat/49300/mp_rank_00_model_states.pt', map_location='cpu')
rank_01 = torch.load('/mnt/glm-ckpts/glm-130b-sat/49300/mp_rank_01_model_states.pt', map_location='cpu')
rank_02 = torch.load('/mnt/glm-ckpts/glm-130b-sat/49300/mp_rank_02_model_states.pt', map_location='cpu')
rank_03 = torch.load('/mnt/glm-ckpts/glm-130b-sat/49300/mp_rank_03_model_states.pt', map_location='cpu')
rank_04 = torch.load('/mnt/glm-ckpts/glm-130b-sat/49300/mp_rank_04_model_states.pt', map_location='cpu')
rank_05 = torch.load('/mnt/glm-ckpts/glm-130b-sat/49300/mp_rank_05_model_states.pt', map_location='cpu')
rank_06 = torch.load('/mnt/glm-ckpts/glm-130b-sat/49300/mp_rank_06_model_states.pt', map_location='cpu')
rank_07 = torch.load('/mnt/glm-ckpts/glm-130b-sat/49300/mp_rank_07_model_states.pt', map_location='cpu')

sd_split={0:rank_00, 1:rank_01, 2:rank_02, 3:rank_03, 4:rank_04, 5:rank_05, 6:rank_06, 7:rank_07}

word_embedding_00 = rank_00['module']['transformer.word_embeddings.weight']
word_embedding_01 = rank_01['module']['transformer.word_embeddings.weight']
word_embedding_02 = rank_02['module']['transformer.word_embeddings.weight']
word_embedding_03 = rank_03['module']['transformer.word_embeddings.weight']
word_embedding_04 = rank_04['module']['transformer.word_embeddings.weight']
word_embedding_05 = rank_05['module']['transformer.word_embeddings.weight']
word_embedding_06 = rank_06['module']['transformer.word_embeddings.weight']
word_embedding_07 = rank_07['module']['transformer.word_embeddings.weight']

word_embedding = torch.cat((word_embedding_00,
                            word_embedding_01,
                            word_embedding_02,
                            word_embedding_03,
                            word_embedding_04,
                            word_embedding_05,
                            word_embedding_06,
                            word_embedding_07,
                            ), dim=0)

output_state_dict['transformer.word_embeddings.weight'] = word_embedding


for layer_id in range(70):

    input_layernorm_weight = 'transformer.layers.' + str(layer_id) + '.input_layernorm.weight'
    output_state_dict[input_layernorm_weight] = rank_00['module'][input_layernorm_weight]
    print(input_layernorm_weight)

    input_layernorm_bias = 'transformer.layers.' + str(layer_id) + '.input_layernorm.bias'
    output_state_dict[input_layernorm_bias] = rank_00['module'][input_layernorm_bias]
    print(input_layernorm_bias)

    self_att_qkv_weight = 'transformer.layers.' + str(
        layer_id) + '.attention.query_key_value.weight'

    self_att_qkv_bias = 'transformer.layers.' + str(
        layer_id) + '.attention.query_key_value.bias'

    self_att_dense_weight = 'transformer.layers.' + str(
        layer_id) + '.attention.dense.weight'

    tmp_qkv_weight = []
    tmp_qkv_bias = []
    tmp_dense_weight = []
    for i in range(8):
        rand_i = sd_split[i]
        tmp_qkv_weight.append(rand_i['module'][self_att_qkv_weight])
        tmp_qkv_bias.append(rand_i['module'][self_att_qkv_bias])
        tmp_dense_weight.append(rand_i['module'][self_att_dense_weight])

    output_state_dict[self_att_qkv_weight] = torch.cat(tmp_qkv_weight, dim=0)
    output_state_dict[self_att_qkv_bias] = torch.cat(tmp_qkv_bias, dim=0)
    output_state_dict[self_att_dense_weight] = torch.cat(tmp_dense_weight, dim=1)
    print(self_att_qkv_weight)
    print(self_att_qkv_bias)
    print(self_att_dense_weight)

    self_att_dense_bias = 'transformer.layers.' + str(layer_id) + '.attention.dense.bias'
    output_state_dict[self_att_dense_bias] = rank_00['module'][self_att_dense_bias]
    print(self_att_dense_bias)

    post_layernorm_weight = 'transformer.layers.' + str(layer_id) + '.post_attention_layernorm.weight'
    output_state_dict[post_layernorm_weight] = rank_00['module'][post_layernorm_weight]
    print(post_layernorm_weight)

    post_layernorm_bias = 'transformer.layers.' + str(layer_id) + '.post_attention_layernorm.bias'
    output_state_dict[post_layernorm_bias] = rank_00['module'][post_layernorm_bias]
    print(post_layernorm_bias)

    mlp_h_weight = 'transformer.layers.' + str(layer_id) + '.mlp.dense_h_to_4h.weight'
    mlp_h_bias = 'transformer.layers.' + str(layer_id) + '.mlp.dense_h_to_4h.bias'
    tmp_mlp_weight = []
    tmp_mlp_bias = []
    for i in range(8):
        rand_i = sd_split[i]
        tmp_mlp_weight.append(rand_i['module'][mlp_h_weight])
        tmp_mlp_bias.append(rand_i['module'][mlp_h_bias])

    output_state_dict[mlp_h_weight] = torch.cat(tmp_mlp_weight, dim=0)
    output_state_dict[mlp_h_bias] = torch.cat(tmp_mlp_bias, dim=0)
    print(mlp_h_weight)
    print(mlp_h_bias)

    mlp_4h_weight = 'transformer.layers.' + str(layer_id) + '.mlp.dense_4h_to_h.weight'
    tmp_mlp_4h_weight = []
    for i in range(8):
        rand_i = sd_split[i]
        tmp_mlp_4h_weight.append(rand_i['module'][mlp_4h_weight])

    output_state_dict[mlp_4h_weight] = torch.cat(tmp_mlp_4h_weight, dim=1)
    print(mlp_4h_weight)

    mlp_4h_bias = 'transformer.layers.' + str(layer_id) + '.mlp.dense_4h_to_h.bias'
    output_state_dict[mlp_4h_bias] = rank_00['module'][mlp_4h_bias]
    print(mlp_4h_bias)

output_state_dict['transformer.final_layernorm.weight'] = rank_00['module'][
    'transformer.final_layernorm.weight']
print('final_layernorm.weight')

output_state_dict['transformer.final_layernorm.bias'] = rank_00['module'][
    'transformer.final_layernorm.bias']
print('final_layernorm.bias')

torch.save(output_state_dict, "/mnt/glm-ckpts/glm-130b-sat/pytorch_model.bin")
print("done")
