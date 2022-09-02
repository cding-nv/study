import os
import sys
import torch
from torch import nn
from reference import BertConfig, BertModel
from oodnlib.mod import ExtBertEncoder, ExtBertPooler


class BertEncoderPooler(nn.Module):
    def __init__(self,
                 config,
                 init_parms=None):
        super(BertEncoderPooler, self).__init__()
        self.encoder = ExtBertEncoder(config, init_parms)
        self.pooler = ExtBertPooler(config, init_parms)

    def forward(self,
                hidden_states,
                attention_mask,
                cls_count=1):
        encoder_outputs = self.encoder(hidden_states,
                                       attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(encoder_outputs[0], cls_count=cls_count)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs


def breakpoint():
    import os, signal
    os.kill(os.getpid(), signal.SIGTRAP)


def cmpTensor(path, name, l, r):
    if not isinstance(l, torch.Tensor) or not isinstance(r, torch.Tensor):
        print("\n>>> input should be torch.Tensor")
        return
    if l.size() != r.size():
        print("\n>>> tensor size mismatch: {} != {}".format(l.size(), r.size()))
        return
    l = l.to(torch.float64)
    r = r.to(torch.float64)
    cos = torch.cosine_similarity(l, r, dim=-1)
    cos_min = torch.min(cos)
    cos_mean = torch.mean(cos)
    cos_std = torch.std(cos, unbiased=False)
    print("{:24}   min: {:.6f}   mean: {:.6f}   std: {:.6f}".format(name, cos_min, cos_mean, cos_std))
    torch.save(l.detach().clone(), os.path.join(path, name))


def cmpTensorList(path, name, l, r):
    if not isinstance(l, tuple) or not isinstance(r, tuple):
        print("\n>>> input should be tuple")
        return
    if len(l) != len(r):
        print("\n>>> tuple size mismatch: {} != {}".format(len(l), len(r)))
        return
    for i in range(len(l)):
        cmpTensor(path, "{}[{:02}]".format(name, i), l[i], r[i])


def RefBert(config,
            hidden_states,
            attention_mask,
            grad):
    use_fp16 = hidden_states.dtype == torch.float16
    bert = BertModel(config).cuda().half() if use_fp16 else BertModel(config).cuda()
    if config.training:
        bert.train()
    else:
        bert.eval()
    bert_parms = list(bert.parameters())

    head_mask = [None] * config.num_hidden_layers
    outputs = bert(hidden_states, attention_mask, head_mask)
    ans = outputs[0] + outputs[1].unsqueeze(1)

    ghost = torch.empty(0, dtype=hidden_states.dtype, device=hidden_states.device)
    if config.training:
        ans.backward(gradient=grad)

        grad_normA_gamma_list = torch.empty(config.num_hidden_layers, config.hidden_size, dtype=ghost.dtype, device=ghost.device)
        grad_normA_beta_list = torch.empty(config.num_hidden_layers, config.hidden_size, dtype=ghost.dtype, device=ghost.device)
        grad_normB_gamma_list = torch.empty(config.num_hidden_layers, config.hidden_size, dtype=ghost.dtype, device=ghost.device)
        grad_normB_beta_list = torch.empty(config.num_hidden_layers, config.hidden_size, dtype=ghost.dtype, device=ghost.device)
        grad_linearA_weight_list = torch.empty(config.num_hidden_layers, 3 * config.hidden_size, config.hidden_size, dtype=ghost.dtype, device=ghost.device)
        grad_linearA_bias_list = torch.empty(config.num_hidden_layers, 3 * config.hidden_size, dtype=ghost.dtype, device=ghost.device)
        grad_linearB_weight_list = torch.empty(config.num_hidden_layers, config.hidden_size, config.hidden_size, dtype=ghost.dtype, device=ghost.device)
        grad_linearB_bias_list = torch.empty(config.num_hidden_layers, config.hidden_size, dtype=ghost.dtype, device=ghost.device)
        grad_linearC_weight_list = torch.empty(config.num_hidden_layers, config.intermediate_size, config.hidden_size, dtype=ghost.dtype, device=ghost.device)
        grad_linearC_bias_list = torch.empty(config.num_hidden_layers, config.intermediate_size, dtype=ghost.dtype, device=ghost.device)
        grad_linearD_weight_list = torch.empty(config.num_hidden_layers, config.hidden_size, config.intermediate_size, dtype=ghost.dtype, device=ghost.device)
        grad_linearD_bias_list = torch.empty(config.num_hidden_layers, config.hidden_size, dtype=ghost.dtype, device=ghost.device)
        grad_norm_gamma = torch.empty(config.hidden_size, dtype=ghost.dtype, device=ghost.device) if config.pre_layernorm else ghost
        grad_norm_beta = torch.empty(config.hidden_size, dtype=ghost.dtype, device=ghost.device) if config.pre_layernorm else ghost
        grad_linear_weight = torch.empty(config.hidden_size, config.hidden_size, dtype=ghost.dtype, device=ghost.device)
        grad_linear_bias = torch.empty(config.hidden_size, dtype=ghost.dtype, device=ghost.device)

        if config.pre_layernorm:
            for i in range(config.num_hidden_layers):
                parms = bert_parms[i*16:]
                grad_normA_gamma_list[i].copy_(parms[6].grad)
                grad_normA_beta_list[i].copy_(parms[7].grad)
                grad_normB_gamma_list[i].copy_(parms[12].grad)
                grad_normB_beta_list[i].copy_(parms[13].grad)
                grad_linearA_weight_list[i].copy_(torch.cat((parms[0].grad, parms[2].grad, parms[4].grad), dim=0))
                grad_linearA_bias_list[i].copy_(torch.cat((parms[1].grad, parms[3].grad, parms[5].grad), dim=0))
                grad_linearB_weight_list[i].copy_(parms[8].grad)
                grad_linearB_bias_list[i].copy_(parms[9].grad)
                grad_linearC_weight_list[i].copy_(parms[10].grad)
                grad_linearC_bias_list[i].copy_(parms[11].grad)
                grad_linearD_weight_list[i].copy_(parms[14].grad)
                grad_linearD_bias_list[i].copy_(parms[15].grad)
            grad_norm_gamma.copy_(bert_parms[-4].grad)
            grad_norm_beta.copy_(bert_parms[-3].grad)
            grad_linear_weight.copy_(bert_parms[-2].grad)
            grad_linear_bias.copy_(bert_parms[-1].grad)
        else:
            for i in range(config.num_hidden_layers):
                parms = bert_parms[i*16:]
                grad_normA_gamma_list[i].copy_(parms[8].grad)
                grad_normA_beta_list[i].copy_(parms[9].grad)
                grad_normB_gamma_list[i].copy_(parms[14].grad)
                grad_normB_beta_list[i].copy_(parms[15].grad)
                grad_linearA_weight_list[i].copy_(torch.cat((parms[0].grad, parms[2].grad, parms[4].grad), dim=0))
                grad_linearA_bias_list[i].copy_(torch.cat((parms[1].grad, parms[3].grad, parms[5].grad), dim=0))
                grad_linearB_weight_list[i].copy_(parms[6].grad)
                grad_linearB_bias_list[i].copy_(parms[7].grad)
                grad_linearC_weight_list[i].copy_(parms[10].grad)
                grad_linearC_bias_list[i].copy_(parms[11].grad)
                grad_linearD_weight_list[i].copy_(parms[12].grad)
                grad_linearD_bias_list[i].copy_(parms[13].grad)
            grad_linear_weight.copy_(bert_parms[-2].grad)
            grad_linear_bias.copy_(bert_parms[-1].grad)

    return (bert_parms,
            outputs,
            grad_normA_gamma_list if config.training else ghost,
            grad_normA_beta_list if config.training else ghost,
            grad_normB_gamma_list if config.training else ghost,
            grad_normB_beta_list if config.training else ghost,
            grad_linearA_weight_list if config.training else ghost,
            grad_linearA_bias_list if config.training else ghost,
            grad_linearB_weight_list if config.training else ghost,
            grad_linearB_bias_list if config.training else ghost,
            grad_linearC_weight_list if config.training else ghost,
            grad_linearC_bias_list if config.training else ghost,
            grad_linearD_weight_list if config.training else ghost,
            grad_linearD_bias_list if config.training else ghost,
            grad_norm_gamma if config.training else ghost,
            grad_norm_beta if config.training else ghost,
            grad_linear_weight if config.training else ghost,
            grad_linear_bias if config.training else ghost)


def ExtBert(config,
            hidden_states,
            attention_mask,
            grad,
            ref_outputs):
    (init_parms,
     ans_outputs,
     ans_grad_normA_gamma_list,
     ans_grad_normA_beta_list,
     ans_grad_normB_gamma_list,
     ans_grad_normB_beta_list,
     ans_grad_linearA_weight_list,
     ans_grad_linearA_bias_list,
     ans_grad_linearB_weight_list,
     ans_grad_linearB_bias_list,
     ans_grad_linearC_weight_list,
     ans_grad_linearC_bias_list,
     ans_grad_linearD_weight_list,
     ans_grad_linearD_bias_list,
     ans_grad_norm_gamma,
     ans_grad_norm_beta,
     ans_grad_linear_weight,
     ans_grad_linear_bias) = ref_outputs

    use_fp16 = hidden_states.dtype == torch.float16
    bert = BertEncoderPooler(config, init_parms).cuda().half() if use_fp16 else BertEncoderPooler(config, init_parms).cuda()
    if config.training:
        bert.train()
    else:
        bert.eval()

    outputs = bert(hidden_states, attention_mask)
    ans = outputs[0] + outputs[1].unsqueeze(1)

    if config.training:
        ans.backward(gradient=grad)

    path = config.out_path
    cmpTensor(path, "sequence_output", outputs[0], ans_outputs[0])
    cmpTensor(path, "pooled_output", outputs[1], ans_outputs[1])
    if config.output_hidden_states:
        print()
        cmpTensorList(path, "all_hidden_states", outputs[2], ans_outputs[2])
    if config.output_attentions:
        print()
        idx = 1 + config.output_hidden_states + config.output_attentions
        cmpTensorList(path, "all_attentions", outputs[idx], ans_outputs[idx])
    if config.training:
        print()
        cmpTensor(path, "grad_normA_gamma_list", bert.encoder.normA_gamma_list.grad, ans_grad_normA_gamma_list)
        cmpTensor(path, "grad_normA_beta_list", bert.encoder.normA_beta_list.grad, ans_grad_normA_beta_list)
        cmpTensor(path, "grad_normB_gamma_list", bert.encoder.normB_gamma_list.grad, ans_grad_normB_gamma_list)
        cmpTensor(path, "grad_normB_beta_list", bert.encoder.normB_beta_list.grad, ans_grad_normB_beta_list)
        cmpTensor(path, "grad_linearA_weight_list", bert.encoder.linearA_weight_list.grad, ans_grad_linearA_weight_list)
        cmpTensor(path, "grad_linearA_bias_list", bert.encoder.linearA_bias_list.grad, ans_grad_linearA_bias_list)
        cmpTensor(path, "grad_linearB_weight_list", bert.encoder.linearB_weight_list.grad, ans_grad_linearB_weight_list)
        cmpTensor(path, "grad_linearB_bias_list", bert.encoder.linearB_bias_list.grad, ans_grad_linearB_bias_list)
        cmpTensor(path, "grad_linearC_weight_list", bert.encoder.linearC_weight_list.grad, ans_grad_linearC_weight_list)
        cmpTensor(path, "grad_linearC_bias_list", bert.encoder.linearC_bias_list.grad, ans_grad_linearC_bias_list)
        cmpTensor(path, "grad_linearD_weight_list", bert.encoder.linearD_weight_list.grad, ans_grad_linearD_weight_list)
        cmpTensor(path, "grad_linearD_bias_list", bert.encoder.linearD_bias_list.grad, ans_grad_linearD_bias_list)
        if config.pre_layernorm:
            cmpTensor(path, "grad_norm_gamma", bert.encoder.norm_gamma.grad, ans_grad_norm_gamma)
            cmpTensor(path, "grad_norm_beta", bert.encoder.norm_beta.grad, ans_grad_norm_beta)
        cmpTensor(path, "grad_linear_weight", bert.pooler.linear_weight.grad, ans_grad_linear_weight)
        cmpTensor(path, "grad_linear_bias", bert.pooler.linear_bias.grad, ans_grad_linear_bias)


def main():
    pre_layernorm = True
    training = True
    P_dropout_prob = 0.0
    H_dropout_prob = 0.0
    drop_path_prob = 0.0
    fast_fp32 = False
    output_attentions = False
    output_hidden_states = True

    inp_path = sys.argv[1]
    out_path = sys.argv[2]
    hidden_states  = torch.load(inp_path + "/hidden_states.pt")
    attention_mask = torch.load(inp_path + "/attention_mask.pt")
    torch.manual_seed(911)
    grad = torch.rand_like(hidden_states)

    config = BertConfig(pre_layernorm=pre_layernorm,
                        attention_probs_dropout_prob=P_dropout_prob,
                        hidden_act="gelu",
                        hidden_dropout_prob=H_dropout_prob,
                        drop_path=drop_path_prob,
                        hidden_size=768,
                        initializer_range=0.02,
                        intermediate_size=3072,
                        num_attention_heads=12,
                        num_hidden_layers=12,
                        vocab_size_or_config_json_file=30522,
                        max_position_embeddings=512,
                        type_vocab_size=2,
                        use_token_type=True,
                        use_position=True,
                        input_dropout_and_layernorm=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states)
    config.amp = True
    if hidden_states.dtype == torch.float16:
        config.opt_level = "O2"
    elif fast_fp32:
        config.opt_level = "O1"
    else:
        config.opt_level = "O0"
    config.training = training
    config.use_fp16 = config.opt_level == "O2"
    config.fast_fp32 = fast_fp32
    config.name = "transformer"
    config.out_path = out_path

    ref_outputs = RefBert(config, hidden_states, attention_mask, grad)
    ExtBert(config, hidden_states, attention_mask, grad, ref_outputs)


if __name__ == "__main__":
    main()

