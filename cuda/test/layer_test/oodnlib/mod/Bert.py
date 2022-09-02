import copy
import torch
import oodnlib.ext.bert
from .Plugin import register_op
from torch import nn


class ExtBertEncoder(nn.Module):
    def __init__(self, config, init_parms=None):
        super(ExtBertEncoder, self).__init__()
        config.fast_fp32 = getattr(config, "fast_fp32", True)
        self.config = copy.deepcopy(config)
        torch.ops.bert.encoder_init(config.fast_fp32)

        self.normA_gamma_list = nn.Parameter(
            torch.Tensor(config.num_hidden_layers, config.hidden_size)
        )
        self.normA_beta_list = nn.Parameter(
            torch.Tensor(config.num_hidden_layers, config.hidden_size)
        )
        self.normB_gamma_list = nn.Parameter(
            torch.Tensor(config.num_hidden_layers, config.hidden_size)
        )
        self.normB_beta_list = nn.Parameter(
            torch.Tensor(config.num_hidden_layers, config.hidden_size)
        )
        self.linearA_weight_list = nn.Parameter(
            torch.Tensor(
                config.num_hidden_layers, 3 * config.hidden_size, config.hidden_size
            )
        )
        self.linearA_bias_list = nn.Parameter(
            torch.Tensor(config.num_hidden_layers, 3 * config.hidden_size)
        )
        self.linearB_weight_list = nn.Parameter(
            torch.Tensor(
                config.num_hidden_layers, config.hidden_size, config.hidden_size
            )
        )
        self.linearB_bias_list = nn.Parameter(
            torch.Tensor(config.num_hidden_layers, config.hidden_size)
        )
        self.linearC_weight_list = nn.Parameter(
            torch.Tensor(
                config.num_hidden_layers, config.intermediate_size, config.hidden_size
            )
        )
        self.linearC_bias_list = nn.Parameter(
            torch.Tensor(config.num_hidden_layers, config.intermediate_size)
        )
        self.linearD_weight_list = nn.Parameter(
            torch.Tensor(
                config.num_hidden_layers, config.hidden_size, config.intermediate_size
            )
        )
        self.linearD_bias_list = nn.Parameter(
            torch.Tensor(config.num_hidden_layers, config.hidden_size)
        )
        self.norm_gamma = nn.Parameter(
            torch.Tensor(config.hidden_size)
            if config.pre_layernorm
            else torch.Tensor(0)
        )
        self.norm_beta = nn.Parameter(
            torch.Tensor(config.hidden_size)
            if config.pre_layernorm
            else torch.Tensor(0)
        )

        if init_parms is None:
            self.normA_gamma_list.data.fill_(1.0)
            self.normA_beta_list.data.zero_()
            self.normB_gamma_list.data.fill_(1.0)
            self.normB_beta_list.data.zero_()
            self.linearA_weight_list.data.normal_(
                mean=0.0, std=config.initializer_range
            )
            self.linearA_bias_list.data.zero_()
            self.linearB_weight_list.data.normal_(
                mean=0.0, std=config.initializer_range
            )
            self.linearB_bias_list.data.zero_()
            self.linearC_weight_list.data.normal_(
                mean=0.0, std=config.initializer_range
            )
            self.linearC_bias_list.data.zero_()
            self.linearD_weight_list.data.normal_(
                mean=0.0, std=config.initializer_range
            )
            self.linearD_bias_list.data.zero_()
            if config.pre_layernorm:
                self.norm_gamma.data.fill_(1.0)
                self.norm_beta.data.zero_()
        elif config.pre_layernorm:
            for i in range(config.num_hidden_layers):
                parms = init_parms[i * 16 :]
                self.normA_gamma_list[i].data.copy_(parms[6])
                self.normA_beta_list[i].data.copy_(parms[7])
                self.normB_gamma_list[i].data.copy_(parms[12])
                self.normB_beta_list[i].data.copy_(parms[13])
                self.linearA_weight_list[i].data.copy_(
                    torch.cat((parms[0], parms[2], parms[4]), dim=0)
                )
                self.linearA_bias_list[i].data.copy_(
                    torch.cat((parms[1], parms[3], parms[5]), dim=0)
                )
                self.linearB_weight_list[i].data.copy_(parms[8])
                self.linearB_bias_list[i].data.copy_(parms[9])
                self.linearC_weight_list[i].data.copy_(parms[10])
                self.linearC_bias_list[i].data.copy_(parms[11])
                self.linearD_weight_list[i].data.copy_(parms[14])
                self.linearD_bias_list[i].data.copy_(parms[15])
            self.norm_gamma.data.copy_(init_parms[-4])
            self.norm_beta.data.copy_(init_parms[-3])
        else:
            for i in range(config.num_hidden_layers):
                parms = init_parms[i * 16 :]
                self.normA_gamma_list[i].data.copy_(parms[8])
                self.normA_beta_list[i].data.copy_(parms[9])
                self.normB_gamma_list[i].data.copy_(parms[14])
                self.normB_beta_list[i].data.copy_(parms[15])
                self.linearA_weight_list[i].data.copy_(
                    torch.cat((parms[0], parms[2], parms[4]), dim=0)
                )
                self.linearA_bias_list[i].data.copy_(
                    torch.cat((parms[1], parms[3], parms[5]), dim=0)
                )
                self.linearB_weight_list[i].data.copy_(parms[6])
                self.linearB_bias_list[i].data.copy_(parms[7])
                self.linearC_weight_list[i].data.copy_(parms[10])
                self.linearC_bias_list[i].data.copy_(parms[11])
                self.linearD_weight_list[i].data.copy_(parms[12])
                self.linearD_bias_list[i].data.copy_(parms[13])

    def forward(self, hidden_states, attention_mask):
        if self.training:
            (
                sequence_output,
                layer_out_list,
                attention_list,
            ) = torch.ops.bert.encoder_train(
                hidden_states,
                attention_mask,
                self.normA_gamma_list,
                self.normA_beta_list,
                self.normB_gamma_list,
                self.normB_beta_list,
                self.linearA_weight_list,
                self.linearA_bias_list,
                self.linearB_weight_list,
                self.linearB_bias_list,
                self.linearC_weight_list,
                self.linearC_bias_list,
                self.linearD_weight_list,
                self.linearD_bias_list,
                self.norm_gamma,
                self.norm_beta,
                self.config.pre_layernorm,
                self.config.num_attention_heads,
                self.config.intermediate_size,
                self.config.attention_probs_dropout_prob,
                self.config.hidden_dropout_prob,
                self.config.drop_path,
                self.config.layer_norm_eps,
            )
        else:
            register_op("bert.encoder_infer")
            (sequence_output, layer_out_list) = torch.ops.bert.encoder_infer(
                hidden_states,
                attention_mask,
                self.normA_gamma_list,
                self.normA_beta_list,
                self.normB_gamma_list,
                self.normB_beta_list,
                self.linearA_weight_list,
                self.linearA_bias_list,
                self.linearB_weight_list,
                self.linearB_bias_list,
                self.linearC_weight_list,
                self.linearC_bias_list,
                self.linearD_weight_list,
                self.linearD_bias_list,
                self.norm_gamma,
                self.norm_beta,
                self.config.pre_layernorm,
                self.config.num_attention_heads,
                self.config.intermediate_size,
                self.config.layer_norm_eps,
                self.config.fast_fp32,
                self.config.num_hidden_layers,
            )

        # construct outputs
        outputs = (sequence_output,)
        if self.config.output_hidden_states:
            all_hidden_states = (hidden_states,) + layer_out_list.unbind()
            outputs += (all_hidden_states,)
        # output_attentions is only available for training
        if self.training and self.config.output_attentions:
            all_attentions = attention_list.unbind()
            outputs += (all_attentions,)
        return outputs  # sequence_output, (all hidden states), (all attentions)


class ExtBertPooler(nn.Module):
    def __init__(self, config, init_parms=None):
        super(ExtBertPooler, self).__init__()
        config.fast_fp32 = getattr(config, "fast_fp32", True)
        self.config = copy.deepcopy(config)
        torch.ops.bert.pooler_init(config.fast_fp32)

        self.linear_weight = nn.Parameter(
            torch.Tensor(config.hidden_size, config.hidden_size)
        )
        self.linear_bias = nn.Parameter(torch.Tensor(config.hidden_size))

        if init_parms is None:
            self.linear_weight.data.normal_(mean=0.0, std=config.initializer_range)
            self.linear_bias.data.zero_()
        else:
            self.linear_weight.data.copy_(init_parms[-2])
            self.linear_bias.data.copy_(init_parms[-1])

    def forward(self, hidden_states, cls_count=1):
        if self.training:
            pooled_output = torch.ops.bert.pooler_train(
                hidden_states, self.linear_weight, self.linear_bias, cls_count
            )
        else:
            register_op("bert.pooler_infer")
            pooled_output = torch.ops.bert.pooler_infer(
                hidden_states,
                self.linear_weight,
                self.linear_bias,
                cls_count,
                self.config.fast_fp32,
            )

        return pooled_output.squeeze(1)
