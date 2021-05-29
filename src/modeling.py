from transformers import AutoModel
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoModel
from transformers.modeling_bert import BertForSequenceClassification, SequenceClassifierOutput, BertModel, \
    BertPreTrainedModel
from transformers import RobertaModel,RobertaForSequenceClassification
from transformers.modeling_roberta import RobertaPreTrainedModel,RobertaClassificationHead
from transformers import ElectraModel,ElectraForSequenceClassification
from transformers.modeling_electra import ElectraPreTrainedModel,ElectraClassificationHead
from transformers import PretrainedConfig, AutoConfig
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np


def real_labels(labels):
    attack_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    orig_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    simplify_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    isMR_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    for i in range(len(labels)):
        if labels[i] > 9:
            attack_labels[i] = 1
            orig_labels[i] = labels[i] - 10
        elif labels[i] > 7:
            orig_labels[i] = labels[i] - 8
        elif labels[i] > 5:
            simplify_labels[i] = 1
            attack_labels[i] = 1
            isMR_labels[i] = 1
            orig_labels[i] = labels[i] - 6
        elif labels[i] > 3:
            simplify_labels[i] = 1
            isMR_labels[i] = 1
            orig_labels[i] = labels[i] - 4
        elif labels[i] > 1:
            attack_labels[i] = 1
            isMR_labels[i] = 1
            orig_labels[i] = labels[i] - 2
        elif labels[i] > -1:
            isMR_labels[i] = 1
            orig_labels[i] = labels[i]
    return attack_labels,orig_labels, simplify_labels, isMR_labels

def real_labels_mnli(labels):
    attack_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    orig_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    simplify_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    for i in range(len(labels)):
        if labels[i] < 3:
            orig_labels[i] = labels[i]
        elif labels[i] < 6 :
            orig_labels[i] = labels[i] - 3
            simplify_labels[i] = 1
        elif labels[i] < 9:
            attack_labels[i] = 1
            orig_labels[i] = labels[i] - 6
        elif labels[i] < 12:
            simplify_labels[i] = 1
            attack_labels[i] = 1
            orig_labels[i] = labels[i] - 9
    return attack_labels, orig_labels, simplify_labels



from transformers.activations import get_activation

class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ElectraForSequenceClassificationAdvV2(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)
        self.classifier1 = ElectraClassificationHead(config, 2)
        self.classifier2 = ElectraClassificationHead(config, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        inference=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        sequence_output = discriminator_hidden_states[0]
        logits1 = self.classifier1(sequence_output)
        logits2 = self.classifier2(sequence_output)
        prob = torch.sigmoid(logits2)

        loss = None
        if labels is not None:
            attack_labels, orig_labels, simplify_labels, isMR_labels = real_labels(labels)
            loss_fct1 = CrossEntropyLoss()
            loss1 = loss_fct1(logits1.view(-1, 2), orig_labels.view(-1))
            loss_fct2 = nn.BCEWithLogitsLoss()
            active_loss2 = simplify_labels.view(-1) == 0
            active_logits2 = logits2.view(-1)[active_loss2]
            active_labels2 = attack_labels.float().view(-1)[active_loss2]
            loss2 = loss_fct2(active_logits2, active_labels2)
            loss = loss1 + loss2

        if inference:
            output = (logits1, prob) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        if not return_dict:
            output = (logits1,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaForSequenceClassificationAdvV2(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier1 = RobertaClassificationHead(config,2)
        self.classifier2 = RobertaClassificationHead(config,1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        inference=False
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits1 = self.classifier1(sequence_output)
        logits2 = self.classifier2(sequence_output)
        prob = torch.sigmoid(logits2)
        loss = None
        if labels is not None:
            attack_labels, orig_labels, simplify_labels, isMR_labels = real_labels(labels)
            loss_fct1 = CrossEntropyLoss()
            loss1 = loss_fct1(logits1.view(-1, 2), orig_labels.view(-1))
            loss_fct2 = nn.BCEWithLogitsLoss()
            active_loss2 = simplify_labels.view(-1) == 0
            active_logits2 = logits2.view(-1)[active_loss2]
            active_labels2 = attack_labels.float().view(-1)[active_loss2]
            loss2 = loss_fct2(active_logits2, active_labels2)
            loss = loss1 + loss2

        if inference:
            output = (logits1, prob) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if not return_dict:
            output = (logits1,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits1,
            hidden_states=outputs[0],
            attentions=outputs.attentions,
        )

class BertForSequenceClassification(BertPreTrainedModel):
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False
            ,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooled_output = outputs[1]
        # attention_loss = 0.001*torch.norm(outputs['attentions'][0])
        attention_loss = 0.01 * torch.reciprocal(torch.norm(outputs['attentions'][0]))
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss = loss + attention_loss

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


class BertForSequenceClassificationAdvV2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 1)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        inference=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)
        prob = torch.sigmoid(logits2)
        loss = None
        if labels is not None:
            attack_labels, orig_labels, simplify_labels, isMR_labels = real_labels(labels)
            loss_fct1 = CrossEntropyLoss()
            loss1 = loss_fct1(logits1.view(-1, 2), orig_labels.view(-1))
            loss_fct2 = nn.BCEWithLogitsLoss()
            active_loss2 = simplify_labels.view(-1) == 0
            active_logits2 = logits2.view(-1)[active_loss2]
            active_labels2 = attack_labels.float().view(-1)[active_loss2]
            loss2 = loss_fct2(active_logits2, active_labels2)
            loss = loss1+loss2

        if inference:
            output = (logits1, prob) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if not return_dict:
            output = (logits1,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits1,
            hidden_states=outputs[0],
            attentions=outputs.attentions,
        )


class BertForSequenceClassificationAdvV2_mnli(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 3)
        self.classifier2 = nn.Linear(config.hidden_size, 1)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        inference=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)
        prob = torch.sigmoid(logits2)
        loss = None
        if labels is not None:
            attack_labels, orig_labels, simplify_labels= real_labels_mnli(labels)
            loss_fct1 = CrossEntropyLoss()
            loss1 = loss_fct1(logits1.view(-1, 3), orig_labels.view(-1))
            loss_fct2 = nn.BCEWithLogitsLoss()
            active_loss2 = simplify_labels.view(-1) == 0
            active_logits2 = logits2.view(-1)[active_loss2]
            active_labels2 = attack_labels.float().view(-1)[active_loss2]
            loss2 = loss_fct2(active_logits2, active_labels2)
            loss = loss1+loss2

        if inference:
            output = (logits1, prob) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if not return_dict:
            output = (logits1,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits1,
            hidden_states=outputs[0],
            attentions=outputs.attentions,
        )


class BertForSequenceClassificationAdvV3(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.pooler1 = Pooler(config)
        self.pooler2 = Pooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.classifier3 = nn.Linear(config.hidden_size, 1)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        inference = False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        output2 = self.pooler1(pooled_output)
        output3 = self.pooler2(pooled_output)
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(output2)
        logits3 = self.classifier3(output3)
        prob = torch.sigmoid(logits3)
        """
        print('prob: \n')
        print(prob)
        print('logits1: \n')
        print(logits1)
        print('logits2: \n')
        print(logits2)
        print('logits: \n')
        print(logits)
        """
        loss = None
        if labels is not None:
            attack_labels, orig_labels, simplify_labels, isMR_labels = real_labels(labels)
            loss_fct1 = CrossEntropyLoss()
            active_loss1_notattack = attack_labels.view(-1) == 0
            active_loss1_isMR = isMR_labels.view(-1) == 1
            active_loss1 = active_loss1_notattack & active_loss1_isMR
            active_logits1 = logits1.view(-1, 2)[active_loss1]
            active_labels1 = orig_labels.view(-1)[active_loss1]
            loss1 = loss_fct1(active_logits1, active_labels1)
            # active_loss2 = attack_labels.view(-1) == 1
            active_loss2_isattack = attack_labels.view(-1) == 1
            active_loss2_isMR = isMR_labels.view(-1) == 1
            active_loss2 = active_loss2_isattack & active_loss2_isMR
            active_logits2 = logits2.view(-1, 2)[active_loss2]
            active_labels2 = orig_labels.view(-1)[active_loss2]
            loss2 = loss_fct1(active_logits2, active_labels2)
            loss_fct2 = nn.BCEWithLogitsLoss()

            active_loss3 = simplify_labels.view(-1) == 0
            active_logits3 = logits3.view(-1)[active_loss3]
            active_labels3 = attack_labels.float().view(-1)[active_loss3]
            loss3 = loss_fct2(active_logits3, active_labels3)

            # loss3 = loss_fct2(logits3.view(-1), attack_labels.float().view(-1))
            loss = loss1+loss2+loss3

        if inference:
            output = (logits1, logits2, prob) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if not return_dict:

            logits = []
            for (i, prob_sentence) in enumerate(prob):
                if prob_sentence[0] <= 0.5:
                    logits.append(logits1[i].tolist())
                else:
                    logits.append(logits2[i].tolist())
            logits = torch.Tensor(logits).cuda()

            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits1,
            hidden_states=outputs[0],
            attentions=outputs.attentions,
        )



class BertForSequenceClassificationAdvBase(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 1)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)
        """
        print('prob: \n')
        print(prob)
        print('logits1: \n')
        print(logits1)
        print('logits2: \n')
        print(logits2)
        print('logits: \n')
        print(logits)
        """
        loss = None
        if labels is not None:
            attack_labels, orig_labels = real_labels(labels)
            loss_fct1 = CrossEntropyLoss()
            active_logits = logits1.view(-1, 2)
            active_labels = orig_labels.view(-1)
            loss1 = loss_fct1(active_logits, active_labels)
            loss = loss1

        if not return_dict:
            output = (logits1,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits1,
            hidden_states=outputs[0],
            attentions=outputs.attentions,
        )



class BertForSequenceClassificationAdv(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.classifier3 = nn.Linear(config.hidden_size, 1)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)
        logits3 = self.classifier3(pooled_output)
        prob = torch.sigmoid(logits3)
        logits = logits1.mul(prob) + logits2.mul(1 - prob)
        """
        print('prob: \n')
        print(prob)
        print('logits1: \n')
        print(logits1)
        print('logits2: \n')
        print(logits2)
        print('logits: \n')
        print(logits)
        """
        loss = None
        if labels is not None:
            attack_labels, orig_labels = real_labels(labels)
            loss_fct1 = CrossEntropyLoss()
            loss1 = loss_fct1(logits.view(-1, self.num_labels), orig_labels.view(-1))
            loss_fct2 = nn.BCEWithLogitsLoss()
            loss2 = loss_fct2(logits3.view(-1), attack_labels.float().view(-1))
            loss3 = 1/torch.norm(logits1-logits2)
            loss = loss1+loss2+loss3

        if not return_dict:
            output = (logits,prob) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs[0],
            attentions=outputs.attentions,
        )

class BertForSequenceClassificationAdvNew(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.classifier3 = nn.Linear(config.hidden_size, 1)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)
        logits3 = self.classifier3(pooled_output)
        prob = torch.sigmoid(logits3)
        logits = logits1.mul(prob) + logits2.mul(1 - prob)
        """
        print('prob: \n')
        print(prob)
        print('logits1: \n')
        print(logits1)
        print('logits2: \n')
        print(logits2)
        print('logits: \n')
        print(logits)
        """
        loss = None
        if labels is not None:
            attack_labels, orig_labels = real_labels(labels)
            loss_fct1 = CrossEntropyLoss()
            loss1 = loss_fct1(logits.view(-1, self.num_labels), orig_labels.view(-1))
            loss_fct2 = nn.BCEWithLogitsLoss()
            loss2 = loss_fct2(logits3.view(-1), attack_labels.float().view(-1))
            loss3 = 1/torch.norm(logits1-logits2)
            loss = loss1+loss2+loss3

        if not return_dict:
            output = (logits,prob) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs[0],
            attentions=outputs.attentions,
        )


class BertForSequenceClassificationRecover(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.classifier3 = nn.Linear(config.hidden_size, 1)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)
        logits3 = self.classifier3(pooled_output)
        prob = torch.sigmoid(logits3)
        logits = logits1.mul(prob) + logits2.mul(1 - prob)
        """
        print('prob: \n')
        print(prob)
        print('logits1: \n')
        print(logits1)
        print('logits2: \n')
        print(logits2)
        print('logits: \n')
        print(logits)
        """
        loss = None
        if labels is not None:
            attack_labels, orig_labels = real_labels(labels)
            loss_fct1 = CrossEntropyLoss()
            loss1 = loss_fct1(logits.view(-1, self.num_labels), orig_labels.view(-1))
            loss_fct2 = nn.BCEWithLogitsLoss()
            loss2 = loss_fct2(logits3.view(-1), attack_labels.float().view(-1))
            loss3 = 1/torch.norm(logits1-logits2)
            loss = loss1+loss2+loss3

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class PrLMForClassificationSvd(AutoModelForSequenceClassification):
    def __init__(self):
        super(PrLMForClassificationSvd, self).__init__()

    def from_pretrained_svd(pretrained_model_name_or_path, from_tf, config, cache_dir):
        model = AutoModelForSequenceClassification.from_pretrained(
            # pretrained_model_name_or_path='bert-base-uncased',
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            from_tf=from_tf,
            config=config,
            cache_dir=cache_dir)
        # model.bert = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        print(config.num_labels)
        if config.svd_reserve_size != 0:
            u, s, v = torch.svd(model.bert.embeddings.word_embeddings.weight.data)
            s_new = torch.zeros([len(s)])
            for i in range(config.svd_reserve_size):
                s_new[i] = s[i]
            weight_new = torch.matmul(torch.matmul(u, torch.diag_embed(s_new)), v.transpose(-2, -1))
            model.bert.embeddings.word_embeddings.weight.data.copy_(weight_new)
            #model.bert.embeddings.word_embeddings.requires_grad_(False)
        return model


class PrLMForClassificationSvdElectra(AutoModelForSequenceClassification):
    def __init__(self):
        super(PrLMForClassificationSvdElectra, self).__init__()

    def from_pretrained_svd(pretrained_model_name_or_path, from_tf, config, cache_dir):
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            from_tf=from_tf,
            config=config,
            cache_dir=cache_dir, )
        if config.svd_reserve_size != 0:
            u, s, v = torch.svd(model.electra.embeddings.word_embeddings.weight.data)
            s_new = torch.zeros([len(s)])
            for i in range(config.svd_reserve_size):
                s_new[i] = s[i]
            weight_new = torch.matmul(torch.matmul(u, torch.diag_embed(s_new)), v.transpose(-2, -1))
            print(model.electra.embeddings.word_embeddings.weight.data-weight_new)
            model.electra.embeddings.word_embeddings.weight.data.copy_(weight_new)
        return model

class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output