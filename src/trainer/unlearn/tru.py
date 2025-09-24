import copy
from trainer.utils import compute_kl_divergence
from trainer.unlearn.base import UnlearnTrainer


class TRU(UnlearnTrainer):
    def __init__(self,alpha=1.0, beta=0.1, gamma = 0.1, retain_loss_type = "NLL", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.retain_loss_type = retain_loss_type
        self.ref_model = None
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)
    
    def _prepare_ref_model(self, model):
        ref_model = copy.deep_copy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        
        return ref_model
    
    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        if self.retain_loss_type == "NLL":
            retain_loss += retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += kl_loss
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )
        return retain_loss
    
    def compute_ga_loss(self, model, inputs):
        outputs = model(**inputs)
        loss = -outputs.loss
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_thinking_inputs = inputs["forget_thinking"]
        forget_thinking_inputs = {
            "input_ids": forget_thinking_inputs["input_ids"],
            "attention_mask": forget_thinking_inputs["attention_mask"],
            "labels": forget_thinking_inputs["labels"],
        }
        forget_thinking_outputs = model(**forget_thinking_inputs)

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }

        forget_inputs = inputs["forget"]
        forget_inputs ={
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"]
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        forget_loss = self.compute_ga_loss(model=model,inputs=forget_inputs)

        loss = self.alpha * forget_thinking_outputs.loss + self.beta * retain_loss + self.gamma * forget_loss

        return (loss, forget_thinking_outputs) if return_outputs else loss
