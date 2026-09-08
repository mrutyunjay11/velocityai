from velocityai.nn.module import Module
import velocityai as vai

class CrossEntropyLoss(Module):
    def forward(self, logits, targets):
        return vai.ops.cross_entropy_loss(logits, targets)
