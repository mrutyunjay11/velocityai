from velocityai.nn.module import Module
import velocityai as vai

class ReLU(Module):
    def forward(self, x):
        return vai.relu(x)
