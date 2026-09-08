#include "csrc/autograd/function.h"

namespace velocityai {
namespace autograd {

class AddBackward : public Function {
public:
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override {
        // gradient of addition is 1 * grad_outputs for both inputs
        return {grad_outputs[0], grad_outputs[0]};
    }
};

class MatMulBackward : public Function {
public:
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override {
        auto grad_out = grad_outputs[0];
        auto x = saved_tensors_[0];
        auto w = saved_tensors_[1];
        
        // dL/dx = dL/dout @ w^T
        // dL/dw = x^T @ dL/dout
        // We will compute this in Python for the prototype to avoid cyclic dependencies.
        return {grad_out, grad_out}; 
    }
};

} // namespace autograd
} // namespace velocityai
