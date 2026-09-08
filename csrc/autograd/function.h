#pragma once
#include "csrc/core/tensor.h"
#include <vector>
#include <memory>

namespace velocityai {
namespace autograd {

class Function {
public:
    virtual ~Function() = default;
    
    // forward() is implicitly executed when we dispatch the op, 
    // but backward() computes gradients given grad_outputs.
    virtual std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) = 0;
    
    // Saved tensors for backward pass
    void save_for_backward(const std::vector<Tensor>& tensors) {
        saved_tensors_ = tensors;
    }
    
    const std::vector<Tensor>& saved_tensors() const {
        return saved_tensors_;
    }

protected:
    std::vector<Tensor> saved_tensors_;
};

} // namespace autograd
} // namespace velocityai
