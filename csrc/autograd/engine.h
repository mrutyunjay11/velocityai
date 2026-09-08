#pragma once
#include "csrc/core/tensor.h"
#include "csrc/autograd/function.h"
#include <unordered_map>
#include <vector>
#include <memory>

namespace velocityai {
namespace autograd {

struct Edge {
    std::shared_ptr<Function> function;
    int input_index;
};

// Represents a node in the backward graph
struct Node {
    std::shared_ptr<Function> func;
    std::vector<Edge> next_edges;
};

class Engine {
public:
    static Engine& get() {
        static Engine instance;
        return instance;
    }
    
    // Execute backward pass starting from roots
    void execute(const std::vector<Tensor>& roots, const std::vector<Tensor>& root_grads);
    
private:
    Engine() = default;
};

} // namespace autograd
} // namespace velocityai
