import functools
from velocityai.compiler.tracer import trace, get_active_tracer
from velocityai.compiler.passes.operator_fusion import OperatorFusionPass
from velocityai.compiler.passes.dead_code_elim import DeadCodeEliminationPass
from velocityai.compiler.lowering import LoweringEngine
from velocityai.compiler.cache import _GLOBAL_CACHE

def compile(fn):
    @functools.wraps(fn)
    def wrapper(*args):
        # We assume args are tensors for now
        # Shape signature
        input_shapes = [arg.shape for arg in args]
        input_dtypes = [arg.dtype for arg in args]
        
        # Check cache
        cached_fn = _GLOBAL_CACHE.get(fn, input_shapes, input_dtypes)
        if cached_fn is not None:
            return cached_fn(*args)
            
        # 1. Trace the function
        with trace() as tracer:
            tracer_args = []
            for i, arg in enumerate(args):
                tracer_args.append(tracer.add_input(arg, name=f"arg_{i}"))
            
            # The tracer is active, so any velocityai tensor ops will be intercepted
            # Actually, we need our tensor methods to check tracer!
            # Since we haven't wired up tensor methods to tracer yet, we will
            # do it in tensor.py
            output = fn(*args)
            
            # Register outputs
            if isinstance(output, tuple):
                tracer.set_outputs(output)
            else:
                tracer.set_outputs([output])
                
        graph = tracer.graph
        
        # 2. Optimization Passes
        graph = OperatorFusionPass().run(graph)
        graph = DeadCodeEliminationPass().run(graph)
        
        # 3. Lowering
        lowering = LoweringEngine()
        compiled_fn = lowering.lower(graph)
        
        # 4. Cache
        _GLOBAL_CACHE.put(fn, input_shapes, input_dtypes, compiled_fn)
        
        # Run it for the first time
        return compiled_fn(*args)
        
    return wrapper
