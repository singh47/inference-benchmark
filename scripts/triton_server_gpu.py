from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig, TritonSecurityConfig  # Add TritonSecurityConfig
import numpy as np
import onnxruntime as ort

# Load ONNX model with GPU
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "/workspace/model_repository/models/1/resnet50-v1-7.onnx",
    sess_options,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# to verify which provider is actually being used
print(f"Available providers: {ort.get_available_providers()}")
print(f"Session using providers: {session.get_providers()}")

@batch
def infer_fn(**inputs):
    data = inputs["data"]
    result = session.run(None, {"data": data})[0]
    return [result]

config = TritonConfig(http_port=8500, grpc_port=8501, metrics_port=8502, allow_vertex_ai=False)
security_config = TritonSecurityConfig(access_token='my-token')  # Disable token requirement

with Triton(config=config, security_config=security_config) as triton:  # Add security_config
    triton.bind(
        model_name="resnet50-v1-7",
        infer_func=infer_fn,
        inputs=[Tensor(name="data", dtype=np.float32, shape=(3, 224, 224))],
        outputs=[Tensor(name="resnetv17_dense0_fwd", dtype=np.float32, shape=(1000,))],
        config=ModelConfig(max_batch_size=1)
    )
    triton.serve()
