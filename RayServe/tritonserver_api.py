from tritonserver import _c as triton_bindings
import json

#@dataclass
class ServerOptions:
    def __init__():
        pass
    
#@dataclass
class InferenceRequest:
    def __init__(server,model,name,version,dictionary):
        pass

#@dataclass
class Tensor:
    pass

    def __dlpack__(self, stream = None):
        pass
    def __dlpack_device__(self):
        pass
    
    
class Model:
    def __init__(self, server, name, version):
        pass

    def inference_request():
        pass
    
    def infer_async(inference_request):
        pass

    def metadata():
        pass
    
    def ready():
        pass

    def batch_properties():
        pass
    


class TritonServer:

    class Options:
        pass
    
    def __init__(self,options):
        pass

    def start(self):
        options = triton_bindings.TRITONSERVER_ServerOptions()
        options.set_model_repository_path("/workspace/models")
        options.set_model_control_mode(triton_bindings.TRITONSERVER_ModelControlMode.POLL)
        options.set_log_verbose(0)
        options.set_exit_timeout(5)
        self._server = triton_bindings.TRITONSERVER_Server(options)
    
    def model(self, name, version):
        pass

    def model_index(self, ready=False):

        models = json.loads(self._server.model_index(ready).serialize_to_json())

        return [Model(self,model["name"],model["version"]) for model in models]
        
#        return json.loads(self._server.model_index(ready).serialize_to_json())
    
    def load_model():
        pass
    
    def unload_model():
        pass
    
    def stop():
        pass
