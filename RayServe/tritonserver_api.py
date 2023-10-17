import json

from tritonserver import _c as triton_bindings


# @dataclass
class ServerOptions:
    def __init__():
        pass


# @dataclass
class InferenceRequest:
    def __init__(server, model, name, version, dictionary):
        pass


# @dataclass
class Tensor:
    pass

    def __dlpack__(self, stream=None):
        pass

    def __dlpack_device__(self):
        pass


class TritonServer:
    class Options:
        pass

    def __init__(self, options):
        pass

    def start(self):
        options = triton_bindings.TRITONSERVER_ServerOptions()
        options.set_model_repository_path("/workspace/models")
        options.set_model_control_mode(
            triton_bindings.TRITONSERVER_ModelControlMode.POLL
        )
        options.set_log_verbose(0)
        options.set_exit_timeout(5)
        self._server = triton_bindings.TRITONSERVER_Server(options)

    def model(self, name, version):
        pass

    def model_index(self, ready=False):
        models = json.loads(self._server.model_index(ready).serialize_to_json())
        print(models)
        return [
            Model(self, model["name"], int(model["version"]), model["state"])
            for model in models
        ]

    #        return json.loads(self._server.model_index(ready).serialize_to_json())

    def load_model():
        pass

    def unload_model():
        pass

    def stop():
        pass


class Model:
    def __init__(
        self, server: TritonServer, name: str, version: int, state: str = None
    ):
        self._name = name
        self._version = version
        self._server = server._server
        self._state = state

    def inference_request():
        pass

    def infer_async(inference_request):
        pass

    def metadata():
        pass

    def ready(self):
        return self._server.model_is_ready(self._name, self._version)

    def batch_properties():
        pass

    def __str__(self):
        return "%s" % (
            {"name": self._name, "version": self._version, "state": self._state}
        )
