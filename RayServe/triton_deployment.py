import requests
from fastapi import FastAPI
from ray import serve
from tritonserver import _c as triton_bindings

# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()

def create_triton_server():
    options = triton_bindings.TRITONSERVER_ServerOptions()
    options.set_model_repository_path("/workspace/models")
    options.set_model_control_mode(triton_bindings.TRITONSERVER_ModelControlMode.POLL)
    options.set_log_verbose(0)
    options.set_exit_timeout(5)
    return triton_bindings.TRITONSERVER_Server(options)

@serve.deployment(route_prefix="/")
@serve.ingress(app)
class TritonDeployment:

    def __init__(self):
        self._triton_server = create_triton_server()
    
    # FastAPI will automatically parse the HTTP request for us.
    @app.get("/hello")
    def say_hello(self, name: str) -> str:
        return f"Hello {name}!"


# 2: Deploy the deployment.
serve.run(TritonDeployment.bind())

# 3: Query the deployment and print the result.
print(requests.get("http://localhost:8000/hello", params={"name": "Theodore"}).json())
# "Hello Theodore!"
