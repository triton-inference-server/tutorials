import requests
from fastapi import FastAPI
from ray import serve
import tritonserver
import os
from pprint import pprint

# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()

S3_BUCKET_URL = None

if "S3_BUCKET_URL" in os.environ:
    S3_BUCKET_URL = os.environ["S3_BUCKET_URL"]


def _print_heading(message):
    print("")
    print(message)
    print("-" * len(message))


@serve.deployment
@serve.ingress(app)
class TritonDeployment:
    def __init__(self):
        self._triton_server = tritonserver

        if S3_BUCKET_URL is not None:
            model_repository = S3_BUCKET_URL
        else:
            model_repository = "/workspace/models"

        self._triton_server = tritonserver.Server(
            model_repository=model_repository,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_info=False,
        )
        self._triton_server.start(wait_until_ready=True)

        _print_heading("Triton Server Started")
        _print_heading("Metadata")
        pprint(self._triton_server.metadata())
        _print_heading("Models")
        pprint(self._triton_server.models())

    @app.get("/test")
    def test(self, name: str) -> str:
        if not self._triton_server.model("test").ready():
            self._test_model = self._triton_server.load("test")

        output = []
        for response in self._test_model.infer(inputs={"string_input": [[name]]}):
            output.append(response.outputs["string_output"].to_string_array()[0][0])

        return "".join(output)

    @app.get("/hello")
    def say_hello(self, name: str) -> str:
        return f"Hello {name}!"


if __name__ == "__main__":
    # 2: Deploy the deployment.
    serve.run(TritonDeployment.bind(), route_prefix="/")

    # 3: Query the deployment and print the result.
    print(
        requests.get("http://localhost:8000/test", params={"name": "Theodore"}).json()
    )
    # "Hello Theodore!"
else:
    triton_app = TritonDeployment.bind()
