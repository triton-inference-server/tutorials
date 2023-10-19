import numpy
import requests
from fastapi import FastAPI
from ray import serve
from tritonserver_api import TritonServer

# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()


@serve.deployment(route_prefix="/")
@serve.ingress(app)
class TritonDeployment:
    def __init__(self):
        server_options = TritonServer.Options()
        self._triton_server = TritonServer(server_options)
        self._triton_server.start()

        self._models = self._triton_server.model_index()
        for model in self._models:
            print(model)

        print(self._models[-1].ready())

        model = self._triton_server.model("test")

        print(model.metadata())

        inference_request = model.inference_request()

        inference_request.inputs["text_input"] = numpy.array(
            ["hello", "world"], dtype=numpy.object_
        )
        inference_request.inputs["fp16_input"] = numpy.array(
            [[0.5], [1], [2], [3]], dtype=numpy.float16
        )

        for response in model.infer_async(inference_request):
            for output in response.outputs.items():
                print(output)

    #        print(self._models)

    #        self._simple = self._triton_server.model("simple")

    #       while (not self._simple.ready()):
    #          time.sleep(0.1)

    #     inference_request = InferenceRequest(server,"simple",1)

    #    inference_request = self._simple.inference_request()

    #   infernce_request.inputs["INPUT_0"] = foo

    #  self._simple.infer_async(inputs={"INPUT0":[]},
    #                          priority=1,
    #                         correlation_id=1)

    #        self._simple.infer_async(inference_request)

    # FastAPI will automatically parse the HTTP request for us.
    @app.get("/hello")
    def say_hello(self, name: str) -> str:
        return f"Hello {name}!"


# 2: Deploy the deployment.
serve.run(TritonDeployment.bind())

# 3: Query the deployment and print the result.
print(requests.get("http://localhost:8000/hello", params={"name": "Theodore"}).json())
# "Hello Theodore!"
