import os
import time

import numpy
import requests
import tritonserver
from fastapi import FastAPI
from PIL import Image
from ray import serve
import json

# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()


@serve.deployment(route_prefix="/", ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class TritonDeployment:
    def __init__(self):

        self._triton_server = tritonserver.Server(
            model_repository=["/workspace/models"]
        )
        self._triton_server.start(blocking=True)
        self._metric = tritonserver.Metric(
            tritonserver.MetricFamily(
                tritonserver.MetricKind.COUNTER, "custom_counter", "custom"
            ),
            {"test": "test"},
        )

        self._metric.increment(5)
        print(self._triton_server.metadata())

        #        while not self._triton_server.ready():

        #    time.sleep(0.5)

        self._models = self._triton_server.models
        for (name, version), model in self._models.items():
            print(model)
            print(model.batch_properties())
            print(model.transaction_properties())
            print(model.config())

    @app.get("/test")
    def test(self, text_input: str, fp16_input: float) -> dict:
        test = self._triton_server.get_model("test")
        responses = test.infer(
            inputs={
                "text_input": numpy.array([text_input], dtype=numpy.object_),
                "fp16_input": numpy.array([[fp16_input]], dtype=numpy.float16),
            }
        )
        for response in responses:
            text_output = response.outputs["text_output"].astype(str)[0]
            fp16_output = response.outputs["fp16_output"].astype(float)[0]

        print(response)
        ret_val = {"text_output": text_output, "fp16_output": list(fp16_output)}
        return ret_val

    @app.get("/async_test")
    async def async_test(self, text_input: str, fp16_input: float) -> dict:
        test = self._triton_server.get_model("test")
        responses = test.async_infer(
            inputs={
                "text_input": numpy.array([text_input], dtype=numpy.object_),
                "fp16_input": numpy.array([[fp16_input]], dtype=numpy.float16),
            }
        )
        async for response in responses:
            text_output = response.outputs["text_output"].astype(str)[0]
            fp16_output = response.outputs["fp16_output"].astype(float)[0]

        ret_val = {"text_output": text_output, "fp16_output": list(fp16_output)}
        return ret_val


    @app.get("/classify")
    def classify(self, image_name: str) -> str:
        model = self._triton_server.models["resnet50_libtorch"]

        # print(model.metadata())

        input_ = numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)

        responses = model.infer(inputs={"INPUT__0": input_})
        for response in responses:
            output_ = response.outputs["OUTPUT__0"]
            max_ = numpy.argmax(output_[0])
        #            print(max_)
        return "max: %s" % (max_)

    @app.get("/generate")
    def generate(self, text_input: str) -> str:
        trt_llm = self._triton_server.get_model("ensemble")
        try:
            if not trt_llm.ready():
                return ""
        except:
            return ""

        responses = trt_llm.infer(
            inputs={
                "text_input": numpy.array([[text_input]], dtype=numpy.object_),
                "max_tokens": numpy.array([[100]], dtype=numpy.uint32),
                "stop_words": numpy.array([[""]], dtype=numpy.object_),
                "bad_words": numpy.array([[""]], dtype=numpy.object_),
            }
        )
        result = []
        for response in responses:
            result.append(response.outputs["text_output"].astype(str)[0])
        return "".join(result)

    @app.get("/hello")
    def say_hello(self, name: str) -> str:
        return f"Hello {name}!"

    @app.get("/metrics")
    def metrics(self) -> str:
        self._metric.increment(10)
        return self._triton_server.metrics()

    def __del__(self):
        pass


if __name__ == "__main__":
    # 2: Deploy the deployment.
    serve.run(TritonDeployment.bind())

    # 3: Query the deployment and print the result.
    print(
        requests.get("http://localhost:8000/hello", params={"name": "Theodore"}).json()
    )
    print(
        requests.get(
            "http://localhost:8000/generate", params={"text_input": "Theodore"}
        ).json()
    )
    print(
        requests.get(
            "http://localhost:8000/test",
            params={"text_input": "Theodore", "fp16_input": 0.5},
        ).json()
    )
    # "Hello Theodore!"

    print(
        requests.get(
            "http://localhost:8000/classify",
            params={"image_name": "Theodore"},
        ).json()
    )

    print(
        requests.get(
            "http://localhost:8000/async_test",
            params={"text_input": "Theodore", "fp16_input": 0.5},
        ).json()
    )


    print(requests.get("http://localhost:8000/metrics").json())


else:
    triton_app = TritonDeployment.bind()
