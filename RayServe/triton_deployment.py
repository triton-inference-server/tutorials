import os
import time

import numpy
import requests
import tritonserver_api
from fastapi import FastAPI
from PIL import Image
from ray import serve

# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()


@serve.deployment(route_prefix="/", ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class TritonDeployment:
    def __init__(self):
        #        options = tritonserver_api.Options(model_repository_paths=["/workspace/models"],
        #                                          model_control_mode=tritonserver_api.ModelControlMode.POLL,
        #                                         server_id="hello"
        #                                        )

        # self._triton_server = tritonserver_api.Server(
        #     server_id="hello",
        #     startup_models=["foo"],
        #     model_repository_paths=["/workspace/models"],
        #     strict_model_config=True,
        #     exit_on_error=False,
        #     log_verbose=False,
        #     log_error=True,
        #     log_warn=True,
        # )

        self._triton_server = tritonserver_api.serve(
            model_repository_paths=["/workspace/models"]
        )
        self._metric = tritonserver_api.Metric(
            tritonserver_api.MetricFamily(
                tritonserver_api.MetricKind.COUNTER, "custom_counter", "custom"
            ),
            {"test": "test"},
        )

        self._metric.increment(5)
        print(self._triton_server.metadata())

        #        while not self._triton_server.ready():

        #    time.sleep(0.5)

        self._models = self._triton_server.model_index()
        for model in self._models:
            print(model)
            print(model.batch_properties())
            print(model.transaction_properties())
            print(model.config())

    @app.get("/test")
    def generate(self, text_input: str, fp16_input: float) -> dict:
        test = self._triton_server.model("test")
        responses = test.infer_async(
            inputs={
                "text_input": numpy.array([text_input], dtype=numpy.object_),
                "fp16_input": numpy.array([[fp16_input]], dtype=numpy.float16),
            }
        )
        for response in responses:
            text_output = response.outputs["text_output"].astype(str)[0]
            fp16_output = response.outputs["fp16_output"].astype(float)[0]

        return {"text_output": text_output, "fp16_output": fp16_output}

    @app.get("/classify")
    def classify(self, image_name: str) -> str:
        model = self._triton_server.model("resnet50_libtorch")

        # print(model.metadata())

        input_ = numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)

        responses = model.infer_async(inputs={"INPUT__0": input_})
        for response in responses:
            output_ = response.outputs["OUTPUT__0"]
            max_ = numpy.argmax(output_[0])
        #            print(max_)
        return "max: %s" % (max_)

    @app.get("/generate")
    def generate(self, text_input: str) -> str:
        trt_llm = self._triton_server.model("ensemble")
        try:
            if not trt_llm.ready():
                return ""
        except:
            return ""

        responses = trt_llm.infer_async(
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

    print(requests.get("http://localhost:8000/metrics").json())


else:
    triton_app = TritonDeployment.bind()
