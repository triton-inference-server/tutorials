# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import argparse
import json
import shutil
import subprocess
from transformers import pipeline, AutoTokenizer


class ServerBuilder:
    def __init__(self, gen_model, class_model, gen_max_len):
        self.gen_model_full = gen_model
        self.gen_model_name = gen_model.split("/")[-1]
        self.class_model_full = class_model
        self.class_model_name = class_model.split("/")[-1]
        self.max_len = gen_max_len

    def ModelExists(self) -> bool:
        """
        Check that the target model exists.
        """
        try:
            generator_test = AutoTokenizer.from_pretrained(self.gen_model_full)
            classifier_test = AutoTokenizer.from_pretrained(self.class_model_full)
        except Exception as e:
            print("**ERROR**")
            print("Model load error: " + str(e))
            return False
        return True

    def CreateDockerFile(self):
        """
        Create a dockerfile to run the server.
        """
        self.ResetDockerFile()  # Clear previous configuration in case of a re-run
        file_params = []
        file_params.append("FROM nvcr.io/nvidia/tritonserver:23.08-py3\n")
        file_params.append("RUN pip install protobuf==3.20.3\n")
        file_params.append("RUN pip install transformers==4.33.2\n")
        file_params.append("RUN pip install sentencepiece==0.1.99\n")
        file_params.append("RUN pip install accelerate==0.23.0\n")
        self.AppendToDockerFile(file_params)

    def AppendToDockerFile(self, parameters: list):
        """
        Append instructions to the docker file.
        """
        df = open("Dockerfile", "a")
        for parameter in parameters:
            df.write(parameter)
        df.close()

    def ResetWorkspace(self):
        """
        Reset the workspace to initial conditions.
        """
        self.ResetDockerFile()
        self.ResetModelDirectory()

    def ResetDockerFile(self):
        """
        Clear the docker file and set build stage.
        """
        try:
            os.remove("./Dockerfile")
        except Exception as e:
            pass  # File will not exist on first attempt

    def ResetModelDirectory(self):
        """
        Clear the the model directory in case of retry.
        """
        try:
            shutil.rmtree("./model_repository/")
        except Exception as e:
            pass  # File will not exist on first attempt

    def BuildDockerImage(self):
        """
        Build the docker image.
        """
        subprocess.run("docker build -t triton_transformer_server .", shell=True)

    def RunDockerImage(self):
        """
        Run the docker image.
        """
        self.server_proc = subprocess.run(
            "docker run --gpus all -it --rm -p 8000:8000 --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/model_repository:/opt/tritonserver/model_repository triton_transformer_server tritonserver --model-repository=model_repository",
            shell=True,
        )

    def CreateModelDirectory(self):
        """
        Create the model directory complete with model and config.pbtxt
        """
        models = [self.gen_model_name, self.class_model_name]
        for model in models:
            config_path = "model_repository/" + model
            model_repo_path = config_path + "/1"
            os.makedirs(model_repo_path)
            is_gen_model = (model == self.gen_model_name)
            self.GenerateConfigFile(config_path, is_gen_model)
            if is_gen_model:
                shutil.copy(
                    "./base_text_generation_model.py", (model_repo_path + "/model.py")
                )
            else:
                shutil.copy(
                    "./base_text_classification_model.py",
                    (model_repo_path + "/model.py"),
                )
        # Precompile so we can delete files later. Otherwise docker creates
        # files as root.
        subprocess.run("python -m compileall ./model_repository/", shell=True)

    def GenerateConfigFile(self, config_path, is_gen_model):
        file_path = config_path + "/config.pbtxt"
        with open(file_path, "w+") as fp:
            fp.write('backend: "python"\n')
            if is_gen_model:
                self.AddModelParameter("huggingface_model", self.gen_model_full, fp)
                self.AddModelParameter("max_length", self.max_len, fp)
            else:
                self.AddModelParameter("huggingface_model", self.class_model_full, fp)

    def AddModelParameter(self, key, value, fp):
        fp.write("parameters: {\n")
        fp.write("\tkey: " + '"' + key + '"' + ",\n")
        fp.write("\tvalue: {string_value: " + '"' + value + '"' + "}\n")
        fp.write("}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen-model",
        type=str,
        required=False,
        default="facebook/opt-125m",
        help="Path to the hugging face hosted model suited for text generation (e.g., facebook/opt-125m)",
    )
    parser.add_argument(
        "--class-model",
        type=str,
        required=False,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Path to the hugging face hosted model suited for text classification (e.g., distilbert-base-uncased-finetuned-sst-2-english)",
    )
    parser.add_argument(
        "--gen-max-len",
        type=str,
        required=False,
        default="10",
        help="Parameter for the text generator. Specifices the max length of the generated output",
    )
    args = parser.parse_args()
    builder = ServerBuilder(args.gen_model, args.class_model, args.gen_max_len)
    if not builder.ModelExists():
        exit(1)
    builder.ResetWorkspace()
    builder.CreateModelDirectory()
    builder.CreateDockerFile()
    builder.BuildDockerImage()
    builder.RunDockerImage()
