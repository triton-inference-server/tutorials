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
    def __init__(self, gen_models, class_models, gen_max_len):
        self.gen_model_paths, self.gen_model_names = self.ParseModelList(gen_models)
        self.class_model_paths, self.class_model_names = self.ParseModelList(
            class_models
        )
        self.max_len = gen_max_len

    def ParseModelList(self, hf_paths: str) -> list | list:
        """
        Parse command line model lists. Return both the full path
        for each model and their name
        """
        model_paths = []
        model_names = []
        # Remove whitespaces (if any) from command line parameter list
        paths = hf_paths.replace(" ","").split(",")
        for path in paths:
            model_paths.append(path)
            model_names.append(path.split("/")[-1])
        return model_paths, model_names

    def ModelExists(self) -> bool:
        """
        Check that the target model exists.
        """
        try:
            for gen_model in self.gen_model_paths:
                generator_test = AutoTokenizer.from_pretrained(gen_model)
            for class_model in self.class_model_paths:
                classifier_test = AutoTokenizer.from_pretrained(class_model)
        except Exception as e:
            print("**ERROR**")
            print("Model load error: " + str(e))
            return False
        return True

    def ResetWorkspace(self):
        """
        Clear the the model directory in case of retry.
        """
        try:
            shutil.rmtree("./model_repository/")
        except Exception as e:
            pass  # File will not exist on first attempt

    def CreateModelDirectory(self):
        """
        Create the model directory complete with model and config.pbtxt
        """
        model_names = self.gen_model_names + self.class_model_names
        model_paths = self.gen_model_paths + self.class_model_paths
        for idx, model in enumerate(model_names):
            config_path = "model_repository/" + model
            model_repo_path = config_path + "/1"
            os.makedirs(model_repo_path)
            is_gen_model = model in self.gen_model_names
            model_path = model_paths[idx]
            self.GenerateConfigFile(config_path, is_gen_model,model_path)
            if is_gen_model:
                shutil.copy(
                    "./text_generation/base_text_generation_model.py",
                    (model_repo_path + "/model.py"),
                )
            else:
                shutil.copy(
                    "./text_classification/base_text_classification_model.py",
                    (model_repo_path + "/model.py"),
                )
        # Precompile so we can delete files later. Otherwise docker creates
        # files as root.
        subprocess.run("python -m compileall ./model_repository/", shell=True)

    def GenerateConfigFile(self, config_path, is_gen_model, model_path):
        """
        Copy and modify the template config.pbtxt file
        """
        template_config = "text_classification/config.pbtxt"
        if is_gen_model:
            template_config = "text_generation/config.pbtxt"
        target_config = config_path + "/config.pbtxt"

        # Replace default tags with script parameter values
        shutil.copy(template_config, target_config)
        safe_model_path = self.SafeModelPath(model_path)
        subprocess.run("sed -i 's/MODEL_TAG/"+safe_model_path+"/g' ./" + target_config, shell=True)
        if is_gen_model:
            subprocess.run("sed -i 's/MAX_LENGTH_TAG/"+self.max_len+"/g' ./" + target_config, shell=True)

    def SafeModelPath(self, model_path):
        """
        Replace '/' in hf model path (if present) for safe use with 'sed'
        """
        return model_path.replace("/", "\/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen-models",
        type=str,
        required=False,
        default="facebook/opt-125m",
        help="Comma separated list of hugging face hosted transformer models suited for text generation (e.g., 'facebook/opt-125m, gpt2')",
    )
    parser.add_argument(
        "--class-models",
        type=str,
        required=False,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Comma separated list of hugging face hosted transformer models suited for text classification (e.g., distilbert-base-uncased-finetuned-sst-2-english)",
    )
    parser.add_argument(
        "--gen-max-len",
        type=str,
        required=False,
        default="15",
        help="Parameter for the text generator. Specifies the max length of the generated output",
    )
    args = parser.parse_args()
    builder = ServerBuilder(args.gen_models, args.class_models, args.gen_max_len)
    if not builder.ModelExists():
        exit(1)
    builder.ResetWorkspace()
    builder.CreateModelDirectory()
