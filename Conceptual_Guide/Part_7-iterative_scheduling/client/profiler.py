# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import time
from queue import SimpleQueue
from threading import Event

import prettytable
import tqdm
from tritonserver import InferenceRequest, Server


class TimeStampedQueue(SimpleQueue):
    def __init__(self, event) -> None:
        super().__init__()
        self.event = event

    def put(self, item):
        current_time = time.time()
        super().put((current_time, item))
        if item.final:
            self.event.set()


class PerfAnalyzer:
    def __init__(self, server, model_name, concurrency, input_data):
        self._concurrency = concurrency
        self._input_data = input_data
        self._requests = []

        self._measurement_interval_seconds = 5
        self._number_of_intervals = 5
        self._model = server.model(model_name)
        self._prepare_requests()
        self._queues = []
        self._request_timestamps = []

        self._is_profiler_thread_running = True

    def _prepare_requests(self):
        for i in range(self._concurrency):
            input_data = self._input_data[i % len(self._input_data)]
            request = InferenceRequest(
                self._model,
                inputs=input_data["inputs"],
                parameters=input_data["parameters"],
            )
            self._requests.append(request)

    def profile(self):
        self._queues = []
        self._request_timestamps = []
        for i in tqdm.tqdm(range(20)):
            results = []
            current_queues = []
            for i, request in enumerate(self._requests):
                input_data = self._input_data[i % len(self._input_data)]
                current_queue = TimeStampedQueue(Event())
                self._queues.append(current_queue)
                current_queues.append(current_queue)
                request = InferenceRequest(
                    self._model,
                    inputs=input_data["inputs"],
                    parameters=input_data["parameters"],
                    response_queue=current_queue,
                )
                time.sleep(0.05)
                self._request_timestamps.append(time.time())
                results.append(self._model.infer(request))

            for queue in current_queues:
                queue.event.wait()

    def _calculate_response_throughput(self, timestamp_lists):
        timestamps = []
        for timestamp_list in timestamp_lists:
            for timestamp in timestamp_list:
                timestamps.append(timestamp)

        start_time = min(timestamps)
        end_time = max(timestamps)

        total_seconds = end_time - start_time
        return len(timestamps) / total_seconds

    def _calculate_time_to_last_response(self, timestamp_lists):
        time_to_last_response = []
        for i, timestamp_list in enumerate(timestamp_lists):
            time_to_last_response.append(
                timestamp_list[-1] - self._request_timestamps[i]
            )

        return (sum(time_to_last_response) / len(time_to_last_response)) * 1000

    def _calculate_inter_token_latency(self, timestamp_lists):
        inter_token_latencies = []
        for _, timestamp_list in enumerate(timestamp_lists):
            before = None
            for timestamp in timestamp_list:
                if before is None:
                    before = timestamp
                else:
                    inter_token_latencies.append(timestamp - before)
        return sum(inter_token_latencies) / len(inter_token_latencies)

    def get_stats(self):
        timestamp_lists = []
        for queue in self._queues:
            timestamp_lists.append([])
            while queue.qsize() > 0:
                timestamp, _ = queue.get_nowait()
                timestamp_lists[-1].append(timestamp)

        return {
            "response_throughput": self._calculate_response_throughput(timestamp_lists),
            "time_to_last_response": self._calculate_time_to_last_response(
                timestamp_lists
            ),
            "inter_token_latency": self._calculate_inter_token_latency(timestamp_lists),
        }


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Profile a model")
    argument_parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to profile",
        action="extend",
        nargs="+",
    )
    argument_parser.add_argument(
        "--model-repository",
        type=str,
        required=True,
        help="Path to the model repository",
    )
    argument_parser.add_argument(
        "--concurrency", type=int, default=10, help="Number of concurrent requests"
    )
    args = argument_parser.parse_args()
    input_data = [
        {
            "inputs": {"text_input": [["Hello, how are you?"]]},
            "parameters": {"ignore_eos": True, "max_tokens": 32},
        }
    ]

    data = []
    server = Server(model_repository=args.model_repository, log_error=True)
    server.start(wait_until_ready=True)
    for model_name in args.model_name:
        perf_analyzer = PerfAnalyzer(server, model_name, args.concurrency, input_data)
        perf_analyzer.profile()
        stats = perf_analyzer.get_stats()
        data.append(stats)

    table = prettytable.PrettyTable(
        [
            "Model Name",
            "Tokens/sec",
            "Time to last token [TTLT] (ms)",
            "Inter token latency [ITL] (ms)",
        ]
    )
    for i, entry in enumerate(data):
        table.add_row(
            [
                args.model_name[i],
                f"{entry['response_throughput']:.2f}",
                f"{entry['time_to_last_response']:.3f}",
                f"{entry['inter_token_latency']:.3f}",
            ]
        )
    print(table)
