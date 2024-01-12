# Triton Inference Server In-Process Python API [BETA]

Starting with release r24.01 Triton Inference Server will include a
Python package enabling developers to embed Triton Inference Server
instances in their Python applications. The in-process Python API is
designed to match the functionality of the in-process C API while
providing a higher level abstraction. At its core in fact the API
relies on a 1:1 python binding of the C API created with pybind-c++
and thus provides all the flexibility and power of the C API with a
simpler to use interface. 

This tutorial repository includes a preview of the API based on the
r23.12 release of Triton.

> [!Note]
> As the API is in BETA please expect some changes as we 
> test out different features and get feedback.
> All feedback is weclome and we look forward to hearing from you!




