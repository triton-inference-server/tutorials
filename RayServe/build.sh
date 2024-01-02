#cp -rf ~/ray_triton_integration/tutorials/RayServe/local-tritonbuild/install/lib/libtritonserver.so ./deps/
#cp -rf ~/ray_triton_integration/tutorials/RayServe/local-tritonbuild/install/python/tritonserver-2.42.0.dev0-py3-none-any.whl ./deps/
#cp -rf ~/ray_triton_integration/tutorials/RayServe/local-tritonbuild/install/backends ./deps/
docker build -t rayserve-triton -f Dockerfile .
