# cp -rf ~/python_beta_api/local-tritonbuild/install/lib/libtritonserver.so ./deps/
# cp -rf ~/python_beta_api/local-tritonbuild/install/python/tritonserver-2.42.0.dev0-py3-none-any.whl ./deps/
# cp -rf ~/python_beta_api/local-tritonbuild/install/backends ./deps/
docker build -t rayserve-triton -f Dockerfile .
