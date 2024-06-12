# Container Generation

The files in this folder are intended to be used to create the Triton Server container image.

Run the following command to create a Triton Server container image.

```bash
docker build --file ./server.containerfile --tag <image_name_here> .
```

Run the following command to create a client load generation container image.

```bash
docker build --file ./client.containerfile --tag <image_name_here> .
```
