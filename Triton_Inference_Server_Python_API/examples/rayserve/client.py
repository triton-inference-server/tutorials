import time

import requests

prompt = "a cute cat is dancing on the grass."
input = "%20".join(prompt.split(" "))

request_count = 100
start = time.time()
for i in range(request_count):
    resp = requests.get(f"http://127.0.0.1:8000/imagine?prompt={input}")
end = time.time()
print(f"Images per second: {request_count/(end-start)}")


# with open("output.png", 'wb') as f:
#   f.write(resp.content)
