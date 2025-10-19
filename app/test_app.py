import requests
import base64

def test_classify_image():
    url = "http://localhost:8001"
    health = requests.get(f"{url}/health")  # Check health endpoint
    print("Health check response:")
    print(health.json().get("status"))
    assert health.status_code == 200
    print("Sending test request to /classify endpoint...")
    with open("/home/72/streetvision-subnet/received_image.jpg", "rb") as f:
        image_bytes = f.read()
    image_data = base64.b64encode(image_bytes).decode('utf-8')
    response = requests.post(f"{url}/classify", json={"image": image_data})
    print("Response JSON:")
    print(response.json())
    assert response.status_code == 200
    assert "result" in response.json()

if __name__ == "__main__":
    test_classify_image()