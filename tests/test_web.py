import os
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_index_html_served():
    """Requesting '/' should return index.html"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<!DOCTYPE html>" in response.text  # basic sanity check

def test_css_file_served():
    """Requesting '/style.css' should return CSS"""
    response = client.get("/style.css")
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]

def test_js_file_served():
    """Requesting '/app.js' should return JS"""
    response = client.get("/app.js")
    assert response.status_code == 200
    content_type = response.headers["content-type"]
    assert "javascript" in content_type, f"Unexpected content-type: {content_type}"

