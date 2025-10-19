const predictBtn = document.getElementById("predictBtn");
const imageFileInput = document.getElementById("imageFile");
const previewDiv = document.getElementById("preview");
const predictionsDiv = document.getElementById("predictions");
const classesDiv = document.getElementById("classes");

const API_URL = "http://127.0.0.1:8000";

// -----------------------------
// Fetch valid classes
// -----------------------------
async function fetchClasses() {
    try {
        const response = await fetch(`${API_URL}/classes`);
        if (!response.ok) throw new Error("Failed to fetch classes");
        const data = await response.json();
        classesDiv.innerHTML = data.classes.join(", ");
    } catch (err) {
        classesDiv.innerHTML = `Error loading classes: ${err}`;
    }
}

fetchClasses();

// -----------------------------
// Helpers
// -----------------------------
function createFormData(file) {
    const formData = new FormData();
    formData.append("file", file);
    return formData;
}

// -----------------------------
// Predict button
// -----------------------------
predictBtn.onclick = async () => {
    const files = imageFileInput.files;
    if (!files.length) {
        alert("Please select an image file.");
        return;
    }

    const file = files[0];

    // Show image preview
    const reader = new FileReader();
    reader.onload = e => {
        previewDiv.innerHTML = `<img src="${e.target.result}" alt="preview">`;
    };
    reader.readAsDataURL(file);

    // Call predict endpoint
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            body: createFormData(file)
        });

        if (!response.ok) {
            const err = await response.json();
            predictionsDiv.innerHTML = `<p style="color:red;">Error: ${err.detail}</p>`;
            return;
        }

        const data = await response.json();
        const preds = data.predictions;

        let html = "<h2>Predictions:</h2><table><tr><th>Class</th><th>Probability</th></tr>";
        preds.forEach(p => {
            html += `<tr><td>${p.class}</td><td>${(p.prob*100).toFixed(2)}%</td></tr>`;
        });
        html += "</table>";
        predictionsDiv.innerHTML = html;

    } catch (error) {
        predictionsDiv.innerHTML = `<p style="color:red;">Error: ${error}</p>`;
    }
};
