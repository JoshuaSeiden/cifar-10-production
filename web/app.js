document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("fileInput");
    const imgPreview = document.getElementById("imgPreview");
    const predictionsList = document.getElementById("predictionsList");
    const topKSelect = document.getElementById("topKSelect");
    const classesNote = document.getElementById("classesNote");

    let CLASS_NAMES = [];

    // Fetch class names from API and populate dropdown & note
    async function fetchClasses() {
        try {
            const res = await fetch("/classes");
            if (!res.ok) throw new Error("Failed to fetch classes");
            const data = await res.json();
            CLASS_NAMES = data.classes;

            classesNote.textContent = `Available classes: ${CLASS_NAMES.join(", ")}`;

            topKSelect.innerHTML = "";
            for (let i = 1; i <= CLASS_NAMES.length; i++) {
                const opt = document.createElement("option");
                opt.value = i;
                opt.textContent = i;
                topKSelect.appendChild(opt);
            }
            topKSelect.value = 1;
        } catch (err) {
            classesNote.textContent = `Error loading classes: ${err.message}`;
        }
    }

    // Preview selected image
    function previewImage(file) {
        if (!file) {
            imgPreview.style.display = "none";
            predictionsList.innerHTML = "";
            return;
        }
        const objectURL = URL.createObjectURL(file);
        imgPreview.src = objectURL;
        imgPreview.style.display = "block";
    }

    // Run prediction
    async function predictImage() {
        const file = fileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);
        formData.append("top_k", topKSelect.value);

        try {
            const res = await fetch("/predict", { method: "POST", body: formData });
            if (!res.ok) throw new Error("Prediction failed");

            const data = await res.json();
            const predictions = data.predictions;

            predictionsList.innerHTML = "";
            predictions.forEach(p => {
                const li = document.createElement("li");
                li.textContent = `${p.class}: ${(p.prob*100).toFixed(2)}%`;
                predictionsList.appendChild(li);
            });
        } catch (err) {
            predictionsList.innerHTML = `<li style="color:red;">${err.message}</li>`;
        }
    }

    // Event listeners
    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        previewImage(file);
        predictImage(); // Auto-predict on file selection
    });

    topKSelect.addEventListener("change", () => {
        if (fileInput.files.length > 0) predictImage();
    });

    fetchClasses();
});
