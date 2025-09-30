const BACKEND_URL = "https://shell-training-edunet-07.onrender.com/predict"; // replace with your Render backend URL

async function sendMessage() {
  const msg = document.getElementById("msg").value;

  try {
    const res = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg })
    });

    if (!res.ok) throw new Error("Network response was not ok");

    const data = await res.json();
    document.getElementById("output").innerText = "Prediction: " + data.prediction;

  } catch (err) {
    console.error("Error connecting to backend:", err);
    document.getElementById("output").innerText = "Error connecting to backend";
  }
}
