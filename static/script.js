const box = document.getElementById("message");
const result = document.getElementById("result");
const predictBtn = document.getElementById("predict-btn");
const clearBtn = document.getElementById("clear-btn");

function show(kind, title, detail) {
  result.hidden = false;
  result.className = "result " + kind;
  result.innerHTML = "";
  const t = document.createElement("p");
  t.className = "verdict";
  t.textContent = title;
  result.appendChild(t);
  if (detail) {
    const d = document.createElement("p");
    d.className = "detail";
    d.textContent = detail;
    result.appendChild(d);
  }
}

async function predict() {
  const message = box.value.trim();
  if (!message) {
    show("err", "Empty message", "Please type or paste a message first.");
    return;
  }
  predictBtn.disabled = true;
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });
    const data = await res.json();
    if (!res.ok) {
      show("err", "Could not check message", data.error);
    } else if (data.is_spam) {
      show("spam", "SPAM", "Model's spam probability: " + data.spam_probability + "%");
    } else {
      show("ok", "NOT SPAM", "Model's spam probability: " + data.spam_probability + "%");
    }
  } catch (e) {
    show("err", "Server not reachable", "Check that the Flask app is running.");
  } finally {
    predictBtn.disabled = false;
  }
}

predictBtn.addEventListener("click", predict);
clearBtn.addEventListener("click", () => {
  box.value = "";
  result.hidden = true;
  box.focus();
});
document.querySelectorAll(".example").forEach(a =>
  a.addEventListener("click", e => {
    e.preventDefault();
    box.value = a.dataset.text;
    result.hidden = true;
  })
);
