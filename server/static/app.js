const $ = (id) => document.getElementById(id);
const form = $("form"), text = $("text"), cipher = $("cipher"), topK = $("top-k");
const submit = $("submit"), statusLine = $("status"), count = $("count");
let maxChars = Infinity;

const pct = (p) => `${(p * 100).toFixed(1)}%`;

function setStatus(message, isError = false) {
  statusLine.textContent = message;
  statusLine.classList.toggle("error", isError);
}

function updateCount() {
  const n = text.value.length;
  const over = n > maxChars;
  count.textContent = `${n.toLocaleString()} characters` +
    (Number.isFinite(maxChars) ? ` of ${maxChars.toLocaleString()} max` : "");
  count.classList.toggle("error", over);
  submit.disabled = !text.value.trim() || over;
}

function renderGuesses(list, rows) {
  list.replaceChildren(...rows.map(({ label, detail, probability }) => {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "label";
    name.textContent = label;
    if (detail) {
      const code = document.createElement("code");
      code.textContent = detail;
      name.append(" ", code);
    }
    const bar = document.createElement("span");
    bar.className = "bar";
    bar.style.setProperty("--p", probability);
    const value = document.createElement("span");
    value.className = "value";
    value.textContent = pct(probability);
    li.append(name, bar, value);
    return li;
  }));
}

function showOptional(el, message) {
  el.hidden = !message;
  el.textContent = message ?? "";
}

function render(r) {
  const known = r.cipher_source === "user_specified";
  $("cipher-heading").textContent = known ? "Cipher (given)" : "Cipher";
  renderGuesses($("ciphers"), r.top_ciphers.map((c) => ({ label: c.id, probability: c.probability })));
  renderGuesses($("languages"), r.top_languages.map((l) => ({ label: l.name, detail: l.code, probability: l.probability })));
  showOptional($("family-note"), r.family_note);
  showOptional($("length-caveat"), r.length_caveat);
  const chunks = r.n_chunks > 1 ? `, averaged over ${r.n_chunks} chunks` : "";
  const words = r.word_level_applicable
    ? "word boundaries preserved" : "no usable word boundaries (letters-only features)";
  $("summary").textContent = `${r.n_chars.toLocaleString()} characters${chunks}; ${words}; ${r.elapsed_ms} ms`;
  $("results").hidden = false;
}

async function errorMessage(res) {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join("; ");
  } catch { /* not JSON */ }
  return `${res.status} ${res.statusText}`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submit.disabled = true;
  setStatus("Identifying…");
  try {
    const res = await fetch("api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text.value, top_k: Number(topK.value), cipher: cipher.value || null }),
    });
    if (!res.ok) throw new Error(await errorMessage(res));
    render(await res.json());
    setStatus("");
  } catch (err) {
    setStatus(`Prediction failed: ${err.message}`, true);
  } finally {
    updateCount();
  }
});

text.addEventListener("input", updateCount);
text.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !submit.disabled) form.requestSubmit();
});
$("file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  text.value = await file.text();
  updateCount();
});

async function loadMeta() {
  try {
    const res = await fetch("api/meta");
    if (!res.ok) throw new Error(await errorMessage(res));
    const meta = await res.json();
    maxChars = meta.max_chars;
    $("n-ciphers").textContent = meta.ciphers.length;
    $("n-languages").textContent = meta.languages.length;
    for (const id of meta.ciphers) cipher.add(new Option(id, id));
    setStatus("");
  } catch (err) {
    setStatus(`Could not reach the model server: ${err.message}`, true);
  }
  updateCount();
}

loadMeta();
