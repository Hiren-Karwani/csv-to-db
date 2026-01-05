/* ============================
   CHATBOT WIDGET LOGIC
============================ */
const chatWidget = document.getElementById('chat-widget');
const chatMsgs = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
let isChatOpen = false;

// Only initialize if widget exists on page
if (chatWidget) {
  // Expose these handlers on `window` so inline attributes (onclick/onkeypress)
  // in templates can call them reliably.
  window.toggleChat = function() {
    isChatOpen = !isChatOpen;
    const icon = document.getElementById('toggle-icon');
    if (isChatOpen) {
      chatWidget.classList.add('open');
      if(icon) icon.innerText = '▼';
    } else {
      chatWidget.classList.remove('open');
      if(icon) icon.innerText = '▲';
    }
  };

  window.handleEnter = function(e) {
    if (e.key === 'Enter') window.sendMessage && window.sendMessage();
  };

  function appendMessage(text, sender) {
    if (!chatMsgs) return;
    const div = document.createElement('div');
    div.className = `msg ${sender}`;
    div.innerText = text;
    chatMsgs.appendChild(div);
    chatMsgs.scrollTop = chatMsgs.scrollHeight;
    // persist in sessionStorage so history survives page navigation
    try {
      const hist = JSON.parse(sessionStorage.getItem('chat_history') || '[]');
      hist.push({sender: sender, text: text, ts: Date.now()});
      sessionStorage.setItem('chat_history', JSON.stringify(hist));
    } catch (e) { console.error('Could not persist chat history', e); }
  }

  window.sendMessage = function() {
    if (!chatInput) return;
    const text = chatInput.value.trim();
    if (!text) return;

    appendMessage(text, 'user');
    chatInput.value = '';

    // Show thinking indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'msg bot';
    loadingDiv.innerText = 'Thinking...';
    loadingDiv.id = 'loading-msg';
    chatMsgs.appendChild(loadingDiv);

    fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    })
    .then(res => res.json())
    .then(data => {
      const loader = document.getElementById('loading-msg');
      if (loader) loader.remove();
      // If server returned details or structured info, include a short summary
      const resp = (data && data.response) ? data.response : 'No response';
      appendMessage(resp, 'bot');
      // optionally store provider details in sessionStorage under a separate key
      if (data && data.details) {
        try {
          const meta = JSON.parse(sessionStorage.getItem('chat_meta') || '[]');
          meta.push({question: text, details: data.details, ts: Date.now()});
          sessionStorage.setItem('chat_meta', JSON.stringify(meta));
        } catch (e) { /* ignore */ }
      }
    })
    .catch(err => {
      const loader = document.getElementById('loading-msg');
      if (loader) loader.remove();
      appendMessage("Error reaching the brain 🧠", 'bot');
      console.error(err);
    });
  };
}

// On load, restore chat history from sessionStorage
(function restoreChatHistory(){
  try {
    const hist = JSON.parse(sessionStorage.getItem('chat_history') || '[]');
    if (Array.isArray(hist) && hist.length && chatMsgs) {
      chatMsgs.innerHTML = '';
      hist.forEach(item => {
        const div = document.createElement('div');
        div.className = `msg ${item.sender}`;
        div.innerText = item.text;
        chatMsgs.appendChild(div);
      });
      chatMsgs.scrollTop = chatMsgs.scrollHeight;
    }
  } catch (e) { console.error('Failed to restore chat history', e); }
})();

/* ============================
   UPLOAD & DRAG-DROP LOGIC
============================ */
let files = [];
const input = document.getElementById("csv-files");
const dropArea = document.getElementById("drop-area");
const fileListDisplay = document.getElementById("file-list-display");
const btn = document.getElementById("uploadBtn");
const bar = document.getElementById("progress");
const progressContainer = document.querySelector(".progress-container");
const resultDiv = document.getElementById("result");

// Only initialize upload logic if we are on the upload page (elements exist)
if (dropArea && input) {
    dropArea.onclick = () => input.click();

    input.onchange = e => {
      handleFiles([...e.target.files]);
    };

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
      e.preventDefault();
      e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
      dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false);
    });

    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
      const dt = e.dataTransfer;
      const newFiles = [...dt.files];
      handleFiles(newFiles);
    }

    function handleFiles(newFiles) {
      files = [...files, ...newFiles];
      updateFileList();
      if (files.length > 0 && btn) {
          btn.removeAttribute("disabled");
          btn.style.opacity = "1";
          btn.innerText = `Upload ${files.length} File(s)`;
      }
    }

    function updateFileList() {
        if (!fileListDisplay) return;
        fileListDisplay.innerHTML = "";
        files.forEach((f, index) => {
            const div = document.createElement("div");
            div.className = "file-item";
            div.innerHTML = `
                <span>📄 ${f.name}</span>
                <span style="font-size:0.8rem; color:#aaa;">${(f.size/1024).toFixed(1)} KB</span>
            `;
            fileListDisplay.appendChild(div);
        });
    }

    if (btn) {
        btn.onclick = () => {
          if (files.length === 0) return;

          // UI Updates
          btn.innerText = "Uploading...";
          btn.disabled = true;
          if (progressContainer) progressContainer.style.display = "block";
          if (resultDiv) resultDiv.innerHTML = "";

          let fd = new FormData();
          files.forEach(f => fd.append("files", f));

          let xhr = new XMLHttpRequest();
          xhr.open("POST", "/upload");

          xhr.upload.onprogress = e => {
            if (e.lengthComputable && bar) {
                const percentComplete = (e.loaded / e.total) * 100;
                bar.style.width = percentComplete + "%";
            }
          };

          xhr.onload = () => {
            if (xhr.status === 200) {
                if (bar) bar.style.width = "100%";
                setTimeout(() => {
                    if (resultDiv) resultDiv.innerHTML = xhr.responseText;
                    btn.innerText = "Upload Complete!";
                }, 500);
            } else {
                if (resultDiv) resultDiv.innerHTML = `<p style="color:var(--secondary-color)">❌ Upload failed. Server Error.</p>`;
                btn.innerText = "Try Again";
                btn.disabled = false;
            }
          };

          xhr.onerror = () => {
              if (resultDiv) resultDiv.innerHTML = `<p style="color:var(--secondary-color)">❌ Network Error.</p>`;
              btn.innerText = "Try Again";
              btn.disabled = false;
          };

          xhr.send(fd);
        };
    }
}

