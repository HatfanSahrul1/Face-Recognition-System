const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const toggleBtn = document.getElementById('toggleCameraBtn');
const testBackendBtn = document.getElementById('testBackendBtn');
const registerBtn = document.getElementById('registerBtn');
const verifyBtn = document.getElementById('verifyBtn');
const registerNameInput = document.getElementById('registerName');
const resultDiv = document.getElementById('result');
const cameraStatus = document.getElementById('cameraStatus');
const cameraStatusText = document.getElementById('cameraStatusText');

let stream = null;

// Helper untuk update status kamera
function updateCameraUI() {
    if (stream) {
        cameraStatus.className = 'status-indicator camera-on';
        cameraStatusText.innerText = 'Kamera menyala';
        toggleBtn.innerText = 'Matikan Kamera';
    } else {
        cameraStatus.className = 'status-indicator camera-off';
        cameraStatusText.innerText = 'Kamera mati';
        toggleBtn.innerText = 'Hidupkan Kamera';
    }
}

// Fungsi toggle kamera
async function toggleCamera() {
    if (stream) {
        stopCamera();
    } else {
        await startCamera();
    }
    updateCameraUI();
}

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (err) {
        console.error(err);
        setResult('Gagal mengakses kamera: ' + err.message, 'error');
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
    }
}

// Capture gambar dari video
function captureImage() {
    if (!stream) {
        setResult('Kamera tidak aktif', 'error');
        return null;
    }
    canvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
    return canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
}

// Set hasil di result box
function setResult(msg, type = 'info') {
    resultDiv.innerText = msg;
    resultDiv.className = 'result-box';
    if (type === 'success') resultDiv.classList.add('success');
    else if (type === 'error') resultDiv.classList.add('error');
}

// Test backend
async function testBackend() {
    try {
        const res = await fetch('http://localhost:8080/health');
        const text = await res.text();
        setResult('Backend: ' + text, 'success');
    } catch (err) {
        setResult('Gagal koneksi ke backend: ' + err.message, 'error');
    }
}

// Daftar wajah
async function registerFace() {
    const name = registerNameInput.value.trim();
    if (!name) {
        setResult('Nama harus diisi', 'error');
        return;
    }
    const imageBase64 = captureImage();
    if (!imageBase64) return;

    try {
        const res = await fetch('http://localhost:8080/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name, image: imageBase64 })
        });
        const result = await res.json();
        if (res.ok) {
            setResult('Pendaftaran berhasil: ' + JSON.stringify(result), 'success');
            registerNameInput.value = '';          // ← kosongkan input
            registerBtn.disabled = true;           // nonaktifkan tombol lagi
        } else {
            setResult('Gagal daftar: ' + (result.error || 'Unknown error'), 'error');
        }
    } catch (err) {
        setResult('Gagal daftar: ' + err.message, 'error');
    }
}

// Verifikasi wajah
async function verifyFace() {
    const imageBase64 = captureImage();
    if (!imageBase64) return;

    try {
        const res = await fetch('http://localhost:8080/verify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageBase64 })
        });
        const result = await res.json();
        if (res.ok) {
            if (result.name) {
                setResult(`Dikenali sebagai: ${result.name} (confidence: ${(result.confidence*100).toFixed(2)}%)`, 'success');
            } else {
                setResult('Wajah tidak dikenali', 'error');
            }
        } else {
            // Tangani error dari backend (spoof, dll)
            setResult('Gagal: ' + (result.error || 'Unknown error'), 'error');
        }
    } catch (err) {
        setResult('Gagal verifikasi: ' + err.message, 'error');
    }
}

// Event listeners
toggleBtn.addEventListener('click', toggleCamera);
testBackendBtn.addEventListener('click', testBackend);
registerBtn.addEventListener('click', registerFace);
verifyBtn.addEventListener('click', verifyFace);

// Enable/disable register button berdasarkan input nama
registerNameInput.addEventListener('input', function() {
    registerBtn.disabled = this.value.trim() === '';
});

// Inisialisasi: coba nyalakan kamera otomatis (opsional)
startCamera().then(() => updateCameraUI()).catch(() => updateCameraUI());