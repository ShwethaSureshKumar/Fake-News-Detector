/*
document.getElementById('captureBtn').addEventListener('click', async () => {
  const statusDiv = document.getElementById('status');
  
  try {
    // Get current tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // Request capture through background script
    const response = await new Promise((resolve) => {
      chrome.runtime.sendMessage({ type: 'captureTab' }, resolve);
    });
    
    if (!response || !response.dataUrl) {
      throw new Error('Failed to capture screenshot');
    }
    
    // Convert data URL to blob
    const res = await fetch(response.dataUrl);
    const blob = await res.blob();
    
    // Send to server
    statusDiv.textContent = 'Sending to server...';
    const serverResponse = await sendToServer(blob);
    
    statusDiv.textContent = `Success: ${serverResponse.message}`;
    statusDiv.className = 'success';
  } catch (error) {
    console.error('Capture error:', error);
    statusDiv.textContent = `Error: ${error.message}`;
    statusDiv.className = 'error';
  }
});

async function sendToServer(blob) {
  const SERVER_URL = 'http://localhost:5000/api/capture';
  
  const formData = new FormData();
  formData.append('screenshot', blob, 'screenshot.jpg');
  
  const response = await fetch(SERVER_URL, {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error('Server request failed');
  }
  
  return response.json();
}
*/
document.getElementById('captureBtn').addEventListener('click', async () => {
  const statusDiv = document.getElementById('status');
  
  try {
    // Get current tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // Request capture through background script
    const response = await new Promise((resolve) => {
      chrome.runtime.sendMessage({ type: 'captureTab' }, resolve);
    });
    
    if (!response || !response.dataUrl) {
      throw new Error('Failed to capture screenshot');
    }
    
    // Convert data URL to blob
    const res = await fetch(response.dataUrl);
    const blob = await res.blob();
    
    // Send to server
    statusDiv.textContent = 'Sending to server...';
    const serverResponse = await sendToServer(blob, response.url, response.title);
    
    statusDiv.textContent = `Success: ${serverResponse.message}`;
    statusDiv.className = 'success';
  } catch (error) {
    console.error('Capture error:', error);
    statusDiv.textContent = `Error: ${error.message}`;
    statusDiv.className = 'error';
  }
});

async function sendToServer(blob, url, title) {
  const SERVER_URL = 'http://localhost:5000/api/capture';
  
  const formData = new FormData();
  formData.append('screenshot', blob, 'screenshot.jpg');
  formData.append('url', url);
  formData.append('title', title);
  
  const response = await fetch(SERVER_URL, {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error('Server request failed');
  }
  
  return response.json();
}