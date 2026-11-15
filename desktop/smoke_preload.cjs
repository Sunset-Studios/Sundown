const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("sundown_smoke", {
  send: (message) => ipcRenderer.send("sundown:smoke", message),
});
