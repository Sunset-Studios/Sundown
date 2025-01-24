const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('config_api', {
    get_config: (file_name) => ipcRenderer.invoke('config:get-config', file_name),
    set_config: (file_name, config) => ipcRenderer.invoke('config:set-config', file_name, config),
});