const { app, BrowserWindow, session, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs/promises');

const CONFIG_BASE_PATH = path.join(app.getAppPath(), 'assets/config');

const is_dev = !app.isPackaged;

ipcMain.handle('config:get-config', async (event, file_name) => {
    try {
        const filePath = path.join(CONFIG_BASE_PATH, `${file_name}.json`);
        const data = await fs.readFile(filePath, 'utf8');
        return JSON.parse(data);
    } catch (err) {
        if (err.code === 'ENOENT') return null;
        throw err;
    }
});

ipcMain.handle('config:set-config', async (event, file_name, config) => {
    try {
        const filePath = path.join(CONFIG_BASE_PATH, `${file_name}.json`);
        await fs.writeFile(filePath, JSON.stringify(config, null, 2), 'utf8');
        return { success: true };
    } catch (err) {
        return { success: false, error: err.message };
    }
});

async function create_window() {
  const win = new BrowserWindow({
    width: 1920,
    height: 1080,
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: true,
      crossOriginIsolated: true,
      preload: path.join(__dirname, '../preload/index.mjs'),
    }
  });

  // Enable SharedArrayBuffer
  win.webContents.session.webRequest.onHeadersReceived((details, callback) => {
      details.responseHeaders['Cross-Origin-Opener-Policy'] = ['same-origin'];
      details.responseHeaders['Cross-Origin-Embedder-Policy'] = ['require-corp'];
      callback({ responseHeaders: details.responseHeaders });
  });

  if (is_dev) {
    // Load webgpu extension in third-party directory
    const extension_path = path.join(app.getAppPath(), 'electron/third_party/webgpu_inspector');
    await session.defaultSession.loadExtension(extension_path, { allowFileAccess: true });
    win.loadFile(path.join(app.getAppPath(), 'dist/index.html'));
  } else {
    const index_path = path.join('file://', app.getAppPath(), 'dist', 'index.html');
    win.loadURL(index_path);
  }
}

app.whenReady().then(async () => {
  await create_window();
  app.on('activate', async () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      await create_window();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
