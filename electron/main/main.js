const { app, BrowserWindow, session } = require('electron');
const path = require('path');

async function create_window() {
  const win = new BrowserWindow({
    width: 1920,
    height: 1080,
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  if (!app.isPackaged) {
    // Load webgpu extension in third-party directory
    console.log(app.getAppPath())
    const extension_path = path.join(app.getAppPath(), 'electron/third_party/webgpu_inspector');
    await session.defaultSession.loadExtension(extension_path, { allowFileAccess: true });
    win.loadFile(path.join(app.getAppPath(), 'dist/index.html'));
  } else {
    const indexPath = path.join('file://', app.getAppPath(), 'dist', 'index.html');
    win.loadURL(indexPath);
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
