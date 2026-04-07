use serde::Serialize;
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FilesystemEntry {
    kind: String,
    name: String,
    path: String,
    size_bytes: u64,
    modified_ms: Option<u64>,
    extension: Option<String>,
    child_count: Option<usize>,
    is_symlink: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DirectoryListing {
    kind: String,
    name: String,
    path: Option<String>,
    parent_path: Option<String>,
    size_bytes: u64,
    modified_ms: Option<u64>,
    child_count: usize,
    children: Vec<FilesystemEntry>,
    virtual_root: bool,
}

fn to_path_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn modified_ms(metadata: &fs::Metadata) -> Option<u64> {
    metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis().min(u64::MAX as u128) as u64)
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| path.to_string_lossy().to_string())
}

fn path_parent_string(path: &Path) -> Option<String> {
    path.parent()
        .map(to_path_string)
        .filter(|parent| !parent.is_empty())
}

fn build_entry_from_path(path: &Path, custom_name: Option<String>) -> Result<FilesystemEntry, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("Failed to read metadata for '{}': {error}", path.display()))?;
    let file_type = metadata.file_type();
    let is_directory = file_type.is_dir();

    Ok(FilesystemEntry {
        kind: if is_directory { "directory" } else { "file" }.to_string(),
        name: custom_name.unwrap_or_else(|| display_name(path)),
        path: to_path_string(path),
        size_bytes: if is_directory { 0 } else { metadata.len() },
        modified_ms: modified_ms(&metadata),
        extension: if is_directory {
            None
        } else {
            path.extension()
                .map(|extension| extension.to_string_lossy().to_string())
                .filter(|extension| !extension.is_empty())
        },
        child_count: None,
        is_symlink: file_type.is_symlink(),
    })
}

fn sort_entries(entries: &mut [FilesystemEntry]) {
    entries.sort_by(|left, right| {
        let left_priority = if left.kind == "directory" { 0 } else { 1 };
        let right_priority = if right.kind == "directory" { 0 } else { 1 };

        left_priority
            .cmp(&right_priority)
            .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
    });
}

fn collect_root_shortcuts() -> Vec<FilesystemEntry> {
    let mut seen = HashSet::new();
    let mut entries = Vec::new();

    let mut push_unique = |path: PathBuf, label: Option<&str>| {
        let canonical = to_path_string(&path);
        if !seen.insert(canonical) {
            return;
        }

        if let Ok(entry) = build_entry_from_path(&path, label.map(str::to_string)) {
            entries.push(entry);
        }
    };

    if let Some(home) = dirs::home_dir() {
        push_unique(home, Some("Home"));
    }
    if let Some(desktop) = dirs::desktop_dir() {
        push_unique(desktop, Some("Desktop"));
    }
    if let Some(documents) = dirs::document_dir() {
        push_unique(documents, Some("Documents"));
    }
    if let Some(downloads) = dirs::download_dir() {
        push_unique(downloads, Some("Downloads"));
    }

    if cfg!(target_os = "windows") {
        for drive_letter in 'A'..='Z' {
            let drive = format!("{drive_letter}:\\");
            let drive_path = PathBuf::from(&drive);
            if drive_path.exists() {
                push_unique(drive_path, Some(&format!("Drive {drive_letter}:")));
            }
        }
    } else {
        push_unique(PathBuf::from("/"), Some("Root"));
    }

    sort_entries(&mut entries);
    entries
}

fn list_virtual_root() -> DirectoryListing {
    let children = collect_root_shortcuts();

    DirectoryListing {
        kind: "directory".to_string(),
        name: "Computer".to_string(),
        path: None,
        parent_path: None,
        size_bytes: 0,
        modified_ms: None,
        child_count: children.len(),
        children,
        virtual_root: true,
    }
}

fn list_directory(path: &Path) -> Result<DirectoryListing, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("Failed to read directory '{}': {error}", path.display()))?;

    if !metadata.is_dir() {
        return Err(format!("'{}' is not a directory.", path.display()));
    }

    let mut children = Vec::new();
    let read_dir = fs::read_dir(path)
        .map_err(|error| format!("Failed to read directory '{}': {error}", path.display()))?;

    for child in read_dir {
        let child = child.map_err(|error| {
            format!(
                "Failed to inspect an entry inside '{}': {error}",
                path.display()
            )
        })?;

        if let Ok(entry) = build_entry_from_path(&child.path(), None) {
            children.push(entry);
        }
    }

    sort_entries(&mut children);

    Ok(DirectoryListing {
        kind: "directory".to_string(),
        name: display_name(path),
        path: Some(to_path_string(path)),
        parent_path: path_parent_string(path),
        size_bytes: 0,
        modified_ms: modified_ms(&metadata),
        child_count: children.len(),
        children,
        virtual_root: false,
    })
}

#[tauri::command]
fn filesystem_list_directory(path: Option<String>) -> Result<DirectoryListing, String> {
    match path {
        Some(path) if !path.trim().is_empty() => list_directory(Path::new(&path)),
        _ => Ok(list_virtual_root()),
    }
}

#[tauri::command]
fn filesystem_open_path(path: String) -> Result<(), String> {
    open::that_detached(path).map_err(|error| format!("Failed to open path: {error}"))
}

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            filesystem_list_directory,
            filesystem_open_path
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
