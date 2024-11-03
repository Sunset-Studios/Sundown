export function write_file(file_path, content) {
    const url = new URL(`${file_path}`, window.location.href);
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', url.href, false);
    xhr.send(content);
}

export function read_file(file_path) {
    let asset = null;
    try {
        const url = new URL(`${file_path}`, window.location.href);
        
        // Check if file exists
        const check_xhr = new XMLHttpRequest();
        check_xhr.open('HEAD', url.href, false);
        check_xhr.send(null);
        
        if (check_xhr.status === 200) {
          // File exists, now fetch its contents
          const get_xhr = new XMLHttpRequest();
          get_xhr.open('GET', url.href, false);
          get_xhr.send(null);
          
          if (get_xhr.status === 200 && !get_xhr.responseText.includes("<!DOCTYPE html>")) {
              asset = get_xhr.responseText;
          }
        }
    } catch (error) {
        // Network error or other issues, continue silently. Let caller handle null asset return. 
    }
    return asset;
}