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

async function fetch_file_response(file_path) {
    try {
        const url = new URL(`${file_path}`, window.location.href);
        const response = await fetch(url.href);
        if (!response.ok) {
            return null;
        }
        return response;
    } catch (error) {
        return null;
    }
}

export async function read_file_async(file_path) {
    const response = await fetch_file_response(file_path);
    if (!response) {
        return null;
    }

    const asset = await response.text();
    if (asset.includes("<!DOCTYPE html>")) {
        return null;
    }
    return asset;
}

export async function read_file_bytes_async(file_path) {
    const response = await fetch_file_response(file_path);
    if (!response) {
        return null;
    }

    const content_type = response.headers.get("content-type") || "";
    if (content_type.includes("text/html")) {
        return null;
    }

    return await response.arrayBuffer();
}
