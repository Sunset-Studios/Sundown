import { log, warn, error } from "../utility/logging.js";

export class ConfigDB {
    static db_name = 'ConfigDatabase';
    static store_name = 'config_store';
    static version = 1;
    static db = null;

    /**
     * Initializes the config database.
     * @param {boolean} delete_db - Whether to delete the database if it already exists
     * @returns {Promise<void>}
     */
    static async init(delete_db = false) {
        if (delete_db) {
            return new Promise((resolve, reject) => {
                const delete_request = indexedDB.deleteDatabase(this.db_name);
                
                delete_request.onerror = () => {
                    reject(new Error('Failed to delete database'));
                };

                delete_request.onsuccess = () => {
                    this._open_db().then(resolve).catch(reject);
                };
            });
        }
        return this._open_db();
    }

    /**
     * Opens the config database.
     * @returns {Promise<void>}
     */
    static async _open_db() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.db_name, this.version);

            request.onerror = () => {
                reject(new Error('Failed to open database'));
            };

            request.onsuccess = (event) => {
                this.db = event.target.result;
                resolve();
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains(this.store_name)) {
                    db.createObjectStore(this.store_name);
                }
            };
        });
    }

    /**
     * Sets the config for the specified file.
     * 
     * @param {string} file_name - The name of the config file
     * @param {any} config - The config object to set
     * @returns {Promise<void>}
     */
    static async set_config(file_name, config) {
        if (!this.db) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.store_name], 'readwrite');
            const store = transaction.objectStore(this.store_name);
            
            const request = store.put(config, file_name);
            
            request.onsuccess = () => resolve();
            request.onerror = () => reject(new Error('Failed to set config'));
        });
    }

    /**
     * Gets the config for the specified file.
     * 
     * @param {string} file_name - The name of the config file
     * @returns {Promise<any>} The config object, or undefined if it doesn't exist
     */
    static async get_config(file_name) {
        if (!this.db) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.store_name], 'readonly');
            const store = transaction.objectStore(this.store_name);
            
            const request = store.get(file_name);
            
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(new Error('Failed to get config'));
        });
    }

    /**
     * Sets a config property in the specified file.
     * 
     * @param {string} file_name - The name of the config file
     * @param {string} property_path - The path to the property (e.g. 'parent.child.property')
     * @param {any} value - The value to set for the property
     * @returns {Promise<void>}
     */
    static async set_config_property(file_name, property_path, value) {
        if (!this.db) await this.init();

        const config = await this.get_config(file_name) || {};
        
        // Handle nested properties using path (e.g., 'parent.child.property')
        const parts = property_path.split('.');
        let current = config;
        
        for (let i = 0; i < parts.length - 1; i++) {
            if (!(parts[i] in current)) {
                current[parts[i]] = {};
            }
            current = current[parts[i]];
        }
        
        current[parts[parts.length - 1]] = value;

        return this.set_config(file_name, config);
    }

    /**
     * Gets a config property from the specified file.
     * 
     * @param {string} file_name - The name of the config file
     * @param {string} property_path - The path to the property (e.g. 'parent.child.property')
     * @returns {Promise<any>} The value of the property, or undefined if it doesn't exist
     */
    static async get_config_property(file_name, property_path) {
        const config = await this.get_config(file_name);
        if (!config) return undefined;

        // Handle nested properties
        const parts = property_path.split('.');
        let current = config;
        
        for (const part of parts) {
            if (current === undefined || current === null) return undefined;
            current = current[part];
        }
        
        return current;
    }

    /**
     * Tests whether a config property exists at the specified path
     * 
     * @param {string} file_name - The name of the config file
     * @param {string} property_path - The path to the property (e.g. 'parent.child.property')
     * @returns {Promise<boolean>} True if the property exists, false otherwise
     */
    static async has_config_property(file_name, property_path) {
        const config = await this.get_config(file_name);
        if (!config) return false;

        // Handle nested properties
        const parts = property_path.split('.');
        let current = config;
        
        for (const part of parts) {
            if (current === undefined || current === null || !(part in current)) {
                return false;
            }
            current = current[part];
        }
        
        return true;
    }
}

export class ConfigSync {
    /**
     * Save the current configuration to the dev server.
     * @param {string} file_name 
     */
    static async save_to_server(file_name) {
        const config = await ConfigDB.get_config(file_name);
        try {
            if (window.config_api) {
                // Electron environment - use IPC
                const result = await window.config_api.set_config(file_name, config);
                if (!result.success) {
                    throw new Error(result.error || 'Failed to save config via IPC.');
                }
            } else {
                // Browser environment - use fetch API
                const response = await fetch('/sundown/dev/save-config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file_name, config }),
                });

                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.error || 'Failed to save config to server.');
                }
            }
        } catch (err) {
            error('Error saving config:', err);
            throw err;
        }
    }

    /**
     * Load the configuration from the dev server.
     * @param {string} file_name 
     */
    static async load_from_server(file_name) {
        try {
            if (window.config_api) {
                // Electron environment - use IPC
                const config = await window.config_api.get_config(file_name);
                if (config) {
                    await ConfigDB.set_config(file_name, config);
                }
            } else {
                // Browser environment - use fetch API
                const response = await fetch(`/sundown/dev/get-config?file_name=${encodeURIComponent(file_name)}`);
                if (!response.ok) {
                    throw new Error('Failed to load config from server.');
                }
                const config = await response.json();
                await ConfigDB.set_config(file_name, config);
            }
        } catch (err) {
            error('Error loading config:', err);
            throw err;
        }
    }
}
