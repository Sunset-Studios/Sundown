import { log, warn, error } from "../utility/logging.js";

const info_name = 'info';
const warn_name = 'warn';
const error_name = 'error';

/**
 * Simple logger for ML operations that supports both console logging

 * and buffered logging for external processing
 */
class Logger {
    _buffer = [];
    _console_enabled = true;
    _buffer_enabled = false;

    /**
     * Enable or disable console logging
     * @param {boolean} enabled 
     */
    set_console_logging(enabled) {
        this._console_enabled = enabled;
    }

    /**
     * Enable or disable buffered logging
     * @param {boolean} enabled 
     */
    set_buffer_logging(enabled) {
        this._buffer_enabled = enabled;
    }

    /**
     * Log a message
     * @param {string} message 
     * @param {string} [level='info'] - Log level (info, warn, error)
     */
    log(message, level = info_name) {
        const formatted_message = `[ML] [${level.toUpperCase()}] ${message}`;
        if (this._console_enabled) {
            switch (level) {
                case warn_name:
                    warn(formatted_message);
                    break;
                case error_name:
                    error(formatted_message);
                    break;
                default:
                    log(formatted_message);
            }
        }
        if (this._buffer_enabled) {
            this._buffer.push({ message: formatted_message, timestamp: Date.now() });
        }
    }

    /**
     * Get and clear the log buffer
     * @returns {Array<{message: string, timestamp: number}>}
     */
    flush() {
        const logs = [...this._buffer];
        this._buffer = [];
        return logs;
    }

    /**
     * Get the current log buffer without clearing it
     * @returns {Array<{message: string, timestamp: number}>}
     */
    peek() {
        return this._buffer;
    }

    /**
     * Pop a log from the buffer
     * @returns {Array<{message: string, timestamp: number}>}
     */
    pop(count = 1) {
        return this._buffer.splice(0, count);
    }

    /**
     * Clear the log buffer
     */
    clear() {
        this._buffer = [];
    }
}

// Export a singleton instance for convenience
export const logger = new Logger(); 