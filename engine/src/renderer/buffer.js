import { Name } from "../utility/names.js";
import { ResourceCache, CacheTypes } from "./resource_cache.js";

/**
 * Flags for buffer resources in the render graph.
 * @enum {number}
 */
export const BufferFlags = Object.freeze({
  /** No flags */
  None: 0,
  /** Indicates a transient buffer resource */
  Transient: 1,
});

export class Buffer {
    config = null;
    buffer = null;

    // Create a GPU buffer to store the data
    init(context, config) {
        this.config = config;

        let buffer_data = null;

        if (config.raw_data) {
            buffer_data = config.raw_data;
        } else {
            const data = config.data.flat();
            buffer_data = new Float32Array(data.length);
            buffer_data.set(data);
        }

        this.config.size = this.config.size || buffer_data.byteLength;

        this.buffer = context.device.createBuffer({
            label: this.config.name,
            size: this.config.size,
            usage: this.config.usage
        });

        this.write(context, buffer_data);
    }

    destroy(context) {
        if (this.buffer) {
            this.buffer = null;
            ResourceCache.get().remove(CacheTypes.BUFFER, Name.from(this.config.name))
        }
    }

    write(context, data, offset = 0, size = null) {
        const is_array_buffer = ArrayBuffer.isView(data);
        const raw_data = is_array_buffer ? data : data.flat();
        const buffer_data = is_array_buffer ? raw_data : new Float32Array(raw_data.length);
        buffer_data.set(raw_data);
        context.device.queue.writeBuffer(this.buffer, offset, buffer_data, 0, size ?? buffer_data.length);
    }

    write_raw(context, data, offset = 0, size = null) {
        context.device.queue.writeBuffer(this.buffer, offset, data, 0, size ?? data.length);
    }

    read(context, data, data_length, offset = 0, data_offset = 0) {
        // TODO: Support more than just Float32Array
        const buffer_data = new Float32Array(this.buffer.getMappedRange());
        data.set(buffer_data.subarray(offset, offset + data_length), data_offset);
        this.buffer.unmap();
    }

    bind_vertex(encoder, slot = 0) {
        encoder.setVertexBuffer(slot, this.buffer);
    }

    get physical_id() {
        return Name.from(this.config.name);
    }

    static create(context, config) {
        let buffer = ResourceCache.get().fetch(CacheTypes.BUFFER, Name.from(config.name));
        if (!buffer) {
            buffer = new Buffer();
            buffer.init(context, config);
            ResourceCache.get().store(CacheTypes.BUFFER, Name.from(config.name), buffer);
        }
        return buffer;
    }
}


