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
        ResourceCache.get().remove(CacheTypes.BUFFER, Name.from(this.config.name))
        this.buffer = null;
    }

    write(context, data, offset = 0, size = null, data_offset = 0, data_type = Float32Array) {
        const is_array_buffer = ArrayBuffer.isView(data);
        const raw_data = is_array_buffer ? data : data.flat();
        const buffer_data = is_array_buffer ? raw_data : new data_type(raw_data);
        context.device.queue.writeBuffer(this.buffer, offset, buffer_data, data_offset, size ?? buffer_data.length);
    }

    write_raw(context, data, offset = 0, size = null, data_offset = 0) {
        try {
            context.device.queue.writeBuffer(this.buffer, offset, data, data_offset, size ?? data.length);
        } catch (e) {
            console.error('Error writing buffer with offset ', offset, ' and size ', size, ': ', e);
        }
    }

    write_large(context, data, offset = 0) {
        this.buffer.mapAsync(GPUMapMode.WRITE).then(() => {
            const buffer_data = new Float32Array(this.buffer.getMappedRange());
            buffer_data.set(data, offset);
            this.buffer.unmap();
        });
    }

    async read(context, data, data_length, offset = 0, data_offset = 0, data_type = Float32Array) {
        await this.buffer.mapAsync(GPUMapMode.READ);
        if (this.buffer) { // Buffer could have been destroyed while waiting for the map
            const buffer_data = new data_type(this.buffer.getMappedRange());
            data.set(buffer_data.subarray(offset, offset + data_length), data_offset);
            this.buffer.unmap();
        }
    }

    copy_texture(encoder, texture, bytes_per_row) {
        encoder.copyTextureToBuffer(
            { texture: texture.image },
            { buffer: this.buffer, bytesPerRow: bytes_per_row },
            { 
                width: texture.config.width,
                height: texture.config.height,
                depthOrArrayLayers: texture.config.depth
            }
        );
    }

    copy_buffer(encoder, offset, buffer, buffer_offset = 0, size = null) {
        encoder.copyBufferToBuffer(
            this.buffer,
            offset,
            buffer.buffer,
            buffer_offset,
            size ?? this.config.size
        );
    }
    bind_vertex(encoder, slot = 0) {
        encoder.setVertexBuffer(slot, this.buffer);
    }

    get physical_id() {
        return Name.from(this.config.name);
    }

    static create(context, config) {
        let existing_buffer = ResourceCache.get().fetch(CacheTypes.BUFFER, Name.from(config.name));

        if (existing_buffer && config.force) {
            existing_buffer.destroy(context)
            existing_buffer = null;
            config.force = false;
        }

        if (!existing_buffer) {
            existing_buffer = new Buffer();
            existing_buffer.init(context, config);
            ResourceCache.get().store(CacheTypes.BUFFER, Name.from(config.name), existing_buffer);
        }

        return existing_buffer;
    }
}
