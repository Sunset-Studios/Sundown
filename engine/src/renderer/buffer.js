import { Name } from "../utility/names.js";
import { Renderer } from "./renderer.js";
import { ResourceCache } from "./resource_cache.js";
import { CacheTypes, BufferFlags } from "./renderer_types.js";

export class Buffer {
    config = null;
    buffer = null;

    // Create a GPU buffer to store the data
    init(config) {
        const renderer = Renderer.get();

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

        this.buffer = renderer.device.createBuffer({
            label: this.config.name,
            size: this.config.size,
            usage: this.config.usage
        });

        this.write(buffer_data);
    }

    destroy() {
        ResourceCache.get().remove(CacheTypes.BUFFER, Name.from(this.config.name))
        this.buffer = null;
    }

    write(data, offset = 0, size = null, data_offset = 0, data_type = Float32Array) {
        const renderer = Renderer.get();
        const is_array_buffer = ArrayBuffer.isView(data);
        const raw_data = is_array_buffer ? data : data.flat();
        const buffer_data = is_array_buffer ? raw_data : new data_type(raw_data);
        renderer.device.queue.writeBuffer(this.buffer, offset, buffer_data, data_offset, size ?? buffer_data.length);
    }

    write_raw(data, offset = 0, size = null, data_offset = 0) {
        const renderer = Renderer.get();
        try {
            renderer.device.queue.writeBuffer(this.buffer, offset, data, data_offset, size ?? data.length);
        } catch (e) {
            console.error('Error writing buffer with offset ', offset, ' and size ', size, ': ', e);
        }
    }

    write_large(data, offset = 0) {
        this.buffer.mapAsync(GPUMapMode.WRITE).then(() => {
            const buffer_data = new Float32Array(this.buffer.getMappedRange());
            buffer_data.set(data, offset);
            this.buffer.unmap();
        });
    }

    async read(data, data_length, offset = 0, data_offset = 0, data_type = Float32Array) {
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

    static create(config) {
        let existing_buffer = ResourceCache.get().fetch(CacheTypes.BUFFER, Name.from(config.name));

        if (existing_buffer && config.force) {
            existing_buffer.destroy()
            existing_buffer = null;
            config.force = false;
        }

        if (!existing_buffer) {
            existing_buffer = new Buffer();
            existing_buffer.init(config);
            ResourceCache.get().store(CacheTypes.BUFFER, Name.from(config.name), existing_buffer);
        }

        return existing_buffer;
    }
}
