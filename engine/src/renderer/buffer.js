import Name from "@/utility/names.js";
import { ResourceCache, CacheTypes } from "@/renderer/resource_cache.js";

export class Buffer {
    config = null;
    buffer = null;
    data = null;

    // Create a GPU buffer to store the data
    init(context, config) {
        this.config = config;
        this.data = config.data.flat();

        const buffer_data = new Float32Array(this.data.length);
        buffer_data.set(this.data);

        this.config.size = buffer_data.byteLength;

        this.buffer = context.device.createBuffer({
            label: this.config.name,
            size: this.config.size,
            usage: this.config.usage
        });

        this.write(context, buffer_data);
    }

    write(context, data, offset = 0) {
        context.device.queue.writeBuffer(this.buffer, offset, data);
    }

    read(context, data, data_length, offset = 0, data_offset = 0) {
        const buffer_data = new Float32Array(this.buffer.getMappedRange());
        data.set(buffer_data.subarray(offset, offset + data_length), data_offset);
        this.buffer.unmap();
    }

    bind_vertex(encoder, slot = 0) {
        encoder.setVertexBuffer(slot, this.buffer);
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


