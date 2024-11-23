export const DataType = {
  FLOAT32: { array: Float32Array, default: 0.0, byte_size: 4 },
  UINT8: { array: Uint8Array, default: 0, byte_size: 1 },
  UINT16: { array: Uint16Array, default: 0, byte_size: 2 },
  UINT32: { array: Uint32Array, default: 0, byte_size: 4 },
  INT8: { array: Int8Array, default: 0, byte_size: 1 },
  INT16: { array: Int16Array, default: 0, byte_size: 2 },
  INT32: { array: Int32Array, default: 0, byte_size: 4 },
  BIGINT64: { array: BigInt64Array, default: 0n, byte_size: 8 },
};

export const BufferType = {
  STORAGE: 'GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST',
  STORAGE_SRC: 'GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC',
  UNIFORM: 'GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST',
  UNIFORM_SRC: 'GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC',
  VERTEX: 'GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST',
  VERTEX_SRC: 'GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC',
  CPU_READ: 'GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST',
  CPU_READ_SRC: 'GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC',
};
