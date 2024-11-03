export const DataType = {
  FLOAT32: { array: Float32Array, default: 0.0, byte_size: 4 },
  UINT8: { array: Uint8Array, default: 0, byte_size: 1 },
  UINT32: { array: Uint32Array, default: 0, byte_size: 4 },
  BIGINT64: { array: BigInt64Array, default: 0n, byte_size: 8 },
};

export const BufferType = {
  STORAGE: 'GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST',
  UNIFORM: 'GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST',
  VERTEX: 'GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST',
  CPU_READ: 'GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST',
};

