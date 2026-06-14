import { InputType } from "../../ml_types.js";
import { Tensor } from "../../math/tensor.js";
import {
  BaseDataProvider,
  DataProviderKind,
  require_shape,
} from "../data_provider.js";

function get_image_shape(image) {
  if (!image) {
    throw new Error("ImageFolderProvider requires image records.");
  }

  if (image.shape) {
    return require_shape(image.shape, "image.shape");
  }

  if (
    !Number.isInteger(image.width) ||
    image.width <= 0 ||
    !Number.isInteger(image.height) ||
    image.height <= 0 ||
    !Number.isInteger(image.channels) ||
    image.channels <= 0
  ) {
    throw new Error("Image records require positive integer width, height, and channels.");
  }

  return [image.width, image.height, image.channels];
}

function normalize_image_data(image, normalize_to_float) {
  if (!ArrayBuffer.isView(image.data)) {
    throw new Error("Image records require typed-array data.");
  }

  if (!normalize_to_float) {
    return image.data;
  }

  const result = new Float32Array(image.data.length);
  for (let i = 0; i < image.data.length; i++) {
    result[i] = Number(image.data[i]) / 255.0;
  }
  return result;
}

export class ImageFolderProvider extends BaseDataProvider {
  constructor(options = {}) {
    if (!Array.isArray(options.images) || options.images.length === 0) {
      throw new Error("ImageFolderProvider.images must be a non-empty array.");
    }

    const images = Array.from(options.images);
    const start = options.start ?? 0;
    if (!Number.isInteger(start) || start < 0 || start > images.length) {
      throw new Error("ImageFolderProvider.start must be an integer within the image range.");
    }

    super({
      kind: DataProviderKind.IMAGE_FOLDER,
      label: options.label ?? "Image Folder",
      input_type: InputType.IMAGE,
      shape: options.shape ?? get_image_shape(images[0]),
      batch_size: options.batch_size,
      finite: true,
      loop: options.loop,
    });

    this.images = images;
    this.cursor = start;
    this.shuffle = !!options.shuffle;
    this.normalize_to_float = options.normalize_to_float !== false;
    this.order = this.images.map((_, index) => index);

    this.validate_images();

    if (this.shuffle) {
      this.shuffle_order();
    }
  }

  static async from_files(files, options = {}) {
    if (!files || files.length === 0) {
      throw new Error("ImageFolderProvider.from_files requires at least one file.");
    }

    const images = [];
    for (const file of Array.from(files)) {
      const image = await ImageFolderProvider.file_to_image_record(file);
      images.push(image);
    }
    return new ImageFolderProvider({ ...options, images });
  }

  static async file_to_image_record(file) {
    if (typeof createImageBitmap !== "function") {
      throw new Error("Image file loading requires a browser environment.");
    }

    const bitmap = await createImageBitmap(file);
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0);
    const image_data = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    canvas.remove();

    return {
      data: new Uint8Array(image_data.data.buffer),
      width: bitmap.width,
      height: bitmap.height,
      channels: 4,
      name: file.name,
    };
  }

  snapshot() {
    return {
      cursor: this.cursor,
      order: this.order.slice(),
      state: this.state,
    };
  }

  restore(snapshot) {
    this.cursor = snapshot.cursor;
    this.order = snapshot.order.slice();
    this.state = snapshot.state;
  }

  reset() {
    super.reset();
    this.cursor = 0;
    this.order = this.images.map((_, index) => index);
    if (this.shuffle) {
      this.shuffle_order();
    }
  }

  validate_images() {
    const expected_shape = this.shape;
    const expected_size = Tensor.sample_size(expected_shape);

    for (let i = 0; i < this.images.length; i++) {
      const image = this.images[i];
      const shape = get_image_shape(image);
      if (
        shape.length !== expected_shape.length ||
        shape.some((dimension, index) => dimension !== expected_shape[index])
      ) {
        throw new Error(`Image ${i} shape does not match provider shape.`);
      }

      if (!ArrayBuffer.isView(image.data)) {
        throw new Error(`Image ${i} data must be a typed array.`);
      }

      if (image.data.length !== expected_size) {
        throw new Error(`Image ${i} data length does not match provider shape.`);
      }
    }
  }

  shuffle_order() {
    for (let i = this.order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.order[i], this.order[j]] = [this.order[j], this.order[i]];
    }
  }

  next(count = this.batch_size) {
    if (this.is_paused()) {
      return null;
    }

    if (this.images.length === 0 || (this.cursor >= this.images.length && !this.loop)) {
      this.state = "exhausted";
      return null;
    }

    const image_sample_size = Tensor.sample_size(this.shape);
    const chunks = [];
    const names = [];

    while (chunks.length < count) {
      if (this.cursor >= this.images.length) {
        if (!this.loop) {
          break;
        }
        this.cursor = 0;
        if (this.shuffle) {
          this.shuffle_order();
        }
      }

      const image_index = this.order[this.cursor++];
      const image = this.images[image_index];
      const image_data = normalize_image_data(image, this.normalize_to_float);

      chunks.push(image_data);
      names.push(image.name ?? image_index);
    }

    if (chunks.length === 0) {
      this.state = "exhausted";
      return null;
    }

    const ArrayType = this.normalize_to_float ? Float32Array : chunks[0].constructor;
    const data = new ArrayType(image_sample_size * chunks.length);

    for (let i = 0; i < chunks.length; i++) {
      data.set(chunks[i], i * image_sample_size);
    }

    if (this.cursor >= this.images.length && !this.loop) {
      this.state = "exhausted";
    }

    return {
      data,
      shape: this.shape,
      batch_size: chunks.length,
      input_type: this.input_type,
      meta: { names },
      done: this.state === "exhausted",
    };
  }
}
