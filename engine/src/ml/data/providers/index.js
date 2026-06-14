import { DataProviderKind, DataProviderRegistry } from "../data_provider.js";
import { FormulaProvider } from "./formula_provider.js";
import { ImageFolderProvider } from "./image_folder_provider.js";
import { NoiseProvider } from "./noise_provider.js";
import { TableProvider } from "./table_provider.js";
import { TextStreamProvider } from "./text_stream_provider.js";

let registered = false;

export function register_default_data_providers() {
  if (registered) {
    return;
  }

  DataProviderRegistry.register(DataProviderKind.FORMULA, FormulaProvider);
  DataProviderRegistry.register(DataProviderKind.IMAGE_FOLDER, ImageFolderProvider);
  DataProviderRegistry.register(DataProviderKind.NOISE, NoiseProvider);
  DataProviderRegistry.register(DataProviderKind.TABLE, TableProvider);
  DataProviderRegistry.register(DataProviderKind.TEXT, TextStreamProvider);

  registered = true;
}

register_default_data_providers();

export {
  FormulaProvider,
  ImageFolderProvider,
  NoiseProvider,
  TableProvider,
  TextStreamProvider,
};
