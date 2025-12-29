/**
 * Model Registry Configuration
 *
 * Central registry for all models supported by the extension.
 * This eliminates duplication between extension.ts and terminalCommands.ts.
 */

/**
 * Model configuration type
 */
export interface ModelConfig {
  /** HuggingFace model ID (used for downloading and API requests) */
  huggingfaceId: string;
  /** Local directory name (where model is stored) */
  localDirName: string;
  /** Subdirectory for Meta's original format (used by Direct API), if applicable */
  originalSubdir?: string;
  /** Display name for UI */
  displayName: string;
  /** Model size for user information */
  size: string;
  /** Model type (llm, image, etc.) */
  type: 'llm' | 'image' | 'multimodal';
}

/**
 * Model Registry
 * Extensible registry of all models supported by the extension.
 * Add new models here to make them available throughout the extension.
 */
export const MODEL_REGISTRY: Record<string, ModelConfig> = {
  'llama-3.1-8b': {
    huggingfaceId: 'meta-llama/Llama-3.1-8B-Instruct',
    localDirName: 'Llama-3.1-8B-Instruct',
    originalSubdir: 'original',
    displayName: 'Llama 3.1 8B Instruct',
    size: '~16GB',
    type: 'llm',
  },
  // Future models can be added here as they become compatible with tt-metal:
  // 'llama-3.2-9b': {
  //   huggingfaceId: 'meta-llama/Llama-3.2-9B-Instruct',
  //   localDirName: 'Llama-3.2-9B-Instruct',
  //   originalSubdir: 'original',
  //   displayName: 'Llama 3.2 9B Instruct',
  //   size: '~18GB',
  //   type: 'llm',
  // },
  // 'sd-3.5-large': {
  //   huggingfaceId: 'stabilityai/stable-diffusion-3.5-large',
  //   localDirName: 'stable-diffusion-3.5-large',
  //   displayName: 'Stable Diffusion 3.5 Large',
  //   size: '~10GB',
  //   type: 'image',
  // },
} as const;

/**
 * Default model key used throughout the extension
 * Change this to switch the default model for all lessons
 */
export const DEFAULT_MODEL_KEY = 'llama-3.1-8b';

/**
 * Helper function to get a model config by key
 */
export function getModelConfig(modelKey: string = DEFAULT_MODEL_KEY): ModelConfig {
  const config = MODEL_REGISTRY[modelKey];
  if (!config) {
    throw new Error(`Model '${modelKey}' not found in MODEL_REGISTRY. Available models: ${Object.keys(MODEL_REGISTRY).join(', ')}`);
  }
  return config;
}

/**
 * Helper functions for model paths
 */
export async function getModelBasePath(modelKey: string = DEFAULT_MODEL_KEY): Promise<string> {
  const os = await import('os');
  const path = await import('path');
  const config = getModelConfig(modelKey);
  return path.join(os.homedir(), 'models', config.localDirName);
}

export async function getModelOriginalPath(modelKey: string = DEFAULT_MODEL_KEY): Promise<string> {
  const path = await import('path');
  const config = getModelConfig(modelKey);
  const basePath = await getModelBasePath(modelKey);

  if (!config.originalSubdir) {
    throw new Error(`Model '${modelKey}' does not have an originalSubdir. This model may not support Direct API.`);
  }

  return path.join(basePath, config.originalSubdir);
}
