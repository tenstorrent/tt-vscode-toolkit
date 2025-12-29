# Adding New Models to the Extension

This guide explains how to add support for new models to the Tenstorrent VS Code extension.

## Model Registry Architecture

The extension uses a centralized **Model Registry** to manage all supported models. This ensures consistency across all commands, paths, and API calls.

## Quick Start: Adding a New Model

### 1. Add Model to Registry (src/extension.ts)

```typescript
const MODEL_REGISTRY: Record<string, ModelConfig> = {
  'llama-3.1-8b': {
    huggingfaceId: 'meta-llama/Llama-3.1-8B-Instruct',
    localDirName: 'Llama-3.1-8B-Instruct',
    originalSubdir: 'original',
    displayName: 'Llama 3.1 8B Instruct',
    size: '~16GB',
    type: 'llm',
  },

  // Add your new model here:
  'llama-3.2-9b': {
    huggingfaceId: 'meta-llama/Llama-3.2-9B-Instruct',
    localDirName: 'Llama-3.2-9B-Instruct',
    originalSubdir: 'original',  // Optional - only if model supports Direct API
    displayName: 'Llama 3.2 9B Instruct',
    size: '~18GB',
    type: 'llm',
  },
};
```

### 2. Add Model to Terminal Commands (src/commands/terminalCommands.ts)

```typescript
const MODEL_REGISTRY: Record<string, ModelConfig> = {
  'llama-3.1-8b': {
    huggingfaceId: 'meta-llama/Llama-3.1-8B-Instruct',
    localDirName: 'Llama-3.1-8B-Instruct',
    displayName: 'Llama 3.1 8B Instruct',
  },

  // Add the same model here:
  'llama-3.2-9b': {
    huggingfaceId: 'meta-llama/Llama-3.2-9B-Instruct',
    localDirName: 'Llama-3.2-9B-Instruct',
    displayName: 'Llama 3.2 9B Instruct',
  },
};
```

### 3. Update Default Model (Optional)

To make the new model the default across all lessons:

```typescript
// In both src/extension.ts and src/commands/terminalCommands.ts
const DEFAULT_MODEL_KEY = 'llama-3.2-9b';  // Change this
```

### 4. Rebuild

```bash
npm run build
```

## Model Configuration Fields

### Required Fields

| Field | Description | Example |
|-------|-------------|---------|
| `huggingfaceId` | Full HuggingFace model ID | `'meta-llama/Llama-3.1-8B-Instruct'` |
| `localDirName` | Directory name in ~/models/ | `'Llama-3.1-8B-Instruct'` |
| `displayName` | User-friendly name | `'Llama 3.1 8B Instruct'` |
| `size` | Download size for user info | `'~16GB'` |
| `type` | Model category | `'llm'`, `'image'`, or `'multimodal'` |

### Optional Fields

| Field | Description | When to Use |
|-------|-------------|-------------|
| `originalSubdir` | Subdirectory for Meta's format | Only for models supporting Direct API (Lessons 4-5) |

## Model Types

### LLM (Large Language Model)

```typescript
'llama-3.1-8b': {
  huggingfaceId: 'meta-llama/Llama-3.1-8B-Instruct',
  localDirName: 'Llama-3.1-8B-Instruct',
  originalSubdir: 'original',  // If it has Meta's original format
  displayName: 'Llama 3.1 8B Instruct',
  size: '~16GB',
  type: 'llm',
}
```

### Image Generation Model

```typescript
'sd-3.5-large': {
  huggingfaceId: 'stabilityai/stable-diffusion-3.5-large',
  localDirName: 'stable-diffusion-3.5-large',
  // No originalSubdir - not used with Direct API
  displayName: 'Stable Diffusion 3.5 Large',
  size: '~10GB',
  type: 'image',
}
```

### Multimodal Model

```typescript
'llama-3.2-vision': {
  huggingfaceId: 'meta-llama/Llama-3.2-11B-Vision-Instruct',
  localDirName: 'Llama-3.2-11B-Vision-Instruct',
  displayName: 'Llama 3.2 11B Vision',
  size: '~22GB',
  type: 'multimodal',
}
```

## Using Models in Code

### Get Model Configuration

```typescript
// Get default model
const config = getModelConfig();

// Get specific model
const config = getModelConfig('llama-3.2-9b');

// Access properties
console.log(config.huggingfaceId);  // 'meta-llama/Llama-3.2-9B-Instruct'
console.log(config.displayName);    // 'Llama 3.2 9B Instruct'
```

### Get Model Paths

```typescript
// Get base path (for HuggingFace format - vLLM)
const basePath = await getModelBasePath();
// Returns: ~/models/Llama-3.1-8B-Instruct

// Get original path (for Meta format - Direct API)
const originalPath = await getModelOriginalPath();
// Returns: ~/models/Llama-3.1-8B-Instruct/original

// Get specific model path
const path = await getModelBasePath('llama-3.2-9b');
```

### Error Handling

The registry includes helpful error messages:

```typescript
// If model doesn't exist:
getModelConfig('nonexistent-model');
// Error: Model 'nonexistent-model' not found in MODEL_REGISTRY.
//        Available models: llama-3.1-8b, llama-3.2-9b

// If model doesn't have originalSubdir:
await getModelOriginalPath('sd-3.5-large');
// Error: Model 'sd-3.5-large' does not have an originalSubdir.
//        This model may not support Direct API.
```

## Creating Model-Specific Lessons

To create a lesson that uses a specific model:

### Option 1: Change Default (affects all lessons)
```typescript
const DEFAULT_MODEL_KEY = 'llama-3.2-9b';
```

### Option 2: Create Model-Specific Commands

```typescript
async function runInferenceWithLlama32(): Promise<void> {
  const modelPath = await getModelOriginalPath('llama-3.2-9b');
  // ... use modelPath in commands
}
```

### Option 3: Add Model Selection UI

```typescript
async function selectModel(): Promise<string> {
  const models = Object.entries(MODEL_REGISTRY)
    .filter(([_, config]) => config.type === 'llm')
    .map(([key, config]) => ({
      label: config.displayName,
      description: config.size,
      detail: config.huggingfaceId,
      modelKey: key,
    }));

  const selected = await vscode.window.showQuickPick(models, {
    placeHolder: 'Select a model to use',
  });

  return selected?.modelKey || DEFAULT_MODEL_KEY;
}
```

## Testing Your New Model

1. **Rebuild the extension:**
   ```bash
   npm run build
   ```

2. **Test model paths:**
   - Verify download command uses correct HuggingFace ID
   - Verify local path matches `localDirName`

3. **Test with different lesson types:**
   - Lesson 3-5: Direct API (needs `originalSubdir`)
   - Lesson 6-7: vLLM (uses base path)

4. **Verify API calls:**
   - Chat participant uses correct `huggingfaceId`
   - Test commands use correct model name

## Example: Adding Llama 3.2 9B

```typescript
// 1. Add to src/extension.ts MODEL_REGISTRY
'llama-3.2-9b': {
  huggingfaceId: 'meta-llama/Llama-3.2-9B-Instruct',
  localDirName: 'Llama-3.2-9B-Instruct',
  originalSubdir: 'original',
  displayName: 'Llama 3.2 9B Instruct',
  size: '~18GB',
  type: 'llm',
},

// 2. Add to src/commands/terminalCommands.ts MODEL_REGISTRY
'llama-3.2-9b': {
  huggingfaceId: 'meta-llama/Llama-3.2-9B-Instruct',
  localDirName: 'Llama-3.2-9B-Instruct',
  displayName: 'Llama 3.2 9B Instruct',
},

// 3. (Optional) Change default in both files
const DEFAULT_MODEL_KEY = 'llama-3.2-9b';

// 4. Rebuild
npm run build

// 5. Test
// - Download command shows correct model
// - Paths resolve to ~/models/Llama-3.2-9B-Instruct
// - API calls use meta-llama/Llama-3.2-9B-Instruct
```

## Benefits of Model Registry

✅ **Single source of truth** - Define model once, use everywhere
✅ **Type safety** - TypeScript ensures all required fields present
✅ **Easy to add models** - Just add to registry, no code changes needed
✅ **Consistent paths** - Helper functions ensure correct path construction
✅ **Clear errors** - Helpful messages when model not found
✅ **Extensible** - Easy to add new model types and properties
✅ **Maintainable** - Change model in one place, updates everywhere

## Future Enhancements

Potential additions to the registry system:

- **Model selection UI** - Let users choose models per lesson
- **Multi-model lessons** - Use different models for different tasks
- **Model requirements** - Check hardware requirements before download
- **Model versioning** - Track model versions and updates
- **Model metadata** - Add more info (license, paper, benchmarks)
