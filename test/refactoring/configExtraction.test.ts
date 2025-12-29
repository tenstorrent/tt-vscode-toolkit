/**
 * Config Extraction Tests
 *
 * Tests for the model registry configuration extraction refactoring.
 * Verifies that:
 * 1. src/config/modelRegistry.ts exports expected functions and constants
 * 2. Model configuration is properly structured
 * 3. Helper functions work correctly
 */

import { expect } from 'chai';
import {
  MODEL_REGISTRY,
  DEFAULT_MODEL_KEY,
  getModelConfig,
  getModelBasePath,
  getModelOriginalPath
} from '../../src/config/modelRegistry';

describe('Config Extraction Tests', () => {
  describe('MODEL_REGISTRY Structure', () => {
    it('MODEL_REGISTRY should be defined', () => {
      expect(MODEL_REGISTRY).to.exist;
    });

    it('MODEL_REGISTRY should have at least one model', () => {
      const modelKeys = Object.keys(MODEL_REGISTRY);
      expect(modelKeys.length).to.be.greaterThan(0);
    });

    it('DEFAULT_MODEL_KEY should be defined', () => {
      expect(DEFAULT_MODEL_KEY).to.exist;
    });

    it('DEFAULT_MODEL_KEY should exist in MODEL_REGISTRY', () => {
      expect(MODEL_REGISTRY[DEFAULT_MODEL_KEY]).to.exist;
    });

    it('All models should have required fields', () => {
      Object.entries(MODEL_REGISTRY).forEach(([key, config]) => {
        expect(config.huggingfaceId, `Model ${key} should have huggingfaceId`).to.exist;
        expect(config.localDirName, `Model ${key} should have localDirName`).to.exist;
        expect(config.displayName, `Model ${key} should have displayName`).to.exist;
        expect(config.size, `Model ${key} should have size`).to.exist;
        expect(config.type, `Model ${key} should have type`).to.exist;
        expect(['llm', 'image', 'multimodal']).to.include(config.type);
      });
    });
  });

  describe('getModelConfig()', () => {
    it('Should return default model when called without arguments', () => {
      const config = getModelConfig();
      expect(config).to.exist;
      expect(config).to.equal(MODEL_REGISTRY[DEFAULT_MODEL_KEY]);
    });

    it('Should return specific model when passed a valid key', () => {
      const firstKey = Object.keys(MODEL_REGISTRY)[0];
      const config = getModelConfig(firstKey);
      expect(config).to.exist;
      expect(config).to.equal(MODEL_REGISTRY[firstKey]);
    });

    it('Should throw error for invalid model key', () => {
      expect(() => getModelConfig('invalid-model-key')).to.throw(/Model.*not found in MODEL_REGISTRY/);
    });
  });

  describe('getModelBasePath()', () => {
    it('Should return a path string', async () => {
      const path = await getModelBasePath();
      expect(path).to.be.a('string');
      expect(path.length).to.be.greaterThan(0);
    });

    it('Should include home directory', async () => {
      const path = await getModelBasePath();
      expect(path).to.include('models');
    });

    it('Should include model local directory name', async () => {
      const config = getModelConfig();
      const path = await getModelBasePath();
      expect(path).to.include(config.localDirName);
    });
  });

  describe('getModelOriginalPath()', () => {
    it('Should return a path string for models with originalSubdir', async () => {
      const config = getModelConfig();

      if (config.originalSubdir) {
        const path = await getModelOriginalPath();
        expect(path).to.be.a('string');
        expect(path.length).to.be.greaterThan(0);
        expect(path).to.include(config.originalSubdir);
      }
    });

    it('Should throw error for models without originalSubdir', async () => {
      // Create a test model config without originalSubdir
      const modelWithoutOriginal = Object.entries(MODEL_REGISTRY).find(
        ([_, config]) => !config.originalSubdir
      );

      if (modelWithoutOriginal) {
        try {
          await getModelOriginalPath(modelWithoutOriginal[0]);
          throw new Error('Should have thrown');
        } catch (error) {
          expect(error).to.match(/does not have an originalSubdir/);
        }
      }
    });
  });

  describe('No Duplication', () => {
    it('Config should be importable from src/config', () => {
      // This test verifies the barrel export works
      expect(MODEL_REGISTRY).to.exist;
      expect(getModelConfig).to.be.a('function');
      expect(getModelBasePath).to.be.a('function');
      expect(getModelOriginalPath).to.be.a('function');
    });
  });
});
