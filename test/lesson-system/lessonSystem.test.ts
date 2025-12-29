/**
 * Lesson System Integration Tests
 *
 * Tests for the custom TreeView + Webview lesson system.
 * These tests are content-agnostic - they verify the system works
 * regardless of how many lessons or categories exist.
 *
 * NOTE: These tests require the VSCode Extension Host environment.
 * They cannot be run with regular mocha - use the VSCode Extension Test Runner.
 * To run: Press F5 in VSCode or use the "Extension Tests" launch configuration.
 */

import * as assert from 'assert';
import * as vscode from 'vscode';
import { LessonRegistry } from '../../src/utils/LessonRegistry';

suite('Lesson System Test Suite', () => {
  let context: vscode.ExtensionContext;

  suiteSetup(async function() {
    this.timeout(30000);

    // Get extension context
    const ext = vscode.extensions.getExtension('tenstorrent.tt-vscode-toolkit');
    assert.ok(ext, 'Extension should be installed');

    await ext.activate();
    context = (ext.exports as any).context;
    assert.ok(context, 'Extension context should be available');
  });

  suite('Extension Activation', () => {
    test('Extension should be present', () => {
      const ext = vscode.extensions.getExtension('tenstorrent.tt-vscode-toolkit');
      assert.ok(ext, 'Extension not found');
    });

    test('Extension should activate', async () => {
      const ext = vscode.extensions.getExtension('tenstorrent.tt-vscode-toolkit');
      await ext!.activate();
      assert.strictEqual(ext!.isActive, true, 'Extension did not activate');
    });
  });

  suite('TreeView Registration', () => {
    test('showLesson command should be registered', async () => {
      const commands = await vscode.commands.getCommands();
      assert.ok(
        commands.includes('tenstorrent.showLesson'),
        'showLesson command not registered'
      );
    });

    test('refreshLessons command should be registered', async () => {
      const commands = await vscode.commands.getCommands();
      assert.ok(
        commands.includes('tenstorrent.refreshLessons'),
        'refreshLessons command not registered'
      );
    });

    test('filterLessons command should be registered', async () => {
      const commands = await vscode.commands.getCommands();
      assert.ok(
        commands.includes('tenstorrent.filterLessons'),
        'filterLessons command not registered'
      );
    });
  });

  suite('Lesson Registry', () => {
    let registry: LessonRegistry;

    setup(() => {
      registry = new LessonRegistry(context);
    });

    test('Should load lesson registry', async () => {
      await registry.load();
      const count = registry.getTotalCount();
      assert.ok(count > 0, 'Should load at least one lesson');
    });

    test('Should have at least one category', async () => {
      await registry.load();
      const categories = registry.getCategories();

      assert.ok(categories.length > 0, 'Should have at least one category');

      // Verify each category has required fields
      categories.forEach(category => {
        assert.ok(category.id, 'Category should have an id');
        assert.ok(category.title, 'Category should have a title');
        assert.ok(typeof category.order === 'number', 'Category should have an order');
      });
    });

    test('Should get lesson by ID', async () => {
      await registry.load();
      const allLessons = registry.getAll();

      // Test with first available lesson
      if (allLessons.length > 0) {
        const firstLesson = allLessons[0];
        const retrieved = registry.get(firstLesson.id);

        assert.ok(retrieved, 'Should retrieve lesson by ID');
        assert.strictEqual(retrieved?.id, firstLesson.id, 'Retrieved lesson should match');
        assert.ok(retrieved?.title, 'Lesson should have a title');
      }
    });

    test('Should filter by category', async () => {
      await registry.load();
      const categories = registry.getCategories();

      // Test filtering with first available category
      if (categories.length > 0) {
        const firstCategory = categories[0];
        const lessonsInCategory = registry.getByCategory(firstCategory.id);

        // Verify all returned lessons belong to the correct category
        lessonsInCategory.forEach(lesson => {
          assert.strictEqual(
            lesson.category,
            firstCategory.id,
            `Lesson ${lesson.id} should belong to category ${firstCategory.id}`
          );
        });
      }
    });

    test('All lessons should have required fields', async () => {
      await registry.load();
      const allLessons = registry.getAll();

      assert.ok(allLessons.length > 0, 'Should have at least one lesson');

      allLessons.forEach(lesson => {
        assert.ok(lesson.id, 'Lesson should have an id');
        assert.ok(lesson.title, 'Lesson should have a title');
        assert.ok(lesson.description, 'Lesson should have a description');
        assert.ok(lesson.category, 'Lesson should have a category');
        assert.ok(lesson.markdownFile, 'Lesson should have a markdown file');
        assert.ok(Array.isArray(lesson.supportedHardware), 'Lesson should have supportedHardware array');
        assert.ok(lesson.status, 'Lesson should have a status');
        assert.ok(Array.isArray(lesson.completionEvents), 'Lesson should have completionEvents array');
        assert.ok(Array.isArray(lesson.tags), 'Lesson should have tags array');
      });
    });
  });

  suite('Integration Tests', () => {
    test('Should execute showLesson command with valid lesson ID', async () => {
      // Load registry to get a valid lesson ID
      const registry = new LessonRegistry(context);
      await registry.load();
      const allLessons = registry.getAll();

      if (allLessons.length > 0) {
        const firstLesson = allLessons[0];

        // This should not throw
        await vscode.commands.executeCommand(
          'tenstorrent.showLesson',
          firstLesson.id
        );

        assert.ok(true, 'showLesson command executed successfully');
      }
    });

    test('Core extension commands should be registered', async () => {
      const commands = await vscode.commands.getCommands();
      const requiredCommands = [
        'tenstorrent.showWelcome',
        'tenstorrent.showFaq',
        'tenstorrent.openWalkthrough',
        'tenstorrent.showLesson',
        'tenstorrent.refreshLessons',
        'tenstorrent.filterLessons'
      ];

      requiredCommands.forEach(cmd => {
        assert.ok(
          commands.includes(cmd),
          `Command ${cmd} should be registered`
        );
      });
    });
  });
});
