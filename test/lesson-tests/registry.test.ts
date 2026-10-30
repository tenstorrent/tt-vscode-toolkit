/**
 * Lesson registry shape: the sidebar is the four golden paths, then Go Deeper, then the
 * project series. Categories that predate the tt CLI rework (first-inference, serving,
 * advanced, compilers) must not come back.
 */
import { expect } from 'chai';
import * as fs from 'fs';
import * as path from 'path';

const registry = JSON.parse(
  fs.readFileSync(path.join(__dirname, '../../content/lesson-registry.json'), 'utf8')
);

const EXPECTED_ORDER = [
  'welcome',
  'setup',
  'serve',
  'connect',
  'build',
  'deeper',
  'applications',
  'cookbook',
  'cs-fundamentals',
  'llm-from-scratch',
  'custom-training',
];

describe('lesson registry categories', () => {
  it('lists exactly the golden-path categories, in sidebar order', () => {
    const ids = [...registry.categories].sort((a: any, b: any) => a.order - b.order).map((c: any) => c.id);
    expect(ids).to.deep.equal(EXPECTED_ORDER);
  });

  it('gives every lesson one of those categories', () => {
    const known = new Set(EXPECTED_ORDER);
    const bad = registry.lessons.filter((l: any) => !known.has(l.category)).map((l: any) => `${l.id}:${l.category}`);
    expect(bad).to.deep.equal([]);
  });

  it('has no lesson left in a retired category', () => {
    const retired = new Set(['first-inference', 'serving', 'advanced', 'compilers']);
    expect(registry.lessons.filter((l: any) => retired.has(l.category)).map((l: any) => l.id)).to.deep.equal([]);
  });
});
