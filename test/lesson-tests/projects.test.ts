/**
 * Projects Directory Validation Tests
 *
 * Ensures projects directory exists and is copied to dist for command usage.
 */

import { expect } from 'chai';
import * as fs from 'fs';
import * as path from 'path';

describe('Projects Directory Tests', () => {
  const projectsDir = path.join(__dirname, '../../content/projects');
  const distProjectsDir = path.join(__dirname, '../../dist/content/projects');

  describe('Source Projects Directory', () => {
    it('content/projects directory should exist', () => {
      expect(fs.existsSync(projectsDir)).to.be.true;
    });

    it('should be a directory', () => {
      const stat = fs.statSync(projectsDir);
      expect(stat.isDirectory()).to.be.true;
    });

    it('should be readable', () => {
      expect(() => fs.readdirSync(projectsDir)).to.not.throw();
    });
  });

  describe('Build Output', () => {
    it('dist/content/projects should exist after build', () => {
      expect(fs.existsSync(distProjectsDir)).to.be.true;
    });

    it('dist/content/projects should be readable for copying', () => {
      expect(() => fs.readdirSync(distProjectsDir)).to.not.throw();
    });
  });
});
