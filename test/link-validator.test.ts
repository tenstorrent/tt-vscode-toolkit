/**
 * Internal Link Validation Test
 *
 * Validates that all internal links in content files point to valid targets:
 * - Command links (command:tenstorrent.showLesson?{...}) reference existing lessons
 * - File paths reference existing files
 * - Lesson cross-references are valid
 *
 * Run with: npm run test:links
 */

import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';

interface LessonRegistry {
    lessons: Array<{
        id: string;
        title: string;
        markdownFile: string;
    }>;
}

interface LinkValidationError {
    file: string;
    line: number;
    linkText: string;
    error: string;
}

describe('Internal Link Validation', () => {
    const projectRoot = path.join(__dirname, '..');
    const contentRoot = path.join(projectRoot, 'content');
    const registryPath = path.join(contentRoot, 'lesson-registry.json');
    let lessonRegistry: LessonRegistry;
    let validLessonIds: Set<string>;
    let errors: LinkValidationError[] = [];

    before(() => {
        // Load lesson registry
        const registryContent = fs.readFileSync(registryPath, 'utf-8');
        lessonRegistry = JSON.parse(registryContent);
        validLessonIds = new Set(lessonRegistry.lessons.map(l => l.id));

        // Scan content directory once for all tests
        scanDirectory(contentRoot);
    });

    /**
     * Extract command links from markdown content
     * Format: [Link Text](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22lesson-id%22%7D)
     */
    function extractCommandLinks(content: string, filePath: string): LinkValidationError[] {
        const errors: LinkValidationError[] = [];
        const lines = content.split('\n');

        // Regex to match command links
        const commandLinkRegex = /\[([^\]]+)\]\(command:tenstorrent\.showLesson\?([^)]+)\)/g;

        lines.forEach((line, index) => {
            let match;
            while ((match = commandLinkRegex.exec(line)) !== null) {
                const linkText = match[1];
                const encodedParams = match[2];

                try {
                    // Decode URL-encoded JSON
                    const decodedParams = decodeURIComponent(encodedParams);
                    const params = JSON.parse(decodedParams);

                    if (params.lessonId && !validLessonIds.has(params.lessonId)) {
                        errors.push({
                            file: filePath,
                            line: index + 1,
                            linkText,
                            error: `Invalid lessonId: "${params.lessonId}" - not found in registry`
                        });
                    }
                } catch (e) {
                    errors.push({
                        file: filePath,
                        line: index + 1,
                        linkText,
                        error: `Failed to parse command link: ${e}`
                    });
                }
            }
        });

        return errors;
    }

    /**
     * Extract relative file paths from markdown content
     * Format: [Alt Text](/path/to/file.png) or ![Alt](../path/to/file.png)
     */
    function extractFilePaths(content: string, filePath: string): LinkValidationError[] {
        const errors: LinkValidationError[] = [];
        const lines = content.split('\n');
        const fileDir = path.dirname(filePath);

        // Track if we're inside a code block
        let inCodeBlock = false;

        // Regex to match markdown links and images
        const filePathRegex = /!?\[([^\]]*)\]\(([^)]+)\)/g;

        lines.forEach((line, index) => {
            // Toggle code block state on triple backticks
            if (line.trim().startsWith('```')) {
                inCodeBlock = !inCodeBlock;
                return;
            }

            // Skip lines inside code blocks
            if (inCodeBlock) {
                return;
            }

            let match;
            while ((match = filePathRegex.exec(line)) !== null) {
                const linkText = match[1];
                const linkTarget = match[2];

                // Skip external URLs and command links
                if (linkTarget.startsWith('http') || linkTarget.startsWith('command:')) {
                    continue;
                }

                // Skip anchor links
                if (linkTarget.startsWith('#')) {
                    continue;
                }

                // Resolve relative path
                let resolvedPath: string;
                if (linkTarget.startsWith('/')) {
                    // Absolute path from project root (e.g., /assets/img/...)
                    // Strip leading / so path.join works correctly
                    resolvedPath = path.join(projectRoot, linkTarget.substring(1));
                } else {
                    // Relative path from current file
                    resolvedPath = path.join(fileDir, linkTarget);
                }

                // Check if file exists
                if (!fs.existsSync(resolvedPath)) {
                    errors.push({
                        file: filePath,
                        line: index + 1,
                        linkText,
                        error: `File not found: ${linkTarget} (resolved to: ${resolvedPath})`
                    });
                }
            }
        });

        return errors;
    }

    /**
     * Scan a directory for markdown files and validate links
     */
    function scanDirectory(dir: string) {
        const entries = fs.readdirSync(dir, { withFileTypes: true });

        for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);

            if (entry.isDirectory()) {
                scanDirectory(fullPath);
            } else if (entry.isFile() && (entry.name.endsWith('.md') || entry.name.endsWith('.html'))) {
                const content = fs.readFileSync(fullPath, 'utf-8');
                const relativePath = path.relative(path.join(__dirname, '..'), fullPath);

                // Validate command links
                errors.push(...extractCommandLinks(content, relativePath));

                // Validate file paths (only for markdown, not HTML)
                if (entry.name.endsWith('.md')) {
                    errors.push(...extractFilePaths(content, fullPath));
                }
            }
        }
    }

    it('should have valid lesson IDs in all command links', () => {
        // Filter to only command link errors
        const commandLinkErrors = errors.filter(e =>
            e.error.includes('Invalid lessonId') || e.error.includes('Failed to parse')
        );

        if (commandLinkErrors.length > 0) {
            const errorMessage = commandLinkErrors.map(e =>
                `  ${e.file}:${e.line} - [${e.linkText}] - ${e.error}`
            ).join('\n');

            assert.fail(`Found ${commandLinkErrors.length} invalid command links:\n${errorMessage}`);
        }
    });

    it('should have valid file paths in all markdown links', () => {
        // Filter to only file path errors
        const filePathErrors = errors.filter(e => e.error.includes('File not found'));

        if (filePathErrors.length > 0) {
            const errorMessage = filePathErrors.map(e =>
                `  ${e.file}:${e.line} - [${e.linkText}] - ${e.error}`
            ).join('\n');

            assert.fail(`Found ${filePathErrors.length} broken file paths:\n${errorMessage}`);
        }
    });

    it('should have all lessons referenced in welcome.html present in registry', () => {
        const welcomePath = path.join(contentRoot, 'pages', 'welcome.html');
        const welcomeContent = fs.readFileSync(welcomePath, 'utf-8');

        // Extract onclick="openWalkthrough('lesson-id')" patterns
        const walkthroughRegex = /onclick="openWalkthrough\('([^']+)'\)"/g;
        const referencedLessons = new Set<string>();
        let match;

        while ((match = walkthroughRegex.exec(welcomeContent)) !== null) {
            referencedLessons.add(match[1]);
        }

        // Check each referenced lesson exists
        const missingLessons: string[] = [];
        for (const lessonId of referencedLessons) {
            if (!validLessonIds.has(lessonId)) {
                missingLessons.push(lessonId);
            }
        }

        if (missingLessons.length > 0) {
            assert.fail(`Welcome page references ${missingLessons.length} lessons not in registry:\n  ${missingLessons.join('\n  ')}`);
        }
    });

    it('should have all lessons in registry referenced somewhere in documentation', () => {
        // This is a warning test - lessons not referenced anywhere might be orphaned
        const allContent: string[] = [];

        function collectContent(dir: string) {
            const entries = fs.readdirSync(dir, { withFileTypes: true });
            for (const entry of entries) {
                const fullPath = path.join(dir, entry.name);
                if (entry.isDirectory()) {
                    collectContent(fullPath);
                } else if (entry.isFile() && (entry.name.endsWith('.md') || entry.name.endsWith('.html'))) {
                    allContent.push(fs.readFileSync(fullPath, 'utf-8'));
                }
            }
        }

        collectContent(contentRoot);
        const combinedContent = allContent.join('\n');

        const unreferencedLessons: string[] = [];
        for (const lessonId of validLessonIds) {
            // Check if lesson ID appears anywhere in content
            if (!combinedContent.includes(lessonId)) {
                unreferencedLessons.push(lessonId);
            }
        }

        // This is a warning, not a hard failure
        if (unreferencedLessons.length > 0) {
            console.warn(`\n⚠️  Warning: ${unreferencedLessons.length} lessons not referenced in any documentation:`);
            console.warn(`  ${unreferencedLessons.join(', ')}`);
            console.warn('  Consider adding them to welcome.html or FAQ.md\n');
        }
    });

    after(() => {
        // Print summary
        if (errors.length === 0) {
            console.log('\n✅ All internal links are valid!\n');
        } else {
            console.log(`\n❌ Found ${errors.length} total link validation issues\n`);
        }
    });
});
