#!/usr/bin/env node

import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('🧪 Swarm Training Prototype Test');
console.log('===============================\n');

// Test 1: Check Node.js version
console.log('1. Checking Node.js version...');
const nodeVersion = process.version;
const majorVersion = parseInt(nodeVersion.replace('v', '').split('.')[0]);
if (majorVersion >= 16) {
    console.log(`   ✅ Node.js ${nodeVersion} (OK)`);
} else {
    console.log(`   ⚠️  Node.js ${nodeVersion} (Recommended: 16+)`);
}

// Test 2: Check dependencies
console.log('\n2. Checking dependencies...');
try {
    const packageJson = JSON.parse(readFileSync('package.json', 'utf8'));
    const nodeModules = join(__dirname, 'node_modules');
    
    const deps = Object.keys(packageJson.dependencies || {});
    const devDeps = Object.keys(packageJson.devDependencies || {});
    
    console.log(`   Dependencies: ${deps.length} packages`);
    console.log(`   Dev dependencies: ${devDeps.length} packages`);
    
    // Check if node_modules exists
    if (existsSync(nodeModules)) {
        console.log('   ✅ node_modules directory exists');
    } else {
        console.log('   ❌ node_modules directory missing');
        console.log('   Run: npm install');
        process.exit(1);
    }
    
} catch (error) {
    console.log(`   ❌ Error checking dependencies: ${error.message}`);
}

// Test 3: Check source files
console.log('\n3. Checking source files...');
const requiredFiles = [
    'src/index.js',
    'src/core/trainer.js',
    'src/core/phase.js',
    'src/core/models.js',
    'src/network/peer.js',
    'src/ui/manager.js',
    'index.html',
    'vite.config.js'
];

let missingFiles = [];
for (const file of requiredFiles) {
    if (existsSync(file)) {
        console.log(`   ✅ ${file}`);
    } else {
        console.log(`   ❌ ${file} (missing)`);
        missingFiles.push(file);
    }
}

if (missingFiles.length > 0) {
    console.log(`\n   ⚠️  Missing ${missingFiles.length} files`);
}

// Test 4: Check build configuration
console.log('\n4. Checking build configuration...');
try {
    // Dynamic import for ES module
    const viteConfig = await import('./vite.config.js');
    console.log('   ✅ Vite configuration loaded');
    
    if (viteConfig.default.server && viteConfig.default.server.port) {
        console.log(`   ✅ Server port: ${viteConfig.default.server.port}`);
    }
} catch (error) {
    console.log(`   ❌ Vite configuration error: ${error.message}`);
}

// Summary
console.log('\n📊 Test Summary');
console.log('===============');

if (missingFiles.length === 0) {
    console.log('✅ All tests passed!');
    console.log('\n🚀 To run the prototype:');
    console.log('   1. Start dev server: npm run dev');
    console.log('   2. Open browser: http://localhost:3000');
    console.log('   3. For testing: Open test.html in browser');
} else {
    console.log(`⚠️  ${missingFiles.length} issues found`);
    console.log('\nPlease fix the missing files and run tests again.');
}

console.log('\n💡 Note: This prototype uses simulated training for demonstration.');
console.log('   Real WebTorch integration would require actual model training.');