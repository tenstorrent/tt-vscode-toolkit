'use strict';
const path = require('path');

/** @type {import('webpack').Configuration} */
const config = {
  target: 'node',          // VSCode extension host runs in Node.js
  mode: 'production',
  entry: './src/extension.ts',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'extension.js',
    libraryTarget: 'commonjs2',  // Required by VSCode extension host
  },
  externals: {
    vscode: 'commonjs vscode',   // The vscode module is provided by VSCode at runtime
    ws: 'commonjs ws',           // ws is a transitive dep of vscode-languageclient; available at runtime
  },
  resolve: {
    extensions: ['.ts', '.js'],
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        exclude: /node_modules/,
        use: 'ts-loader',
      },
    ],
  },
};

module.exports = config;
