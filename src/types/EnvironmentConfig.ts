/**
 * Python environment configuration for Tenstorrent terminals
 *
 * This module defines Python environments associated with each terminal context.
 * Each environment has a specific activation command and visual representation.
 */

/**
 * Python environment configuration
 */
export interface EnvironmentConfig {
  /** Unique identifier */
  id: string;

  /** Display name shown in UI */
  displayName: string;

  /** Activation command to source the environment */
  activationCommand: string;

  /** Deactivation command (if needed) */
  deactivationCommand?: string;

  /** Path to venv directory (for existence checks) */
  venvPath?: string;

  /** Whether this environment requires unset of TT_METAL_HOME */
  unsetTTMetalHome?: boolean;

  /** Description for quick pick UI */
  description?: string;

  /** Icon for status bar (codicon name) */
  icon?: string;
}

/**
 * Terminal context types
 * Each context corresponds to a specific project/purpose
 */
export type TerminalContext =
  | 'tt-metal'      // TT-Metal setup, demos, TTNN, cookbook
  | 'tt-forge'      // TT-Forge build, test, image classification
  | 'tt-xla'        // JAX demos and testing
  | 'vllm-server'   // Production inference with vLLM
  | 'api-server'    // Direct API, Flask servers
  | 'explore';      // Manual exploration, curl, ad-hoc testing

/**
 * Registry of Python environments for each terminal context
 *
 * This is the single source of truth for environment configuration.
 * Each terminal context automatically activates its associated environment.
 */
export const ENVIRONMENT_REGISTRY: Record<TerminalContext, EnvironmentConfig> = {
  'tt-metal': {
    id: 'tt-metal',
    displayName: 'TT-Metal',
    activationCommand: 'export PYTHONPATH=~/tt-metal:$PYTHONPATH && source ~/tt-metal/python_env/setup-metal.sh',
    description: 'TT-Metal Python environment (PYTHONPATH + setup-metal.sh)',
    icon: '$(chip)',
  },

  'tt-forge': {
    id: 'tt-forge',
    displayName: 'TT-Forge',
    activationCommand: 'source ~/tt-forge-venv/bin/activate',
    deactivationCommand: 'deactivate',
    venvPath: '~/tt-forge-venv',
    unsetTTMetalHome: true,
    description: 'TT-Forge Python 3.11 virtual environment',
    icon: '$(flame)',
  },

  'tt-xla': {
    id: 'tt-xla',
    displayName: 'TT-XLA',
    activationCommand: 'source ~/tt-xla-venv/bin/activate',
    deactivationCommand: 'deactivate',
    venvPath: '~/tt-xla-venv',
    description: 'TT-XLA JAX virtual environment',
    icon: '$(circuit-board)',
  },

  'vllm-server': {
    id: 'vllm-server',
    displayName: 'vLLM',
    activationCommand: 'source ~/tt-vllm-venv/bin/activate',
    deactivationCommand: 'deactivate',
    venvPath: '~/tt-vllm-venv',
    description: 'vLLM production inference environment',
    icon: '$(rocket)',
  },

  'api-server': {
    id: 'api-server',
    displayName: 'TT-Metal (API)',
    activationCommand: 'export PYTHONPATH=~/tt-metal:$PYTHONPATH && source ~/tt-metal/python_env/setup-metal.sh',
    description: 'TT-Metal environment for API servers',
    icon: '$(server)',
  },

  'explore': {
    id: 'system',
    displayName: 'System Python',
    activationCommand: '', // No activation needed
    description: 'System default Python (no venv)',
    icon: '$(terminal)',
  },
};
