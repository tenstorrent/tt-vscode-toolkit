/**
 * Environment detection for lesson commands.
 *
 * Readers land in one of two places this module cares about:
 *   - the Tenstorrent Cloud pod (the tt-code-server image): one n150, no Docker, and a
 *     rule not to change drivers or firmware (its WELCOME.md says so);
 *   - a host they own (TT-QuietBox 2, a workstation), where `tt update` and `tt serve`
 *     do their full job.
 *
 * The pod is recognised by env vars its own Dockerfile sets (`TT_VLLM_VENV` and
 * `TT_INSTALLER_VENV`). Detection never relies on a tool being absent: "no docker" is
 * also true of a half-installed workstation, and a guard that fires there would hide a
 * real problem behind a reassuring message.
 *
 * No VSCode import here, so the module is unit-testable; the notifier is injected.
 */

export type Environment = 'cloud-pod' | 'host';

/** The pod's Dockerfile `ENV` lines that mark it. Both must be present. */
const POD_MARKERS = ['TT_VLLM_VENV', 'TT_INSTALLER_VENV'] as const;

export function detectEnvironment(env: NodeJS.ProcessEnv = process.env): Environment {
  return POD_MARKERS.every((k) => typeof env[k] === 'string' && env[k] !== '') ? 'cloud-pod' : 'host';
}

/** The one line a reader sees when a guarded button is pressed in the pod. */
export const CLOUD_GUARD_MESSAGE = (action: string): string =>
  `${action} is not available in the Tenstorrent Cloud pod: it changes firmware or needs Docker. See the Where Am I? lesson.`;

/**
 * Returns true when `action` may run here. On the cloud pod it shows the message via
 * `notify` and returns false; the caller then runs nothing.
 *
 * @param action  what the button would run, as the reader sees it (`tt update`)
 * @param env     process env (injected for tests)
 * @param notify  how to tell the reader (extension passes vscode.window.showInformationMessage)
 */
export function guardCloud(
  action: string,
  env: NodeJS.ProcessEnv = process.env,
  notify: (message: string) => unknown = () => undefined
): boolean {
  if (detectEnvironment(env) === 'host') return true;
  notify(CLOUD_GUARD_MESSAGE(action));
  return false;
}
