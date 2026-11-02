/**
 * Environment guard tests.
 *
 * The Tenstorrent Cloud pod (tt-code-server image) must not run `tt update` (it would
 * try to flash firmware, which the pod forbids) or anything that pulls a container
 * (no Docker there). The pod is recognised by env vars its own Dockerfile sets, never
 * by the absence of a tool. No VSCode here: the notifier is injected.
 */
import { expect } from 'chai';
import { detectEnvironment, guardCloud, CLOUD_GUARD_MESSAGE } from '../../src/utils/environment';

const POD = { TT_VLLM_VENV: '/home/user/tt-metal/build/python_env_vllm', TT_INSTALLER_VENV: '/home/user/.tt-installer-venv' };

describe('environment guard', () => {
  it('detects the cloud pod from both image env vars', () => {
    expect(detectEnvironment(POD)).to.equal('cloud-pod');
  });

  it('is a host when either var is missing', () => {
    expect(detectEnvironment({ TT_VLLM_VENV: POD.TT_VLLM_VENV })).to.equal('host');
    expect(detectEnvironment({ TT_INSTALLER_VENV: POD.TT_INSTALLER_VENV })).to.equal('host');
    expect(detectEnvironment({})).to.equal('host');
  });

  it('guardCloud lets a host proceed without a message', () => {
    const shown: string[] = [];
    expect(guardCloud('tt update', {}, (m) => shown.push(m))).to.equal(true);
    expect(shown).to.deep.equal([]);
  });

  it('guardCloud refuses on the pod with the one-line message', () => {
    const shown: string[] = [];
    expect(guardCloud('tt update', POD, (m) => shown.push(m))).to.equal(false);
    expect(shown).to.deep.equal([CLOUD_GUARD_MESSAGE('tt update')]);
    expect(shown[0]).to.equal(
      'tt update is not available in the Tenstorrent Cloud pod: it changes firmware or needs Docker. See the Where Am I? lesson.'
    );
  });
});
