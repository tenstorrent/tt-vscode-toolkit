# Making Koyeb Images Public and Usable by Anyone

This guide explains how to make Docker images publicly accessible so anyone can deploy to Koyeb with one click.

## Current State

**What's already configured:**
- ✅ GitHub Actions builds and publishes images to `ghcr.io`
- ✅ Deploy buttons configured in documentation
- ✅ Release automation working

**What's missing:**
- ⚠️ GitHub Container Registry packages are **private by default**
- ⚠️ Koyeb-optimized image (`Dockerfile.koyeb`) not published to registry

---

## Step 1: Make GitHub Container Registry Packages Public

### Why This Matters

By default, all packages published to `ghcr.io` are **private**. This means:
- ❌ Anonymous users can't pull images
- ❌ Deploy button requires authentication
- ❌ Users can't easily deploy to Koyeb

**After making public:**
- ✅ Anyone can pull images anonymously
- ✅ Deploy button works without authentication
- ✅ True one-click deployment

### How to Make Packages Public

#### For Personal Account (tenstorrent user)

1. Navigate to: https://github.com/tenstorrent?tab=packages
2. Click on `tt-vscode-toolkit` package
3. Click "Package settings" (right sidebar)
4. Scroll to "Danger Zone"
5. Click "Change visibility"
6. Select "Public"
7. Confirm the change

#### For Organization (tenstorrent org)

**Prerequisites:** Organization must allow public packages

1. **Enable in Organization Settings** (admin required):
   ```
   https://github.com/organizations/tenstorrent/settings/member_privileges
   ```
   - Under "Package creation"
   - Enable "Public packages"

2. **Make Package Public:**
   - Navigate to: https://github.com/orgs/tenstorrent/packages
   - Click `tt-vscode-toolkit` package
   - Click "Package settings"
   - Scroll to "Danger Zone"
   - Click "Change visibility"
   - Select "Public"

**⚠️ Important:** Once a package is public, it **cannot be made private again**. Ensure you're ready for public distribution.

### Verify Public Access

After making public, test anonymous access:

```bash
# Should work without authentication
docker pull ghcr.io/tenstorrent/tt-vscode-toolkit:latest

# If it works, image is truly public!
```

---

## Step 2: Publish Koyeb-Optimized Images

Currently, `Dockerfile.koyeb` is only built during Koyeb deployment. To make it available as a public image:

### Add Koyeb Image to GitHub Actions

**Update:** `.github/workflows/docker-build.yml`

Add a new job:

```yaml
build-koyeb:
  name: Build Koyeb Image
  runs-on: ubuntu-latest
  needs: test-extension
  permissions:
    contents: read
    packages: write

  steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '18'
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Build extension
      run: |
        npm run build
        npm run package

    - name: Log in to container registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch,suffix=-koyeb
          type=ref,event=pr,suffix=-koyeb
          type=semver,pattern={{version}},suffix=-koyeb
          type=semver,pattern={{major}}.{{minor}},suffix=-koyeb
          type=sha,prefix=koyeb-

    - name: Build and push Koyeb image
      uses: docker/build-push-action@v5
      with:
        context: .
        file: ./Dockerfile.koyeb
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
```

### Update Release Workflow

**Update:** `.github/workflows/release.yml`

Add Koyeb image tagging:

```yaml
- name: Tag and push Koyeb image
  run: |
    docker pull ghcr.io/tenstorrent/tt-vscode-toolkit:main-koyeb
    docker tag ghcr.io/tenstorrent/tt-vscode-toolkit:main-koyeb \
      ghcr.io/tenstorrent/tt-vscode-toolkit:${{ steps.version.outputs.version }}-koyeb
    docker tag ghcr.io/tenstorrent/tt-vscode-toolkit:main-koyeb \
      ghcr.io/tenstorrent/tt-vscode-toolkit:latest-koyeb
    docker push ghcr.io/tenstorrent/tt-vscode-toolkit:${{ steps.version.outputs.version }}-koyeb
    docker push ghcr.io/tenstorrent/tt-vscode-toolkit:latest-koyeb
```

### Update Deploy Button

**Update:** `docs/deployment/README.md`

Add Koyeb-optimized option:

```markdown
### Option 3: Koyeb-Optimized Image (Best Performance)

[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=docker&image=ghcr.io/tenstorrent/tt-vscode-toolkit:latest-koyeb&name=tt-vscode-toolkit&ports=8080;http;/&env[PASSWORD]=changeme)

Uses Koyeb-optimized Dockerfile with tt-metal startup installation.
```

---

## Step 3: Configure Repository Settings

### Repository Visibility

Ensure the repository itself is public:

1. Go to: https://github.com/tenstorrent/tt-vscode-toolkit/settings
2. Scroll to "Danger Zone"
3. Verify visibility is set to "Public"

### Package Permissions

After making package public, configure inheritance:

1. Go to package settings
2. Under "Manage Actions access"
3. Select "Inherit access from source repository"
4. This allows GitHub Actions to push without additional secrets

---

## Step 4: Test Public Deployment

### Test Anonymous Docker Pull

```bash
# Should work without docker login
docker pull ghcr.io/tenstorrent/tt-vscode-toolkit:latest
docker pull ghcr.io/tenstorrent/tt-vscode-toolkit:latest-full
docker pull ghcr.io/tenstorrent/tt-vscode-toolkit:latest-koyeb
```

### Test Deploy Button

1. Open incognito/private browser window
2. Visit: `docs/deployment/README.md` (on GitHub)
3. Click "Deploy to Koyeb" button
4. Should redirect to Koyeb without authentication errors
5. Complete deployment flow

### Test Specific Version

```bash
# After release v0.0.255
docker pull ghcr.io/tenstorrent/tt-vscode-toolkit:0.0.255
docker pull ghcr.io/tenstorrent/tt-vscode-toolkit:0.0.255-koyeb
```

---

## Summary of Changes Needed

### 1. One-Time Configuration (Admin Required)

**In GitHub:**
- [ ] Enable public packages in organization settings
- [ ] Make `tt-vscode-toolkit` package public
- [ ] Verify repository is public
- [ ] Configure package to inherit repository permissions

**Estimated time:** 5 minutes

### 2. Code Changes (Developer)

**Add to workflows:**
- [ ] Add `build-koyeb` job to `docker-build.yml`
- [ ] Add Koyeb image tagging to `release.yml`
- [ ] Update deploy button in `README.md`

**Estimated time:** 15 minutes

### 3. Testing (Developer + Users)

- [ ] Test anonymous image pulls
- [ ] Test deploy button in incognito mode
- [ ] Deploy to Koyeb from public image
- [ ] Verify all three image variants work

**Estimated time:** 30 minutes

---

## Image Variants (After Implementation)

Once fully public, three image variants will be available:

| Image | Purpose | Size | Build Time | Best For |
|-------|---------|------|------------|----------|
| `latest` | Basic image | ~500MB | Fast | Self-hosted, Docker users |
| `latest-full` | Full with tt-metal | ~3GB | Slow | Quick start, all features |
| `latest-koyeb` | Koyeb-optimized | ~500MB build<br>~10min startup | Fast build | **Koyeb deployments** ✅ |

### Deploy Button Options

After making public:

```markdown
## Deploy to Koyeb

Choose your deployment method:

### Quick Start (Recommended)
[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=docker&image=ghcr.io/tenstorrent/tt-vscode-toolkit:latest-koyeb&name=tt-vscode-toolkit&ports=8080;http;/&env[PASSWORD]=changeme)

**Koyeb-optimized** - Fast build, tt-metal installs on startup

### Production Ready
[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=docker&image=ghcr.io/tenstorrent/tt-vscode-toolkit:latest-full&name=tt-vscode-toolkit&ports=8080;http;/&env[PASSWORD]=changeme)

**Full image** - Everything pre-installed, larger download

### From Source
[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=git&builder=dockerfile&repository=github.com/tenstorrent/tt-vscode-toolkit&branch=main&name=tt-vscode-toolkit)

**Git-based** - Always latest, builds from source
```

---

## Benefits of Public Images

### For Users
- ✅ True one-click deployment (no auth required)
- ✅ Faster deployment (no build step with Docker images)
- ✅ Version control (can deploy specific versions)
- ✅ Transparent and trustworthy (source code visible)

### For Project
- ✅ Lower barrier to entry (anyone can try it)
- ✅ Better discoverability (public packages in search)
- ✅ Community growth (easier for contributors)
- ✅ Professional presentation (polished deployment)

### For Tenstorrent
- ✅ Showcase N300 hardware capabilities
- ✅ Easy demo for potential customers
- ✅ Developer community building
- ✅ Open-source ecosystem growth

---

## Security Considerations

### What's Safe to Make Public

✅ **Safe:**
- Base Docker images (Ubuntu, code-server)
- VSCode extension (open source)
- tt-smi (publicly available)
- Documentation and examples

✅ **Already Public:**
- Source code (GitHub repository)
- Dockerfiles (in repo)
- Deployment scripts (in repo)

### What to Keep Private

⚠️ **Never include in public images:**
- API keys or tokens
- Passwords (users set their own)
- Private keys
- Internal URLs or credentials

✅ **Current images are safe:**
- All secrets passed as environment variables at runtime
- No hardcoded credentials
- Users provide their own PASSWORD

### Image Scanning

GitHub automatically scans public images for vulnerabilities:
- Security advisories appear in package page
- Dependabot alerts for base image issues
- Regular security updates recommended

---

## Troubleshooting

### Package Not Visible After Publishing

**Problem:** Image builds successfully but doesn't appear in packages

**Solution:**
1. Check package visibility setting
2. Wait a few minutes (can take time to appear)
3. Verify organization allows public packages
4. Check workflow permissions in GitHub Actions

### Deploy Button Authentication Error

**Problem:** Deploy button asks for authentication

**Solution:**
1. Verify package is marked as "Public"
2. Test with: `docker pull ghcr.io/.../image:latest` (no login)
3. Check repository visibility
4. Ensure package inherits repository permissions

### Image Pull Rate Limits

**Problem:** Too many pulls, hitting rate limits

**Solution:**
1. GitHub's rate limits are generous for public packages
2. Authenticated pulls have higher limits
3. Consider caching in CI/CD pipelines
4. Monitor usage in package insights

---

## Resources

**GitHub Documentation:**
- [Working with Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Making Packages Public](https://github.com/orgs/community/discussions/26014)

**Koyeb Documentation:**
- [Deploy to Koyeb Button](https://www.koyeb.com/docs/build-and-deploy/deploy-to-koyeb-button)
- [Docker Deployments](https://www.koyeb.com/docs/build-and-deploy/build-from-docker-image)

**Our Documentation:**
- [Deploy Button Guide](./DEPLOY_BUTTON_GUIDE.md)
- [Container Registry Guide](./CONTAINER_REGISTRY_GUIDE.md)

---

## Next Steps

1. **Make Package Public** (5 min)
   - Organization admin enables public packages
   - Change package visibility to public

2. **Add Koyeb Image Build** (15 min)
   - Update GitHub Actions workflows
   - Test build locally first

3. **Test Deployment** (30 min)
   - Anonymous image pull
   - Deploy button test
   - Full deployment to Koyeb

4. **Update Documentation** (10 min)
   - Add new deploy button option
   - Update README with public images

**Total time:** ~1 hour to fully public deployment

---

## Checklist

### Pre-Flight
- [ ] Repository is public
- [ ] Source code reviewed (no secrets)
- [ ] Dockerfiles reviewed (no credentials)
- [ ] Organization ready for public packages

### Configuration
- [ ] Organization settings: Enable public packages
- [ ] Package settings: Change to public
- [ ] Package settings: Inherit repository permissions
- [ ] GitHub Actions: Add Koyeb image build

### Testing
- [ ] Anonymous Docker pull works
- [ ] Deploy button works in incognito
- [ ] Koyeb deployment succeeds
- [ ] All image variants available

### Documentation
- [ ] README updated with deploy buttons
- [ ] Release notes include image info
- [ ] User guide updated
- [ ] Troubleshooting guide complete

### Launch
- [ ] Announce public availability
- [ ] Monitor package downloads
- [ ] Track deployment issues
- [ ] Gather user feedback
