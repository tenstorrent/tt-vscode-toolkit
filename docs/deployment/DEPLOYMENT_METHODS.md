# Koyeb Deployment Methods

There are **3 ways** to deploy to Koyeb. Choose based on your needs:

## Quick Comparison

| Method | Registry Login? | Build Location | Speed | Use Case |
|--------|----------------|----------------|-------|----------|
| **Direct Deploy** | ❌ No | Koyeb cloud | Fast | **Testing/Dev** |
| **Registry Deploy** | ✅ Yes | Local | Slower | Production, CI/CD |
| **UI Deploy** | ✅ Yes | Local | Slowest | Manual, one-off |

---

## Method 1: Direct Deploy ⭐ (Best for Testing)

**No registry login needed!** Koyeb builds the image for you.

```bash
./koyeb-deploy-direct.sh
```

**How it works:**
1. Builds extension locally
2. Uploads directory to Koyeb
3. Koyeb builds Docker image remotely
4. Koyeb deploys automatically

**Pros:**
- ✅ No registry authentication needed
- ✅ No manual push to registry
- ✅ Fastest for testing
- ✅ Simple workflow

**Cons:**
- ⚠️ Slower first build (Koyeb builds remotely)
- ⚠️ No image caching locally
- ⚠️ Uploads entire directory each time

**Perfect for:**
- Quick testing
- Development iterations
- When you don't want to set up registry access

---

## Method 2: Registry Deploy (Production)

**Registry login required.** You build and push, Koyeb pulls.

```bash
# One-time setup
./koyeb-registry-login.sh

# Then deploy
./quick-deploy-koyeb.sh
```

**How it works:**
1. Builds extension locally
2. Builds Docker image locally
3. Pushes to registry.koyeb.com
4. Koyeb pulls and deploys

**Pros:**
- ✅ Image caching (faster rebuilds)
- ✅ Can use in CI/CD pipelines
- ✅ Version control of images
- ✅ Can deploy same image multiple times

**Cons:**
- ⚠️ Requires registry authentication
- ⚠️ More steps
- ⚠️ Need to manage API tokens

**Perfect for:**
- Production deployments
- CI/CD pipelines
- When you want image versioning
- Multiple deployments of same image

---

## Method 3: UI Deploy

**Manual deployment through Koyeb dashboard.**

1. Build and push image to registry
2. Go to https://app.koyeb.com
3. Click "Create Service"
4. Enter image URL
5. Configure options
6. Deploy

**Pros:**
- ✅ Visual interface
- ✅ See all options clearly
- ✅ Good for learning

**Cons:**
- ⚠️ Slowest method
- ⚠️ Manual steps
- ⚠️ Not scriptable
- ⚠️ Need to remember settings

**Perfect for:**
- First-time users
- Visual learners
- One-off deployments

---

## Detailed Workflows

### Direct Deploy (No Registry)

```bash
# Setup (one-time)
curl -fsSL https://cli.koyeb.com/install.sh | sh
koyeb login
export KOYEB_ORG=your-org

# Deploy (every time)
./koyeb-deploy-direct.sh

# That's it! No registry needed!
```

**Output:**
```
🚀 Koyeb Direct Deploy (No Registry!)

App:     tt-vscode-toolkit
Service: vscode
Pass:    abc123xyz456

1/2 Building extension...
2/2 Deploying to Koyeb...
   (Building and deploying in one step)

✅ Deployed!

🌐 https://vscode-yourorg.koyeb.app
🔑 abc123xyz456
```

---

### Registry Deploy (With Caching)

```bash
# Setup (one-time)
curl -fsSL https://cli.koyeb.com/install.sh | sh
koyeb login
export KOYEB_ORG=your-org

# Registry login (one-time)
./koyeb-registry-login.sh
# Or: podman login registry.koyeb.com

# Deploy (every time)
./quick-deploy-koyeb.sh
```

**Output:**
```
🚀 Quick Deploy to Koyeb

Service: vscode
Image:   registry.koyeb.com/yourorg/tt-vscode-toolkit:latest
Pass:    abc123xyz456

1/4 Building extension...
2/4 Building container...
3/4 Pushing to registry...
4/4 Deploying to Koyeb...

✅ Deployed!

🌐 https://vscode-yourorg.koyeb.app
🔑 abc123xyz456
```

---

## When to Use Each Method

### Use Direct Deploy if:
- ✅ You're testing/developing
- ✅ You want quick iterations
- ✅ You don't want to set up registry access
- ✅ You're doing one-off deployments

### Use Registry Deploy if:
- ✅ You're deploying to production
- ✅ You have CI/CD pipelines
- ✅ You want to deploy the same image multiple times
- ✅ You need version control of images
- ✅ You want faster rebuilds (caching)

### Use UI Deploy if:
- ✅ You're learning Koyeb
- ✅ You want to see all options visually
- ✅ You're doing a one-time manual deployment

---

## Switching Between Methods

You can use both methods! They're not mutually exclusive:

```bash
# Test with direct deploy
./koyeb-deploy-direct.sh

# Once happy, deploy to production with registry
./koyeb-registry-login.sh
./quick-deploy-koyeb.sh
```

Or:

```bash
# Develop with direct deploy
./koyeb-deploy-direct.sh

# Set up CI/CD with registry method
# (in GitHub Actions, etc.)
```

---

## Cost Considerations

Both methods cost the same - Koyeb charges for:
- Compute (instance type)
- Hardware (N300 accelerator if requested)
- Network bandwidth

The deployment method doesn't affect cost.

---

## Speed Comparison

**First deployment:**
- Direct: ~5-8 minutes (uploads + remote build)
- Registry: ~5-10 minutes (local build + push + deploy)

**Subsequent deployments:**
- Direct: ~5-8 minutes (always rebuilds remotely)
- Registry: ~3-5 minutes (uses cached layers)

**Winner:** Direct for first time, Registry for iterations

---

## Storage Considerations

**Direct Deploy:**
- Koyeb stores build cache remotely
- No local storage used (except source code)

**Registry Deploy:**
- Images stored in Koyeb registry
- Counts toward registry quota
- Can manage/delete old images

---

## CI/CD Integration

### Direct Deploy in CI/CD

```yaml
# GitHub Actions
- name: Deploy to Koyeb
  run: |
    koyeb deploy . default/tt-vscode \
      --archive-builder docker \
      --archive-docker-dockerfile Dockerfile.koyeb
```

### Registry Deploy in CI/CD

```yaml
# GitHub Actions
- name: Build and Push
  run: |
    podman build -t registry.koyeb.com/org/tt-vscode:${{ github.sha }} .
    podman push registry.koyeb.com/org/tt-vscode:${{ github.sha }}

- name: Deploy
  run: |
    koyeb services update tt-vscode \
      --docker-image registry.koyeb.com/org/tt-vscode:${{ github.sha }}
```

---

## Summary

**For Testing/Development:**
→ Use `./koyeb-deploy-direct.sh` (no registry needed!)

**For Production:**
→ Use `./quick-deploy-koyeb.sh` (with registry)

**For Learning:**
→ Use Koyeb dashboard UI

---

## Quick Reference Commands

```bash
# Direct deploy (testing)
./koyeb-deploy-direct.sh

# Registry deploy (production)
./koyeb-registry-login.sh  # one-time
./quick-deploy-koyeb.sh    # every deployment

# Interactive deploy
./deploy-to-koyeb.sh       # full configuration

# Check status
koyeb services get vscode

# Watch logs
koyeb services logs vscode -f

# Delete service
koyeb services delete vscode
```

---

**Start with direct deploy for testing, then move to registry deploy for production!** 🚀
