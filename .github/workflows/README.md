# CI/CD Workflows

This directory contains GitHub Actions workflows for automated testing, building, and deployment.

## Workflows

### 1. CI (Continuous Integration) - `ci.yml`

Runs on every push and pull request to `main` and `develop` branches.

**Jobs:**
- **Test**: Runs tests across Python 3.9, 3.10, and 3.11
  - Installs PyTorch CPU version
  - Runs pytest with coverage
  - Uploads coverage to Codecov
- **Lint**: Checks code quality
  - Runs pre-commit hooks (black, isort, flake8, mypy)

**Status:** Automatically triggered on push/PR

### 2. Docker Build and Push - `docker.yml`

Builds and pushes Docker images to GitHub Container Registry (and optionally DockerHub).

**Triggers:**
- Push to `main` branch
- Git tags (e.g., `v1.0.0`)
- Manual trigger via workflow_dispatch

**Features:**
- Multi-registry support (GHCR + DockerHub)
- Automatic semantic versioning from git tags
- Docker layer caching for faster builds
- SHA-based tags for traceability

**Image Tags Generated:**
- `latest` (on main branch)
- `v1.2.3` (on version tags)
- `main-abc1234` (SHA-based)

**Status:** Automatically triggered on main branch push

### 3. Deploy to AWS EC2 - `deploy.yml`

Manually triggered deployment to AWS EC2 K3s cluster.

**Trigger:** Manual via GitHub Actions UI

**Inputs:**
- **environment**: staging or production
- **image_tag**: Docker image tag to deploy (default: latest)

**Steps:**
1. Configures AWS credentials
2. SSH into EC2 instance
3. Pulls specified Docker image
4. Updates K3s deployments
5. Waits for rollout completion
6. Runs health check

**Status:** Manual trigger only (safe deployment)

## Setup Instructions

### Prerequisites

1. **GitHub Repository Secrets**

Navigate to: `Settings → Secrets and variables → Actions → New repository secret`

Add the following secrets:

#### For CI (Optional)
```
CODECOV_TOKEN          # Get from codecov.io after signing up
```

#### For Docker (Optional - only if using DockerHub)
```
DOCKERHUB_USERNAME     # Your DockerHub username
DOCKERHUB_TOKEN        # DockerHub access token (not password)
```

#### For Deployment
```
AWS_ACCESS_KEY_ID      # AWS IAM user access key
AWS_SECRET_ACCESS_KEY  # AWS IAM user secret key
AWS_REGION             # e.g., us-east-1

EC2_HOST              # EC2 public IP or hostname
EC2_USER              # EC2 username (usually 'ubuntu')
EC2_SSH_KEY           # Private SSH key for EC2 access (full content)
```

### Setting Up Secrets

#### 1. Codecov (for coverage reports)

```bash
# Sign up at codecov.io
# Connect your GitHub repository
# Copy the token and add to GitHub secrets
```

#### 2. DockerHub (optional)

```bash
# Create access token at hub.docker.com → Account Settings → Security
# Add username and token to GitHub secrets
```

#### 3. AWS Credentials

```bash
# Create IAM user with EC2 access
# Generate access key
# Add credentials to GitHub secrets
```

#### 4. EC2 SSH Key

```bash
# Copy your private key content
cat ~/.ssh/your-ec2-key.pem

# Paste entire content (including BEGIN/END lines) into EC2_SSH_KEY secret
```

## Usage

### Running CI Tests Locally

Before pushing, test locally:

```bash
# Run tests
pytest tests/ -v --cov=src/sentiment_analyzer

# Run linting
pre-commit run --all-files
```

### Building Docker Image Locally

```bash
# Build image
docker build -t sentiment-analyzer:local .

# Test image
docker run -p 8000:8000 sentiment-analyzer:local
```

### Triggering Manual Deployment

1. Go to: `Actions → Deploy to AWS EC2 → Run workflow`
2. Select branch (usually `main`)
3. Choose environment: `staging` or `production`
4. Enter image tag: `latest`, `v1.0.0`, or specific SHA
5. Click "Run workflow"

### Monitoring Workflows

Check workflow status:
- **Repository Actions tab**: See all workflow runs
- **Pull Request checks**: See CI status before merge
- **README badges**: Quick status overview

## Workflow Permissions

The workflows use the following permissions:

- `GITHUB_TOKEN`: Automatically provided, used for:
  - Pushing to GitHub Container Registry
  - Posting check results
- Custom secrets: Manually configured (see Setup above)

## Troubleshooting

### CI Tests Failing

```bash
# Check test output in Actions tab
# Run locally to reproduce:
pytest tests/ -v
```

### Docker Build Failing

```bash
# Check if Dockerfile builds locally:
docker build -t test .
```

### Deployment Failing

**Common issues:**

1. **SSH Connection Failed**
   - Verify `EC2_HOST` is correct and accessible
   - Check `EC2_SSH_KEY` is complete (includes headers)
   - Verify security group allows SSH from GitHub Actions IPs

2. **Docker Pull Failed**
   - Ensure image exists in registry
   - Check image tag is correct
   - Verify EC2 has access to GHCR

3. **K3s Rollout Failed**
   - SSH into EC2: `ssh -i key.pem ubuntu@<EC2_IP>`
   - Check pod logs: `kubectl logs -n sentiment-analysis <pod-name>`
   - Check pod status: `kubectl describe pod -n sentiment-analysis <pod-name>`

4. **Health Check Failed**
   - Verify API is running: `kubectl get pods -n sentiment-analysis`
   - Check service: `curl http://localhost:30800/health`
   - Review pod logs for errors

## Best Practices

1. **Always run tests locally** before pushing
2. **Use descriptive commit messages** (triggers appear in Actions)
3. **Create tags for releases**: `git tag v1.0.0 && git push --tags`
4. **Monitor Actions tab** for workflow failures
5. **Review security**: Rotate secrets periodically
6. **Use staging** before production deployments

## GitHub Environments (Optional)

For additional safety, configure environments:

1. Go to: `Settings → Environments → New environment`
2. Create `staging` and `production` environments
3. Add protection rules:
   - Required reviewers for production
   - Wait timer before deployment
   - Environment-specific secrets

## Future Improvements

- Add automated integration tests post-deployment
- Implement blue-green deployment strategy
- Add Slack/email notifications on failures
- Implement automatic rollback on health check failure
- Add performance testing in CI pipeline
