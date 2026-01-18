# Git Setup Instructions

Follow these steps to push your project to a new Git repository.

## Prerequisites
- You have created a new repository on GitHub/GitLab/Bitbucket (e.g., `HomeAnomalyDetection`)
- You have the repository URL (e.g., `https://github.com/yourusername/HomeAnomalyDetection.git`)

## Step-by-Step Instructions

### Step 1: Initialize Git Repository
```bash
cd /Users/sukhsagarshukla/Documents/Sukh/projects/HomeAnomalyDetection
git init
```

### Step 2: Add All Files
```bash
git add .
```

### Step 3: Make Initial Commit
```bash
git commit -m "Initial commit: Hotel Booking Anomaly Detection System"
```

### Step 4: Add Remote Repository
Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual values:

**For GitHub:**
```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

**For GitLab:**
```bash
git remote add origin https://gitlab.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

**For Bitbucket:**
```bash
git remote add origin https://bitbucket.org/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### Step 5: Push to Remote Repository
```bash
git branch -M main
git push -u origin main
```

## Alternative: If Your Repository Uses 'master' Branch
If your repository uses `master` instead of `main`:
```bash
git branch -M master
git push -u origin master
```

## Complete Command Sequence (Copy-Paste Ready)

Replace `YOUR_REPO_URL` with your actual repository URL:

```bash
cd /Users/sukhsagarshukla/Documents/Sukh/projects/HomeAnomalyDetection
git init
git add .
git commit -m "Initial commit: Hotel Booking Anomaly Detection System"
git remote add origin YOUR_REPO_URL
git branch -M main
git push -u origin main
```

## Troubleshooting

### If you get "remote origin already exists"
```bash
git remote remove origin
git remote add origin YOUR_REPO_URL
```

### If you get authentication errors
- **GitHub**: Use Personal Access Token instead of password
- **GitLab**: Use Personal Access Token
- **Bitbucket**: Use App Password

### If you need to update remote URL
```bash
git remote set-url origin YOUR_NEW_REPO_URL
```

## Next Steps After Pushing

1. **Verify**: Check your repository on GitHub/GitLab to see all files
2. **Add README**: Update README.md with project description
3. **Add License**: Add LICENSE file if needed
4. **Set Branch Protection**: Configure branch protection rules if working in a team

## Future Updates

After initial push, use these commands for future updates:

```bash
git add .
git commit -m "Your commit message"
git push
```
