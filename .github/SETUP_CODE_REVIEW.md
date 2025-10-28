# Claude Code Review Setup Guide

This repository is configured to automatically review Pull Requests using Claude AI.

## How It Works

When a PR is opened, synchronized, or reopened, the GitHub Actions workflow:
1. Fetches the PR diff and changed files
2. Sends the code changes to Claude AI for review
3. Posts a comprehensive review comment on the PR

## Setup Instructions

### 1. Add Anthropic API Key to GitHub Secrets

You need to add your Anthropic API key as a GitHub repository secret:

1. Go to your repository settings on GitHub
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `ANTHROPIC_API_KEY`
5. Value: Your Anthropic API key (get one from https://console.anthropic.com/)
6. Click **Add secret**

### 2. Get an Anthropic API Key

If you don't have an API key:
1. Visit https://console.anthropic.com/
2. Sign up or log in
3. Go to **API Keys** section
4. Create a new API key
5. Copy the key (you won't be able to see it again!)

### 3. Enable GitHub Actions

Ensure GitHub Actions are enabled for your repository:
1. Go to **Settings** → **Actions** → **General**
2. Under "Actions permissions", ensure actions are allowed
3. Under "Workflow permissions", ensure "Read and write permissions" is selected

## What Gets Reviewed

Claude analyzes:
- ✅ **Code Quality**: Architecture, design patterns, organization
- 🔒 **Security**: Vulnerabilities, data handling, auth/authz
- ⚡ **Performance**: Bottlenecks, optimization opportunities
- 📚 **Best Practices**: Conventions, error handling, logging
- 🧪 **Testing**: Coverage, edge cases, test quality
- 📖 **Documentation**: Comments, docstrings, README updates
- 🐛 **Bugs & Edge Cases**: Potential issues, error scenarios
- 💡 **Suggestions**: Specific improvements with examples

## Customization

### Modify Review Criteria

Edit [.github/scripts/claude_review.py](.github/scripts/claude_review.py) and update the prompt to focus on specific aspects of your codebase.

### Change Claude Model

In `claude_review.py`, line with `model=`, you can change to:
- `claude-sonnet-4-20250514` (default - balanced speed/quality)
- `claude-opus-4-20250514` (most capable, slower)
- `claude-haiku-4-20250514` (fastest, lighter reviews)

### Adjust Token Limits

In `claude_review.py`, modify:
- `max_tokens=4000` - Maximum length of review
- `diff[:15000]` - Amount of diff to analyze
- `content[:5000]` - Amount of file content per file
- `[:10]` - Number of files to include for context

## Testing the Workflow

Create a test PR to verify the setup:

```bash
git checkout -b test-claude-review
echo "# Test" >> TEST.md
git add TEST.md
git commit -m "Test Claude code review"
git push origin test-claude-review
gh pr create --title "Test: Claude Code Review" --body "Testing automated code review"
```

Then check:
1. GitHub Actions tab to see if the workflow runs
2. The PR for the review comment from Claude

## Troubleshooting

### Workflow doesn't run
- Check that GitHub Actions are enabled
- Verify the workflow file is in `.github/workflows/`
- Ensure the PR targets the main branch

### "ANTHROPIC_API_KEY not set" error
- Add the API key to GitHub Secrets (see step 1 above)
- Ensure the secret name is exactly `ANTHROPIC_API_KEY`

### API rate limits
- Claude has rate limits based on your plan
- Consider adding delays or reducing review frequency if needed
- Monitor usage at https://console.anthropic.com/

### Review comment not posted
- Check workflow logs in the Actions tab
- Ensure repository has "Read and write" workflow permissions
- Verify `GITHUB_TOKEN` permissions in workflow file

## Cost Considerations

- Each review costs based on tokens used
- Typical PR review: ~$0.05 - $0.20 per review
- Monitor usage in Anthropic Console
- Consider limiting to specific file types or PR sizes

## Disabling Reviews

To temporarily disable:
1. Go to `.github/workflows/claude-code-review.yml`
2. Comment out the `on:` section or the entire file
3. Or delete the workflow file

## Support

For issues:
- Check GitHub Actions logs
- Review Anthropic API status: https://status.anthropic.com/
- File an issue in this repository

---

**Note**: This uses Claude Sonnet 4.5, Anthropic's most capable model. Reviews are thorough, constructive, and provide actionable feedback.
