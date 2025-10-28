#!/usr/bin/env python3
"""
Claude Code Review Script
Analyzes PR changes and provides comprehensive code review using Claude AI.
"""

import os
import sys
import subprocess
import json
import time
try:
    from anthropic import Anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False
import requests

def get_pr_diff():
    """Get the diff for the PR."""
    try:
        with open('pr_diff.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        print("Error: pr_diff.txt not found")
        return None

def get_changed_files():
    """Get list of changed files in the PR."""
    base_ref = os.getenv('GITHUB_BASE_REF', 'main')
    head_sha = os.getenv('GITHUB_SHA', 'HEAD')

    result = subprocess.run(
        ['git', 'diff', '--name-only', f'origin/{base_ref}...{head_sha}'],
        capture_output=True,
        text=True
    )

    return result.stdout.strip().split('\n') if result.stdout else []

def read_file_content(filepath):
    """Read content of a file if it exists."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except (FileNotFoundError, UnicodeDecodeError, IsADirectoryError):
        return None

def generate_review(diff, changed_files, pr_title, pr_body):
    """Generate code review using Claude."""

    # Check for Claude session tokens (preferred for Claude Max accounts)
    access_token = os.getenv('CLAUDE_ACCESS_TOKEN')

    # Fall back to Anthropic API key if session tokens not available
    api_key = os.getenv('ANTHROPIC_API_KEY')

    if not access_token and not api_key:
        print("Error: Neither CLAUDE_ACCESS_TOKEN nor ANTHROPIC_API_KEY is set")
        sys.exit(1)

    use_session_auth = bool(access_token)

    # Prepare file contents for context
    file_contents = []
    for filepath in changed_files[:10]:  # Limit to 10 files to avoid token limits
        content = read_file_content(filepath)
        if content:
            file_contents.append(f"### File: {filepath}\n```\n{content[:5000]}\n```\n")

    files_context = "\n".join(file_contents) if file_contents else "No file contents available."

    prompt = f"""You are a senior software engineer conducting a thorough code review. Analyze this Pull Request and provide constructive feedback.

**PR Title:** {pr_title}

**PR Description:**
{pr_body or 'No description provided'}

**Changed Files:**
{', '.join(changed_files)}

**Diff:**
```diff
{diff[:15000]}  # Limit diff size
```

**File Contents (for context):**
{files_context}

Please provide a comprehensive code review covering:

1. **Code Quality**: Architecture, design patterns, code organization
2. **Security**: Potential vulnerabilities, data handling, authentication/authorization
3. **Performance**: Bottlenecks, optimization opportunities, scalability concerns
4. **Best Practices**: Following Python/JavaScript conventions, error handling, logging
5. **Testing**: Test coverage, edge cases, test quality
6. **Documentation**: Code comments, docstrings, README updates
7. **Bugs & Edge Cases**: Potential issues, error scenarios
8. **Suggestions**: Specific improvements with code examples where helpful

Format your review as follows:
- Use CHECK for things done well
- Use WARNING for warnings/concerns
- Use X for critical issues
- Use LIGHTBULB for suggestions

Be constructive, specific, and provide actionable feedback. If the code looks good, say so!
"""

    try:
        if use_session_auth:
            # Use Claude.ai web API with session token
            print("Using Claude session authentication (Claude Max account)")

            # Create a new conversation
            create_response = requests.post(
                'https://claude.ai/api/organizations',
                headers={
                    'Cookie': f'sessionKey={access_token}',
                    'Content-Type': 'application/json'
                },
                timeout=30
            )

            if create_response.status_code != 200:
                print(f"Failed to get organization: {create_response.status_code}")
                return f"""## WARNING Code Review Error

Failed to authenticate with Claude using session token.
Status: {create_response.status_code}

Please check that your CLAUDE_ACCESS_TOKEN is valid and not expired.
You may need to refresh your token from https://claude.ai/
"""

            orgs = create_response.json()
            if not orgs:
                return """## WARNING Code Review Error

No organizations found for this Claude account.
"""

            org_uuid = orgs[0]['uuid']

            # Create conversation
            conv_response = requests.post(
                f'https://claude.ai/api/organizations/{org_uuid}/chat_conversations',
                headers={
                    'Cookie': f'sessionKey={access_token}',
                    'Content-Type': 'application/json'
                },
                json={'name': 'Code Review', 'uuid': str(time.time())},
                timeout=30
            )

            if conv_response.status_code != 201:
                print(f"Failed to create conversation: {conv_response.status_code}")
                return f"""## WARNING Code Review Error

Failed to create conversation.
Status: {conv_response.status_code}
"""

            conv_uuid = conv_response.json()['uuid']

            # Send message
            message_response = requests.post(
                f'https://claude.ai/api/organizations/{org_uuid}/chat_conversations/{conv_uuid}/completion',
                headers={
                    'Cookie': f'sessionKey={access_token}',
                    'Content-Type': 'application/json'
                },
                json={
                    'prompt': prompt,
                    'model': 'claude-sonnet-4-20250514',
                    'timezone': 'UTC'
                },
                timeout=120,
                stream=True
            )

            if message_response.status_code != 200:
                print(f"Claude API error: {message_response.status_code} - {message_response.text}")
                return f"""## WARNING Code Review Error

Failed to generate review using Claude session authentication.
Status: {message_response.status_code}

Please check that your CLAUDE_ACCESS_TOKEN is valid and not expired.
"""

            # Parse streaming response
            review = ""
            for line in message_response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        try:
                            data = json.loads(line_text[6:])
                            if 'completion' in data:
                                review = data['completion']
                        except json.JSONDecodeError:
                            continue
        else:
            # Use Anthropic SDK with API key
            print("Using Anthropic API key authentication")
            if not ANTHROPIC_SDK_AVAILABLE:
                return """## WARNING Code Review Error

Anthropic SDK not available. Please install it with: pip install anthropic
"""
            client = Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            review = message.content[0].text

        # Format the review as a GitHub comment
        github_comment = f"""## ROBOT Claude Code Review

{review}

---
*Automated review by Claude AI - [Learn more](https://claude.ai)*
"""

        return github_comment

    except Exception as e:
        print(f"Error calling Claude API: {e}")
        import traceback
        traceback.print_exc()
        return f"""## WARNING Code Review Error

An error occurred while generating the code review:
```
{str(e)}
```

Please check the workflow logs for more details.
"""

def main():
    """Main function to orchestrate the code review."""

    pr_title = os.getenv('PR_TITLE', 'No title')
    pr_body = os.getenv('PR_BODY', '')

    # Get PR diff
    diff = get_pr_diff()
    if not diff:
        print("No diff found, skipping review")
        with open('review_comment.md', 'w') as f:
            f.write("No changes detected in this PR.")
        return

    # Get changed files
    changed_files = get_changed_files()

    print(f"Reviewing PR: {pr_title}")
    print(f"Changed files: {len(changed_files)}")

    # Generate review
    review = generate_review(diff, changed_files, pr_title, pr_body)

    # Write review to file for GitHub Actions to post
    with open('review_comment.md', 'w') as f:
        f.write(review)

    print("Review generated successfully")

if __name__ == '__main__':
    main()
