
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
        github_comment = f"""## 🤖 Claude Code Review

{review}

---
*Automated review by Claude AI • [Learn more](https://claude.ai)*
"""

        return github_comment

    except Exception as e:
        print(f"Error calling Claude API: {e}")
        import traceback
        traceback.print_exc()
        return f"""## ⚠️ Code Review Error

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
