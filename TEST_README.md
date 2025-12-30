# Link Validation Test

This script validates all links in markdown files across the repository.

## Purpose

The `test_markdown_links.py` script ensures that:
- All relative file paths referenced in markdown files exist
- Image links point to valid files
- External URLs are documented (accessibility checked if network available)

## Usage

Run the test from the repository root:

```bash
python test_markdown_links.py
```

## Exit Codes

- **0**: All relative/local links are valid
- **1**: One or more relative/local links are broken

## Output

The test provides:
- List of all markdown files scanned
- Status of each link (✓ OK or ✗ BROKEN)
- Summary of total/broken links
- External link accessibility check (when network available)
- Detailed information about any broken links

## CI Integration

This test can be added to CI pipelines to prevent broken links from being merged:

```yaml
- name: Validate Markdown Links
  run: python test_markdown_links.py
```

## Notes

- External links (http/https) are checked but won't fail the test if inaccessible
- This is intentional as external sites may be temporarily down or require authentication
- The test focuses on ensuring repository-internal links are valid
