#!/usr/bin/env python3
"""
Test script to verify all links in markdown files are valid.
Tests both relative file paths and external URLs.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple
import urllib.request
import urllib.error

# Repository root
REPO_ROOT = Path(__file__).parent.absolute()


def find_markdown_files() -> List[Path]:
    """Find all markdown files in the repository."""
    markdown_files = []
    for root, dirs, files in os.walk(REPO_ROOT):
        # Skip hidden directories and common ignore patterns
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__']]
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(Path(root) / file)
    return markdown_files


def extract_links_from_markdown(file_path: Path) -> List[Tuple[str, str, int]]:
    """
    Extract all links from a markdown file.
    Returns list of tuples: (link_text, link_url, line_number)
    """
    links = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Match markdown links: [text](url)
            pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            matches = re.finditer(pattern, line)
            for match in matches:
                link_text = match.group(1)
                link_url = match.group(2)
                links.append((link_text, link_url, line_num))
            
            # Also match plain image links: ![alt](url)
            img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
            img_matches = re.finditer(img_pattern, line)
            for match in img_matches:
                alt_text = match.group(1)
                img_url = match.group(2)
                links.append((f"Image: {alt_text}", img_url, line_num))
    
    return links


def check_relative_path(base_path: Path, relative_url: str) -> Tuple[bool, str]:
    """
    Check if a relative path exists.
    Returns (exists, absolute_path)
    """
    # Remove anchor links
    url_without_anchor = relative_url.split('#')[0]
    
    if not url_without_anchor:
        return True, "Anchor link only"
    
    # Resolve the path relative to the markdown file's directory
    resolved_path = (base_path.parent / url_without_anchor).resolve()
    
    # Check if path exists
    if resolved_path.exists():
        return True, str(resolved_path)
    else:
        return False, str(resolved_path)


def check_external_url(url: str, timeout: int = 10) -> Tuple[bool, str]:
    """
    Check if an external URL is accessible.
    Returns (accessible, status_message)
    """
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; LinkChecker/1.0)'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, f"URL Error: {e.reason}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def test_all_markdown_links():
    """Test all links in all markdown files."""
    print("="*80)
    print("TESTING MARKDOWN LINKS")
    print("="*80)
    print()
    
    markdown_files = find_markdown_files()
    print(f"Found {len(markdown_files)} markdown file(s)\n")
    
    total_links = 0
    broken_links = []
    external_links = []
    
    for md_file in markdown_files:
        rel_path = md_file.relative_to(REPO_ROOT)
        print(f"\nChecking: {rel_path}")
        print("-" * 80)
        
        links = extract_links_from_markdown(md_file)
        
        if not links:
            print("  No links found")
            continue
        
        for link_text, link_url, line_num in links:
            total_links += 1
            
            # Determine if link is external or relative
            if link_url.startswith(('http://', 'https://')):
                external_links.append((rel_path, link_text, link_url, line_num))
                print(f"  Line {line_num}: [External] {link_text}")
                print(f"           URL: {link_url}")
            else:
                # Check relative path
                exists, resolved = check_relative_path(md_file, link_url)
                status = "✓ OK" if exists else "✗ BROKEN"
                print(f"  Line {line_num}: [{status}] {link_text}")
                print(f"           Path: {link_url}")
                
                if not exists:
                    broken_links.append({
                        'file': str(rel_path),
                        'line': line_num,
                        'text': link_text,
                        'url': link_url,
                        'resolved': resolved
                    })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total links found: {total_links}")
    print(f"Relative/local links: {total_links - len(external_links)}")
    print(f"External links: {len(external_links)}")
    print(f"Broken links: {len(broken_links)}")
    
    if broken_links:
        print("\n" + "="*80)
        print("BROKEN LINKS")
        print("="*80)
        for link in broken_links:
            print(f"\nFile: {link['file']}")
            print(f"Line: {link['line']}")
            print(f"Text: {link['text']}")
            print(f"URL:  {link['url']}")
            print(f"Resolved to: {link['resolved']}")
        print()
    
    if external_links:
        print("\n" + "="*80)
        print("EXTERNAL LINKS (checking accessibility...)")
        print("="*80)
        ext_broken = []
        for rel_path, link_text, link_url, line_num in external_links:
            accessible, status = check_external_url(link_url)
            status_symbol = "✓" if accessible else "✗"
            print(f"{status_symbol} {link_url}")
            print(f"  Status: {status}")
            print(f"  In: {rel_path}:{line_num}")
            if not accessible:
                ext_broken.append((rel_path, link_text, link_url, line_num))
        
        if ext_broken:
            print(f"\n⚠ Warning: {len(ext_broken)} external link(s) not accessible")
            print("Note: External links may be temporarily unavailable or require authentication")
    
    print("\n" + "="*80)
    
    # Exit with error code if broken links found
    if broken_links:
        print("❌ TEST FAILED: Found broken relative/local links")
        return False
    else:
        print("✅ TEST PASSED: All relative/local links are valid")
        return True


if __name__ == "__main__":
    success = test_all_markdown_links()
    sys.exit(0 if success else 1)
