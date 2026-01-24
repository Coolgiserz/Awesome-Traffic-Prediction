# AGENTS.md

This file contains guidelines for agentic coding assistants working in the Awesome Traffic Prediction repository.

## Project Overview

This is a **documentation-only repository** - an "Awesome List" curated collection of resources for traffic prediction research. The project contains:
- Academic papers (2015-2022) on traffic prediction
- Dataset descriptions and documentation  
- Toolkit and library references
- Research groups and conferences
- Related repositories and resources

**Important**: This repository contains NO executable code, build processes, or traditional software development infrastructure.

## Repository Structure

```
Awesome-Traffic-Prediction/
├── README.md              # Main documentation with extensive paper list
├── CONTRIBUTION.md         # Basic contribution guidelines  
├── AGENTS.md              # This file
├── .github/
│   └── FUNDING.yml        # GitHub Sponsors configuration
├── datasets/              # Dataset documentation files
│   ├── METR.md           # Los Angeles traffic dataset
│   ├── PEMS.md           # Bay Area traffic dataset
│   ├── NYC-Bike.md       # (empty placeholder)
│   ├── NYC-Taxi.md       # (empty placeholder)
│   ├── images/           # Contains dataset images
│   └── resources/        # Contains dataset files
├── models/               # Model documentation (mostly empty)
└── papers/              # PDF research papers (5 files)
```

## Build/Test/Lint Commands

**NONE** - This is a documentation repository with no build system, tests, or linting tools.

## Documentation Style Guidelines

### File Organization
- Use descriptive filenames in English (e.g., `METR.md`, `PEMS.md`)
- Group related content in appropriate directories (`datasets/`, `models/`, `papers/`)
- Maintain the established "Awesome List" categorization structure

### Markdown Formatting
- Use standard GitHub-flavored markdown
- Follow the existing hierarchical structure (0x00, 0x01, etc. for main sections)
- Use consistent heading levels (`#`, `##`, `###`)
- Include proper link formatting: `[text](url)` and `[[paper]](url)` for academic papers

### Content Standards
- **Bilingual support**: The repository contains both English and Chinese content. Maintain this mix where appropriate.
- **Paper citations**: Use the format `[Conference Year] Paper Title [[paper]](link) [[code]](link)`
- **Dataset descriptions**: Include both English descriptions and Chinese explanations where present
- **Links**: Ensure all external links are valid and properly formatted

### Naming Conventions
- **Files**: Use kebab-case for markdown files (e.g., `nyc-taxi.md`)
- **Sections**: Use the established 0x00, 0x01 numbering scheme for main sections
- **References**: Maintain consistent citation format across all entries

### Content Addition Guidelines
1. **Papers**: Add to appropriate year subsection, maintain chronological order
2. **Datasets**: Create dedicated .md files in `datasets/` directory following existing pattern
3. **Code links**: When adding papers, include official code repositories when available
4. **Descriptions**: Provide brief, informative descriptions for all entries

### Link Maintenance
- Regularly verify external links are accessible
- Update broken links or mark as unavailable
- Ensure paper links point to accessible sources (arXiv, conference proceedings, etc.)

### Version Control
- Follow semantic versioning as mentioned in CONTRIBUTION.md
- Update version numbers in README.md badges when making significant changes
- Use descriptive commit messages following established patterns

### Quality Standards
- Verify academic paper citations are accurate (conference, year, title)
- Ensure dataset descriptions are complete and helpful
- Maintain consistent formatting throughout the document
- Check for duplicate entries before adding new content

## Contribution Process

1. **Discuss changes**: Open an issue before making significant changes
2. **Update documentation**: Modify README.md or relevant .md files
3. **Version updates**: Update version numbers in README.md badges
4. **Pull request**: Follow the process outlined in CONTRIBUTION.md

## Tools and Dependencies

**NONE** - This repository requires no special tools, dependencies, or development environment setup. All content is standard markdown that can be edited with any text editor.

## Common Tasks

- **Adding papers**: Update the appropriate year section in README.md
- **Adding datasets**: Create new .md file in `datasets/` directory and update README.md
- **Updating links**: Edit existing entries to fix broken or outdated links
- **Reorganizing content**: Maintain the established section structure and numbering

## Important Notes

- This is NOT a code repository - do not add executable code, scripts, or build files
- Maintain the academic research focus and professional tone
- Preserve the bilingual (English/Chinese) content mix where established
- All content should be relevant to traffic prediction research